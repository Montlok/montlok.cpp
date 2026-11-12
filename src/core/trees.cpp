// Copyright 2026 Montlok. Apache-2.0.
// The "tree_ensemble" family: gradient-boosted trees and random forests from
// LightGBM, XGBoost and scikit-learn (montlok.export converts them).
//
// Tensors (prefix from config "tensors", default "trees"):
//   root   i32[T]  first node of each tree
//   output i32[T]  output column each tree adds to
//   feature i32[N] split feature, -1 for a leaf
//   threshold f64[N], left i32[N], right i32[N], value f64[N] (leaf value)
//   flags u8[N]    bit 0: missing values go left;
//                  bits 1-2: 0 NaN is treated as 0.0, 1 NaN takes the default
//                  branch, 2 NaN and |x| <= 1e-35 take it (LightGBM "Zero")
// Children always have larger indices than their parent, which every source
// format satisfies and which bounds each walk.
//
// At load time the trees are re-laid out for traversal (Packed): breadth-first
// per tree, the two children of a split adjacent, every decision in the form
// "right when v > threshold". A leaf points to itself, reads the -inf at the
// end of the prepared row and keeps its value in `threshold`, so a walk is a
// fixed number of branch-free steps. One row walks kGroup trees at a time; a
// block of rows walks one tree at a time. Both reach the leaves of the
// reference walk (leaf()), which serves models with infinite thresholds or
// shared nodes.
//
// Sums: each output adds its trees in tree order within blocks of kTreeBlock
// trees, the first block starting from the base score, and then adds the block
// sums in block order. A single row can therefore split its blocks across
// latency threads (run_parts) and still give the same bits.
#include <algorithm>
#include <cmath>
#include <limits>

#include "kernels.hpp"
#include "runtime.hpp"

namespace montlok::core {
namespace {

enum class Link { Identity, Sigmoid, Softmax, Exp };

struct Node {
    double threshold;
    double value;
    int32_t feature;
    int32_t left;
    int32_t right;
    uint8_t flags;
};

// LightGBM's kZeroThreshold is the float literal 1e-35f.
constexpr double kZero = static_cast<double>(1e-35f);

struct Packed {
    double threshold;  // right when v > threshold; a leaf's value
    uint32_t feature;  // index into the prepared row; bit 31: NaN goes right
    uint32_t left;     // left child, followed by the right child; a leaf's own index
};
static_assert(sizeof(Packed) == 16, "Packed is 16 bytes");

constexpr uint32_t kNanRight = 0x80000000u;
constexpr size_t kGroup = 16;      // trees walked together for one row
constexpr size_t kTreeBlock = 64;  // trees per partial sum
constexpr size_t kBlock = 64;      // rows walked together for one tree
constexpr size_t kRowWalk = 8;     // blocks smaller than this walk row by row
constexpr size_t kPartSteps = 384; // fewest walk steps worth a latency thread
static_assert(kTreeBlock % kGroup == 0, "a tree block is whole groups");

class TreeEnsemble final : public FamilyModel {
public:
    explicit TreeEnsemble(const BuildContext& ctx) {
        if (ctx.sequence > 1) fail(Status::Shape, "tree_ensemble models are tabular: inputs.sequence must be 0 or 1");
        const json::Value& config = ctx.config;
        shape_.features = ctx.features;
        shape_.sequence = ctx.sequence;
        const json::Value* outputs = ctx.spec.find("outputs");
        shape_.outputs = json_size(*outputs, "size", 1);
        shape_.streamable = true;

        const std::string decision = json_string(config, "decision", "le");
        if (decision != "le" && decision != "lt") fail(Status::Format, "tree_ensemble decision must be \"le\" or \"lt\"");
        less_equal_ = decision == "le";
        const std::string cast = json_string(config, "feature_cast", "f64");
        if (cast != "f64" && cast != "f32") fail(Status::Format, "feature_cast must be \"f64\" or \"f32\"");
        cast_f32_ = cast == "f32";
        const std::string aggregate = json_string(config, "aggregate", "sum");
        if (aggregate != "sum" && aggregate != "mean") fail(Status::Format, "aggregate must be \"sum\" or \"mean\"");
        mean_ = aggregate == "mean";
        const std::string link = json_string(config, "link", "identity");
        if (link == "identity") link_ = Link::Identity;
        else if (link == "sigmoid") link_ = Link::Sigmoid;
        else if (link == "softmax") link_ = Link::Softmax;
        else if (link == "exp") link_ = Link::Exp;
        else fail(Status::Unsupported, "unknown tree_ensemble link \"" + link + "\"");
        sigmoid_scale_ = json_double(config, "sigmoid_scale", 1.0);

        base_.assign(shape_.outputs, 0.0);
        if (const json::Value* base = config.find("base_score"); base && !base->is_null()) {
            if (base->is_number()) {
                std::fill(base_.begin(), base_.end(), base->as_number());
            } else if (base->is_array() && base->as_array().size() == shape_.outputs) {
                for (size_t k = 0; k < shape_.outputs; ++k) base_[k] = base->as_array()[k].as_number();
            } else {
                fail(Status::Format, "base_score must be a number or one number per output");
            }
        }

        const std::string p = json_string(config, "tensors", "trees");
        const Archive& a = *ctx.archive;
        roots_ = a.i32(p + ".root", {-1});
        tree_output_ = a.i32(p + ".output", {static_cast<int64_t>(roots_.size())});
        const auto feature = a.i32(p + ".feature", {-1});
        const auto n = static_cast<int64_t>(feature.size());
        const auto threshold = a.f64(p + ".threshold", {n});
        const auto left = a.i32(p + ".left", {n});
        const auto right = a.i32(p + ".right", {n});
        const auto value = a.f64(p + ".value", {n});
        const auto flags = a.u8(p + ".flags", {n});
        if (roots_.empty() || feature.empty()) fail(Status::Format, "tree_ensemble needs at least one tree");

        nodes_.resize(feature.size());
        for (size_t i = 0; i < nodes_.size(); ++i) {
            Node& node = nodes_[i];
            node.feature = feature[i];
            node.threshold = threshold[i];
            node.left = left[i];
            node.right = right[i];
            node.value = value[i];
            node.flags = flags[i];
            const auto self = static_cast<int32_t>(i);
            if (node.feature >= 0) {
                if (static_cast<size_t>(node.feature) >= shape_.features) {
                    fail(Status::Shape, "node " + std::to_string(i) + " splits on feature " +
                                            std::to_string(node.feature) + " of " + std::to_string(shape_.features));
                }
                if (node.left <= self || node.right <= self || node.left >= n || node.right >= n) {
                    fail(Status::Format, "node " + std::to_string(i) + " must point to later nodes");
                }
                if (std::isnan(node.threshold)) fail(Status::Format, "node " + std::to_string(i) + " has a NaN threshold");
                if (((node.flags >> 1) & 3u) == 3u) fail(Status::Format, "node " + std::to_string(i) + " has bad flags");
            } else if (!std::isfinite(node.value)) {
                fail(Status::Format, "leaf " + std::to_string(i) + " has a non-finite value");
            }
        }
        counts_.assign(shape_.outputs, 0);
        for (size_t t = 0; t < roots_.size(); ++t) {
            if (roots_[t] < 0 || roots_[t] >= n) fail(Status::Format, "tree " + std::to_string(t) + " root is out of range");
            if (tree_output_[t] < 0 || static_cast<size_t>(tree_output_[t]) >= shape_.outputs) {
                fail(Status::Format, "tree " + std::to_string(t) + " adds to a missing output");
            }
            ++counts_[static_cast<size_t>(tree_output_[t])];
        }
        if (link_ == Link::Softmax && shape_.outputs < 2) fail(Status::Format, "softmax needs at least two outputs");
        cost_ = static_cast<double>(roots_.size()) * 8.0;
        blocks_ = (roots_.size() + kTreeBlock - 1) / kTreeBlock;
        compile();
    }

    Shape shape() const override { return shape_; }
    double cost_per_sample() const override { return cost_; }

    void predict(const double* x, size_t batch, double* out) const override {
        const size_t K = shape_.outputs, F = shape_.features;
        thread_local std::vector<double> prepared;
        for (size_t s0 = 0; s0 < batch; s0 += kBlock) {
            const size_t n = std::min(kBlock, batch - s0);
            const double* rows = x + s0 * F;
            double* acc = out + s0 * K;
            for (size_t i = 0; i < n; ++i) {
                if (mean_) std::fill_n(acc + i * K, K, 0.0);
                else std::copy(base_.begin(), base_.end(), acc + i * K);
            }
            if (packed_.empty()) {
                reference_block(rows, n, acc, prepared);
            } else {
                prepared.resize(n * width_);
                bool nan = false;
                for (size_t i = 0; i < n; ++i) nan |= prepare(rows + i * F, prepared.data() + i * width_);
                if (n < kRowWalk) {
                    if (nan) walk_trees<true>(prepared.data(), n, acc);
                    else walk_trees<false>(prepared.data(), n, acc);
                } else if (nan) {
                    walk_rows<true>(prepared.data(), n, acc);
                } else {
                    walk_rows<false>(prepared.data(), n, acc);
                }
            }
            for (size_t i = 0; i < n; ++i) {
                double* row = acc + i * K;
                if (mean_) {
                    for (size_t k = 0; k < K; ++k) {
                        row[k] = base_[k] + (counts_[k] ? row[k] / static_cast<double>(counts_[k]) : 0.0);
                    }
                }
                apply_link(row);
            }
        }
    }

    std::unique_ptr<FamilyStream> stream(size_t batch) const override {
        struct Stateless final : FamilyStream {
            Stateless(const TreeEnsemble& m, size_t b) : model(m), batch(b) {}
            void step(const double* x, double* out) override { model.predict(x, batch, out); }
            void reset() override {}
            const TreeEnsemble& model;
            size_t batch;
        };
        return std::make_unique<Stateless>(*this, batch);
    }

private:
    // Builds the traversal layout. It stays empty, and predict() takes the
    // reference walk, when a split threshold is infinite or trees share nodes.
    void compile() {
        for (const Node& node : nodes_) {
            if (node.feature >= 0 && !std::isfinite(node.threshold)) return;
        }
        const auto F = static_cast<uint32_t>(shape_.features);
        bool zero_copy = false;
        for (const Node& node : nodes_) zero_copy |= node.feature >= 0 && ((node.flags >> 1) & 3u) == 2u;
        const size_t width = static_cast<size_t>(F) * (zero_copy ? 2 : 1) + 1;
        const auto pad = static_cast<uint32_t>(width - 1);
        constexpr double kInf = std::numeric_limits<double>::infinity();

        const size_t T = roots_.size();
        std::vector<Packed> packed{{0.0, pad, 0}};
        std::vector<uint32_t> tree_root(T), tree_depth(T), group_depth((T + kGroup - 1) / kGroup, 0);
        struct Item {
            int32_t source;
            uint32_t target;
            uint32_t depth;
        };
        std::vector<Item> queue;
        for (size_t t = 0; t < T; ++t) {
            tree_root[t] = static_cast<uint32_t>(packed.size());
            packed.emplace_back();
            queue.assign(1, Item{roots_[t], tree_root[t], 0});
            for (size_t q = 0; q < queue.size(); ++q) {
                const Item item = queue[q];
                const Node& node = nodes_[static_cast<size_t>(item.source)];
                if (node.feature < 0) {
                    packed[item.target] = Packed{node.value, pad, item.target};
                    tree_depth[t] = std::max(tree_depth[t], item.depth);
                    continue;
                }
                if (packed.size() + 2 > nodes_.size() + T + 1) return;
                const auto left = static_cast<uint32_t>(packed.size());
                packed.resize(packed.size() + 2);
                const double threshold = less_equal_ ? node.threshold : std::nextafter(node.threshold, -kInf);
                const unsigned missing = (node.flags >> 1) & 3u;
                const bool nan_right = missing == 0 ? 0.0 > threshold : (node.flags & 1u) == 0;
                const auto feature = static_cast<uint32_t>(node.feature) + (missing == 2 ? F : 0u);
                packed[item.target] = Packed{threshold, feature | (nan_right ? kNanRight : 0u), left};
                queue.push_back(Item{node.left, left, item.depth + 1});
                queue.push_back(Item{node.right, left + 1, item.depth + 1});
            }
            group_depth[t / kGroup] = std::max(group_depth[t / kGroup], tree_depth[t]);
        }
        steps_ = 0;
        for (const uint32_t depth : group_depth) steps_ += depth * kGroup;
        packed_ = std::move(packed);
        tree_root_ = std::move(tree_root);
        tree_depth_ = std::move(tree_depth);
        group_depth_ = std::move(group_depth);
        zero_copy_ = zero_copy;
        width_ = width;
    }

    // The row as the walks read it: cast, then (with LightGBM "Zero" splits)
    // a copy where |v| <= kZero reads as NaN, then the -inf that leaves read.
    // Returns whether a walk can meet a NaN.
    bool prepare(const double* x, double* row) const noexcept {
        const size_t F = shape_.features;
        bool nan = false;
        for (size_t f = 0; f < F; ++f) {
            const double v = cast_f32_ ? static_cast<double>(static_cast<float>(x[f])) : x[f];
            row[f] = v;
            nan |= v != v;
        }
        if (zero_copy_) {
            for (size_t f = 0; f < F; ++f) {
                const double v = row[f];
                const double z = std::fabs(v) <= kZero ? std::numeric_limits<double>::quiet_NaN() : v;
                row[F + f] = z;
                nan |= z != z;
            }
        }
        row[width_ - 1] = -std::numeric_limits<double>::infinity();
        return nan;
    }

    template <bool kNan>
    static uint32_t step(const Packed* nodes, uint32_t index, const double* row) noexcept {
        const Packed& node = nodes[index];
        const double v = row[node.feature & ~kNanRight];
        uint32_t right = v > node.threshold;
        if constexpr (kNan) right |= static_cast<uint32_t>(v != v) & (node.feature >> 31);
        return node.left + right;
    }

    // A few rows: every tree block of every row is one unit of work, and the
    // blocks are split across latency threads. Each block's sums get their
    // own cache line.
    template <bool kNan>
    void walk_trees(const double* rows, size_t n, double* acc) const noexcept {
        const size_t K = shape_.outputs, B = blocks_;
        const size_t stride = (K + 7) / 8 * 8;
        thread_local std::vector<double> buffer;
        buffer.resize(n * B * stride + 8);
        const auto aligned = (reinterpret_cast<uintptr_t>(buffer.data()) + 63) & ~uintptr_t{63};
        double* partial = reinterpret_cast<double*>(aligned);
        const size_t max_parts = std::min(B, std::max<size_t>(1, steps_ * n / kPartSteps));
        run_parts(max_parts, [&](size_t part, size_t parts) {
            const size_t b0 = B * part / parts, b1 = B * (part + 1) / parts;
            for (size_t i = 0; i < n; ++i) {
                for (size_t b = b0; b < b1; ++b) {
                    double* sums = partial + (i * B + b) * stride;
                    if (b == 0) std::copy_n(acc + i * K, K, sums);
                    else std::fill_n(sums, K, 0.0);
                    walk_block<kNan>(rows + i * width_, b, sums);
                }
            }
        });
        for (size_t i = 0; i < n; ++i) {
            const double* row = partial + i * B * stride;
            for (size_t k = 0; k < K; ++k) {
                double sum = row[k];
                for (size_t b = 1; b < B; ++b) sum += row[b * stride + k];
                acc[i * K + k] = sum;
            }
        }
    }

    // Adds the leaves of block b to sums, kGroup trees at a time, every tree
    // of a group for as many steps as the group's deepest leaf. A single
    // output accumulates in a register.
    template <bool kNan>
    void walk_block(const double* row, size_t b, double* sums) const noexcept {
        const size_t t0 = b * kTreeBlock, t1 = std::min(tree_root_.size(), t0 + kTreeBlock);
        if (shape_.outputs == 1) {
            double sum = sums[0];
            walk_groups<kNan>(row, t0, t1, [&sum](size_t, double value) { sum += value; });
            sums[0] = sum;
        } else {
            walk_groups<kNan>(row, t0, t1, [this, sums](size_t t, double value) {
                sums[static_cast<size_t>(tree_output_[t])] += value;
            });
        }
    }

    template <bool kNan, typename Add>
    void walk_groups(const double* row, size_t t0, size_t t1, Add&& add) const noexcept {
        const Packed* nodes = packed_.data();
        for (size_t g0 = t0; g0 < t1; g0 += kGroup) {
            const size_t g = std::min(kGroup, t1 - g0);
            uint32_t index[kGroup];
            for (size_t j = 0; j < kGroup; ++j) index[j] = j < g ? tree_root_[g0 + j] : 0;
            for (uint32_t level = group_depth_[g0 / kGroup]; level > 0; --level) {
                for (size_t j = 0; j < kGroup; ++j) index[j] = step<kNan>(nodes, index[j], row);
            }
            for (size_t j = 0; j < g; ++j) add(g0 + j, nodes[index[j]].threshold);
        }
    }

    // A block of rows, one tree at a time; the first tree block adds into acc,
    // each later one into `partial`, which is then added to acc.
    template <bool kNan>
    void walk_rows(const double* rows, size_t n, double* acc) const noexcept {
        const Packed* nodes = packed_.data();
        const size_t K = shape_.outputs, T = tree_root_.size();
        thread_local std::vector<double> partial;
        partial.resize(n * K);
        uint32_t index[kBlock];
        for (size_t b = 0; b < blocks_; ++b) {
            double* sums = b == 0 ? acc : partial.data();
            if (b > 0) std::fill_n(sums, n * K, 0.0);
            for (size_t t = b * kTreeBlock; t < std::min(T, (b + 1) * kTreeBlock); ++t) {
                std::fill_n(index, n, tree_root_[t]);
                for (uint32_t level = tree_depth_[t]; level > 0; --level) {
                    for (size_t i = 0; i < n; ++i) index[i] = step<kNan>(nodes, index[i], rows + i * width_);
                }
                const auto k = static_cast<size_t>(tree_output_[t]);
                for (size_t i = 0; i < n; ++i) sums[i * K + k] += nodes[index[i]].threshold;
            }
            if (b > 0) {
                for (size_t j = 0; j < n * K; ++j) acc[j] += sums[j];
            }
        }
    }

    void reference_block(const double* rows, size_t n, double* acc, std::vector<double>& narrowed) const {
        const size_t K = shape_.outputs, F = shape_.features, T = roots_.size();
        if (cast_f32_) {
            narrowed.resize(n * F);
            for (size_t i = 0; i < n * F; ++i) narrowed[i] = static_cast<double>(static_cast<float>(rows[i]));
            rows = narrowed.data();
        }
        std::vector<double> partial(n * K);
        for (size_t b = 0; b < blocks_; ++b) {
            double* sums = b == 0 ? acc : partial.data();
            if (b > 0) std::fill(partial.begin(), partial.end(), 0.0);
            for (size_t t = b * kTreeBlock; t < std::min(T, (b + 1) * kTreeBlock); ++t) {
                const auto k = static_cast<size_t>(tree_output_[t]);
                for (size_t i = 0; i < n; ++i) sums[i * K + k] += leaf(roots_[t], rows + i * F);
            }
            if (b > 0) {
                for (size_t j = 0; j < n * K; ++j) acc[j] += partial[j];
            }
        }
    }

    double leaf(int32_t index, const double* row) const noexcept {
        const Node* node = &nodes_[static_cast<size_t>(index)];
        while (node->feature >= 0) {
            double v = row[node->feature];
            const unsigned missing = (node->flags >> 1) & 3u;
            bool go_left;
            if (std::isnan(v) && missing != 0) {
                go_left = node->flags & 1u;
            } else {
                if (std::isnan(v)) v = 0.0;
                if (missing == 2 && v >= -kZero && v <= kZero) {
                    go_left = node->flags & 1u;
                } else {
                    go_left = less_equal_ ? v <= node->threshold : v < node->threshold;
                }
            }
            node = &nodes_[static_cast<size_t>(go_left ? node->left : node->right)];
        }
        return node->value;
    }

    void apply_link(double* acc) const noexcept {
        const size_t K = shape_.outputs;
        switch (link_) {
            case Link::Identity: return;
            case Link::Sigmoid:
                for (size_t k = 0; k < K; ++k) acc[k] *= sigmoid_scale_;
                kernels::sigmoid_n(acc, K);
                return;
            case Link::Exp:
                kernels::map4(acc, K, [](kernels::v4d v) { return kernels::vexp(v); });
                return;
            case Link::Softmax: {
                double peak = acc[0];
                for (size_t k = 1; k < K; ++k) peak = std::max(peak, acc[k]);
                for (size_t k = 0; k < K; ++k) acc[k] -= peak;
                kernels::map4(acc, K, [](kernels::v4d v) { return kernels::vexp(v); });
                double total = 0;
                for (size_t k = 0; k < K; ++k) total += acc[k];
                for (size_t k = 0; k < K; ++k) acc[k] /= total;
                return;
            }
        }
    }

    Shape shape_;
    bool less_equal_ = true, cast_f32_ = false, mean_ = false, zero_copy_ = false;
    Link link_ = Link::Identity;
    double sigmoid_scale_ = 1.0;
    std::vector<double> base_;
    std::vector<int32_t> roots_, tree_output_;
    std::vector<size_t> counts_;
    std::vector<Node> nodes_;
    std::vector<Packed> packed_;
    std::vector<uint32_t> tree_root_, tree_depth_, group_depth_;
    size_t width_ = 0, blocks_ = 0, steps_ = 0;
    double cost_ = 0;
};

std::unique_ptr<FamilyModel> build(const BuildContext& ctx) { return std::make_unique<TreeEnsemble>(ctx); }

}  // namespace

void register_tree_ensemble(Registry& registry) { registry.add("tree_ensemble", &build, "builtin"); }

}  // namespace montlok::core
