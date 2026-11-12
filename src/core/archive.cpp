// Copyright 2026 Montlok. Apache-2.0.
// safetensors reader. Layout: an 8-byte little-endian header length, the JSON
// header, then the tensor bytes addressed by "data_offsets" relative to the
// end of the header. The model spec is the JSON string
// __metadata__["montlok"].
#include <cstring>
#include <fstream>
#include <limits>

#include "runtime.hpp"

namespace montlok::core {

namespace {

constexpr uint64_t kMaxHeaderBytes = 64ull << 20;
constexpr size_t kMaxRank = 8;

bool parse_dtype(const std::string& name, DType& out) noexcept {
    if (name == "F64") out = DType::F64;
    else if (name == "F32") out = DType::F32;
    else if (name == "I64") out = DType::I64;
    else if (name == "I32") out = DType::I32;
    else if (name == "U8") out = DType::U8;
    else if (name == "BOOL") out = DType::Bool;
    else return false;
    return true;
}

std::string shape_text(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

}  // namespace

size_t dtype_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::F64:
        case DType::I64: return 8;
        case DType::F32:
        case DType::I32: return 4;
        case DType::U8:
        case DType::Bool: return 1;
    }
    return 0;
}

const char* dtype_name(DType dtype) noexcept {
    switch (dtype) {
        case DType::F64: return "F64";
        case DType::F32: return "F32";
        case DType::I64: return "I64";
        case DType::I32: return "I32";
        case DType::U8: return "U8";
        case DType::Bool: return "BOOL";
    }
    return "?";
}

size_t Tensor::numel() const noexcept {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    return n;
}

std::string Tensor::describe() const { return name + " " + dtype_name(dtype) + shape_text(shape); }

std::shared_ptr<const Archive> Archive::from_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) fail(Status::Io, "cannot open model file: " + path);
    const std::streamoff end = file.tellg();
    if (end < 0) fail(Status::Io, "cannot size model file: " + path);
    const auto size = static_cast<size_t>(end);
    auto archive = std::shared_ptr<Archive>(new Archive());
    archive->allocate(size);
    file.seekg(0);
    if (size && !file.read(reinterpret_cast<char*>(archive->storage_.get()), static_cast<std::streamsize>(size))) {
        fail(Status::Io, "cannot read model file: " + path);
    }
    archive->parse();
    return archive;
}

std::shared_ptr<const Archive> Archive::from_bytes(const void* data, size_t size) {
    if (!data && size) fail(Status::InvalidArgument, "null model buffer");
    auto archive = std::shared_ptr<Archive>(new Archive());
    archive->allocate(size);
    if (size) std::memcpy(archive->storage_.get(), data, size);
    archive->parse();
    return archive;
}

void Archive::allocate(size_t size) {
    // 64-byte alignment of the buffer keeps every tensor aligned to its element
    // size when the writer laid them out that way (montlok.export does).
    storage_.reset(static_cast<uint8_t*>(::operator new[](size > 0 ? size : 1, std::align_val_t(64))));
    size_ = size;
}

void Archive::parse() {
    const uint8_t* bytes = storage_.get();
    if (size_ < 8) fail(Status::Format, "not a safetensors file: shorter than 8 bytes");
    uint64_t header_len = 0;
    for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | bytes[i];
    if (header_len > kMaxHeaderBytes || header_len > size_ - 8) {
        fail(Status::Format, "not a safetensors file: header length out of range");
    }
    const size_t data_start = 8 + static_cast<size_t>(header_len);
    const size_t data_size = size_ - data_start;

    json::Value header;
    try {
        header = json::Value::parse(
            std::string_view(reinterpret_cast<const char*>(bytes + 8), static_cast<size_t>(header_len)));
    } catch (const std::exception& e) {
        fail(Status::Format, std::string("safetensors header: ") + e.what());
    }
    if (!header.is_object()) fail(Status::Format, "safetensors header is not a JSON object");

    const json::Value* metadata = nullptr;
    for (const auto& [name, entry] : header.as_object()) {
        if (name == "__metadata__") {
            metadata = &entry;
            continue;
        }
        if (!entry.is_object()) fail(Status::Format, "tensor entry is not an object: " + name);
        const json::Value* dtype = entry.find("dtype");
        const json::Value* shape = entry.find("shape");
        const json::Value* offsets = entry.find("data_offsets");
        if (!dtype || !dtype->is_string() || !shape || !shape->is_array() || !offsets || !offsets->is_array()) {
            fail(Status::Format, "tensor entry lacks dtype/shape/data_offsets: " + name);
        }
        Tensor t;
        t.name = name;
        if (!parse_dtype(dtype->as_string(), t.dtype)) {
            fail(Status::Unsupported, "tensor " + name + " has unsupported dtype " + dtype->as_string());
        }
        if (shape->as_array().size() > kMaxRank) fail(Status::Format, "tensor rank above 8: " + name);
        size_t numel = 1;
        try {
            for (const auto& d : shape->as_array()) {
                const int64_t extent = d.as_int();
                if (extent < 0) fail(Status::Format, "negative extent in tensor " + name);
                if (extent > 0 && numel > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent)) {
                    fail(Status::Format, "tensor too large: " + name);
                }
                numel *= static_cast<size_t>(extent);
                t.shape.push_back(extent);
            }
            if (offsets->as_array().size() != 2) fail(Status::Format, "data_offsets must have two entries: " + name);
            const int64_t begin = offsets->as_array()[0].as_int();
            const int64_t end = offsets->as_array()[1].as_int();
            if (begin < 0 || end < begin || static_cast<uint64_t>(end) > data_size) {
                fail(Status::Format, "data_offsets out of range: " + name);
            }
            t.data = bytes + data_start + static_cast<size_t>(begin);
            t.nbytes = static_cast<size_t>(end - begin);
        } catch (const Error&) {
            throw;
        } catch (const std::exception& e) {
            fail(Status::Format, "tensor " + name + ": " + e.what());
        }
        if (numel > std::numeric_limits<size_t>::max() / dtype_size(t.dtype) || numel * dtype_size(t.dtype) != t.nbytes) {
            fail(Status::Format, "tensor " + name + " byte size does not match its shape");
        }
        tensors_.push_back(std::move(t));
    }

    if (!metadata || !metadata->is_object()) fail(Status::Format, "not a montlok model: no __metadata__");
    const json::Value* spec = metadata->find("montlok");
    if (!spec || !spec->is_string()) fail(Status::Format, "not a montlok model: __metadata__.montlok is missing");
    spec_text_ = spec->as_string();
    try {
        spec_ = json::Value::parse(spec_text_);
    } catch (const std::exception& e) {
        fail(Status::Format, std::string("model spec: ") + e.what());
    }
    if (!spec_.is_object()) fail(Status::Format, "model spec is not a JSON object");
    const json::Value* schema = spec_.find("schema");
    if (!schema || !schema->is_string() || schema->as_string() != "montlok.model/1") {
        fail(Status::Unsupported, "model spec schema must be \"montlok.model/1\"");
    }
}

const Tensor* Archive::find(std::string_view name) const noexcept {
    for (const auto& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

const Tensor& Archive::tensor(std::string_view name) const {
    const Tensor* t = find(name);
    if (!t) fail(Status::Format, "model file has no tensor named " + std::string(name));
    return *t;
}

const Tensor& Archive::checked(std::string_view name, const std::vector<int64_t>& shape) const {
    const Tensor& t = tensor(name);
    bool ok = t.shape.size() == shape.size();
    for (size_t i = 0; ok && i < shape.size(); ++i) ok = shape[i] < 0 || shape[i] == t.shape[i];
    if (!ok) fail(Status::Shape, "tensor " + t.describe() + " does not have shape " + shape_text(shape));
    return t;
}

std::vector<double> Archive::f64(std::string_view name, const std::vector<int64_t>& shape) const {
    const Tensor& t = checked(name, shape);
    std::vector<double> out(t.numel());
    if (t.dtype == DType::F64) {
        std::memcpy(out.data(), t.data, t.nbytes);
    } else if (t.dtype == DType::F32) {
        for (size_t i = 0; i < out.size(); ++i) {
            float v;
            std::memcpy(&v, t.data + 4 * i, 4);
            out[i] = v;
        }
    } else {
        fail(Status::Format, "tensor " + t.describe() + " must be F64 or F32");
    }
    return out;
}

std::vector<int32_t> Archive::i32(std::string_view name, const std::vector<int64_t>& shape) const {
    const Tensor& t = checked(name, shape);
    std::vector<int32_t> out(t.numel());
    if (t.dtype == DType::I32) {
        std::memcpy(out.data(), t.data, t.nbytes);
    } else if (t.dtype == DType::I64) {
        for (size_t i = 0; i < out.size(); ++i) {
            int64_t v;
            std::memcpy(&v, t.data + 8 * i, 8);
            if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max()) {
                fail(Status::Format, "tensor " + t.describe() + " has values outside int32");
            }
            out[i] = static_cast<int32_t>(v);
        }
    } else {
        fail(Status::Format, "tensor " + t.describe() + " must be I32 or I64");
    }
    return out;
}

std::vector<uint8_t> Archive::u8(std::string_view name, const std::vector<int64_t>& shape) const {
    const Tensor& t = checked(name, shape);
    if (t.dtype != DType::U8 && t.dtype != DType::Bool) fail(Status::Format, "tensor " + t.describe() + " must be U8");
    return std::vector<uint8_t>(t.data, t.data + t.nbytes);
}

}  // namespace montlok::core
