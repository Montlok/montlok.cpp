// Copyright 2026 Montlok. Apache-2.0.
#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace montlok::json {

const char* kind_name(Value::Kind kind) noexcept {
    switch (kind) {
        case Value::Kind::Null: return "null";
        case Value::Kind::Bool: return "bool";
        case Value::Kind::Number: return "number";
        case Value::Kind::String: return "string";
        case Value::Kind::Array: return "array";
        case Value::Kind::Object: return "object";
    }
    return "unknown";
}

namespace {

[[noreturn]] void kind_error(Value::Kind want, Value::Kind got) {
    throw std::runtime_error(std::string("JSON value is ") + kind_name(got) + ", expected " + kind_name(want));
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

void dump_string(std::string& out, const std::string& s) {
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    out += buffer;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
}

void dump_value(std::string& out, const Value& v) {
    switch (v.kind()) {
        case Value::Kind::Null: out += "null"; return;
        case Value::Kind::Bool: out += v.as_bool() ? "true" : "false"; return;
        case Value::Kind::Number: {
            char buffer[32];
            const double x = v.as_number();
            // Integral values print without an exponent or fraction; others
            // round-trip with 17 significant digits.
            if (std::nearbyint(x) == x && std::fabs(x) < 9.007199254740992e15) {
                std::snprintf(buffer, sizeof(buffer), "%.0f", x);
            } else {
                std::snprintf(buffer, sizeof(buffer), "%.17g", x);
            }
            out += buffer;
            return;
        }
        case Value::Kind::String: dump_string(out, v.as_string()); return;
        case Value::Kind::Array: {
            out.push_back('[');
            bool first = true;
            for (const auto& item : v.as_array()) {
                if (!first) out.push_back(',');
                first = false;
                dump_value(out, item);
            }
            out.push_back(']');
            return;
        }
        case Value::Kind::Object: {
            out.push_back('{');
            bool first = true;
            for (const auto& [key, item] : v.as_object()) {
                if (!first) out.push_back(',');
                first = false;
                dump_string(out, key);
                out.push_back(':');
                dump_value(out, item);
            }
            out.push_back('}');
            return;
        }
    }
}

}  // namespace

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Value parse_document() {
        skip_ws();
        Value value = parse_value(0);
        skip_ws();
        if (pos_ != text_.size()) fail("trailing characters");
        return value;
    }

private:
    static constexpr int kMaxDepth = 128;

    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("invalid JSON at byte ") + std::to_string(pos_) + ": " + what);
    }

    void skip_ws() noexcept {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
            ++pos_;
        }
    }

    bool consume(char c) noexcept {
        if (pos_ < text_.size() && text_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect_literal(std::string_view literal) {
        if (text_.substr(pos_, literal.size()) != literal) fail("unexpected token");
        pos_ += literal.size();
    }

    Value parse_value(int depth) {
        if (depth > kMaxDepth) fail("nesting too deep");
        if (pos_ >= text_.size()) fail("unexpected end");
        Value v;
        switch (text_[pos_]) {
            case 'n': expect_literal("null"); return v;
            case 't':
                expect_literal("true");
                v.kind_ = Value::Kind::Bool;
                v.bool_ = true;
                return v;
            case 'f':
                expect_literal("false");
                v.kind_ = Value::Kind::Bool;
                v.bool_ = false;
                return v;
            case '"':
                v.kind_ = Value::Kind::String;
                v.string_ = parse_string();
                return v;
            case '[': return parse_array(depth);
            case '{': return parse_object(depth);
            default: return parse_number();
        }
    }

    Value parse_number() {
        const size_t start = pos_;
        consume('-');
        if (pos_ >= text_.size()) fail("bad number");
        if (text_[pos_] == '0') {
            ++pos_;
        } else if (text_[pos_] >= '1' && text_[pos_] <= '9') {
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        } else {
            fail("bad number");
        }
        if (consume('.')) {
            const size_t digits = pos_;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
            if (pos_ == digits) fail("bad fraction");
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (!consume('+')) consume('-');
            const size_t digits = pos_;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
            if (pos_ == digits) fail("bad exponent");
        }
        // The grammar is checked above; the conversion runs in the classic locale.
        double number = 0;
        std::istringstream stream(std::string(text_.substr(start, pos_ - start)));
        stream.imbue(std::locale::classic());
        stream >> number;
        if (stream.fail() || !std::isfinite(number)) fail("number out of range");
        Value v;
        v.kind_ = Value::Kind::Number;
        v.number_ = number;
        return v;
    }

    uint32_t parse_hex4() {
        if (pos_ + 4 > text_.size()) fail("bad unicode escape");
        uint32_t cp = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_++];
            cp <<= 4;
            if (c >= '0' && c <= '9') cp |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') cp |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') cp |= static_cast<uint32_t>(c - 'A' + 10);
            else fail("bad unicode escape");
        }
        return cp;
    }

    std::string parse_string() {
        ++pos_;  // opening quote
        std::string out;
        while (true) {
            if (pos_ >= text_.size()) fail("unterminated string");
            const unsigned char c = static_cast<unsigned char>(text_[pos_++]);
            if (c == '"') return out;
            if (c < 0x20) fail("control character in string");
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (pos_ >= text_.size()) fail("unterminated escape");
            const char e = text_[pos_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = parse_hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (!(consume('\\') && consume('u'))) fail("unpaired surrogate");
                        const uint32_t low = parse_hex4();
                        if (low < 0xDC00 || low > 0xDFFF) fail("unpaired surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("unpaired surrogate");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }

    Value parse_array(int depth) {
        ++pos_;
        auto items = std::make_shared<Array>();
        skip_ws();
        if (!consume(']')) {
            while (true) {
                skip_ws();
                items->push_back(parse_value(depth + 1));
                skip_ws();
                if (consume(']')) break;
                if (!consume(',')) fail("expected ',' or ']'");
            }
        }
        Value v;
        v.kind_ = Value::Kind::Array;
        v.array_ = std::move(items);
        return v;
    }

    Value parse_object(int depth) {
        ++pos_;
        auto members = std::make_shared<Object>();
        skip_ws();
        if (!consume('}')) {
            while (true) {
                skip_ws();
                if (pos_ >= text_.size() || text_[pos_] != '"') fail("expected member name");
                std::string key = parse_string();
                for (const auto& member : *members) {
                    if (member.first == key) fail("duplicate member name");
                }
                skip_ws();
                if (!consume(':')) fail("expected ':'");
                skip_ws();
                Value item = parse_value(depth + 1);
                members->emplace_back(std::move(key), std::move(item));
                skip_ws();
                if (consume('}')) break;
                if (!consume(',')) fail("expected ',' or '}'");
            }
        }
        Value v;
        v.kind_ = Value::Kind::Object;
        v.object_ = std::move(members);
        return v;
    }

    std::string_view text_;
    size_t pos_ = 0;
};

Value Value::parse(std::string_view text) { return Parser(text).parse_document(); }

bool Value::as_bool() const {
    if (kind_ != Kind::Bool) kind_error(Kind::Bool, kind_);
    return bool_;
}

double Value::as_number() const {
    if (kind_ != Kind::Number) kind_error(Kind::Number, kind_);
    return number_;
}

int64_t Value::as_int() const {
    const double x = as_number();
    if (std::nearbyint(x) != x || std::fabs(x) > 9.007199254740992e15) {
        throw std::runtime_error("JSON number is not an exact integer");
    }
    return static_cast<int64_t>(x);
}

const std::string& Value::as_string() const {
    if (kind_ != Kind::String) kind_error(Kind::String, kind_);
    return string_;
}

const Array& Value::as_array() const {
    if (kind_ != Kind::Array) kind_error(Kind::Array, kind_);
    return *array_;
}

const Object& Value::as_object() const {
    if (kind_ != Kind::Object) kind_error(Kind::Object, kind_);
    return *object_;
}

const Value* Value::find(std::string_view key) const noexcept {
    if (kind_ != Kind::Object) return nullptr;
    for (const auto& member : *object_) {
        if (member.first == key) return &member.second;
    }
    return nullptr;
}

std::string Value::dump() const {
    std::string out;
    dump_value(out, *this);
    return out;
}

}  // namespace montlok::json
