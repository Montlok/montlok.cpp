// Copyright 2026 Montlok. Apache-2.0.
// Minimal immutable JSON values for model specs and safetensors headers.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace montlok::json {

class Value;
using Array = std::vector<Value>;
using Member = std::pair<std::string, Value>;
using Object = std::vector<Member>;

class Value {
public:
    enum class Kind : uint8_t { Null, Bool, Number, String, Array, Object };

    Value() = default;

    // Strict RFC 8259 parsing.
    // Throws std::runtime_error naming the byte offset of the problem.
    static Value parse(std::string_view text);

    Kind kind() const noexcept { return kind_; }
    bool is_null() const noexcept { return kind_ == Kind::Null; }
    bool is_bool() const noexcept { return kind_ == Kind::Bool; }
    bool is_number() const noexcept { return kind_ == Kind::Number; }
    bool is_string() const noexcept { return kind_ == Kind::String; }
    bool is_array() const noexcept { return kind_ == Kind::Array; }
    bool is_object() const noexcept { return kind_ == Kind::Object; }

    // Typed access; each throws std::runtime_error on a kind mismatch.
    bool as_bool() const;
    double as_number() const;
    // An integral number that fits int64_t exactly.
    int64_t as_int() const;
    const std::string& as_string() const;
    const Array& as_array() const;
    const Object& as_object() const;

    // Member lookup on objects; nullptr when absent or not an object.
    const Value* find(std::string_view key) const noexcept;

    std::string dump() const;

private:
    friend class Parser;
    Kind kind_ = Kind::Null;
    bool bool_ = false;
    double number_ = 0;
    std::string string_;
    std::shared_ptr<const Array> array_;
    std::shared_ptr<const Object> object_;
};

const char* kind_name(Value::Kind kind) noexcept;

}  // namespace montlok::json
