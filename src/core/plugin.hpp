// Copyright 2026 Montlok. Apache-2.0.
#pragma once

#include <string>

#include "montlok/montlok.h"

namespace montlok::core {

// Registers a C family (montlok_register_family). The struct is copied.
void register_family_v1(const montlok_family_v1* family, const std::string& origin);

// Loads a plugin library and registers its families (montlok_plugin_load).
void load_plugin(const std::string& path);

}  // namespace montlok::core
