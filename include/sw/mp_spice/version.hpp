#pragma once
// mp-spice -- mixed-precision SPICE modernization (MTL5 + Universal)
//
// Header-only composition layer. As shared utilities emerge (matrix loaders,
// mixed-precision solve harnesses, SPICE netlist front-ends), they live under
// sw::mp_spice. For now this header carries only version metadata.

namespace sw::mp_spice {

inline constexpr int version_major = 0;
inline constexpr int version_minor = 1;
inline constexpr int version_patch = 0;

} // namespace sw::mp_spice
