# `lib/constants/` — Physical-Constants Registry

The one place shared physical constants live (design doc §3.4): WGS84
(`a`, `f`, `GM`, `ω⊕`), time offsets (`TAI−GPS`, `TT−TAI`), astronomical unit,
body radii/GMs, *c* — each value sourced in a comment. Not a hardware catalog:
device numbers live in `config/hardware/` (§19.4); these are constants of
nature and of reference systems.

| File | Role |
|---|---|
| `constants.hpp` | `constants::wgs84`, `constants::time`, `constants::bodies`, … all `constexpr` |
