# `lib/sitl/` — Plant↔FSW SITL Contract

The two-process software-in-the-loop wire contract (§2.2, §2.4, §22.1), shared
verbatim by the truth sim and the F´ deployment so the two sides cannot drift.
Header-only: there is nothing here to link, only a layout both processes agree
on and the decode logic that reads it.

| File | Role |
|---|---|
| `wire.hpp` | The **frozen v1** message layout: POD records memcpy'd little-endian with static-asserted sizes, carried inside standard F´ `Svc::FprimeProtocol` frames over the dedicated SITL socket (not the GDS ground link). Truth-side diagnostics a real part could not report deliberately do not cross (§2.3, REQ-SIM-004). Frozen by `SitlWire.RecordSizesAreTheFrozenV1Layout` |
| `handler.hpp` | `SitlHandler` — the FSW-side validate/decode/reply logic, factored out of `flight::SitlBridge` so the byte protocol is testable without a running topology. STEP is split into decode and `buildStepReply` so the caller runs the rate group *between* the halves: that split is the §2.4 barrier |

Flight-safe (it ships inside the deployment): fixed-size records, no heap, no
exceptions, every length validated against the HELLO-declared suite before use —
a mismatched message is rejected whole, leaving the previous state intact.
