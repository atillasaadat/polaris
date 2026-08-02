SITL protocol — ``polaris::sitl``
=================================

The two-process software-in-the-loop contract shared verbatim by both sides (design doc §2.2, §2.4, §18.11). ``wire.hpp`` is the frozen v1 message layout — POD records memcpy'd little-endian with static-asserted sizes, riding inside standard F´ frames over the dedicated SITL TCP socket rather than the GDS ground link; the truth sim serializes it and the F´ ``SitlBridge`` component deserializes it, so the two cannot drift. Truth-side diagnostics a real part would not report deliberately do not cross (REQ-SIM-004, §2.3). ``SitlHandler`` is the FSW-side decode/reply logic factored out of the component so the byte protocol is testable without a running topology: it validates every message before building a reply, and splits STEP into decode plus ``buildStepReply`` so the rate group runs between the two halves of the §2.4 barrier. ``scripted_profile.hpp`` is the Phase-4 placeholder command profile — a pure function of sim epoch and unit index, shared by the flight ``ScriptedCmdSource`` and the in-process integration test so both runs apply byte-identical commands; it is deleted when real GNC lands.

.. doxygennamespace:: polaris::sitl
   :members:
