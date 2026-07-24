# Configuring a Vehicle

A vehicle is defined entirely in configuration — there is no hardware catalog in
code (design doc §19.4). A spacecraft YAML references parts by `model_id`; the
config compiler ({doc}`/api/python`) resolves them against the hardware library,
validates against a Pydantic schema, and emits provenance-stamped artifacts. The
truth sim ({doc}`/api/sim_scenario`) reads only the compiled artifact.

How one hardware value travels from a datasheet to a running model — every layer
between the compiler and `fromParams` passes the parameter map through
uninterpreted:

```{mermaid}
flowchart LR
    DS[Vendor datasheet] -->|transcribed, cited| HW["config/hardware/&lt;kind&gt;/&lt;model&gt;.yaml"]
    SC["spacecraft YAML<br/>(model_id refs)"] --> CC
    HW --> CC["config compiler<br/>tools/configc (Pydantic)"]
    CC -->|provenance hash| SJ["sim_setup.json"]
    SJ --> LC["SimConfig loader<br/>sim/scenario/sim_config"]
    LC -->|"param map (uninterpreted)"| FP["Spec::fromParams<br/>(one place: SI conversion)"]
    FP --> M["truth model<br/>(Imu, StarTracker, RW, ...)"]
    CC -.->|same resolved object| FP2["F´ params / analysis<br/>artifacts"]
```

The hardware-library authoring contract — file layout, required keys and their
units, the datasheet-comment standard, and the step-by-step *add a catalog
entry* walkthrough — is the repository's `config/hardware/README.md`, included
here:

```{include} ../../config/hardware/README.md
```
