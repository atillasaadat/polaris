# Configuring a Vehicle

A vehicle is defined entirely in configuration — there is no hardware catalog in
code (design doc §19.4). A spacecraft YAML references parts by `model_id`; the
config compiler ({doc}`/api/python`) resolves them against the hardware library,
validates against a Pydantic schema, and emits provenance-stamped artifacts. The
truth sim ({doc}`/api/sim_scenario`) reads only the compiled artifact.

The hardware-library authoring contract — file layout, required keys and their
units, the datasheet-comment standard, and the step-by-step *add a catalog
entry* walkthrough — is the repository's `config/hardware/README.md`, included
here:

```{include} ../../config/hardware/README.md
```
