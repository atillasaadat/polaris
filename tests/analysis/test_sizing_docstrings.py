"""The sizing package documents its dimensions (``analysis/CLAUDE.md``).

In a package whose entire job is dimensional reasoning, a number without its
unit is not under-documented — it is unverifiable, and a formula that mixes
degrees into a radian expression or millitesla into a tesla one produces a
perfectly plausible margin. The repo's numpydoc convention says every public
entry point states its **units and frames**; these tests hold the five
calculation modules to it mechanically, so the convention survives the next
refactor without a reviewer having to re-read every docstring.

Three claims, each checkable and none of them a style preference:

* every public callable is documented with ``Parameters`` and ``Returns``;
* every one that returns a bare ``float`` names a unit somewhere in its
  docstring — the composite returns are exempt because their units live on the
  returned dataclass, which is the third claim;
* every public dataclass carries an ``Attributes`` block naming **every** field,
  which is where the units of a composite result actually are.

References
----------
``analysis/CLAUDE.md`` (numpydoc, units and frames); design doc §12.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re

import pytest

#: The calculation layer — the modules whose outputs are physical quantities.
#: The rendering modules (``html``, ``plots``, ``interactive``, ``report``) are
#: deliberately outside this: they carry no dimensions of their own, they format
#: what these five computed.
MODULES = ("envelope", "disturbances", "magnetorquers", "parameters", "wheels")

#: The subset whose numbers carry a fixed unit. ``envelope`` is excluded from the
#: unit rule alone, and for a stated reason rather than a convenience: it is
#: **unit-generic by construction** — hand it momentum capacity and every radius
#: is in N·m·s, hand it dipole and they are in A·m² — so its docstrings name
#: :attr:`~analysis.sizing.envelope.Envelope.capacity` where the others name a
#: unit. Requiring a bracketed unit there would push the module towards
#: documenting a unit it does not have.
DIMENSIONAL_MODULES = tuple(m for m in MODULES if m != "envelope")

#: A unit is written in brackets, per numpydoc: ``[N·m]``, ``[rad/s]``, ``[-]``
#: for a dimensionless ratio. The bracket is what makes the claim greppable,
#: which is the reason the convention picked it.
UNIT = re.compile(r"\[[^\]]+\]")

#: Return **annotations** that mean "a bare number, with no dataclass to carry
#: its units for it": ``float`` and tuples of floats. Taken from the signature
#: rather than from the prose, so the check cannot be satisfied by editing the
#: docstring it is checking.
BARE_FLOAT = re.compile(r"^(float|tuple\[float(,\s*float)*\])$")


def _public_members(module_name):
    """Every public function, property and dataclass defined in @p module_name."""
    module = importlib.import_module(f"analysis.sizing.{module_name}")
    functions, properties, dataclasses_ = [], [], []
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            functions.append((f"{module_name}.{name}", obj))
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            if dataclasses.is_dataclass(obj):
                dataclasses_.append((f"{module_name}.{name}", obj))
            for prop_name, member in vars(obj).items():
                if isinstance(member, property) and not prop_name.startswith("_"):
                    properties.append((f"{module_name}.{name}.{prop_name}", member))
    return functions, properties, dataclasses_


@pytest.mark.parametrize("module_name", MODULES)
def test_every_public_function_documents_its_parameters_and_return(module_name):
    """No public entry point is undocumented, and none skips its signature.

    The Sphinx site is compiled from these, so a missing section is a hole in the
    published API reference as well as in the source — and the docs build treats
    warnings as errors, which makes this the cheaper place to find it.
    """
    functions, _, _ = _public_members(module_name)
    assert functions, f"{module_name}: nothing public found — the probe is wrong"
    for name, function in functions:
        doc = inspect.getdoc(function) or ""
        assert doc, f"{name} has no docstring"
        assert "Parameters" in doc, f"{name} documents no parameters"
        assert "Returns" in doc, f"{name} documents no return value"


@pytest.mark.parametrize("module_name", DIMENSIONAL_MODULES)
def test_every_bare_float_return_states_its_unit(module_name):
    """A function handing back a naked number must say what the number is in.

    This is the one that would have caught a real defect rather than a style
    lapse: ``bdot_noise_floor``'s rate and ``implied_bandwidth``'s frequency are
    both plain floats, and a caller who assumes degrees for one that returns
    radians is out by 57 with nothing in the type system to object. Composite
    returns are exempt here and covered by the ``Attributes`` test below, which
    is where their units genuinely live.
    """
    functions, properties, _ = _public_members(module_name)
    for name, member in functions + properties:
        function = member.fget if isinstance(member, property) else member
        annotation = str(function.__annotations__.get("return", "")).replace(" ", "")
        if not BARE_FLOAT.match(annotation):
            continue
        doc = inspect.getdoc(function) or ""
        assert UNIT.search(doc), f"{name} returns {annotation} and names no unit"


@pytest.mark.parametrize("module_name", MODULES)
def test_every_public_dataclass_documents_every_field(module_name):
    """An ``Attributes`` block naming all of them — the units of a composite result.

    A result object is where most of this package's numbers surface, and a field
    added without a line in the block is a quantity with no stated unit reaching
    the report. Checked against ``dataclasses.fields`` rather than by reading, so
    it fails on the commit that adds the field rather than on review.
    """
    _, _, found = _public_members(module_name)
    assert found, f"{module_name}: no public dataclass found — the probe is wrong"
    for name, cls in found:
        doc = inspect.getdoc(cls) or ""
        assert "Attributes" in doc, f"{name} has no Attributes block"
        missing = [f.name for f in dataclasses.fields(cls) if f.name not in doc]
        assert not missing, f"{name} documents no units for {missing}"
