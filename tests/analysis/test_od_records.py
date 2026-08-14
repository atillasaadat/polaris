"""The orbit-OD campaign loader (design doc §8.3, §23.2).

Synthetic shards, not a flown campaign: a 7-day 30-run campaign is hours of
wall clock and a test nobody runs is a test that does not exist. What is under
test is the loader's *policy* — which malformed input is fatal, which is
ordinary, and what an absent field becomes — because every statistic downstream
inherits it. A loader that turns a missing measurement into a zero produces a
report that is wrong in a direction no later test can see.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import od_sample, write_od_shard

from analysis.od.records import load_campaign


def test_meta_and_samples_group_into_one_run(tmp_path: Path) -> None:
    """A meta line opens a group and the samples after it belong to it."""
    shard = write_od_shard(
        tmp_path / "shard.jsonl",
        [(3, "nominal", [od_sample(i) for i in range(5)])],
    )

    campaign = load_campaign(shard)

    assert len(campaign.runs) == 1
    run = campaign.runs[0]
    assert (run.run, run.scenario, run.samples) == (3, "nominal", 5)
    assert run.intent == "synthetic nominal"
    assert run.cycle_period_s == 10.0
    assert campaign.scenarios == ("nominal",)


def test_shards_of_disjoint_runs_reassemble(tmp_path: Path) -> None:
    """A campaign sharded across processes is one campaign when read back.

    This is how every real campaign is flown — one process per run, writing its
    own file — so the reassembly is not a convenience feature.
    """
    for run in (0, 1, 2):
        write_od_shard(
            tmp_path / f"shard_{run:02d}.jsonl",
            [
                (run, name, [od_sample(i) for i in range(3)])
                for name in ("nominal", "outage")
            ],
        )

    campaign = load_campaign(tmp_path)

    assert len(campaign.paths) == 3
    assert {r.run for r in campaign.runs} == {0, 1, 2}
    assert campaign.scenarios == ("nominal", "outage")
    assert len(campaign.of("outage")) == 3


def test_a_sample_before_its_meta_line_is_fatal(tmp_path: Path) -> None:
    """An orphaned sample means an interleaved or damaged shard.

    Refused rather than dropped: a campaign silently missing its first scenario
    reads as a clean campaign that happened to be short, which is the failure
    mode a verdict must never be built on.
    """
    shard = tmp_path / "orphan.jsonl"
    shard.write_text(json.dumps(od_sample(0)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="before its meta line"):
        load_campaign(shard)


def test_a_malformed_line_mid_file_is_fatal(tmp_path: Path) -> None:
    """Corruption anywhere but the final line stays an error."""
    shard = write_od_shard(
        tmp_path / "shard.jsonl", [(0, "nominal", [od_sample(i) for i in range(3)])]
    )
    lines = shard.read_text(encoding="utf-8").splitlines()
    lines[2] = "{not json"
    shard.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_campaign(shard)


def test_a_half_written_final_line_is_tolerated_and_named(tmp_path: Path) -> None:
    """The one malformed line with an ordinary cause: a campaign still flying.

    Dropped rather than fatal, because reading a live campaign is routine — but
    recorded in ``truncated`` so a report can say the campaign is incomplete
    instead of presenting a partial run as a finished one.
    """
    shard = write_od_shard(
        tmp_path / "shard.jsonl",
        [(0, "nominal", [od_sample(i) for i in range(6)])],
        truncate_last=True,
    )

    campaign = load_campaign(shard)

    assert campaign.truncated == (shard,)
    # The half-record is gone; every complete one before it survives.
    assert campaign.runs[0].samples == 5


def test_an_absent_nis_is_nan_and_not_zero(tmp_path: Path) -> None:
    """No measurement judged is not the same as a perfect innovation.

    A zero would drag every NIS average down by the fraction of cycles with no
    fix, which on a campaign flying hour-long outages is most of them.
    """
    rows = [od_sample(0), od_sample(1)]
    del rows[1]["nis"]
    shard = write_od_shard(tmp_path / "shard.jsonl", [(0, "nominal", rows)])

    run = load_campaign(shard).runs[0]

    assert run.nis[0] == 3.0
    assert np.isnan(run.nis[1])


def test_absent_ric_columns_are_nan(tmp_path: Path) -> None:
    """A shard written before the RIC decomposition existed contributes nothing.

    NaN rather than zero, so a stale shard is excluded from the ensemble
    statistics rather than pulling the measured spread towards zero — which
    would read as a filter whose error is smaller than it is.
    """
    rows = [od_sample(i) for i in range(3)]
    for row in rows:
        del row["err_ric_m"]
        del row["sigma_ric_m"]
    shard = write_od_shard(tmp_path / "stale.jsonl", [(0, "nominal", rows)])

    run = load_campaign(shard).runs[0]

    assert run.err_ric_m.shape == (3, 3)
    assert np.isnan(run.err_ric_m).all()
    assert np.isnan(run.sigma_ric_m).all()


def test_an_empty_directory_is_a_harness_failure(tmp_path: Path) -> None:
    """No shards at all is refused, never reported as a campaign with no faults."""
    with pytest.raises(FileNotFoundError):
        load_campaign(tmp_path)


def test_regime_mask_selects_only_its_own_samples(tmp_path: Path) -> None:
    """The mask every per-regime statistic is built on."""
    rows = [od_sample(i, regime="nominal") for i in range(4)]
    rows += [od_sample(i, regime="outage") for i in range(4, 7)]
    shard = write_od_shard(tmp_path / "shard.jsonl", [(0, "outage_scenario", rows)])

    run = load_campaign(shard).runs[0]

    assert int(run.mask("nominal").sum()) == 4
    assert int(run.mask("outage").sum()) == 3
    assert not run.mask("spoof").any()
