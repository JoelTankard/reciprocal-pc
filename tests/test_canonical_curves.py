from __future__ import annotations

from pfc.baselines import BLUE_CURVE, GPT_CURVE, RECIPROCAL_PC_CURVE, load_curve


EXPECTED = {
    GPT_CURVE: (1.1953982590593182, 0.43665138466651676, 50000.0),
    BLUE_CURVE: (1.2772543161894683, 0.4062503054770819, 0.0),
    RECIPROCAL_PC_CURVE: (1.2839073546437536, 0.4045396338185124, 0.0),
}


def test_canonical_curves_are_dense_measured_50k_series() -> None:
    expected_steps = [float(step) for step in range(100, 50001, 100)]
    for path, (bpb, accuracy, global_backprops) in EXPECTED.items():
        rows = load_curve(path)
        assert len(rows) == 500
        assert [row["step"] for row in rows] == expected_steps
        assert rows[-1]["val_bpb"] == bpb
        assert rows[-1]["accuracy"] == accuracy
        assert rows[-1]["global_backprops"] == global_backprops


def test_reciprocalpc_public_name_is_stable() -> None:
    rows = load_curve(RECIPROCAL_PC_CURVE)
    assert {row["display_name"] for row in rows} == {"ReciprocalPC"}
    assert all("stage_reciprocal" in str(row["source_run_tag"]) for row in rows)
