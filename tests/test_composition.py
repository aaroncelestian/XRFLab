"""Tests for bulk-rock composition grouping, averaging, and plot coords."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.composition import (
    GroupMode,
    VALUE_WT,
    apply_value_source,
    assign_samples,
    close_to_100,
    convert_values,
    correlation,
    correlation_matrix,
    numbered_replicate_names,
    oxide_label,
    ratio_points,
    rows_from_batch_results,
    scatter_points,
    strip_replicate_suffix,
    summarize_samples,
    ternary_points,
    ternary_xy,
)


def _batch_result(name, path, conc, ok=True):
    return SimpleNamespace(
        spectrum_name=name,
        spectrum_path=path,
        concentrations=conc,
        fit_success=ok,
    )


def test_strip_replicate_suffix():
    assert strip_replicate_suffix("B01_1") == "B01"
    assert strip_replicate_suffix("B01_rep2") == "B01"
    assert strip_replicate_suffix("Basalt01-spot3") == "Basalt01"
    assert strip_replicate_suffix("B01_a") == "B01"
    assert strip_replicate_suffix("steel_sample") == "steel_sample"
    assert strip_replicate_suffix("Spectrum24") == "Spectrum24"


def test_numbered_replicate_names():
    assert numbered_replicate_names("B01", 1) == ["B01"]
    assert numbered_replicate_names("B01", 4) == ["B01_1", "B01_2", "B01_3", "B01_4"]
    assert numbered_replicate_names("B01", 12)[0] == "B01_01"
    assert numbered_replicate_names("B01", 12)[-1] == "B01_12"
    assert numbered_replicate_names("", 4) == []
    assert numbered_replicate_names("B01", 0) == []
    for name in numbered_replicate_names("Basalt01", 4):
        assert strip_replicate_suffix(name) == "Basalt01"
    for name in numbered_replicate_names("Basalt01", 12):
        assert strip_replicate_suffix(name) == "Basalt01"


def test_folder_grouping_20_basalts():
    results = []
    for i in range(1, 21):
        sample = f"B{i:02d}"
        for rep in range(1, 5):
            results.append(
                _batch_result(
                    f"{sample}_rep{rep}",
                    f"/data/basalts/{sample}/spot{rep}.txt",
                    {"Si": 50.0, "Al": 15.0, "Fe": 10.0 + i * 0.1, "Mg": 8.0},
                )
            )
    rows = rows_from_batch_results(results)
    used = assign_samples(rows, GroupMode.AUTO)
    assert used == GroupMode.FOLDER
    summaries = summarize_samples(rows)
    assert len(summaries) == 20
    assert all(s.n == 4 for s in summaries)
    assert summaries[0].sample == "B01"


def test_prefix_grouping_flat_folder():
    results = []
    for sample in ("B01", "B02"):
        for rep in ("a", "b", "c", "d"):
            results.append(
                _batch_result(
                    f"{sample}_{rep}",
                    f"/data/pellets/{sample}_{rep}.txt",
                    {"Si": 48.0, "Fe": 12.0},
                )
            )
    rows = rows_from_batch_results(results)
    used = assign_samples(rows, GroupMode.AUTO)
    assert used == GroupMode.PREFIX
    summaries = summarize_samples(rows)
    assert [s.sample for s in summaries] == ["B01", "B02"]
    assert [s.n for s in summaries] == [4, 4]


def test_average_and_std():
    results = [
        _batch_result("B01_1", "/B01/1.txt", {"Fe": 10.0, "Si": 50.0}),
        _batch_result("B01_2", "/B01/2.txt", {"Fe": 12.0, "Si": 50.0}),
        _batch_result("B01_3", "/B01/3.txt", {"Fe": 14.0, "Si": 50.0}),
        _batch_result("B01_4", "/B01/4.txt", {"Fe": 16.0, "Si": 50.0}),
    ]
    rows = rows_from_batch_results(results)
    assign_samples(rows, GroupMode.FOLDER)
    summaries = summarize_samples(rows)
    assert len(summaries) == 1
    assert summaries[0].mean["Fe"] == 13.0
    assert summaries[0].mean["Si"] == 50.0
    np.testing.assert_allclose(summaries[0].std["Fe"], np.std([10, 12, 14, 16], ddof=1))
    assert summaries[0].std["Si"] == 0.0


def test_failed_fits_excluded():
    results = [
        _batch_result("B01_1", "/B01/1.txt", {"Fe": 10.0}),
        _batch_result("B01_2", "/B01/2.txt", {}, ok=False),
        _batch_result("B01_3", "/B01/3.txt", {"Fe": 12.0}),
    ]
    rows = rows_from_batch_results(results)
    assign_samples(rows, GroupMode.FOLDER)
    summaries = summarize_samples(rows)
    assert summaries[0].n == 2
    assert summaries[0].mean["Fe"] == 11.0


def test_regex_named_group():
    results = [
        _batch_result("rockA_spot01", "/x/rockA_spot01.txt", {"Si": 1.0}),
        _batch_result("rockA_spot02", "/x/rockA_spot02.txt", {"Si": 1.0}),
        _batch_result("rockB_spot01", "/x/rockB_spot01.txt", {"Si": 1.0}),
    ]
    rows = rows_from_batch_results(results)
    assign_samples(rows, GroupMode.REGEX, regex=r"^(?P<sample>rock[AB])")
    summaries = summarize_samples(rows)
    assert {s.sample: s.n for s in summaries} == {"rockA": 2, "rockB": 1}


def test_sequential_10_samples_3_spots():
    results = []
    for i in range(1, 31):
        results.append(
            _batch_result(
                f"Spectrum {i}",
                f"/project/Spectrum {i}.txt",
                {"Si": 50.0, "Fe": 10.0 + (i - 1) // 3},
            )
        )
    rows = rows_from_batch_results(results)
    used = assign_samples(
        rows, GroupMode.SEQUENTIAL, spots_per_sample=3, n_samples=10
    )
    assert used == GroupMode.SEQUENTIAL
    summaries = summarize_samples(rows)
    assert [s.sample for s in summaries] == [f"Sample {i:02d}" for i in range(1, 11)]
    assert all(s.n == 3 for s in summaries)
    assert {r.name for r in summaries[0].rows} == {
        "Spectrum 1",
        "Spectrum 2",
        "Spectrum 3",
    }
    assert {r.name for r in summaries[9].rows} == {
        "Spectrum 28",
        "Spectrum 29",
        "Spectrum 30",
    }


def test_sequential_sorts_by_spectrum_number():
    """Alphabetical Spectrum 1, 10, 2… still chunks as 1–3, 4–6, …"""
    order = [1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9]
    results = [
        _batch_result(f"Spectrum {i}", f"/p/Spectrum {i}.txt", {"Si": 1.0})
        for i in order
    ]
    rows = rows_from_batch_results(results)
    assign_samples(rows, GroupMode.SEQUENTIAL, spots_per_sample=3, n_samples=4)
    summaries = summarize_samples(rows)
    assert [s.sample for s in summaries] == [
        "Sample 01",
        "Sample 02",
        "Sample 03",
        "Sample 04",
    ]
    by_sample = {s.sample: {r.name for r in s.rows} for s in summaries}
    assert by_sample["Sample 01"] == {"Spectrum 1", "Spectrum 2", "Spectrum 3"}
    assert by_sample["Sample 04"] == {"Spectrum 10", "Spectrum 11", "Spectrum 12"}


def test_sequential_leftover_becomes_extra_sample():
    results = [
        _batch_result(f"Spectrum {i}", f"/p/{i}.txt", {"Si": 1.0})
        for i in range(1, 32)
    ]
    rows = rows_from_batch_results(results)
    assign_samples(rows, GroupMode.SEQUENTIAL, spots_per_sample=3, n_samples=10)
    summaries = summarize_samples(rows)
    assert len(summaries) == 11
    assert summaries[-1].sample == "Sample 11"
    assert summaries[-1].n == 1
    assert summaries[-1].rows[0].name == "Spectrum 31"


def test_sequential_custom_prefix():
    results = [
        _batch_result(f"Spectrum {i}", f"/p/{i}.txt", {"Fe": 1.0})
        for i in range(1, 7)
    ]
    rows = rows_from_batch_results(results)
    assign_samples(
        rows,
        GroupMode.SEQUENTIAL,
        spots_per_sample=2,
        n_samples=3,
        prefix="Pellet",
    )
    summaries = summarize_samples(rows)
    assert [s.sample for s in summaries] == ["Pellet 01", "Pellet 02", "Pellet 03"]
    assert all(s.n == 2 for s in summaries)


def test_ternary_vertices():
    x, y = ternary_xy(1, 0, 0)
    np.testing.assert_allclose((x, y), (0.0, 0.0), atol=1e-12)
    x, y = ternary_xy(0, 1, 0)
    np.testing.assert_allclose((x, y), (1.0, 0.0), atol=1e-12)
    x, y = ternary_xy(0, 0, 1)
    np.testing.assert_allclose((x, y), (0.5, np.sqrt(3) / 2), atol=1e-12)
    x, y = ternary_xy(0, 0, 0)
    assert np.isnan(x) and np.isnan(y)


def test_ternary_points_from_means():
    results = []
    for sample, fe in (("B01", 10.0), ("B02", 20.0)):
        for rep in range(4):
            results.append(
                _batch_result(
                    f"{sample}_{rep}",
                    f"/{sample}/{rep}.txt",
                    {"Si": 50.0, "Al": 15.0, "Fe": fe, "Mg": 5.0},
                )
            )
    rows = rows_from_batch_results(results)
    assign_samples(rows, GroupMode.FOLDER)
    summaries = summarize_samples(rows)
    pts = ternary_points(summaries, "Si", "Al", "Fe")
    assert len(pts) == 2
    y = {p[0]: p[2] for p in pts}
    assert y["B02"] > y["B01"]


def test_oxide_convert_and_close():
    vals = convert_values({"Si": 50.0, "Al": 10.0}, as_oxides=True, close=False)
    assert "SiO2" in vals and "Al2O3" in vals
    np.testing.assert_allclose(vals["SiO2"], 50.0 * 2.1393, rtol=1e-4)
    closed = close_to_100(vals)
    np.testing.assert_allclose(sum(closed.values()), 100.0)
    assert oxide_label("Fe", "FeO") == "FeO"
    assert oxide_label("Fe", "Fe2O3") == "Fe2O3"
    assert oxide_label("Fe", "Fe3O4") == "Fe3O4"


def test_scatter_and_ratio_and_matrix():
    results = []
    for i, sample in enumerate(("B01", "B02", "B03")):
        for rep in range(4):
            results.append(
                _batch_result(
                    f"{sample}_{rep}",
                    f"/{sample}/{rep}.txt",
                    {"Si": 40.0 + i, "Fe": 10.0 + 2 * i, "Mg": 8.0},
                )
            )
    rows = rows_from_batch_results(results)
    assign_samples(rows, GroupMode.FOLDER)
    summaries = summarize_samples(rows)
    pts = scatter_points(summaries, "Si", "Fe")
    assert len(pts) == 3
    r, rho = correlation(
        np.array([p[1] for p in pts]), np.array([p[2] for p in pts])
    )
    assert r > 0.99 and rho > 0.99
    ratios = ratio_points(summaries, "Fe", "Mg", "Si", "Mg")
    assert len(ratios) == 3
    matrix = correlation_matrix(summaries, ["Si", "Fe", "Mg"])
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-9)


def test_export_sample_means_csv(tmp_path):
    from core.composition import export_sample_means_csv

    results = [
        _batch_result("B01_1", "/B01/1.txt", {"Fe": 10.0, "Si": 50.0}),
        _batch_result("B01_2", "/B01/2.txt", {"Fe": 12.0, "Si": 50.0}),
    ]
    rows = rows_from_batch_results(results)
    assign_samples(rows, GroupMode.FOLDER)
    summaries = summarize_samples(rows)
    out = tmp_path / "means.csv"
    export_sample_means_csv(out, summaries)
    text = out.read_text()
    assert "Sample" in text and "B01" in text
    assert "Fe" in text and "Si" in text


def test_rows_keep_fp_wt_and_relative():
    result = SimpleNamespace(
        spectrum_name="spot",
        spectrum_path="/spot.txt",
        concentrations={"Si": 80.0, "Fe": 20.0},
        fp_wt={"Si": 30.0, "Fe": 10.0, "O": 60.0},
        fp_formula_wt={"SiO2": 64.2, "FeO": 12.9},
        fit_success=True,
    )
    rows = rows_from_batch_results([result])
    assert rows[0].relative["Si"] == 80.0
    assert rows[0].values["Si"] == 80.0
    assert rows[0].wt["Si"] == 30.0
    assert rows[0].formula_wt["SiO2"] == 64.2

    apply_value_source(rows, VALUE_WT, as_oxides=False)
    assert rows[0].values["Si"] == 30.0
    assert "O" in rows[0].values

    apply_value_source(rows, VALUE_WT, as_oxides=True)
    assert "SiO2" in rows[0].values
    assert rows[0].values["SiO2"] == 64.2

    apply_value_source(rows, "relative")
    assert rows[0].values["Si"] == 80.0


def test_composition_row_roundtrip_includes_wt():
    from core.composition import CompositionRow

    row = CompositionRow(
        name="a",
        source_id="/a.txt",
        sample="a",
        values={"Si": 80.0},
        relative={"Si": 80.0},
        wt={"Si": 30.0, "O": 70.0},
        formula_wt={"SiO2": 100.0},
    )
    restored = CompositionRow.from_dict(row.to_dict())
    assert restored.wt["Si"] == 30.0
    assert restored.formula_wt["SiO2"] == 100.0
    assert restored.relative["Si"] == 80.0
