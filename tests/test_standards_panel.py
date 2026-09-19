"""Standards panel: spots list stays in sync with the selected standard."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from core.spectrum import Spectrum
from core.standards_calibration import StandardRecord
from ui.standards_panel import StandardsPanel


def _app():
    return QApplication.instance() or QApplication([])


def _entry(tmp_path, i: int) -> dict:
    energy = np.linspace(0.0, 10.0, 32)
    path = str(tmp_path / f"oreas466 {i}.txt")
    return {
        "path": path,
        "name": f"oreas466 {i}.txt",
        "spectrum": Spectrum(energy=energy, counts=np.ones(32)),
    }


def test_adding_spots_to_selected_standard_refreshes_list(tmp_path, monkeypatch):
    """Replicates added while the standard stays selected must appear in the list."""
    _app()
    monkeypatch.setattr(StandardsPanel, "_auto_load", lambda self: None)
    monkeypatch.setattr(StandardsPanel, "_auto_save_standards_set", lambda self: None)

    panel = StandardsPanel()
    try:
        first = _entry(tmp_path, 1)
        record = StandardRecord(
            name="oreas 466",
            concentrations={"Fe": 10.0},
            spectrum_paths=[first["path"]],
        )
        panel.calibration.add_standard(record)
        panel.spectra["oreas 466"] = [first]
        panel._after_standards_changed(select="oreas 466")
        assert panel.spots_list.count() == 1
        assert panel.standards_table.item(0, 2).text() == "1"

        extras = [_entry(tmp_path, i) for i in (2, 3, 4)]
        monkeypatch.setattr(panel, "_pick_spectrum_files", lambda title: [e["path"] for e in extras])
        monkeypatch.setattr(panel, "_load_spectra_from_paths", lambda paths: (extras, []))
        panel._add_spectra_to_standard("oreas 466")

        assert panel.standards_table.item(0, 2).text() == "4"
        assert panel.spots_list.count() == 4
        labels = [panel.spots_list.item(i).text() for i in range(panel.spots_list.count())]
        assert labels[0].startswith("Spot 1:")
        assert labels[-1].startswith("Spot 4:")
        assert "oreas466 4.txt" in labels[-1]
    finally:
        panel.close()
        panel.deleteLater()


def _write_standard_folder(root, name, n_spots=2):
    folder = root / name
    folder.mkdir()
    paths = []
    for i in range(1, n_spots + 1):
        path = folder / f"{name}_{i:02d}.txt"
        path.write_text("placeholder", encoding="utf-8")
        paths.append(str(path))
    (folder / f"{name}_elements.csv").write_text(
        "Element,Symbol,Concentration\nIron,Fe,10.0\n", encoding="utf-8"
    )
    return folder, paths


def test_import_folders_loads_each_selected_standard(tmp_path, monkeypatch):
    _app()
    monkeypatch.setattr(StandardsPanel, "_auto_load", lambda self: None)
    monkeypatch.setattr(StandardsPanel, "_auto_save_standards_set", lambda self: None)

    a, a_paths = _write_standard_folder(tmp_path, "OREAS_460b", n_spots=2)
    b, b_paths = _write_standard_folder(tmp_path, "NIST_SRM_2586", n_spots=3)

    def fake_load(paths):
        entries = [
            {"path": p, "name": Path(p).name, "spectrum": Spectrum(
                energy=np.linspace(0.0, 10.0, 8), counts=np.ones(8)
            )}
            for p in paths
        ]
        return entries, []

    panel = StandardsPanel()
    try:
        monkeypatch.setattr(panel, "_pick_folders", lambda title: [str(a), str(b)])
        monkeypatch.setattr(panel, "_load_spectra_from_paths", fake_load)
        monkeypatch.setattr(panel, "_log", lambda text: None)
        panel._add_folders()

        assert set(panel.calibration.standards) == {"OREAS_460b", "NIST_SRM_2586"}
        assert len(panel.calibration.standards["OREAS_460b"].spectrum_paths) == 2
        assert len(panel.calibration.standards["NIST_SRM_2586"].spectrum_paths) == 3
        assert panel.calibration.standards["OREAS_460b"].concentrations["Fe"] == 10.0
        assert panel.standards_table.rowCount() == 2
    finally:
        panel.close()
        panel.deleteLater()


def test_import_parent_folder_imports_children(tmp_path, monkeypatch):
    _app()
    monkeypatch.setattr(StandardsPanel, "_auto_load", lambda self: None)
    monkeypatch.setattr(StandardsPanel, "_auto_save_standards_set", lambda self: None)
    _write_standard_folder(tmp_path, "LKSD_1")
    _write_standard_folder(tmp_path, "PACS_2")
    (tmp_path / "foils").mkdir()
    (tmp_path / "foils" / "Fe.txt").write_text("placeholder", encoding="utf-8")

    def fake_load(self, paths):
        entries = [
            {"path": p, "name": Path(p).name, "spectrum": Spectrum(
                energy=np.linspace(0.0, 10.0, 8), counts=np.ones(8)
            )}
            for p in paths
        ]
        return entries, []

    panel = StandardsPanel()
    try:
        monkeypatch.setattr(StandardsPanel, "_load_spectra_from_paths", fake_load)
        added, updated, skipped, errors = panel._import_standard_folders([str(tmp_path)])
        assert set(added) == {"LKSD_1", "PACS_2"}
        assert updated == []
        assert errors == []
        assert any(path.name == "foils" for path, _reason in skipped)
    finally:
        panel.close()
        panel.deleteLater()
