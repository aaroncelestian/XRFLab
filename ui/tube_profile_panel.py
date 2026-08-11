"""
Tube profile calibration panel — measure Rh (etc.) scatter line ratios
at each instrument voltage mode (15 / 30 / 50 kV).
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QFileDialog, QTextEdit, QMessageBox,
    QFormLayout,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtCore import QStandardPaths

from core.tube_profile import (
    TubeProfileLibrary, DEFAULT_TUBE_KVS, measure_tube_profile_from_spectrum,
    default_tube_profile,
)
from utils.io_handler import IOHandler


class TubeProfilePanel(QWidget):
    """Measure and manage per-kV tube scatter profiles from blank spectra."""

    library_changed = Signal(object)  # TubeProfileLibrary

    def __init__(self, parent=None):
        super().__init__(parent)
        self.library = TubeProfileLibrary()
        self.io = IOHandler()
        self._blank_path = None
        self._setup_ui()
        self._auto_load()

    @staticmethod
    def get_default_path():
        app_data = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if not app_data:
            app_data = str(Path.home() / ".xrflab")
        cal_dir = Path(app_data) / "calibrations"
        cal_dir.mkdir(parents=True, exist_ok=True)
        return cal_dir / "tube_profiles.json"

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        info = QLabel(
            "Measure a <b>blank / scatter</b> spectrum at each tube voltage. "
            "Relative intensities of Rh K, L, and Compton lines become the "
            "instrument tube profile. Deviations during sample fitting flag "
            "overlaps (e.g. sample peaks under tube lines)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Mode / tube settings
        mode_group = QGroupBox("Instrument mode")
        form = QFormLayout(mode_group)

        self.tube_combo = QComboBox()
        self.tube_combo.addItems(["Rh", "W", "Mo", "Ag", "Cr", "Cu"])
        self.tube_combo.setCurrentText("Rh")
        form.addRow("Tube anode:", self.tube_combo)

        self.kv_combo = QComboBox()
        for kv in DEFAULT_TUBE_KVS:
            self.kv_combo.addItem(f"{kv:g} kV", float(kv))
        self.kv_combo.setCurrentIndex(2)  # 50 default
        self.kv_combo.setToolTip(
            "Your instrument modes. Select the voltage used for the blank spectrum."
        )
        form.addRow("Tube voltage:", self.kv_combo)

        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(30, 150)
        self.angle_spin.setValue(90)
        self.angle_spin.setSuffix("°")
        form.addRow("Scatter angle:", self.angle_spin)

        self.compton_fwhm_spin = QDoubleSpinBox()
        self.compton_fwhm_spin.setRange(100, 800)
        self.compton_fwhm_spin.setValue(250)
        self.compton_fwhm_spin.setSuffix(" eV")
        form.addRow("Compton FWHM:", self.compton_fwhm_spin)

        layout.addWidget(mode_group)

        # Blank spectrum
        blank_group = QGroupBox("Blank / scatter spectrum")
        blank_layout = QVBoxLayout(blank_group)
        self.blank_label = QLabel("No spectrum loaded")
        self.blank_label.setStyleSheet("color: #888;")
        blank_layout.addWidget(self.blank_label)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load Blank Spectrum…")
        load_btn.clicked.connect(self._load_blank)
        btn_row.addWidget(load_btn)

        measure_btn = QPushButton("Measure Profile")
        measure_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 6px; }"
        )
        measure_btn.clicked.connect(self._measure_profile)
        btn_row.addWidget(measure_btn)
        blank_layout.addLayout(btn_row)
        layout.addWidget(blank_group)

        # Library status
        status_group = QGroupBox("Saved profiles")
        status_layout = QVBoxLayout(status_group)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(160)
        self.status_text.setFontFamily("Courier")
        status_layout.addWidget(self.status_text)

        save_row = QHBoxLayout()
        save_btn = QPushButton("Save Library…")
        save_btn.clicked.connect(self._save_library)
        save_row.addWidget(save_btn)
        load_lib_btn = QPushButton("Load Library…")
        load_lib_btn.clicked.connect(self._load_library_dialog)
        save_row.addWidget(load_lib_btn)
        apply_btn = QPushButton("Apply to Analysis")
        apply_btn.clicked.connect(self._apply_library)
        save_row.addWidget(apply_btn)
        status_layout.addLayout(save_row)
        layout.addWidget(status_group, stretch=1)

        self._refresh_status()

    def _current_kv(self) -> float:
        return float(self.kv_combo.currentData())

    def _load_blank(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Blank / Scatter Spectrum",
            "",
            "Spectra (*.txt *.csv *.mca *.msa *.emsa);;All Files (*)",
        )
        if not path:
            return
        try:
            spectrum = self.io.load_spectrum(path)
            self._blank_path = path
            self._blank_spectrum = spectrum
            # Prefer metadata voltage if present
            meta = getattr(spectrum, 'metadata', {}) or {}
            if 'excitation_energy' in meta:
                exc = float(meta['excitation_energy'])
                # Snap combo to nearest mode
                best_i = 0
                best_d = 1e9
                for i in range(self.kv_combo.count()):
                    d = abs(float(self.kv_combo.itemData(i)) - exc)
                    if d < best_d:
                        best_d = d
                        best_i = i
                self.kv_combo.setCurrentIndex(best_i)
            self.blank_label.setText(Path(path).name)
            self.blank_label.setStyleSheet("color: #1b7a1b; font-weight: bold;")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _measure_profile(self):
        if not getattr(self, '_blank_spectrum', None):
            QMessageBox.warning(
                self,
                "No Blank",
                "Load a blank / scatter spectrum first.",
            )
            return

        kv = self._current_kv()
        tube = self.tube_combo.currentText()
        try:
            profile = measure_tube_profile_from_spectrum(
                energy=self._blank_spectrum.energy,
                counts=self._blank_spectrum.counts,
                tube_element=tube,
                tube_kv=kv,
                scatter_angle_deg=self.angle_spin.value(),
                compton_fwhm_kev=self.compton_fwhm_spin.value() / 1000.0,
                spectrum_path=self._blank_path,
            )
            self.library.tube_element = tube
            self.library.set_profile(profile)
            self._auto_save()
            self._refresh_status()
            self.library_changed.emit(self.library)

            lines = "\n".join(
                f"  {name}: {val:.3f}"
                for name, val in sorted(
                    profile.line_ratios.items(), key=lambda kv: -kv[1]
                )
            )
            QMessageBox.information(
                self,
                "Profile Measured",
                f"{tube} @ {kv:g} kV ({profile.source})\n"
                f"Reference: {profile.reference_line}\n"
                f"Compton scale: {profile.compton_scale:.3f}\n\n"
                f"Relative intensities:\n{lines}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Measure Error", str(e))

    def _refresh_status(self):
        lines = [
            f"Tube: {self.library.tube_element}",
            f"Modes: {', '.join(f'{k:g}' for k in self.library.available_kvs)} kV",
            "",
        ]
        for kv in self.library.available_kvs:
            p = self.library.profiles.get(TubeProfileLibrary._key(kv))
            if p is None:
                lines.append(f"{kv:g} kV:  — not measured (defaults will be used)")
                continue
            n = len(p.line_ratios)
            lines.append(
                f"{kv:g} kV:  {p.source}  ref={p.reference_line}  "
                f"{n} lines  Compton={p.compton_scale:.2f}"
            )
            top = sorted(p.line_ratios.items(), key=lambda x: -x[1])[:6]
            for name, val in top:
                lines.append(f"         {name:12s}  {val:.3f}")
            lines.append("")
        self.status_text.setPlainText("\n".join(lines))

    def _auto_save(self):
        try:
            self.library.save(self.get_default_path())
        except Exception as e:
            print(f"Warning: could not auto-save tube profiles: {e}")

    def _auto_load(self):
        path = self.get_default_path()
        if path.exists():
            try:
                self.library = TubeProfileLibrary.load(path)
                self._refresh_status()
                # Defer emit until connected by main window
            except Exception as e:
                print(f"Warning: could not load tube profiles: {e}")
                # Seed defaults for each mode so Analysis always has something
                for kv in DEFAULT_TUBE_KVS:
                    self.library.set_profile(default_tube_profile('Rh', kv))
        else:
            for kv in DEFAULT_TUBE_KVS:
                self.library.set_profile(default_tube_profile('Rh', kv))
            self._refresh_status()

    def _save_library(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Tube Profile Library",
            str(Path.home() / "tube_profiles.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            self.library.save(path)
            self._auto_save()
            QMessageBox.information(self, "Saved", f"Saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_library_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Tube Profile Library",
            str(Path.home()),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            self.library = TubeProfileLibrary.load(path)
            self._auto_save()
            self._refresh_status()
            self.library_changed.emit(self.library)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _apply_library(self):
        self._auto_save()
        self.library_changed.emit(self.library)
        QMessageBox.information(
            self,
            "Applied",
            "Tube profiles applied to Analysis.\n"
            "The profile nearest to each spectrum's tube voltage will be used.",
        )

    def get_library(self) -> TubeProfileLibrary:
        return self.library
