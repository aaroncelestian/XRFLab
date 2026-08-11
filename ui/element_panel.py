"""
Element selection panel for XRF analysis
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QComboBox, QDoubleSpinBox, QTreeWidget, QTreeWidgetItem,
    QPushButton, QCheckBox, QTabWidget, QDialog, QTextEdit, QDialogButtonBox,
    QFormLayout, QListWidget, QListWidgetItem, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from ui.periodic_table_widget import PeriodicTableWidget
from core.xray_data import get_element_lines, get_element_info


class ElementPanel(QWidget):
    """Panel for sample information, element selection, and experimental parameters"""
    
    elements_changed = Signal(list)  # Emitted when selected elements change
    fit_requested = Signal()  # Emitted when fit button is clicked
    peak_find_requested = Signal()  # Preview auto peak detection only
    element_clicked = Signal(str, int)  # Emitted when element clicked (symbol, Z)
    peak_list_changed = Signal()  # Emitted when peak list is edited (delete/clear)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.selected_elements = []
        self._peak_list_data = []  # List of peak dicts shown in the UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the panel layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)  # Reduced margins
        layout.setSpacing(3)  # Tighter spacing between groups
        
        # Sample information group
        sample_group = self._create_sample_info_group()
        layout.addWidget(sample_group)
        
        # Experimental parameters group
        exp_params_group = self._create_exp_params_group()
        layout.addWidget(exp_params_group)
        
        # Element selection group
        element_group = self._create_element_selection_group()
        layout.addWidget(element_group, stretch=1)
        
        # Fitting controls group
        fitting_group = self._create_fitting_controls_group()
        layout.addWidget(fitting_group)
    
    def _create_sample_info_group(self):
        """Create sample information input group"""
        group = QGroupBox("Sample Information")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)  # Reduced top margin
        layout.setSpacing(3)  # Tighter spacing
        
        # Sample name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.sample_name_edit = QLineEdit()
        self.sample_name_edit.setPlaceholderText("Enter sample name")
        name_layout.addWidget(self.sample_name_edit)
        layout.addLayout(name_layout)
        
        # Sample type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.sample_type_combo = QComboBox()
        self.sample_type_combo.addItems([
            "Unknown",
            "Bulk",
            "Thin Film",
            "Powder",
            "Liquid"
        ])
        type_layout.addWidget(self.sample_type_combo)
        layout.addLayout(type_layout)
        
        # Sample thickness (for thin films)
        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(QLabel("Thickness:"))
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0, 10000)
        self.thickness_spin.setSuffix(" µm")
        self.thickness_spin.setEnabled(False)
        thickness_layout.addWidget(self.thickness_spin)
        layout.addLayout(thickness_layout)
        
        # Connect sample type change
        self.sample_type_combo.currentTextChanged.connect(self._on_sample_type_changed)
        
        return group
    
    def _create_exp_params_group(self):
        """Create experimental parameters group"""
        group = QGroupBox("Experimental Parameters")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)  # Reduced top margin
        layout.setSpacing(3)  # Tighter spacing
        
        # Excitation energy
        energy_layout = QHBoxLayout()
        energy_layout.addWidget(QLabel("Excitation:"))
        self.excitation_spin = QDoubleSpinBox()
        self.excitation_spin.setRange(1, 100)
        self.excitation_spin.setValue(20)
        self.excitation_spin.setSuffix(" keV")
        self.excitation_spin.setToolTip("X-ray tube voltage")
        energy_layout.addWidget(self.excitation_spin)
        layout.addLayout(energy_layout)
        
        # Tube current
        current_layout = QHBoxLayout()
        current_layout.addWidget(QLabel("Current:"))
        self.current_spin = QDoubleSpinBox()
        self.current_spin.setRange(0.1, 10)
        self.current_spin.setValue(1.0)
        self.current_spin.setSuffix(" mA")
        current_layout.addWidget(self.current_spin)
        layout.addLayout(current_layout)
        
        # Acquisition time
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Live Time:"))
        self.live_time_spin = QDoubleSpinBox()
        self.live_time_spin.setRange(1, 10000)
        self.live_time_spin.setValue(100)
        self.live_time_spin.setSuffix(" s")
        time_layout.addWidget(self.live_time_spin)
        layout.addLayout(time_layout)
        
        # Detector type
        detector_layout = QHBoxLayout()
        detector_layout.addWidget(QLabel("Detector:"))
        self.detector_combo = QComboBox()
        self.detector_combo.addItems([
            "Si(Li)",
            "SDD",
            "HPGe",
            "Proportional Counter"
        ])
        detector_layout.addWidget(self.detector_combo)
        layout.addLayout(detector_layout)
        
        # Incident angle
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Angle:"))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0, 90)
        self.angle_spin.setValue(45)
        self.angle_spin.setSuffix(" °")
        self.angle_spin.setToolTip("Incident angle")
        angle_layout.addWidget(self.angle_spin)
        layout.addLayout(angle_layout)
        
        return group
    
    def _create_element_selection_group(self):
        """Create element selection with periodic table"""
        group = QGroupBox("Element Selection")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(3, 6, 3, 3)
        layout.setSpacing(2)
        
        # Create periodic table widget
        self.periodic_table = PeriodicTableWidget()
        self.periodic_table.elements_changed.connect(self._on_periodic_table_changed)
        self.periodic_table.element_clicked.connect(self.element_clicked.emit)
        self.periodic_table.element_info_requested.connect(self._show_element_info)
        layout.addWidget(self.periodic_table)
        
        return group
    
    def _create_fitting_controls_group(self):
        """Create fitting controls group"""
        group = QGroupBox("Fitting Controls")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)  # Reduced top margin
        layout.setSpacing(3)  # Tighter spacing
        
        # Background method
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("Background:"))
        self.background_combo = QComboBox()
        self.background_combo.addItems([
            "SNIP",
            "Polynomial",
            "Linear",
            "None"
        ])
        bg_layout.addWidget(self.background_combo)
        layout.addLayout(bg_layout)
        
        # Peak shape
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("Peak Shape:"))
        self.peak_shape_combo = QComboBox()
        self.peak_shape_combo.addItems([
            "Gaussian",
            "Voigt",
            "Pseudo-Voigt",
            "Hypermet",
            "Tail-Gaussian"
        ])
        self.peak_shape_combo.setCurrentText("Voigt")  # Set Voigt as default
        self.peak_shape_combo.setToolTip(
            "Gaussian: Simple symmetric peak\n"
            "Voigt: More accurate for X-ray peaks\n"
            "Pseudo-Voigt: Fast approximation of Voigt\n"
            "Hypermet: Includes low-energy tail\n"
            "Tail-Gaussian: Simplified tail model"
        )
        shape_layout.addWidget(self.peak_shape_combo)
        layout.addLayout(shape_layout)

        # FWHM calibration status (locked widths when active)
        self.fwhm_status_label = QLabel(
            "FWHM: no calibration — widths free in LS"
        )
        self.fwhm_status_label.setWordWrap(True)
        self.fwhm_status_label.setStyleSheet(
            "color: #994400; font-weight: bold; padding: 4px;"
        )
        self.fwhm_status_label.setToolTip(
            "When a FWHM calibration is loaded, Analysis locks peak widths\n"
            "to FWHM(E) during least-squares (amplitude/center only).\n"
            "Without calibration, width is free to refine within bounds."
        )
        layout.addWidget(self.fwhm_status_label)
        
        # Escape peaks checkbox
        self.escape_peaks_check = QCheckBox("Include Escape Peaks")
        self.escape_peaks_check.setChecked(True)
        layout.addWidget(self.escape_peaks_check)
        
        # Pile-up correction checkbox
        self.pileup_check = QCheckBox("Pile-up Correction")
        layout.addWidget(self.pileup_check)
        
        # X-ray tube lines
        tube_layout = QHBoxLayout()
        self.tube_lines_check = QCheckBox("Include Tube Lines:")
        self.tube_lines_check.setChecked(True)
        self.tube_lines_check.setToolTip("Model X-ray tube characteristic lines (excluded from quantification)")
        tube_layout.addWidget(self.tube_lines_check)
        
        self.tube_element_combo = QComboBox()
        self.tube_element_combo.addItems(["Rh", "W", "Mo", "Ag", "Cr", "Cu"])
        self.tube_element_combo.setCurrentText("Rh")
        self.tube_element_combo.setToolTip("X-ray tube anode element")
        tube_layout.addWidget(self.tube_element_combo)
        layout.addLayout(tube_layout)

        # Compton (inelastic) tube scatter — broad ~19 keV feature for Rh
        compton_row = QHBoxLayout()
        self.compton_check = QCheckBox("Include Compton:")
        self.compton_check.setChecked(True)
        self.compton_check.setToolTip(
            "Model inelastic tube scatter (e.g. Rh Compton ~18.8 keV at 90°).\n"
            "Uses a wide fixed FWHM (~250 eV), excluded from quantification.\n"
            "Peak find skips this region so it does not seed false peaks."
        )
        compton_row.addWidget(self.compton_check)

        compton_row.addWidget(QLabel("θ:"))
        self.scatter_angle_spin = QDoubleSpinBox()
        self.scatter_angle_spin.setRange(30.0, 150.0)
        self.scatter_angle_spin.setDecimals(0)
        self.scatter_angle_spin.setSingleStep(5.0)
        self.scatter_angle_spin.setValue(90.0)
        self.scatter_angle_spin.setSuffix("°")
        self.scatter_angle_spin.setToolTip(
            "Tube–sample–detector scatter angle for Compton energy.\n"
            "90° is typical for many XRF geometries."
        )
        compton_row.addWidget(self.scatter_angle_spin)

        compton_row.addWidget(QLabel("FWHM:"))
        self.compton_fwhm_spin = QDoubleSpinBox()
        self.compton_fwhm_spin.setRange(100.0, 800.0)
        self.compton_fwhm_spin.setDecimals(0)
        self.compton_fwhm_spin.setSingleStep(25.0)
        self.compton_fwhm_spin.setValue(250.0)
        self.compton_fwhm_spin.setSuffix(" eV")
        self.compton_fwhm_spin.setToolTip(
            "Fixed width for Compton peaks (much broader than detector FWHM)."
        )
        compton_row.addWidget(self.compton_fwhm_spin)
        layout.addLayout(compton_row)

        self.tube_lines_check.toggled.connect(self._on_tube_lines_toggled)
        self._on_tube_lines_toggled(self.tube_lines_check.isChecked())
        
        # Peak detection fine-tuning
        detect_group = QGroupBox("Peak Detection")
        detect_layout = QFormLayout(detect_group)
        detect_layout.setContentsMargins(5, 8, 5, 5)
        detect_layout.setSpacing(4)
        
        self.auto_find_check = QCheckBox("Auto-find unknown peaks")
        self.auto_find_check.setChecked(True)
        self.auto_find_check.setToolTip(
            "Also detect peaks that are not in the selected-element line list.\n"
            "Turn off to fit only selected element (and tube) lines.\n"
            "Peaks under Compton tube scatter are excluded when Compton is on."
        )
        detect_layout.addRow(self.auto_find_check)
        
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.1, 50.0)
        self.prominence_spin.setDecimals(1)
        self.prominence_spin.setSingleStep(0.5)
        self.prominence_spin.setValue(2.0)
        self.prominence_spin.setSuffix(" %")
        self.prominence_spin.setToolTip(
            "Minimum peak prominence as a percent of the tallest peak.\n"
            "Lower = find more/weaker peaks; higher = only strong peaks."
        )
        detect_layout.addRow("Prominence:", self.prominence_spin)
        
        self.min_height_spin = QDoubleSpinBox()
        self.min_height_spin.setRange(0, 1e9)
        self.min_height_spin.setDecimals(0)
        self.min_height_spin.setSingleStep(10)
        self.min_height_spin.setValue(0)
        self.min_height_spin.setSpecialValueText("Off")
        self.min_height_spin.setToolTip(
            "Minimum absolute peak height in counts (after background subtraction).\n"
            "0 / Off disables this filter."
        )
        detect_layout.addRow("Min height:", self.min_height_spin)
        
        self.min_separation_spin = QDoubleSpinBox()
        self.min_separation_spin.setRange(10, 2000)
        self.min_separation_spin.setDecimals(0)
        self.min_separation_spin.setSingleStep(10)
        self.min_separation_spin.setValue(80)
        self.min_separation_spin.setSuffix(" eV")
        self.min_separation_spin.setToolTip(
            "Minimum energy separation between auto-detected peaks.\n"
            "Lower finds closer peaks; raise to avoid splitting one peak."
        )
        detect_layout.addRow("Min separation:", self.min_separation_spin)
        
        self.show_markers_check = QCheckBox("Show peak markers on spectrum")
        self.show_markers_check.setChecked(True)
        self.show_markers_check.setToolTip(
            "Draw vertical markers for fitted / previewed peaks on the spectrum plot."
        )
        detect_layout.addRow(self.show_markers_check)

        # Post-fit smart ID / overlap analysis
        smart_group = QGroupBox("Post-fit Smart ID")
        smart_layout = QFormLayout(smart_group)
        smart_layout.setContentsMargins(5, 8, 5, 5)
        smart_layout.setSpacing(4)

        self.smart_id_check = QCheckBox("Analyze overlaps & multi-line IDs after fit")
        self.smart_id_check.setChecked(False)
        self.smart_id_check.setToolTip(
            "After fitting, flag peaks whose FWHM is broader than the detector\n"
            "model (possible unresolved overlap), use shape hints (η / tail),\n"
            "and check Kβ (and other) lines to confirm or challenge labels."
        )
        smart_layout.addRow(self.smart_id_check)

        self.fwhm_excess_spin = QDoubleSpinBox()
        self.fwhm_excess_spin.setRange(5.0, 200.0)
        self.fwhm_excess_spin.setDecimals(0)
        self.fwhm_excess_spin.setSingleStep(5.0)
        self.fwhm_excess_spin.setValue(30.0)
        self.fwhm_excess_spin.setSuffix(" eV")
        self.fwhm_excess_spin.setToolTip(
            "Flag a peak as an overlap suspect when measured FWHM exceeds\n"
            "the expected detector FWHM by more than this amount."
        )
        smart_layout.addRow("FWHM excess:", self.fwhm_excess_spin)

        self.smart_id_apply_check = QCheckBox("Apply suggestions (relabel + overlap seeds)")
        self.smart_id_apply_check.setChecked(False)
        self.smart_id_apply_check.setToolTip(
            "When checked, high-confidence multi-line suggestions update peak\n"
            "labels, and overlap suspects get an extra peak-list seed so you\n"
            "can Fit again to resolve the envelope. Review Elements afterward."
        )
        smart_layout.addRow(self.smart_id_apply_check)

        layout.addWidget(detect_group)
        layout.addWidget(smart_group)
        
        # Action buttons
        preview_button = QPushButton("Preview Peak Find")
        preview_button.setToolTip(
            "Run peak detection only (no fit) and mark found peaks on the spectrum.\n"
            "Use this to tune prominence / height / separation."
        )
        preview_button.clicked.connect(self.peak_find_requested.emit)
        layout.addWidget(preview_button)
        
        # Fit button
        self.fit_button = QPushButton("Fit Spectrum")
        self.fit_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.fit_button.clicked.connect(self.fit_requested.emit)
        layout.addWidget(self.fit_button)

        # Found / candidate peaks list (editable)
        peaks_group = QGroupBox("Found Peaks")
        peaks_layout = QVBoxLayout(peaks_group)
        peaks_layout.setContentsMargins(5, 8, 5, 5)
        peaks_layout.setSpacing(3)

        self.peak_list_widget = QListWidget()
        self.peak_list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.peak_list_widget.setMinimumHeight(120)
        self.peak_list_widget.setToolTip(
            "Peaks that will be fitted.\n"
            "Use Preview Peak Find or Fit Spectrum to populate.\n"
            "Select and Delete to remove peaks before fitting."
        )
        peaks_layout.addWidget(self.peak_list_widget)

        self.use_peak_list_check = QCheckBox("Fit using peak list")
        self.use_peak_list_check.setChecked(False)
        self.use_peak_list_check.setToolTip(
            "When checked, Fit Spectrum uses the peaks listed here\n"
            "(after any deletions) instead of rebuilding from auto-find.\n"
            "Labels are always rebuilt from the current Elements selection:\n"
            "uncheck false IDs, check missing ones, then Fit again.\n"
            "Automatically enabled when you Preview Peak Find or delete a peak."
        )
        peaks_layout.addWidget(self.use_peak_list_check)

        peak_btn_row = QHBoxLayout()
        delete_peak_btn = QPushButton("Delete Selected")
        delete_peak_btn.setToolTip("Remove selected peaks from the list")
        delete_peak_btn.clicked.connect(self._delete_selected_peaks)
        peak_btn_row.addWidget(delete_peak_btn)

        clear_peaks_btn = QPushButton("Clear")
        clear_peaks_btn.setToolTip("Clear the peak list (next fit will rebuild)")
        clear_peaks_btn.clicked.connect(self._clear_peak_list)
        peak_btn_row.addWidget(clear_peaks_btn)
        peaks_layout.addLayout(peak_btn_row)

        layout.addWidget(peaks_group)
        
        return group

    def set_peak_list(self, peaks, enable_use_list=False):
        """
        Populate the found-peaks list from peak dicts or Peak objects.

        Args:
            peaks: Iterable of dicts or Peak-like objects with energy/element/line
            enable_use_list: If True, check "Fit using peak list"
        """
        self._peak_list_data = []
        self.peak_list_widget.clear()

        for p in peaks or []:
            if hasattr(p, 'energy'):
                entry = {
                    'energy': float(p.energy),
                    'element': getattr(p, 'element', None),
                    'line': getattr(p, 'line', None),
                    'is_tube_line': bool(getattr(p, 'is_tube_line', False)),
                }
                fixed = getattr(p, 'fixed_fwhm', None)
                if fixed is None and getattr(p, 'line', None):
                    if str(p.line).startswith('Compton'):
                        fixed = getattr(p, 'fwhm', None)
                if fixed is not None:
                    entry['fixed_fwhm'] = float(fixed)
                    entry['exclusion_half_width_kev'] = max(0.30, 1.5 * float(fixed))
            else:
                entry = {
                    'energy': float(p['energy']),
                    'element': p.get('element'),
                    'line': p.get('line'),
                    'is_tube_line': bool(p.get('is_tube_line', False)),
                }
                if p.get('fixed_fwhm') is not None:
                    entry['fixed_fwhm'] = float(p['fixed_fwhm'])
                if p.get('exclusion_half_width_kev') is not None:
                    entry['exclusion_half_width_kev'] = float(
                        p['exclusion_half_width_kev']
                    )
            self._peak_list_data.append(entry)
            self.peak_list_widget.addItem(self._format_peak_item(entry))

        if enable_use_list and self._peak_list_data:
            self.use_peak_list_check.setChecked(True)

    def get_peak_list(self):
        """Return a copy of the current peak list (dicts)."""
        return [dict(p) for p in self._peak_list_data]

    def should_use_peak_list(self):
        """True if Fit should use the listed peaks instead of rebuilding."""
        return (
            self.use_peak_list_check.isChecked()
            and len(self._peak_list_data) > 0
        )

    def _format_peak_item(self, entry):
        energy = entry['energy']
        element = entry.get('element')
        line = entry.get('line')
        is_tube = entry.get('is_tube_line', False)
        fixed = entry.get('fixed_fwhm')

        tags = []
        if is_tube:
            tags.append("tube")
        if fixed is not None:
            tags.append(f"wide {fixed*1000:.0f} eV")
        tag_txt = f" [{', '.join(tags)}]" if tags else ""

        if element and line:
            label = f"{energy:.3f} keV  {element} {line}{tag_txt}"
        elif element:
            label = f"{energy:.3f} keV  {element}{tag_txt}"
        else:
            label = f"{energy:.3f} keV  (unknown)"
        return label

    def _on_tube_lines_toggled(self, checked):
        self.compton_check.setEnabled(checked)
        self.scatter_angle_spin.setEnabled(checked and self.compton_check.isChecked())
        self.compton_fwhm_spin.setEnabled(checked and self.compton_check.isChecked())
        if not hasattr(self, '_compton_angle_linked'):
            self.compton_check.toggled.connect(
                lambda c: self._on_tube_lines_toggled(self.tube_lines_check.isChecked())
            )
            self._compton_angle_linked = True

    def _delete_selected_peaks(self):
        rows = sorted(
            {idx.row() for idx in self.peak_list_widget.selectedIndexes()},
            reverse=True,
        )
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self._peak_list_data):
                del self._peak_list_data[row]
            self.peak_list_widget.takeItem(row)
        if self._peak_list_data:
            self.use_peak_list_check.setChecked(True)
        else:
            self.use_peak_list_check.setChecked(False)
        self.peak_list_changed.emit()

    def _clear_peak_list(self):
        self._peak_list_data = []
        self.peak_list_widget.clear()
        self.use_peak_list_check.setChecked(False)
        self.peak_list_changed.emit()
    
    def _on_periodic_table_changed(self, elements):
        """Handle periodic table selection changes"""
        self.selected_elements = elements
        self.elements_changed.emit(self.selected_elements)
    
    def _show_element_info(self, symbol, z):
        """Show detailed element information dialog"""
        # Get element data
        info = get_element_info(symbol, z)
        lines = get_element_lines(symbol, z)
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{info['name']} ({symbol}) - Element Information")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Element info
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(150)
        
        info_html = f"""
        <h2>{info['name']} ({symbol})</h2>
        <p><b>Atomic Number:</b> {z}</p>
        <p><b>Atomic Weight:</b> {info['atomic_weight']:.4f} g/mol</p>
        <p><b>Density:</b> {info['density']:.4f} g/cm³</p>
        """
        info_text.setHtml(info_html)
        layout.addWidget(info_text)
        
        # Emission lines
        lines_text = QTextEdit()
        lines_text.setReadOnly(True)
        lines_text.setFont(QFont("Courier", 10))
        
        lines_content = "<h3>X-ray Emission Lines</h3>"
        
        for series in ['K', 'L', 'M', 'N']:
            if lines[series]:
                lines_content += f"<p><b>{series} Series:</b></p><ul>"
                for line in lines[series]:
                    lines_content += f"<li>{line['name']}: {line['energy']:.3f} keV</li>"
                lines_content += "</ul>"
        
        if not any(lines.values()):
            lines_content += "<p><i>No emission line data available</i></p>"
        
        lines_text.setHtml(lines_content)
        layout.addWidget(lines_text)
        
        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.exec()
    
    def _on_sample_type_changed(self, sample_type):
        """Enable/disable thickness input based on sample type"""
        self.thickness_spin.setEnabled(sample_type == "Thin Film")
    
    def get_selected_elements(self):
        """Return list of selected elements"""
        return self.selected_elements

    def set_selected_elements(self, symbols):
        """
        Select elements on the periodic table by symbol list.

        Used after a fit to populate the Elements tab with identified
        sample elements so the user can uncheck false IDs and refit.
        """
        symbols = [s for s in (symbols or []) if s]
        self.periodic_table.set_selected_elements(symbols)
        # Keep local cache in sync (periodic table emits elements_changed)
        self.selected_elements = self.periodic_table.get_selected_elements()
    
    def update_from_spectrum_metadata(self, metadata: dict):
        """
        Auto-populate experimental parameters from spectrum metadata
        
        Args:
            metadata: Spectrum metadata dictionary
        """
        # Update excitation energy
        if 'excitation_energy' in metadata:
            self.excitation_spin.setValue(float(metadata['excitation_energy']))
        
        # Update tube current
        if 'tube_current' in metadata:
            # Convert from nA to mA
            # PROBECUR in EMSA files is in nanoamps (nA)
            current = float(metadata['tube_current'])
            print(f"  DEBUG: Raw tube_current from metadata: {current} nA")
            # Convert nA → mA (divide by 1,000,000)
            current = current / 1000000.0
            print(f"  DEBUG: Converted to mA: {current}")
            self.current_spin.setValue(current)
            print(f"  DEBUG: Set current spin to: {current} mA")
        
        # Update live time
        if 'live_time' in metadata:
            self.live_time_spin.setValue(float(metadata['live_time']))
        
        # Update incident angle
        if 'incident_angle' in metadata:
            self.angle_spin.setValue(float(metadata['incident_angle']))
        
        # Note: takeoff angle is in metadata but not in UI (could add if needed)
        
        print(f"Updated experimental parameters from spectrum metadata:")
        print(f"  Excitation: {self.excitation_spin.value()} keV")
        print(f"  Current: {self.current_spin.value()} mA")
        print(f"  Live time: {self.live_time_spin.value()} s")
        print(f"  Incident angle: {self.angle_spin.value()}°")
    
    def update_fwhm_status(self, fwhm_calibration=None):
        """
        Update Fitting-tab FWHM status from an applied calibration (or clear it).

        When calibration is active, Analysis locks peak widths to FWHM(E).
        """
        if fwhm_calibration is None:
            self.fwhm_status_label.setText(
                "FWHM: no calibration — widths free in LS"
            )
            self.fwhm_status_label.setStyleSheet(
                "color: #994400; font-weight: bold; padding: 4px;"
            )
            return

        if getattr(fwhm_calibration, 'model_type', None) == 'detector':
            params = getattr(fwhm_calibration, 'parameters', {}) or {}
            fwhm_0_ev = float(params.get('fwhm_0', 0.0)) * 1000.0
            epsilon = float(params.get('epsilon', 0.0))
            r2 = getattr(fwhm_calibration, 'r_squared', None)
            r2_txt = f"  R²={r2:.4f}" if r2 is not None else ""
            text = (
                f"FWHM locked: detector model  "
                f"FWHM₀={fwhm_0_ev:.1f} eV  ε={epsilon:.4g}{r2_txt}"
            )
        else:
            model = getattr(fwhm_calibration, 'model_type', 'unknown')
            r2 = getattr(fwhm_calibration, 'r_squared', None)
            r2_txt = f"  R²={r2:.4f}" if r2 is not None else ""
            text = f"FWHM locked: {model} model{r2_txt}"

        # Example width at Fe Kα for intuition
        try:
            from core.peak_fitting import PeakFitter
            fe_fwhm_ev = float(PeakFitter.calculate_fwhm(6.403)) * 1000.0
            text += f"  (≈{fe_fwhm_ev:.0f} eV @ Fe Kα)"
        except Exception:
            pass

        self.fwhm_status_label.setText(text)
        self.fwhm_status_label.setStyleSheet(
            "color: #1b7a1b; font-weight: bold; padding: 4px;"
        )

    def get_experimental_params(self):
        """Return dictionary of experimental parameters"""
        return {
            'excitation_energy': self.excitation_spin.value(),
            'tube_current': self.current_spin.value(),
            'live_time': self.live_time_spin.value(),
            'detector_type': self.detector_combo.currentText(),
            'incident_angle': self.angle_spin.value()
        }
    
    def get_fitting_params(self):
        """Return dictionary of fitting parameters"""
        # Convert UI peak shape names to internal format
        peak_shape_map = {
            'Gaussian': 'gaussian',
            'Voigt': 'voigt',
            'Pseudo-Voigt': 'pseudo_voigt',
            'Hypermet': 'hypermet',
            'Tail-Gaussian': 'tail_gaussian'
        }
        peak_shape = self.peak_shape_combo.currentText()
        
        min_height = self.min_height_spin.value()
        return {
            'background_method': self.background_combo.currentText(),
            'peak_shape': peak_shape_map.get(peak_shape, peak_shape.lower()),
            'include_escape_peaks': self.escape_peaks_check.isChecked(),
            'pileup_correction': self.pileup_check.isChecked(),
            'include_tube_lines': self.tube_lines_check.isChecked(),
            'tube_element': self.tube_element_combo.currentText(),
            'excitation_kv': self.excitation_spin.value(),
            'include_compton': (
                self.tube_lines_check.isChecked() and self.compton_check.isChecked()
            ),
            'scatter_angle_deg': self.scatter_angle_spin.value(),
            'compton_fwhm_kev': self.compton_fwhm_spin.value() / 1000.0,
            'auto_find_peaks': self.auto_find_check.isChecked(),
            'prominence_percent': self.prominence_spin.value(),
            'min_height': None if min_height <= 0 else min_height,
            'min_separation_ev': self.min_separation_spin.value(),
            'show_peak_markers': self.show_markers_check.isChecked(),
            'use_peak_list': self.should_use_peak_list(),
            'smart_id_after_fit': self.smart_id_check.isChecked(),
            'fwhm_excess_ev': self.fwhm_excess_spin.value(),
            'smart_id_apply': self.smart_id_apply_check.isChecked(),
        }
