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
    identify_on_plot_toggled = Signal(bool)  # Click-spectrum identify mode
    identify_add_element = Signal(str)  # Add candidate element from identify list
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.selected_elements = []
        self._peak_list_data = []  # List of peak dicts shown in the UI
        self._identify_candidates = []
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
        
        # Peak find (detect + auto-ID) then fitting
        peak_find_group = self._create_peak_find_group()
        layout.addWidget(peak_find_group)
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
        """Create element selection with periodic table + plot identify tool."""
        group = QGroupBox("Element Selection")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(3, 6, 3, 3)
        layout.setSpacing(2)
        
        # Create periodic table widget
        # Click = show lines; double-click = add/remove for fitting
        self.periodic_table = PeriodicTableWidget()
        self.periodic_table.elements_changed.connect(self._on_periodic_table_changed)
        self.periodic_table.element_clicked.connect(self.element_clicked.emit)
        self.periodic_table.element_info_requested.connect(self._show_element_info)
        self.periodic_table.setToolTip(
            "Click an element to preview its emission lines on the spectrum.\n"
            "Double-click to add or remove it from the fitting list."
        )
        layout.addWidget(self.periodic_table)

        identify_group = QGroupBox("Identify on Plot")
        identify_layout = QVBoxLayout(identify_group)
        identify_layout.setContentsMargins(5, 8, 5, 5)
        identify_layout.setSpacing(4)

        self.identify_on_plot_btn = QPushButton("Click Spectrum to Identify")
        self.identify_on_plot_btn.setCheckable(True)
        self.identify_on_plot_btn.setToolTip(
            "When on, click the spectrum plot to list the most likely\n"
            "emission-line candidates at that energy. Double-click a\n"
            "candidate (or use Add) to select it on the periodic table."
        )
        self.identify_on_plot_btn.setStyleSheet("""
            QPushButton {
                padding: 6px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #c62828;
                color: white;
            }
        """)
        self.identify_on_plot_btn.toggled.connect(self._on_identify_on_plot_toggled)
        identify_layout.addWidget(self.identify_on_plot_btn)

        self.identify_energy_label = QLabel("Click energy: —")
        self.identify_energy_label.setStyleSheet("color: #555;")
        identify_layout.addWidget(self.identify_energy_label)

        self.identify_candidates_list = QListWidget()
        self.identify_candidates_list.setMinimumHeight(100)
        self.identify_candidates_list.setToolTip(
            "Candidates ranked by |ΔE| from the clicked energy.\n"
            "Double-click to add the element to the selection."
        )
        self.identify_candidates_list.itemDoubleClicked.connect(
            self._on_identify_candidate_activated
        )
        identify_layout.addWidget(self.identify_candidates_list)

        add_row = QHBoxLayout()
        self.identify_add_btn = QPushButton("Add Selected Element")
        self.identify_add_btn.setEnabled(False)
        self.identify_add_btn.setToolTip(
            "Add the highlighted candidate’s element to the periodic-table selection"
        )
        self.identify_add_btn.clicked.connect(self._on_identify_add_clicked)
        add_row.addWidget(self.identify_add_btn)
        identify_layout.addLayout(add_row)

        self.identify_candidates_list.itemSelectionChanged.connect(
            lambda: self.identify_add_btn.setEnabled(
                self.identify_candidates_list.currentItem() is not None
            )
        )

        layout.addWidget(identify_group)
        
        return group

    def _on_identify_on_plot_toggled(self, checked):
        self.identify_on_plot_toggled.emit(bool(checked))
        if checked:
            self.identify_energy_label.setText(
                "Click energy: click a point on the spectrum…"
            )
        else:
            self.identify_energy_label.setText("Click energy: —")
            self.identify_candidates_list.clear()
            self.identify_add_btn.setEnabled(False)

    def set_identify_candidates(self, energy_kev, candidates):
        """
        Populate the identify-on-plot candidate list.

        Args:
            energy_kev: Clicked energy
            candidates: List of dicts from candidates_at_energy()
        """
        self._identify_candidates = list(candidates or [])
        self.identify_energy_label.setText(f"Click energy: {float(energy_kev):.3f} keV")
        self.identify_candidates_list.clear()
        if not self._identify_candidates:
            item = QListWidgetItem("No common-XRF lines within tolerance")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.identify_candidates_list.addItem(item)
            self.identify_add_btn.setEnabled(False)
            return

        for hit in self._identify_candidates:
            delta_ev = hit.get('delta_ev', 0.0)
            sign = "+" if delta_ev >= 0 else "−"
            text = (
                f"{hit['symbol']}-{hit['line']}  "
                f"{hit['line_energy']:.3f} keV  "
                f"(Δ {sign}{abs(delta_ev):.0f} eV)"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, hit)
            self.identify_candidates_list.addItem(item)
        self.identify_candidates_list.setCurrentRow(0)
        self.identify_add_btn.setEnabled(True)

    def _on_identify_candidate_activated(self, item):
        hit = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not hit:
            return
        self.identify_add_element.emit(hit['symbol'])

    def _on_identify_add_clicked(self):
        item = self.identify_candidates_list.currentItem()
        if item is None:
            return
        self._on_identify_candidate_activated(item)

    def add_selected_element(self, symbol):
        """Add one element to the current periodic-table selection."""
        if not symbol or not hasattr(self, 'periodic_table'):
            return
        current = [
            e.get('symbol') for e in (self.get_selected_elements() or [])
            if e.get('symbol')
        ]
        if symbol not in current:
            current.append(symbol)
        self.set_selected_elements(current)
    
    def _create_peak_find_group(self):
        """Peak detection controls, find button, and editable peak list."""
        group = QGroupBox("Peak Find")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(3)

        hint = QLabel(
            "1) Find peaks → 2) review Elements (auto-ID) → 3) Fit on the Fitting tab"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555; font-size: 11px; padding: 2px;")
        layout.addWidget(hint)

        # Background used for detection (shared with Fitting via same widget attrs)
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

        detect_group = QGroupBox("Peak Detection")
        detect_layout = QFormLayout(detect_group)
        detect_layout.setContentsMargins(5, 8, 5, 5)
        detect_layout.setSpacing(4)

        self.auto_find_check = QCheckBox("Auto-find unknown peaks")
        self.auto_find_check.setChecked(True)
        self.auto_find_check.setToolTip(
            "Detect peaks from the spectrum (not only pre-selected element lines).\n"
            "Recommended for the Peak Find → Elements workflow."
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
            "Minimum energy separation between auto-detected peaks."
        )
        detect_layout.addRow("Min separation:", self.min_separation_spin)

        self.show_markers_check = QCheckBox("Show peak markers on spectrum")
        self.show_markers_check.setChecked(True)
        detect_layout.addRow(self.show_markers_check)

        self.auto_id_check = QCheckBox("Auto-ID peaks (common XRF elements)")
        self.auto_id_check.setChecked(True)
        self.auto_id_check.setToolTip(
            "After peak find, match unknown peaks to common XRF emission lines\n"
            "and select those elements on the Elements tab for review."
        )
        detect_layout.addRow(self.auto_id_check)

        layout.addWidget(detect_group)

        # Tube options affect which seeds are added during peak find
        tube_layout = QHBoxLayout()
        self.tube_lines_check = QCheckBox("Include Tube Lines:")
        self.tube_lines_check.setChecked(True)
        self.tube_lines_check.setToolTip(
            "Include X-ray tube characteristic lines in the peak list"
        )
        tube_layout.addWidget(self.tube_lines_check)

        self.tube_element_combo = QComboBox()
        self.tube_element_combo.addItems(["Rh", "W", "Mo", "Ag", "Cr", "Cu"])
        self.tube_element_combo.setCurrentText("Rh")
        tube_layout.addWidget(self.tube_element_combo)
        layout.addLayout(tube_layout)

        compton_row = QHBoxLayout()
        self.compton_check = QCheckBox("Include Compton:")
        self.compton_check.setChecked(True)
        compton_row.addWidget(self.compton_check)
        compton_row.addWidget(QLabel("θ:"))
        self.scatter_angle_spin = QDoubleSpinBox()
        self.scatter_angle_spin.setRange(30.0, 150.0)
        self.scatter_angle_spin.setDecimals(0)
        self.scatter_angle_spin.setSingleStep(5.0)
        self.scatter_angle_spin.setValue(90.0)
        self.scatter_angle_spin.setSuffix("°")
        compton_row.addWidget(self.scatter_angle_spin)
        compton_row.addWidget(QLabel("FWHM:"))
        self.compton_fwhm_spin = QDoubleSpinBox()
        self.compton_fwhm_spin.setRange(100.0, 800.0)
        self.compton_fwhm_spin.setDecimals(0)
        self.compton_fwhm_spin.setSingleStep(25.0)
        self.compton_fwhm_spin.setValue(250.0)
        self.compton_fwhm_spin.setSuffix(" eV")
        compton_row.addWidget(self.compton_fwhm_spin)
        layout.addLayout(compton_row)

        self.tube_lines_check.toggled.connect(self._on_tube_lines_toggled)
        self._on_tube_lines_toggled(self.tube_lines_check.isChecked())

        find_button = QPushButton("Find Peaks + Auto-ID")
        find_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #0D47A1; }
        """)
        find_button.setToolTip(
            "Detect peaks, auto-ID against common XRF lines, then open Elements\n"
            "so you can confirm labels before Fitting."
        )
        find_button.clicked.connect(self.peak_find_requested.emit)
        layout.addWidget(find_button)

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
            "Delete false peaks before Fitting."
        )
        peaks_layout.addWidget(self.peak_list_widget)

        self.use_peak_list_check = QCheckBox("Fit using peak list")
        self.use_peak_list_check.setChecked(False)
        self.use_peak_list_check.setToolTip(
            "When checked, Fit Spectrum uses this list instead of rebuilding.\n"
            "Enabled automatically after Find Peaks."
        )
        peaks_layout.addWidget(self.use_peak_list_check)

        peak_btn_row = QHBoxLayout()
        delete_peak_btn = QPushButton("Delete Selected")
        delete_peak_btn.clicked.connect(self._delete_selected_peaks)
        peak_btn_row.addWidget(delete_peak_btn)
        clear_peaks_btn = QPushButton("Clear")
        clear_peaks_btn.clicked.connect(self._clear_peak_list)
        peak_btn_row.addWidget(clear_peaks_btn)
        peaks_layout.addLayout(peak_btn_row)
        layout.addWidget(peaks_group)

        return group

    def _create_fitting_controls_group(self):
        """Create fitting controls group (shape, tube status, Fit button)."""
        group = QGroupBox("Fitting Controls")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(3)

        hint = QLabel(
            "Confirm Elements first, then Fit. Semi-Quant is on the Results tab."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555; font-size: 11px; padding: 2px;")
        layout.addWidget(hint)

        bg_note = QLabel("Background & tube/Compton options: Peak Find tab")
        bg_note.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(bg_note)

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
        self.peak_shape_combo.setCurrentText("Voigt")
        shape_layout.addWidget(self.peak_shape_combo)
        layout.addLayout(shape_layout)

        self.fwhm_status_label = QLabel(
            "FWHM: no calibration — widths free in LS"
        )
        self.fwhm_status_label.setWordWrap(True)
        self.fwhm_status_label.setStyleSheet(
            "color: #994400; font-weight: bold; padding: 4px;"
        )
        layout.addWidget(self.fwhm_status_label)

        self.tube_profile_status_label = QLabel(
            "Tube profile: defaults (measure blanks at 15/30/50 kV)"
        )
        self.tube_profile_status_label.setWordWrap(True)
        self.tube_profile_status_label.setStyleSheet(
            "color: #994400; font-weight: bold; padding: 4px;"
        )
        layout.addWidget(self.tube_profile_status_label)

        self.escape_peaks_check = QCheckBox("Include Escape Peaks")
        self.escape_peaks_check.setChecked(True)
        layout.addWidget(self.escape_peaks_check)

        self.pileup_check = QCheckBox("Pile-up Correction")
        layout.addWidget(self.pileup_check)

        # Ensure tube widgets exist if Peak Find group was not built yet
        if not hasattr(self, 'tube_lines_check'):
            tube_layout = QHBoxLayout()
            self.tube_lines_check = QCheckBox("Include Tube Lines:")
            self.tube_lines_check.setChecked(True)
            tube_layout.addWidget(self.tube_lines_check)
            self.tube_element_combo = QComboBox()
            self.tube_element_combo.addItems(["Rh", "W", "Mo", "Ag", "Cr", "Cu"])
            self.tube_element_combo.setCurrentText("Rh")
            tube_layout.addWidget(self.tube_element_combo)
            layout.addLayout(tube_layout)
            self.compton_check = QCheckBox("Include Compton:")
            self.compton_check.setChecked(True)
            layout.addWidget(self.compton_check)
            self.scatter_angle_spin = QDoubleSpinBox()
            self.scatter_angle_spin.setRange(30.0, 150.0)
            self.scatter_angle_spin.setValue(90.0)
            self.compton_fwhm_spin = QDoubleSpinBox()
            self.compton_fwhm_spin.setRange(100.0, 800.0)
            self.compton_fwhm_spin.setValue(250.0)

        smart_group = QGroupBox("Post-fit Smart ID")
        smart_layout = QFormLayout(smart_group)
        smart_layout.setContentsMargins(5, 8, 5, 5)
        smart_layout.setSpacing(4)

        self.smart_id_check = QCheckBox("Analyze overlaps & multi-line IDs after fit")
        self.smart_id_check.setChecked(True)
        self.smart_id_check.setToolTip(
            "After fitting, flag broad peaks and check multi-line IDs."
        )
        smart_layout.addRow(self.smart_id_check)

        self.fwhm_excess_spin = QDoubleSpinBox()
        self.fwhm_excess_spin.setRange(5.0, 200.0)
        self.fwhm_excess_spin.setDecimals(0)
        self.fwhm_excess_spin.setSingleStep(5.0)
        self.fwhm_excess_spin.setValue(30.0)
        self.fwhm_excess_spin.setSuffix(" eV")
        smart_layout.addRow("FWHM excess:", self.fwhm_excess_spin)

        self.smart_id_apply_check = QCheckBox("Apply suggestions (relabel + overlap seeds)")
        self.smart_id_apply_check.setChecked(False)
        smart_layout.addRow(self.smart_id_apply_check)
        layout.addWidget(smart_group)

        self.fit_button = QPushButton("Fit Spectrum")
        self.fit_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
        """)
        self.fit_button.setToolTip(
            "Fit peaks using the Elements selection and the Peak Find list."
        )
        self.fit_button.clicked.connect(self.fit_requested.emit)
        layout.addWidget(self.fit_button)

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
                if p.get('inferred'):
                    entry['inferred'] = True
                if p.get('relative_intensity') is not None:
                    entry['relative_intensity'] = float(p['relative_intensity'])
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
        if entry.get('inferred'):
            tags.append("expected")
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
        self._drop_unselected_peak_labels()

    def _drop_unselected_peak_labels(self):
        """Drop labels (and expected seeds) for elements the user just unchecked."""
        if not getattr(self, "_peak_list_data", None):
            return
        if not hasattr(self, "peak_list_widget"):
            return
        allowed = {
            e.get("symbol")
            for e in (self.selected_elements or [])
            if e.get("symbol")
        }
        changed = False
        kept = []
        for entry in self._peak_list_data:
            if entry.get("is_tube_line"):
                kept.append(entry)
                continue
            el = entry.get("element")
            if el and el not in allowed:
                if entry.get("inferred"):
                    changed = True
                    continue
                entry["element"] = None
                entry["line"] = None
                entry.pop("relative_intensity", None)
                changed = True
            kept.append(entry)
        if not changed:
            return
        self._peak_list_data = kept
        self.peak_list_widget.clear()
        for entry in self._peak_list_data:
            self.peak_list_widget.addItem(self._format_peak_item(entry))
        self.peak_list_changed.emit()
    
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

        Used by Peak Find / Auto-ID to seed the Elements tab. Fitting does
        not overwrite this list — unchecking an element must stick.
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
        metadata = metadata or {}

        # Excitation energy / tube voltage (keV or kV)
        if "excitation_energy" in metadata:
            self.excitation_spin.setValue(float(metadata["excitation_energy"]))
        elif "kv" in metadata:
            self.excitation_spin.setValue(float(metadata["kv"]))

        # Tube current: prefer explicit mA (IPJ); EMSA PROBECUR is nanoamps
        if "tube_current_ma" in metadata:
            current = float(metadata["tube_current_ma"])
            if current > self.current_spin.maximum():
                self.current_spin.setMaximum(max(current, 50.0))
            self.current_spin.setValue(current)
        elif "ma" in metadata:
            current = float(metadata["ma"])
            if current > self.current_spin.maximum():
                self.current_spin.setMaximum(max(current, 50.0))
            self.current_spin.setValue(current)
        elif "tube_current" in metadata:
            # Convert from nA to mA (EMSA / some vendor files)
            current = float(metadata["tube_current"])
            if current > 1.0:
                current = current / 1_000_000.0
            if current > self.current_spin.maximum():
                self.current_spin.setMaximum(max(current, 50.0))
            self.current_spin.setValue(current)

        # Live time (seconds)
        if "live_time" in metadata:
            live = float(metadata["live_time"])
            if live > self.live_time_spin.maximum():
                self.live_time_spin.setMaximum(max(live, 10000.0))
            self.live_time_spin.setValue(live)

        # Incident angle
        if "incident_angle" in metadata:
            self.angle_spin.setValue(float(metadata["incident_angle"]))
    
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

    def update_tube_profile_status(self, library=None):
        """Show which per-kV tube profiles are available for Analysis."""
        if library is None or not getattr(library, 'profiles', None):
            self.tube_profile_status_label.setText(
                "Tube profile: defaults (measure blanks at 15/30/50 kV)"
            )
            self.tube_profile_status_label.setStyleSheet(
                "color: #994400; font-weight: bold; padding: 4px;"
            )
            return

        parts = []
        n_meas = 0
        for kv in library.available_kvs:
            p = library.profiles.get(library._key(kv))
            if p is None:
                parts.append(f"{kv:g}=—")
            elif p.source == 'measured':
                parts.append(f"{kv:g}=✓")
                n_meas += 1
            else:
                parts.append(f"{kv:g}=def")

        text = (
            f"Tube profile ({library.tube_element}): "
            + "  ".join(parts)
            + f"  — {n_meas} measured"
        )
        color = "#1b7a1b" if n_meas else "#994400"
        self.tube_profile_status_label.setText(text)
        self.tube_profile_status_label.setStyleSheet(
            f"color: {color}; font-weight: bold; padding: 4px;"
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
            'auto_id_after_peak_find': (
                hasattr(self, 'auto_id_check') and self.auto_id_check.isChecked()
            ),
            'smart_id_after_fit': self.smart_id_check.isChecked(),
            'fwhm_excess_ev': self.fwhm_excess_spin.value(),
            'smart_id_apply': self.smart_id_apply_check.isChecked(),
        }

    def capture_state(self) -> dict:
        """Snapshot of Sample / Peak Find / Elements / Fitting controls."""
        symbols = []
        for item in self.selected_elements or []:
            if isinstance(item, dict):
                sym = item.get("symbol")
            else:
                sym = item
            if sym:
                symbols.append(str(sym))
        return {
            "sample_name": self.sample_name_edit.text(),
            "sample_type": self.sample_type_combo.currentText(),
            "thickness": float(self.thickness_spin.value()),
            "experimental": self.get_experimental_params(),
            "selected_elements": symbols,
            "peak_list": list(self._peak_list_data or []),
            "background": self.background_combo.currentText(),
            "peak_shape": self.peak_shape_combo.currentText(),
            "escape_peaks": self.escape_peaks_check.isChecked(),
            "pileup": self.pileup_check.isChecked(),
            "tube_lines": self.tube_lines_check.isChecked(),
            "tube_element": self.tube_element_combo.currentText(),
            "compton": self.compton_check.isChecked(),
            "scatter_angle": float(self.scatter_angle_spin.value()),
            "compton_fwhm_ev": float(self.compton_fwhm_spin.value()),
            "auto_find": self.auto_find_check.isChecked(),
            "prominence": float(self.prominence_spin.value()),
            "min_height": float(self.min_height_spin.value()),
            "min_separation_ev": float(self.min_separation_spin.value()),
            "show_markers": self.show_markers_check.isChecked(),
            "use_peak_list": self.use_peak_list_check.isChecked(),
            "auto_id": bool(
                hasattr(self, "auto_id_check") and self.auto_id_check.isChecked()
            ),
            "smart_id": self.smart_id_check.isChecked(),
            "fwhm_excess_ev": float(self.fwhm_excess_spin.value()),
            "smart_id_apply": self.smart_id_apply_check.isChecked(),
        }

    def restore_state(self, state: dict) -> None:
        if not state:
            return
        self.sample_name_edit.setText(str(state.get("sample_name") or ""))
        sample_type = state.get("sample_type")
        if sample_type:
            self.sample_type_combo.setCurrentText(str(sample_type))
        if state.get("thickness") is not None:
            self.thickness_spin.setValue(float(state["thickness"]))
        exp = state.get("experimental") or {}
        if "excitation_energy" in exp:
            self.excitation_spin.setValue(float(exp["excitation_energy"]))
        if "tube_current" in exp:
            self.current_spin.setValue(float(exp["tube_current"]))
        if "live_time" in exp:
            self.live_time_spin.setValue(float(exp["live_time"]))
        if exp.get("detector_type"):
            self.detector_combo.setCurrentText(str(exp["detector_type"]))
        if "incident_angle" in exp:
            self.angle_spin.setValue(float(exp["incident_angle"]))
        self.set_selected_elements(state.get("selected_elements") or [])
        if state.get("background"):
            self.background_combo.setCurrentText(str(state["background"]))
        if state.get("peak_shape"):
            self.peak_shape_combo.setCurrentText(str(state["peak_shape"]))
        self.escape_peaks_check.setChecked(bool(state.get("escape_peaks", True)))
        self.pileup_check.setChecked(bool(state.get("pileup", False)))
        self.tube_lines_check.setChecked(bool(state.get("tube_lines", True)))
        if state.get("tube_element"):
            self.tube_element_combo.setCurrentText(str(state["tube_element"]))
        self.compton_check.setChecked(bool(state.get("compton", True)))
        if state.get("scatter_angle") is not None:
            self.scatter_angle_spin.setValue(float(state["scatter_angle"]))
        if state.get("compton_fwhm_ev") is not None:
            self.compton_fwhm_spin.setValue(float(state["compton_fwhm_ev"]))
        self.auto_find_check.setChecked(bool(state.get("auto_find", True)))
        if state.get("prominence") is not None:
            self.prominence_spin.setValue(float(state["prominence"]))
        if state.get("min_height") is not None:
            self.min_height_spin.setValue(float(state["min_height"]))
        if state.get("min_separation_ev") is not None:
            self.min_separation_spin.setValue(float(state["min_separation_ev"]))
        self.show_markers_check.setChecked(bool(state.get("show_markers", True)))
        if hasattr(self, "auto_id_check"):
            self.auto_id_check.setChecked(bool(state.get("auto_id", True)))
        self.smart_id_check.setChecked(bool(state.get("smart_id", True)))
        if state.get("fwhm_excess_ev") is not None:
            self.fwhm_excess_spin.setValue(float(state["fwhm_excess_ev"]))
        self.smart_id_apply_check.setChecked(bool(state.get("smart_id_apply", False)))
        peaks = state.get("peak_list") or []
        self.set_peak_list(peaks, enable_use_list=bool(state.get("use_peak_list")))
