"""
Element selection panel for XRF analysis
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QComboBox, QDoubleSpinBox, QTreeWidget, QTreeWidgetItem,
    QPushButton, QCheckBox, QTabWidget, QDialog, QTextEdit, QDialogButtonBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from ui.periodic_table_widget import PeriodicTableWidget
from core.xray_data import get_element_lines, get_element_info


class ElementPanel(QWidget):
    """Panel for sample information, element selection, and experimental parameters"""
    
    elements_changed = Signal(list)  # Emitted when selected elements change
    fit_requested = Signal()  # Emitted when fit button is clicked
    element_clicked = Signal(str, int)  # Emitted when element clicked (symbol, Z)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.selected_elements = []
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the panel layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
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
        
        return group
    
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
            # Convert from nA to mA if needed
            current = float(metadata['tube_current'])
            print(f"  DEBUG: Raw tube_current from metadata: {current}")
            if current > 1000:  # Likely in nA
                current = current / 1000.0  # Convert to µA
                print(f"  DEBUG: After nA→µA conversion: {current}")
            if current > 1000:  # Still large, likely in µA
                current = current / 1000.0  # Convert to mA
                print(f"  DEBUG: After µA→mA conversion: {current}")
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
        
        return {
            'background_method': self.background_combo.currentText(),
            'peak_shape': peak_shape_map.get(peak_shape, peak_shape.lower()),
            'include_escape_peaks': self.escape_peaks_check.isChecked(),
            'pileup_correction': self.pileup_check.isChecked(),
            'include_tube_lines': self.tube_lines_check.isChecked(),
            'tube_element': self.tube_element_combo.currentText(),
            'excitation_kv': self.excitation_spin.value()
        }
