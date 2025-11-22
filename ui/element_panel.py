"""
Element selection panel for XRF analysis
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QComboBox, QDoubleSpinBox, QTreeWidget, QTreeWidgetItem,
    QPushButton, QCheckBox
)
from PySide6.QtCore import Qt, Signal


class ElementPanel(QWidget):
    """Panel for sample information, element selection, and experimental parameters"""
    
    elements_changed = Signal(list)  # Emitted when selected elements change
    fit_requested = Signal()  # Emitted when fit button is clicked
    
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
        """Create element selection tree"""
        group = QGroupBox("Element Selection")
        layout = QVBoxLayout(group)
        
        # Search box
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.element_search = QLineEdit()
        self.element_search.setPlaceholderText("Type element symbol or name")
        self.element_search.textChanged.connect(self._filter_elements)
        search_layout.addWidget(self.element_search)
        layout.addLayout(search_layout)
        
        # Element tree
        self.element_tree = QTreeWidget()
        self.element_tree.setHeaderLabels(["Element", "Z", "Lines"])
        self.element_tree.setColumnWidth(0, 100)
        self.element_tree.setColumnWidth(1, 40)
        self.element_tree.itemChanged.connect(self._on_element_selection_changed)
        layout.addWidget(self.element_tree)
        
        # Populate with common elements
        self._populate_element_tree()
        
        # Quick selection buttons
        button_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all_elements)
        button_layout.addWidget(self.select_all_btn)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self._clear_all_elements)
        button_layout.addWidget(self.clear_all_btn)
        layout.addLayout(button_layout)
        
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
            "Pseudo-Voigt"
        ])
        shape_layout.addWidget(self.peak_shape_combo)
        layout.addLayout(shape_layout)
        
        # Escape peaks checkbox
        self.escape_peaks_check = QCheckBox("Include Escape Peaks")
        self.escape_peaks_check.setChecked(True)
        layout.addWidget(self.escape_peaks_check)
        
        # Pile-up correction checkbox
        self.pileup_check = QCheckBox("Pile-up Correction")
        layout.addWidget(self.pileup_check)
        
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
    
    def _populate_element_tree(self):
        """Populate element tree with common elements"""
        # Common elements organized by category
        categories = {
            "Light Elements": [
                ("C", 6, "K"),
                ("N", 7, "K"),
                ("O", 8, "K"),
                ("F", 9, "K"),
                ("Na", 11, "K"),
                ("Mg", 12, "K"),
                ("Al", 13, "K"),
                ("Si", 14, "K"),
                ("P", 15, "K"),
                ("S", 16, "K"),
                ("Cl", 17, "K"),
                ("K", 19, "K"),
            ],
            "Transition Metals": [
                ("Ti", 22, "K, L"),
                ("V", 23, "K, L"),
                ("Cr", 24, "K, L"),
                ("Mn", 25, "K, L"),
                ("Fe", 26, "K, L"),
                ("Co", 27, "K, L"),
                ("Ni", 28, "K, L"),
                ("Cu", 29, "K, L"),
                ("Zn", 30, "K, L"),
            ],
            "Heavy Elements": [
                ("Sr", 38, "K, L"),
                ("Y", 39, "K, L"),
                ("Zr", 40, "K, L"),
                ("Mo", 42, "K, L"),
                ("Ag", 47, "K, L"),
                ("Sn", 50, "K, L"),
                ("Ba", 56, "K, L"),
                ("W", 74, "L, M"),
                ("Pb", 82, "L, M"),
            ]
        }
        
        for category, elements in categories.items():
            category_item = QTreeWidgetItem(self.element_tree, [category])
            category_item.setFlags(category_item.flags() | Qt.ItemIsAutoTristate)
            
            for symbol, z, lines in elements:
                element_item = QTreeWidgetItem(category_item, [symbol, str(z), lines])
                element_item.setFlags(element_item.flags() | Qt.ItemIsUserCheckable)
                element_item.setCheckState(0, Qt.Unchecked)
        
        self.element_tree.expandAll()
    
    def _filter_elements(self, text):
        """Filter element tree based on search text"""
        text = text.lower()
        
        for i in range(self.element_tree.topLevelItemCount()):
            category = self.element_tree.topLevelItem(i)
            category_visible = False
            
            for j in range(category.childCount()):
                element = category.child(j)
                symbol = element.text(0).lower()
                z = element.text(1)
                
                if text in symbol or text in z:
                    element.setHidden(False)
                    category_visible = True
                else:
                    element.setHidden(True)
            
            category.setHidden(not category_visible)
    
    def _on_element_selection_changed(self, item, column):
        """Handle element selection changes"""
        if item.childCount() > 0:  # Category item
            return
        
        self.selected_elements = []
        for i in range(self.element_tree.topLevelItemCount()):
            category = self.element_tree.topLevelItem(i)
            for j in range(category.childCount()):
                element = category.child(j)
                if element.checkState(0) == Qt.Checked:
                    self.selected_elements.append({
                        'symbol': element.text(0),
                        'z': int(element.text(1)),
                        'lines': element.text(2)
                    })
        
        self.elements_changed.emit(self.selected_elements)
    
    def _select_all_elements(self):
        """Select all visible elements"""
        for i in range(self.element_tree.topLevelItemCount()):
            category = self.element_tree.topLevelItem(i)
            for j in range(category.childCount()):
                element = category.child(j)
                if not element.isHidden():
                    element.setCheckState(0, Qt.Checked)
    
    def _clear_all_elements(self):
        """Clear all element selections"""
        for i in range(self.element_tree.topLevelItemCount()):
            category = self.element_tree.topLevelItem(i)
            for j in range(category.childCount()):
                element = category.child(j)
                element.setCheckState(0, Qt.Unchecked)
    
    def _on_sample_type_changed(self, sample_type):
        """Enable/disable thickness input based on sample type"""
        self.thickness_spin.setEnabled(sample_type == "Thin Film")
    
    def get_selected_elements(self):
        """Return list of selected elements"""
        return self.selected_elements
    
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
        return {
            'background_method': self.background_combo.currentText(),
            'peak_shape': self.peak_shape_combo.currentText(),
            'include_escape_peaks': self.escape_peaks_check.isChecked(),
            'pileup_correction': self.pileup_check.isChecked()
        }
