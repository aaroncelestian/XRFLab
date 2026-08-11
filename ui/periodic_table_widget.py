"""
Interactive periodic table widget for element selection
"""

from PySide6.QtWidgets import (
    QWidget, QGridLayout, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QButtonGroup, QScrollArea, QFrame, QMenu, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QFont, QColor, QContextMenuEvent


class ElementButton(QPushButton):
    """Custom button for periodic table elements"""
    
    element_right_clicked = Signal(str, int)  # symbol, atomic_number
    
    def __init__(self, symbol, name, atomic_number, group=None):
        super().__init__()
        
        self.symbol = symbol
        self.name = name
        self.atomic_number = atomic_number
        self.group = group
        
        self.setCheckable(True)
        # Compact tiles: 18 cols × 26px + 1px spacing ≈ 485px
        self.setFixedSize(26, 26)
        
        # Set text - just symbol for compact view
        self.setText(symbol)
        
        # Set font - smaller for compact view
        font = QFont("Arial", 7)
        font.setBold(True)
        self.setFont(font)
        
        # Set tooltip
        self.setToolTip(f"{name} ({symbol})\nZ = {atomic_number}\nRight-click for details")
        
        # Apply styling based on element group
        self._apply_group_styling()
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
    
    def _show_context_menu(self, position):
        """Show context menu on right-click"""
        self.element_right_clicked.emit(self.symbol, self.atomic_number)
    
    def _apply_group_styling(self):
        """Apply color coding based on element group"""
        colors = {
            'alkali': '#FF6B6B',           # Red
            'alkaline': '#FFA07A',         # Light coral
            'transition': '#FFD93D',       # Yellow
            'post-transition': '#95E1D3',  # Mint
            'metalloid': '#A8E6CF',        # Light green
            'nonmetal': '#87CEEB',         # Sky blue
            'halogen': '#DDA0DD',          # Plum
            'noble': '#E6E6FA',            # Lavender
            'lanthanide': '#FFDAB9',       # Peach
            'actinide': '#FFB6C1',         # Light pink
        }
        
        bg_color = colors.get(self.group, '#E0E0E0')
        
        self.setStyleSheet(f"""
            ElementButton {{
                background-color: {bg_color};
                border: 1px solid #999999;
                border-radius: 2px;
                color: #333333;
                padding: 0px;
            }}
            ElementButton:hover {{
                border: 1px solid #2196F3;
                background-color: {self._lighten_color(bg_color)};
            }}
            ElementButton:checked {{
                border: 2px solid #4CAF50;
                background-color: {self._darken_color(bg_color)};
                font-weight: bold;
            }}
        """)    
    def _lighten_color(self, hex_color):
        """Lighten a hex color"""
        color = QColor(hex_color)
        h, s, v, a = color.getHsv()
        return QColor.fromHsv(h, max(0, s - 20), min(255, v + 20), a).name()
    
    def _darken_color(self, hex_color):
        """Darken a hex color"""
        color = QColor(hex_color)
        h, s, v, a = color.getHsv()
        return QColor.fromHsv(h, min(255, s + 20), max(0, v - 20), a).name()


class PeriodicTableWidget(QWidget):
    """Interactive periodic table for element selection"""
    
    elements_changed = Signal(list)  # Emitted when selection changes
    element_clicked = Signal(str, int)  # Emitted when element is clicked (symbol, Z)
    element_info_requested = Signal(str, int)  # Emitted when right-click (symbol, Z)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.element_buttons = {}
        self.selected_elements = []
        
        self._setup_ui()
        self._create_periodic_table()
    
    def _setup_ui(self):
        """Setup the widget layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)  # Minimal margins
        layout.setSpacing(2)  # Very tight spacing
        
        # No title - save vertical space
        
        # Scroll area for periodic table
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Container for periodic table — keep tiles tight; absorb slack at edges
        self.table_widget = QWidget()
        self.table_layout = QGridLayout(self.table_widget)
        self.table_layout.setSpacing(1)
        self.table_layout.setContentsMargins(1, 1, 1, 1)
        self.table_layout.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        # Fixed-size element columns/rows; stretch empty trailing cells instead
        for col in range(1, 19):
            self.table_layout.setColumnStretch(col, 0)
        for row in range(9):
            self.table_layout.setRowStretch(row, 0)
        self.table_layout.setColumnStretch(19, 1)
        self.table_layout.setRowStretch(9, 1)
        
        scroll.setWidget(self.table_widget)
        layout.addWidget(scroll, stretch=1)
        
        # Control buttons - compact layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(3)
        
        self.select_all_btn = QPushButton("All")
        self.select_all_btn.setToolTip("Select all elements")
        self.select_all_btn.setMaximumHeight(24)
        self.select_all_btn.clicked.connect(self._select_all)
        button_layout.addWidget(self.select_all_btn)
        
        self.clear_all_btn = QPushButton("Clear")
        self.clear_all_btn.setToolTip("Clear all selections")
        self.clear_all_btn.setMaximumHeight(24)
        self.clear_all_btn.clicked.connect(self._clear_all)
        button_layout.addWidget(self.clear_all_btn)
        
        self.select_common_btn = QPushButton("Common")
        self.select_common_btn.setToolTip("Select commonly analyzed elements in XRF")
        self.select_common_btn.setMaximumHeight(24)
        self.select_common_btn.clicked.connect(self._select_common_xrf)
        button_layout.addWidget(self.select_common_btn)
        
        layout.addLayout(button_layout)
        
        # Legend
        legend_layout = self._create_legend()
        layout.addLayout(legend_layout)
    
    def _create_periodic_table(self):
        """Create the periodic table layout"""
        # Element data: (symbol, name, row, col, group)
        elements = [
            # Period 1
            ('H', 'Hydrogen', 1, 0, 1, 'nonmetal'),
            ('He', 'Helium', 2, 0, 18, 'noble'),
            
            # Period 2
            ('Li', 'Lithium', 3, 1, 1, 'alkali'),
            ('Be', 'Beryllium', 4, 1, 2, 'alkaline'),
            ('B', 'Boron', 5, 1, 13, 'metalloid'),
            ('C', 'Carbon', 6, 1, 14, 'nonmetal'),
            ('N', 'Nitrogen', 7, 1, 15, 'nonmetal'),
            ('O', 'Oxygen', 8, 1, 16, 'nonmetal'),
            ('F', 'Fluorine', 9, 1, 17, 'halogen'),
            ('Ne', 'Neon', 10, 1, 18, 'noble'),
            
            # Period 3
            ('Na', 'Sodium', 11, 2, 1, 'alkali'),
            ('Mg', 'Magnesium', 12, 2, 2, 'alkaline'),
            ('Al', 'Aluminum', 13, 2, 13, 'post-transition'),
            ('Si', 'Silicon', 14, 2, 14, 'metalloid'),
            ('P', 'Phosphorus', 15, 2, 15, 'nonmetal'),
            ('S', 'Sulfur', 16, 2, 16, 'nonmetal'),
            ('Cl', 'Chlorine', 17, 2, 17, 'halogen'),
            ('Ar', 'Argon', 18, 2, 18, 'noble'),
            
            # Period 4
            ('K', 'Potassium', 19, 3, 1, 'alkali'),
            ('Ca', 'Calcium', 20, 3, 2, 'alkaline'),
            ('Sc', 'Scandium', 21, 3, 3, 'transition'),
            ('Ti', 'Titanium', 22, 3, 4, 'transition'),
            ('V', 'Vanadium', 23, 3, 5, 'transition'),
            ('Cr', 'Chromium', 24, 3, 6, 'transition'),
            ('Mn', 'Manganese', 25, 3, 7, 'transition'),
            ('Fe', 'Iron', 26, 3, 8, 'transition'),
            ('Co', 'Cobalt', 27, 3, 9, 'transition'),
            ('Ni', 'Nickel', 28, 3, 10, 'transition'),
            ('Cu', 'Copper', 29, 3, 11, 'transition'),
            ('Zn', 'Zinc', 30, 3, 12, 'transition'),
            ('Ga', 'Gallium', 31, 3, 13, 'post-transition'),
            ('Ge', 'Germanium', 32, 3, 14, 'metalloid'),
            ('As', 'Arsenic', 33, 3, 15, 'metalloid'),
            ('Se', 'Selenium', 34, 3, 16, 'nonmetal'),
            ('Br', 'Bromine', 35, 3, 17, 'halogen'),
            ('Kr', 'Krypton', 36, 3, 18, 'noble'),
            
            # Period 5
            ('Rb', 'Rubidium', 37, 4, 1, 'alkali'),
            ('Sr', 'Strontium', 38, 4, 2, 'alkaline'),
            ('Y', 'Yttrium', 39, 4, 3, 'transition'),
            ('Zr', 'Zirconium', 40, 4, 4, 'transition'),
            ('Nb', 'Niobium', 41, 4, 5, 'transition'),
            ('Mo', 'Molybdenum', 42, 4, 6, 'transition'),
            ('Tc', 'Technetium', 43, 4, 7, 'transition'),
            ('Ru', 'Ruthenium', 44, 4, 8, 'transition'),
            ('Rh', 'Rhodium', 45, 4, 9, 'transition'),
            ('Pd', 'Palladium', 46, 4, 10, 'transition'),
            ('Ag', 'Silver', 47, 4, 11, 'transition'),
            ('Cd', 'Cadmium', 48, 4, 12, 'transition'),
            ('In', 'Indium', 49, 4, 13, 'post-transition'),
            ('Sn', 'Tin', 50, 4, 14, 'post-transition'),
            ('Sb', 'Antimony', 51, 4, 15, 'metalloid'),
            ('Te', 'Tellurium', 52, 4, 16, 'metalloid'),
            ('I', 'Iodine', 53, 4, 17, 'halogen'),
            ('Xe', 'Xenon', 54, 4, 18, 'noble'),
            
            # Period 6
            ('Cs', 'Cesium', 55, 5, 1, 'alkali'),
            ('Ba', 'Barium', 56, 5, 2, 'alkaline'),
            ('La', 'Lanthanum', 57, 5, 3, 'lanthanide'),
            ('Hf', 'Hafnium', 72, 5, 4, 'transition'),
            ('Ta', 'Tantalum', 73, 5, 5, 'transition'),
            ('W', 'Tungsten', 74, 5, 6, 'transition'),
            ('Re', 'Rhenium', 75, 5, 7, 'transition'),
            ('Os', 'Osmium', 76, 5, 8, 'transition'),
            ('Ir', 'Iridium', 77, 5, 9, 'transition'),
            ('Pt', 'Platinum', 78, 5, 10, 'transition'),
            ('Au', 'Gold', 79, 5, 11, 'transition'),
            ('Hg', 'Mercury', 80, 5, 12, 'transition'),
            ('Tl', 'Thallium', 81, 5, 13, 'post-transition'),
            ('Pb', 'Lead', 82, 5, 14, 'post-transition'),
            ('Bi', 'Bismuth', 83, 5, 15, 'post-transition'),
            ('Po', 'Polonium', 84, 5, 16, 'metalloid'),
            ('At', 'Astatine', 85, 5, 17, 'halogen'),
            ('Rn', 'Radon', 86, 5, 18, 'noble'),
            
            # Period 7
            ('Fr', 'Francium', 87, 6, 1, 'alkali'),
            ('Ra', 'Radium', 88, 6, 2, 'alkaline'),
            ('Ac', 'Actinium', 89, 6, 3, 'actinide'),
            ('Rf', 'Rutherfordium', 104, 6, 4, 'transition'),
            ('Db', 'Dubnium', 105, 6, 5, 'transition'),
            ('Sg', 'Seaborgium', 106, 6, 6, 'transition'),
            ('Bh', 'Bohrium', 107, 6, 7, 'transition'),
            ('Hs', 'Hassium', 108, 6, 8, 'transition'),
            
            # Lanthanides (row 7 — directly under main table, no spacer row)
            ('Ce', 'Cerium', 58, 7, 4, 'lanthanide'),
            ('Pr', 'Praseodymium', 59, 7, 5, 'lanthanide'),
            ('Nd', 'Neodymium', 60, 7, 6, 'lanthanide'),
            ('Pm', 'Promethium', 61, 7, 7, 'lanthanide'),
            ('Sm', 'Samarium', 62, 7, 8, 'lanthanide'),
            ('Eu', 'Europium', 63, 7, 9, 'lanthanide'),
            ('Gd', 'Gadolinium', 64, 7, 10, 'lanthanide'),
            ('Tb', 'Terbium', 65, 7, 11, 'lanthanide'),
            ('Dy', 'Dysprosium', 66, 7, 12, 'lanthanide'),
            ('Ho', 'Holmium', 67, 7, 13, 'lanthanide'),
            ('Er', 'Erbium', 68, 7, 14, 'lanthanide'),
            ('Tm', 'Thulium', 69, 7, 15, 'lanthanide'),
            ('Yb', 'Ytterbium', 70, 7, 16, 'lanthanide'),
            ('Lu', 'Lutetium', 71, 7, 17, 'lanthanide'),
            
            # Actinides (row 8)
            ('Th', 'Thorium', 90, 8, 4, 'actinide'),
            ('Pa', 'Protactinium', 91, 8, 5, 'actinide'),
            ('U', 'Uranium', 92, 8, 6, 'actinide'),
            ('Np', 'Neptunium', 93, 8, 7, 'actinide'),
            ('Pu', 'Plutonium', 94, 8, 8, 'actinide'),
            ('Am', 'Americium', 95, 8, 9, 'actinide'),
            ('Cm', 'Curium', 96, 8, 10, 'actinide'),
            ('Bk', 'Berkelium', 97, 8, 11, 'actinide'),
            ('Cf', 'Californium', 98, 8, 12, 'actinide'),
        ]
        
        # Create element buttons
        for symbol, name, z, row, col, group in elements:
            btn = ElementButton(symbol, name, z, group)
            btn.toggled.connect(self._on_element_toggled)
            btn.clicked.connect(lambda checked, s=symbol, znum=z: self.element_clicked.emit(s, znum))
            btn.element_right_clicked.connect(self.element_info_requested.emit)
            
            self.table_layout.addWidget(btn, row, col)
            self.element_buttons[symbol] = btn
        
        # Compact labels for lanthanides and actinides
        label_font = QFont("Arial", 7)
        lanthanide_label = QLabel("Ln →")
        lanthanide_label.setFont(label_font)
        lanthanide_label.setToolTip("Lanthanides")
        lanthanide_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table_layout.addWidget(lanthanide_label, 7, 2)
        
        actinide_label = QLabel("An →")
        actinide_label.setFont(label_font)
        actinide_label.setToolTip("Actinides")
        actinide_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table_layout.addWidget(actinide_label, 8, 2)
    
    def _create_legend(self):
        """Create color legend for element groups"""
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        legend_items = [
            ('Alkali', '#FF6B6B'),
            ('Alkaline', '#FFA07A'),
            ('Trans.', '#FFD93D'),
            ('Post-T.', '#95E1D3'),
            ('Metalloid', '#A8E6CF'),
            ('Nonmet.', '#87CEEB'),
            ('Halogen', '#DDA0DD'),
            ('Noble', '#E6E6FA'),
            ('Ln', '#FFDAB9'),
            ('An', '#FFB6C1'),
        ]
        
        for name, color in legend_items:
            frame = QFrame()
            frame.setFixedSize(8, 8)
            frame.setStyleSheet(f"background-color: {color}; border: 1px solid #999;")
            
            label = QLabel(name)
            label.setFont(QFont("Arial", 6))
            
            item_layout = QHBoxLayout()
            item_layout.addWidget(frame)
            item_layout.addWidget(label)
            item_layout.setSpacing(1)
            item_layout.setContentsMargins(0, 0, 4, 0)
            
            layout.addLayout(item_layout)
        
        layout.addStretch()
        return layout
    
    def _on_element_toggled(self, checked):
        """Handle element button toggle"""
        self._update_selected_elements()
    
    def _update_selected_elements(self):
        """Update the list of selected elements and emit signal"""
        self.selected_elements = []
        
        for symbol, btn in self.element_buttons.items():
            if btn.isChecked():
                self.selected_elements.append({
                    'symbol': symbol,
                    'z': btn.atomic_number,
                    'name': btn.name
                })
        
        # Sort by atomic number
        self.selected_elements.sort(key=lambda x: x['z'])
        
        self.elements_changed.emit(self.selected_elements)
    
    def _select_all(self):
        """Select all elements"""
        for btn in self.element_buttons.values():
            btn.setChecked(True)
    
    def _clear_all(self):
        """Clear all selections"""
        for btn in self.element_buttons.values():
            btn.setChecked(False)
    
    def _select_common_xrf(self):
        """Select commonly analyzed elements in XRF"""
        common_elements = [
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca',
            'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
            'Ag', 'Cd', 'Sn', 'Sb', 'Ba', 'W', 'Pb', 'Bi'
        ]
        
        # Clear all first
        self._clear_all()
        
        # Select common elements
        for symbol in common_elements:
            if symbol in self.element_buttons:
                self.element_buttons[symbol].setChecked(True)
    
    def get_selected_elements(self):
        """Return list of selected elements"""
        return self.selected_elements
    
    def set_selected_elements(self, symbols):
        """
        Set selected elements by symbol list
        
        Args:
            symbols: List of element symbols to select
        """
        # Block per-button signals so we emit once at the end
        for btn in self.element_buttons.values():
            btn.blockSignals(True)
        try:
            for btn in self.element_buttons.values():
                btn.setChecked(False)
            for symbol in symbols:
                if symbol in self.element_buttons:
                    self.element_buttons[symbol].setChecked(True)
        finally:
            for btn in self.element_buttons.values():
                btn.blockSignals(False)
        self._update_selected_elements()
