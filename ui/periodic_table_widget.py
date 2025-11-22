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
        self.setFixedSize(35, 35)  # Reduced from 55x55 to 35x35
        
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
                border: 2px solid #999999;
                border-radius: 4px;
                color: #333333;
                padding: 2px;
            }}
            ElementButton:hover {{
                border: 2px solid #2196F3;
                background-color: {self._lighten_color(bg_color)};
            }}
            ElementButton:checked {{
                border: 3px solid #4CAF50;
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
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("Periodic Table - Select Elements")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Scroll area for periodic table
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Container for periodic table
        self.table_widget = QWidget()
        self.table_layout = QGridLayout(self.table_widget)
        self.table_layout.setSpacing(2)
        self.table_layout.setContentsMargins(5, 5, 5, 5)
        
        scroll.setWidget(self.table_widget)
        layout.addWidget(scroll, stretch=1)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        button_layout.addWidget(self.select_all_btn)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self._clear_all)
        button_layout.addWidget(self.clear_all_btn)
        
        self.select_common_btn = QPushButton("Common XRF")
        self.select_common_btn.setToolTip("Select commonly analyzed elements in XRF")
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
            
            # Lanthanides (row 8)
            ('Ce', 'Cerium', 58, 8, 4, 'lanthanide'),
            ('Pr', 'Praseodymium', 59, 8, 5, 'lanthanide'),
            ('Nd', 'Neodymium', 60, 8, 6, 'lanthanide'),
            ('Pm', 'Promethium', 61, 8, 7, 'lanthanide'),
            ('Sm', 'Samarium', 62, 8, 8, 'lanthanide'),
            ('Eu', 'Europium', 63, 8, 9, 'lanthanide'),
            ('Gd', 'Gadolinium', 64, 8, 10, 'lanthanide'),
            ('Tb', 'Terbium', 65, 8, 11, 'lanthanide'),
            ('Dy', 'Dysprosium', 66, 8, 12, 'lanthanide'),
            ('Ho', 'Holmium', 67, 8, 13, 'lanthanide'),
            ('Er', 'Erbium', 68, 8, 14, 'lanthanide'),
            ('Tm', 'Thulium', 69, 8, 15, 'lanthanide'),
            ('Yb', 'Ytterbium', 70, 8, 16, 'lanthanide'),
            ('Lu', 'Lutetium', 71, 8, 17, 'lanthanide'),
            
            # Actinides (row 9)
            ('Th', 'Thorium', 90, 9, 4, 'actinide'),
            ('Pa', 'Protactinium', 91, 9, 5, 'actinide'),
            ('U', 'Uranium', 92, 9, 6, 'actinide'),
            ('Np', 'Neptunium', 93, 9, 7, 'actinide'),
            ('Pu', 'Plutonium', 94, 9, 8, 'actinide'),
            ('Am', 'Americium', 95, 9, 9, 'actinide'),
            ('Cm', 'Curium', 96, 9, 10, 'actinide'),
            ('Bk', 'Berkelium', 97, 9, 11, 'actinide'),
            ('Cf', 'Californium', 98, 9, 12, 'actinide'),
        ]
        
        # Create element buttons
        for symbol, name, z, row, col, group in elements:
            btn = ElementButton(symbol, name, z, group)
            btn.toggled.connect(self._on_element_toggled)
            btn.clicked.connect(lambda checked, s=symbol, znum=z: self.element_clicked.emit(s, znum))
            btn.element_right_clicked.connect(self.element_info_requested.emit)
            
            self.table_layout.addWidget(btn, row, col)
            self.element_buttons[symbol] = btn
        
        # Add labels for lanthanides and actinides
        lanthanide_label = QLabel("Lanthanides →")
        lanthanide_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table_layout.addWidget(lanthanide_label, 8, 2)
        
        actinide_label = QLabel("Actinides →")
        actinide_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table_layout.addWidget(actinide_label, 9, 2)
    
    def _create_legend(self):
        """Create color legend for element groups"""
        layout = QHBoxLayout()
        
        legend_items = [
            ('Alkali', '#FF6B6B'),
            ('Alkaline', '#FFA07A'),
            ('Transition', '#FFD93D'),
            ('Post-Trans.', '#95E1D3'),
            ('Metalloid', '#A8E6CF'),
            ('Nonmetal', '#87CEEB'),
            ('Halogen', '#DDA0DD'),
            ('Noble', '#E6E6FA'),
            ('Lanthanide', '#FFDAB9'),
            ('Actinide', '#FFB6C1'),
        ]
        
        for name, color in legend_items:
            frame = QFrame()
            frame.setFixedSize(12, 12)
            frame.setStyleSheet(f"background-color: {color}; border: 1px solid #999;")
            
            label = QLabel(name)
            label.setFont(QFont("Arial", 8))
            
            item_layout = QHBoxLayout()
            item_layout.addWidget(frame)
            item_layout.addWidget(label)
            item_layout.setSpacing(3)
            item_layout.setContentsMargins(0, 0, 5, 0)
            
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
        self._clear_all()
        for symbol in symbols:
            if symbol in self.element_buttons:
                self.element_buttons[symbol].setChecked(True)
