"""
Dialog for manual entry of element concentrations for reference standards
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QHeaderView, QMessageBox, QDoubleSpinBox)
from PySide6.QtCore import Qt
from typing import Dict


class ConcentrationEntryDialog(QDialog):
    """Dialog for entering element concentrations manually"""
    
    def __init__(self, standard_name: str, parent=None):
        super().__init__(parent)
        self.standard_name = standard_name
        self.concentrations = {}
        
        self.setWindowTitle(f"Enter Concentrations - {standard_name}")
        self.setMinimumSize(500, 400)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Info
        info = QLabel(
            f"<b>Enter element concentrations for {self.standard_name}</b><br>"
            "Add elements and their concentrations (wt%). "
            "Leave concentration as 0 to remove an element."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Table for element/concentration pairs
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Element", "Concentration (wt%)", "Actions"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(250)
        layout.addWidget(self.table)
        
        # Add row button
        add_row_btn = QPushButton("+ Add Element")
        add_row_btn.clicked.connect(self._add_row)
        layout.addWidget(add_row_btn)
        
        # Add some common elements by default
        self._add_initial_rows()
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._on_ok)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def _add_initial_rows(self):
        """Add some common elements as starting rows"""
        common_elements = ["Si", "Al", "Fe", "Ca", "Mg", "Na", "K", "Ti"]
        for element in common_elements:
            self._add_row(element, 0.0)
    
    def _add_row(self, element="", concentration=0.0):
        """Add a new row to the table"""
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # Element symbol
        element_item = QTableWidgetItem(element)
        self.table.setItem(row, 0, element_item)
        
        # Concentration spin box
        conc_spin = QDoubleSpinBox()
        conc_spin.setRange(0, 100)
        conc_spin.setValue(concentration)
        conc_spin.setDecimals(4)
        conc_spin.setSuffix(" wt%")
        self.table.setCellWidget(row, 1, conc_spin)
        
        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_row(row))
        self.table.setCellWidget(row, 2, remove_btn)
    
    def _remove_row(self, row):
        """Remove a row from the table"""
        self.table.removeRow(row)
    
    def _on_ok(self):
        """Validate and accept the dialog"""
        self.concentrations = {}
        
        # Extract concentrations from table
        for row in range(self.table.rowCount()):
            element_item = self.table.item(row, 0)
            if not element_item:
                continue
            
            element = element_item.text().strip()
            if not element:
                continue
            
            conc_widget = self.table.cellWidget(row, 1)
            if not conc_widget:
                continue
            
            concentration = conc_widget.value()
            
            # Only include non-zero concentrations
            if concentration > 0:
                self.concentrations[element] = concentration
        
        if not self.concentrations:
            QMessageBox.warning(
                self,
                "No Data",
                "Please enter at least one element with a non-zero concentration."
            )
            return
        
        self.accept()
    
    def get_concentrations(self) -> Dict[str, float]:
        """Get the entered concentrations"""
        return self.concentrations
