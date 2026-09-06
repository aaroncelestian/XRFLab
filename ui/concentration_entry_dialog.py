"""
Editable certified composition for a reference standard.

CSV import is a convenience for filling the table — the user always sees
and confirms the element list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QDoubleSpinBox, QFileDialog, QAbstractItemView,
)
from PySide6.QtCore import Qt

from core.reference_composition import load_composition_csv


class ConcentrationEntryDialog(QDialog):
    """Review / edit element concentrations (wt%) for a standard."""

    def __init__(
        self,
        standard_name: str,
        parent=None,
        concentrations: Optional[Dict[str, float]] = None,
        source_label: str = "",
    ):
        super().__init__(parent)
        self.standard_name = standard_name
        self.concentrations = {}
        self._initial = dict(concentrations or {})
        self._source_label = source_label

        self.setWindowTitle(f"Composition — {standard_name}")
        self.setMinimumSize(560, 440)
        self._init_ui()
        if self._initial:
            self._load_rows(self._initial)
        else:
            self._add_initial_rows()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            f"<b>Certified composition for {self.standard_name}</b><br>"
            "Confirm the elements and wt%. Load a CSV to fill the table, "
            "or type values. Same-filename CSVs are used only as a starting point."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.source_label = QLabel(self._source_label or "No CSV loaded — enter values or load a file.")
        self.source_label.setWordWrap(True)
        self.source_label.setStyleSheet("color: #555;")
        layout.addWidget(self.source_label)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Element", "wt%", ""])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(250)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        tools = QHBoxLayout()
        add_row_btn = QPushButton("+ Add element")
        add_row_btn.clicked.connect(lambda: self._add_row())
        tools.addWidget(add_row_btn)

        load_btn = QPushButton("Load CSV…")
        load_btn.setToolTip(
            "NIST-style (Symbol, Concentration_mg_kg) or Element, Concentration"
        )
        load_btn.clicked.connect(self._browse_csv)
        tools.addWidget(load_btn)
        tools.addStretch()
        layout.addLayout(tools)

        buttons = QHBoxLayout()
        ok_btn = QPushButton("Use this composition")
        ok_btn.clicked.connect(self._on_ok)
        ok_btn.setDefault(True)
        buttons.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        buttons.addStretch()
        layout.addLayout(buttons)

    def _add_initial_rows(self):
        for element in ("Si", "Al", "Fe", "Ca", "Mg", "Na", "K", "Ti"):
            self._add_row(element, 0.0)

    def _load_rows(self, concentrations: Dict[str, float]):
        self.table.setRowCount(0)
        for element, wt in concentrations.items():
            self._add_row(element, float(wt))

    def _add_row(self, element="", concentration=0.0):
        row = self.table.rowCount()
        self.table.insertRow(row)

        element_item = QTableWidgetItem(element)
        self.table.setItem(row, 0, element_item)

        conc_spin = QDoubleSpinBox()
        conc_spin.setRange(0, 100)
        conc_spin.setValue(min(max(concentration, 0.0), 100.0))
        conc_spin.setDecimals(6)
        conc_spin.setSingleStep(0.01)
        self.table.setCellWidget(row, 1, conc_spin)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda _=False, r=row: self._remove_row_button(r))
        self.table.setCellWidget(row, 2, remove_btn)

    def _remove_row_button(self, row: int):
        # Row indices shift after removals — identify by sender's row
        btn = self.sender()
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 2) is btn:
                self.table.removeRow(r)
                return
        if 0 <= row < self.table.rowCount():
            self.table.removeRow(row)

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load composition CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        self.apply_csv(path)

    def apply_csv(self, csv_path: str | Path, *, announce: bool = True) -> bool:
        try:
            loaded = load_composition_csv(csv_path)
        except Exception as exc:
            QMessageBox.critical(self, "CSV Error", f"Could not read composition:\n{exc}")
            return False
        if not loaded:
            QMessageBox.warning(
                self,
                "No Data",
                "No element concentrations found in that CSV.\n\n"
                "Expected Symbol + Concentration (wt% or mg/kg).",
            )
            return False
        self._load_rows(loaded)
        name = Path(csv_path).name
        self.source_label.setText(f"Filled from {name} ({len(loaded)} elements). Edit if needed.")
        if announce:
            self.source_label.setStyleSheet("color: #1b7a3d;")
        return True

    def _on_ok(self):
        self.concentrations = {}
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
            if concentration > 0:
                self.concentrations[element] = concentration

        if not self.concentrations:
            QMessageBox.warning(
                self,
                "No Data",
                "Enter at least one element with a non-zero concentration.",
            )
            return
        self.accept()

    def get_concentrations(self) -> Dict[str, float]:
        return self.concentrations
