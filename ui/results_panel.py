"""
Results panel for displaying quantification results and fit statistics
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel, QTextEdit, QComboBox, QDoubleSpinBox,
    QGridLayout
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QBrush

from core.matrix_model import MatrixAssumptions, MatrixKind


class ResultsPanel(QWidget):
    """Panel for displaying analysis results and statistics"""
    
    element_selected = Signal(str)  # Element symbol clicked in results table
    quantify_requested = Signal()  # Emitted when Run Quant is clicked
    fp_quantify_requested = Signal()
    matrix_assumptions_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.results_data = []
        self._fp_live = False
        self._updating_controls = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._emit_matrix_changed)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the panel layout with vertical stacking"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        stats_group = self._create_statistics_group()
        main_layout.addWidget(stats_group)

        matrix_group = self._create_matrix_group()
        main_layout.addWidget(matrix_group)
        
        results_group = self._create_results_table_group()
        main_layout.addWidget(results_group, stretch=2)
        
        peaks_group = self._create_peaks_group()
        main_layout.addWidget(peaks_group, stretch=1)
        
        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        
        self.quantify_button = QPushButton("Semi-Quant")
        self.quantify_button.setToolTip(
            "Area-normalized relative intensities (semi-quantitative).\n"
            "Not fundamental-parameters concentrations."
        )
        self.quantify_button.setStyleSheet("""
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
        self.quantify_button.clicked.connect(
            lambda _checked=False: self.quantify_requested.emit()
        )
        button_row.addWidget(self.quantify_button)

        self.fp_button = QPushButton("FP Composition")
        self.fp_button.setToolTip(
            "Fundamental-parameters wt% using the matrix model above.\n"
            "H2O, OH, and CO2 are user assumptions (not measured).\n"
            "After the first run, changing knobs recomputes live."
        )
        self.fp_button.setStyleSheet("""
            QPushButton {
                background-color: #00897B;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #00796B;
            }
            QPushButton:pressed {
                background-color: #00695C;
            }
        """)
        self.fp_button.clicked.connect(
            lambda _checked=False: self.fp_quantify_requested.emit()
        )
        button_row.addWidget(self.fp_button)
        
        self.export_button = QPushButton("Export Results")
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        button_row.addWidget(self.export_button)
        
        main_layout.addLayout(button_row)

    def _create_matrix_group(self):
        group = QGroupBox("Matrix (for FP composition)")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        self.matrix_combo = QComboBox()
        self.matrix_combo.addItem("Measured only (metal / sulfide)", MatrixKind.MEASURED)
        self.matrix_combo.addItem("Oxide / silicate", MatrixKind.OXIDE)
        self.matrix_combo.addItem("Carbonate", MatrixKind.CARBONATE)
        self.matrix_combo.addItem("Hydroxide", MatrixKind.HYDROXIDE)
        self.matrix_combo.setToolTip(
            "Galena (PbS) and metals need Measured only.\n"
            "Silicates use Oxide. Calcite uses Carbonate.\n"
            "Goethite / clays use Hydroxide, then tune OH and H2O."
        )
        self.matrix_combo.currentIndexChanged.connect(self._on_matrix_control_changed)
        grid.addWidget(QLabel("Model"), 0, 0)
        grid.addWidget(self.matrix_combo, 0, 1, 1, 3)

        self.fe_combo = QComboBox()
        self.fe_combo.addItems(["FeO", "Fe2O3"])
        self.fe_combo.setEnabled(False)
        self.fe_combo.setToolTip("Iron oxidation state for the oxide model")
        self.fe_combo.currentTextChanged.connect(self._on_matrix_control_changed)
        grid.addWidget(QLabel("Fe as"), 1, 0)
        grid.addWidget(self.fe_combo, 1, 1)

        self.h2o_spin = self._light_spin("Molecular water (wt%). Default 0.")
        self.oh_spin = self._light_spin("Extra hydroxide (wt% OH), on top of the hydroxide model.")
        self.co2_spin = self._light_spin("Extra CO2 (wt%), on top of the carbonate model.")
        grid.addWidget(QLabel("H₂O"), 2, 0)
        grid.addWidget(self.h2o_spin, 2, 1)
        grid.addWidget(QLabel("OH"), 2, 2)
        grid.addWidget(self.oh_spin, 2, 3)
        grid.addWidget(QLabel("CO₂"), 3, 0)
        grid.addWidget(self.co2_spin, 3, 1)

        layout.addLayout(grid)

        self.matrix_hint = QLabel()
        self.matrix_hint.setWordWrap(True)
        self.matrix_hint.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(self.matrix_hint)
        self._refresh_matrix_hint()
        return group

    def _light_spin(self, tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 80.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        spin.setSuffix(" wt%")
        spin.setValue(0.0)
        spin.setToolTip(tooltip)
        spin.valueChanged.connect(self._on_matrix_control_changed)
        return spin
    
    def _create_statistics_group(self):
        """Create fit statistics display group"""
        group = QGroupBox("Fit Statistics")
        layout = QHBoxLayout(group)  # Changed to horizontal for compact display
        layout.setSpacing(15)
        
        # Chi-squared
        self.chi_squared_label = QLabel("χ²: --")
        self.chi_squared_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.chi_squared_label)
        
        # R-squared
        self.r_squared_label = QLabel("R²: --")
        self.r_squared_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.r_squared_label)
        
        # Reduced chi-squared
        self.reduced_chi_label = QLabel("χ²ᵣ: --")
        self.reduced_chi_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.reduced_chi_label)
        
        # Iterations
        self.iterations_label = QLabel("Iterations: --")
        self.iterations_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.iterations_label)
        
        layout.addStretch()  # Push stats to the left
        
        return group
    
    def _create_results_table_group(self):
        """Create quantification results table"""
        group = QGroupBox("Quantification")
        layout = QVBoxLayout(group)
        
        # Create table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Element",
            "Rel. Intensity",
            "Uncertainty",
            "Line"
        ])
        
        # Configure table appearance
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setToolTip("Click an element to show its emission lines on the spectrum (click again to clear)")
        self.results_table.cellClicked.connect(self._on_result_cell_clicked)
        
        layout.addWidget(self.results_table)

        self.formula_label = QLabel("")
        self.formula_label.setWordWrap(True)
        self.formula_label.setFont(QFont("Arial", 9))
        self.formula_label.setStyleSheet("color: #333;")
        layout.addWidget(self.formula_label)
        
        # Method note
        self.method_label = QLabel(
            "Method: area-normalized semi-quant (not FP wt%)"
        )
        self.method_label.setFont(QFont("Arial", 9))
        self.method_label.setStyleSheet("color: #666;")
        layout.addWidget(self.method_label)

        # Total concentration label
        self.total_label = QLabel("Sum of relative intensities: -- %")
        self.total_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.total_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.total_label)
        
        return group
    
    def _create_peaks_group(self):
        """Create peak identification list"""
        group = QGroupBox("Identified Peaks")
        layout = QVBoxLayout(group)
        
        self.peaks_text = QTextEdit()
        self.peaks_text.setReadOnly(True)
        self.peaks_text.setMinimumHeight(100)
        self.peaks_text.setPlaceholderText("No peaks identified yet")
        layout.addWidget(self.peaks_text)
        
        return group

    def get_matrix_assumptions(self) -> MatrixAssumptions:
        kind = self.matrix_combo.currentData()
        if not isinstance(kind, MatrixKind):
            kind = MatrixKind.MEASURED
        return MatrixAssumptions(
            kind=kind,
            fe_as=self.fe_combo.currentText() or "FeO",
            h2o_wt=float(self.h2o_spin.value()),
            oh_wt=float(self.oh_spin.value()),
            co2_wt=float(self.co2_spin.value()),
        )

    def set_fp_live(self, enabled: bool) -> None:
        self._fp_live = bool(enabled)

    def _on_matrix_control_changed(self, *_args):
        if self._updating_controls:
            return
        self._refresh_matrix_hint()
        if self._fp_live:
            self._debounce.start()

    def _emit_matrix_changed(self):
        self.matrix_assumptions_changed.emit()

    def _refresh_matrix_hint(self):
        assumptions = self.get_matrix_assumptions()
        self.fe_combo.setEnabled(assumptions.kind == MatrixKind.OXIDE)
        self.matrix_hint.setText(assumptions.hint())
    
    def set_fit_statistics(self, statistics):
        """
        Update fit statistics
        
        Args:
            statistics: Dictionary with chi_squared, reduced_chi_squared, r_squared, etc.
        """
        chi_squared = statistics.get('chi_squared', 0)
        r_squared = statistics.get('r_squared', 0)
        reduced_chi = statistics.get('reduced_chi_squared', 0)
        iterations = statistics.get('iterations', 1)
        
        self.chi_squared_label.setText(f"χ²: {chi_squared:.4f}")
        self.r_squared_label.setText(f"R²: {r_squared:.4f}")
        self.reduced_chi_label.setText(f"χ²ᵣ: {reduced_chi:.4f}")
        self.iterations_label.setText(f"Iterations: {iterations}")

        warnings = statistics.get('tube_overlap_warnings') or []
        if warnings:
            warn_txt = "Tube ratio flags:\n" + "\n".join(f"• {w}" for w in warnings)
            self.chi_squared_label.setToolTip(warn_txt)
            self.reduced_chi_label.setStyleSheet("color: #cc6600; font-weight: bold;")
        else:
            self.chi_squared_label.setToolTip("")
            self.reduced_chi_label.setStyleSheet("")
    
    def set_results(self, results):
        """
        Update quantification results table
        
        Args:
            results: List of dictionaries with keys:
                     'element', 'concentration', 'error', 'line'
        """
        self.results_data = results
        self.results_table.setRowCount(len(results))

        method = "semi_quant_area"
        if results:
            method = results[0].get("method", method)
        is_fp = method == "fp_matrix"

        if is_fp:
            self.results_table.setHorizontalHeaderLabels([
                "Element", "wt%", "Source", "Line"
            ])
        else:
            self.results_table.setHorizontalHeaderLabels([
                "Element", "Rel. Intensity", "Uncertainty", "Line"
            ])
        
        total_concentration = 0.0
        assumed_brush = QBrush(QColor("#f3f3f3"))
        
        for i, result in enumerate(results):
            # Element symbol
            element_item = QTableWidgetItem(result['element'])
            element_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(i, 0, element_item)
            
            conc = result['concentration']
            conc_item = QTableWidgetItem(f"{conc:.3f} %")
            conc_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results_table.setItem(i, 1, conc_item)

            role = result.get("role")
            error = result.get('error', None)
            if is_fp:
                source = "assumed" if role == "assumed" else "measured"
                mid_item = QTableWidgetItem(source)
            elif error is None:
                mid_item = QTableWidgetItem("—")
            else:
                mid_item = QTableWidgetItem(f"± {error:.3f} %")
            mid_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results_table.setItem(i, 2, mid_item)
            
            line_item = QTableWidgetItem(result.get('line', 'K'))
            line_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(i, 3, line_item)

            if role == "assumed":
                for col in range(4):
                    item = self.results_table.item(i, col)
                    if item is not None:
                        item.setBackground(assumed_brush)
            
            total_concentration += conc
        
        self.total_label.setText(
            f"{'Analytical total' if is_fp else 'Sum of relative intensities'}: "
            f"{total_concentration:.2f} %"
        )
        
        if 98 <= total_concentration <= 102:
            self.total_label.setStyleSheet("color: green;")
        elif 95 <= total_concentration <= 105:
            self.total_label.setStyleSheet("color: orange;")
        else:
            self.total_label.setStyleSheet("color: red;")
        
        if hasattr(self, "method_label"):
            if is_fp:
                self.method_label.setText(
                    "Method: standardless FP wt% (relative Sherman + matrix assumptions)"
                )
            else:
                self.method_label.setText(
                    "Method: area-normalized semi-quant (not FP wt%)"
                )

    def set_formula_summary(self, text: str) -> None:
        self.formula_label.setText(text or "")
    
    def set_peaks(self, peaks):
        """
        Update identified peaks list from Peak objects
        
        Args:
            peaks: List of Peak objects from fitting
        """
        text_lines = []
        for peak in peaks:
            if peak.element and peak.line:
                tube = ""
                if getattr(peak, 'is_tube_line', False):
                    tube = " [TUBE]"
                    if getattr(peak, 'fixed_fwhm', None) is not None:
                        tube = " [TUBE, wide]"
                text_lines.append(
                    f"{peak.element}-{peak.line}: {peak.energy:.3f} keV "
                    f"(Area={peak.area:.0f}, FWHM={peak.fwhm:.3f} keV){tube}"
                )
            else:
                text_lines.append(
                    f"Peak at {peak.energy:.3f} keV "
                    f"(Area={peak.area:.0f}, FWHM={peak.fwhm:.3f} keV)"
                )

        self.peaks_text.setPlainText("\n".join(text_lines) if text_lines else "No peaks")
    
    def set_tube_overlap_flags(self, flags):
        """Append tube-profile overlap warnings under the peaks list."""
        if not flags:
            return
        current = self.peaks_text.toPlainText()
        block = "\n\n--- Tube profile ratio checks ---\n" + "\n".join(
            f"⚠ {f.get('message', f)}" for f in flags
        )
        self.peaks_text.setPlainText(current + block)

    def set_tube_constraint_notes(self, notes):
        """Append soft-prior / doublet notes under the peaks list."""
        if not notes:
            return
        current = self.peaks_text.toPlainText()
        block = "\n\n--- Tube constraints ---\n" + "\n".join(f"• {n}" for n in notes)
        self.peaks_text.setPlainText(current + block)
    
    def set_quantification(self, concentrations):
        """
        Update quantification results from concentration dictionary
        
        Args:
            concentrations: Dict with element symbols as keys, each containing
                          'concentration', 'error', 'lines' (list), 'total_area'
        """
        results = []
        for element, data in concentrations.items():
            lines = [str(line) for line in data.get('lines', []) if line]
            role = data.get("role")
            line = data.get("line") or (', '.join(lines) if lines else '--')
            if role == "assumed" and (not line or line == "--"):
                line = "assumed"
            results.append({
                'element': element,
                'concentration': data['concentration'],
                'error': data.get('error'),
                'line': line,
                'method': data.get('method', 'semi_quant_area'),
                'role': role,
            })
        
        self.set_results(results)
    
    def clear_results(self):
        """Clear all results and statistics"""
        self.results_table.setRowCount(0)
        self.results_data = []
        self._fp_live = False
        self.total_label.setText("Sum of relative intensities: -- %")
        self.total_label.setStyleSheet("")
        if hasattr(self, "method_label"):
            self.method_label.setText(
                "Method: area-normalized semi-quant (not FP wt%)"
            )
        if hasattr(self, "formula_label"):
            self.formula_label.setText("")
        self.chi_squared_label.setText("χ²: --")
        self.r_squared_label.setText("R²: --")
        self.reduced_chi_label.setText("χ²ᵣ: --")
        self.iterations_label.setText("Iterations: --")
        self.peaks_text.clear()
    
    def get_results(self):
        """Return current results data"""
        return self.results_data
    
    def _on_result_cell_clicked(self, row, _column):
        """Emit selected element so the spectrum can show/clear its lines"""
        if 0 <= row < len(self.results_data):
            symbol = self.results_data[row].get('element')
            if symbol:
                self.element_selected.emit(symbol)
    
    def add_result_row(self, element, concentration, error, line):
        """
        Add a single result row
        
        Args:
            element: Element symbol
            concentration: Concentration value
            error: Error/uncertainty
            line: X-ray line (K, L, M)
        """
        result = {
            'element': element,
            'concentration': concentration,
            'error': error,
            'line': line
        }
        self.results_data.append(result)
        self.set_results(self.results_data)
