"""
Results panel for displaying quantification results and fit statistics
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel, QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class ResultsPanel(QWidget):
    """Panel for displaying analysis results and statistics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.results_data = []
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the panel layout with three columns"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create horizontal layout for three columns
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(5)
        
        # Column 1: Fit statistics
        stats_group = self._create_statistics_group()
        columns_layout.addWidget(stats_group, stretch=1)
        
        # Column 2: Quantification results
        results_group = self._create_results_table_group()
        columns_layout.addWidget(results_group, stretch=2)
        
        # Column 3: Peak identification
        peaks_group = self._create_peaks_group()
        columns_layout.addWidget(peaks_group, stretch=1)
        
        main_layout.addLayout(columns_layout)
        
        # Export button at the bottom (full width)
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
        main_layout.addWidget(self.export_button)
    
    def _create_statistics_group(self):
        """Create fit statistics display group"""
        group = QGroupBox("Fit Statistics")
        layout = QVBoxLayout(group)
        
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
        
        return group
    
    def _create_results_table_group(self):
        """Create quantification results table"""
        group = QGroupBox("Quantification Results")
        layout = QVBoxLayout(group)
        
        # Create table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Element",
            "Concentration",
            "Error",
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
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        layout.addWidget(self.results_table)
        
        # Total concentration label
        self.total_label = QLabel("Total: -- %")
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
        self.peaks_text.setMaximumHeight(120)
        self.peaks_text.setPlaceholderText("No peaks identified yet")
        layout.addWidget(self.peaks_text)
        
        return group
    
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
    
    def set_results(self, results):
        """
        Update quantification results table
        
        Args:
            results: List of dictionaries with keys:
                     'element', 'concentration', 'error', 'line'
        """
        self.results_data = results
        self.results_table.setRowCount(len(results))
        
        total_concentration = 0.0
        
        for i, result in enumerate(results):
            # Element symbol
            element_item = QTableWidgetItem(result['element'])
            element_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(i, 0, element_item)
            
            # Concentration
            conc = result['concentration']
            conc_item = QTableWidgetItem(f"{conc:.3f} %")
            conc_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results_table.setItem(i, 1, conc_item)
            
            # Error
            error = result.get('error', 0.0)
            error_item = QTableWidgetItem(f"± {error:.3f} %")
            error_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results_table.setItem(i, 2, error_item)
            
            # Line
            line_item = QTableWidgetItem(result.get('line', 'K'))
            line_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(i, 3, line_item)
            
            total_concentration += conc
        
        # Update total
        self.total_label.setText(f"Total: {total_concentration:.2f} %")
        
        # Color code based on total (should be close to 100%)
        if 98 <= total_concentration <= 102:
            self.total_label.setStyleSheet("color: green;")
        elif 95 <= total_concentration <= 105:
            self.total_label.setStyleSheet("color: orange;")
        else:
            self.total_label.setStyleSheet("color: red;")
    
    def set_peaks(self, peaks):
        """
        Update identified peaks list from Peak objects
        
        Args:
            peaks: List of Peak objects from fitting
        """
        text_lines = []
        for peak in peaks:
            if peak.element and peak.line:
                text_lines.append(
                    f"{peak.element}-{peak.line}: {peak.energy:.3f} keV "
                    f"(Area={peak.area:.0f}, FWHM={peak.fwhm:.3f} keV)"
                )
            else:
                text_lines.append(
                    f"Unknown: {peak.energy:.3f} keV "
                    f"(Area={peak.area:.0f}, FWHM={peak.fwhm:.3f} keV)"
                )
        
        if text_lines:
            self.peaks_text.setPlainText("\n".join(text_lines))
        else:
            self.peaks_text.setPlainText("No peaks identified")
    
    def set_quantification(self, concentrations):
        """
        Update quantification results from concentration dictionary
        
        Args:
            concentrations: Dict with element symbols as keys, each containing
                          'concentration', 'error', 'lines' (list), 'total_area'
        """
        results = []
        for element, data in concentrations.items():
            # Format lines list as comma-separated string
            lines_str = ', '.join(data.get('lines', []))
            
            results.append({
                'element': element,
                'concentration': data['concentration'],
                'error': data['error'],
                'line': lines_str  # All contributing lines
            })
        
        self.set_results(results)
    
    def clear_results(self):
        """Clear all results and statistics"""
        self.results_table.setRowCount(0)
        self.results_data = []
        self.total_label.setText("Total: -- %")
        self.total_label.setStyleSheet("")
        self.chi_squared_label.setText("χ²: --")
        self.r_squared_label.setText("R²: --")
        self.reduced_chi_label.setText("χ²ᵣ: --")
        self.iterations_label.setText("Iterations: --")
        self.peaks_text.clear()
    
    def get_results(self):
        """Return current results data"""
        return self.results_data
    
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
