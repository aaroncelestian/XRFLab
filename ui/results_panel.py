"""
Results panel for displaying quantification results and fit statistics
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
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
        """Setup the panel layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Fit statistics group
        stats_group = self._create_statistics_group()
        layout.addWidget(stats_group)
        
        # Results table group
        results_group = self._create_results_table_group()
        layout.addWidget(results_group, stretch=1)
        
        # Peak identification group
        peaks_group = self._create_peaks_group()
        layout.addWidget(peaks_group)
        
        # Export button
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
        layout.addWidget(self.export_button)
    
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
    
    def set_fit_statistics(self, chi_squared, r_squared, iterations):
        """
        Update fit statistics display
        
        Args:
            chi_squared: Chi-squared value
            r_squared: R-squared value
            iterations: Number of iterations
        """
        self.chi_squared_label.setText(f"χ²: {chi_squared:.4f}")
        self.r_squared_label.setText(f"R²: {r_squared:.4f}")
        
        # Calculate reduced chi-squared (assuming degrees of freedom)
        # This is a placeholder - actual calculation needs proper DOF
        reduced_chi = chi_squared / max(1, iterations)
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
    
    def set_identified_peaks(self, peaks):
        """
        Update identified peaks list
        
        Args:
            peaks: List of dictionaries with keys:
                   'energy', 'element', 'line', 'intensity'
        """
        text_lines = []
        for peak in peaks:
            energy = peak['energy']
            element = peak['element']
            line = peak['line']
            intensity = peak.get('intensity', 0)
            
            text_lines.append(
                f"{element}-{line}: {energy:.3f} keV (I={intensity:.0f})"
            )
        
        self.peaks_text.setPlainText("\n".join(text_lines))
    
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
