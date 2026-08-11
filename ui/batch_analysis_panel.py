"""
Batch Analysis Panel UI

This panel handles bulk spectral fitting and quantification of multiple XRF spectra.
Users can process many spectra at once and review individual fit quality.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QPushButton, QLabel, QFileDialog, QProgressBar,
                               QMessageBox, QSplitter, QTabWidget, QListWidget,
                               QListWidgetItem, QTextEdit, QTableWidget, QTableWidgetItem,
                               QHeaderView, QCheckBox, QComboBox, QScrollArea, QFormLayout)
from PySide6.QtCore import Qt, Signal, QThread
from pathlib import Path
import pyqtgraph as pg
import numpy as np

from core.batch_processing import BatchProcessor, BatchProcessingConfig, BatchFitResult
from ui.element_panel import ElementPanel


class BatchProcessingWorker(QThread):
    """Worker thread for batch processing"""
    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(list)  # List of BatchFitResult
    error = Signal(str)
    
    def __init__(self, processor, file_paths):
        super().__init__()
        self.processor = processor
        self.file_paths = file_paths
    
    def run(self):
        """Run batch processing in background"""
        try:
            results = self.processor.process_file_list(
                self.file_paths,
                progress_callback=self.progress.emit
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class BatchAnalysisPanel(QWidget):
    """Panel for batch spectral fitting and quantification"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.config = BatchProcessingConfig()
        self.processor = None
        self.worker = None
        self.results = []
        self.current_result = None
        self.element_panel = None  # Will be set from main window
        
        self._init_ui()
    
    def set_element_panel(self, element_panel):
        """Set reference to Analysis tab's element panel"""
        self.element_panel = element_panel
        self._update_settings_summary()
    
    def _init_ui(self):
        """Initialize the user interface with sub-tabs"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)

        left_tab_widget = QTabWidget()
        left_tab_widget.setMinimumWidth(280)
        left_tab_widget.setMaximumWidth(380)

        left_tab_widget.addTab(self._create_setup_tab(), "Setup")
        left_tab_widget.addTab(self._create_results_tab(), "Results")

        splitter.addWidget(left_tab_widget)
        splitter.addWidget(self._create_plot_widget())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 900])

        layout.addWidget(splitter)

    def _create_setup_tab(self):
        """Create Setup tab with files and settings summary"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        layout.addWidget(self._create_settings_summary_group())
        layout.addWidget(self._create_file_selection_group(), stretch=1)
        layout.addWidget(self._create_processing_controls_group())

        return widget

    def _create_results_tab(self):
        """Create Results tab with sub-tabs"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        results_subtabs = QTabWidget()
        results_subtabs.addTab(self._create_summary_subtab(), "Summary")
        results_subtabs.addTab(self._create_trends_subtab(), "Trends")
        layout.addWidget(results_subtabs)

        return widget

    def _create_summary_subtab(self):
        """Create summary sub-tab with statistics and spectrum list"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        layout.addWidget(self._create_summary_group())
        layout.addWidget(self._create_spectrum_list_group(), stretch=2)
        layout.addWidget(self._create_export_group())

        return widget

    def _create_trends_subtab(self):
        """Create concentration trends sub-tab"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        layout.addWidget(self._create_element_trends_selection())
        self.trends_plot_widget = self._create_trends_plot_widget()
        layout.addWidget(self.trends_plot_widget, stretch=1)

        return widget

    def _create_settings_summary_group(self):
        """Compact settings snapshot from Analysis tab"""
        group = QGroupBox("Settings (from Analysis)")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(3)
        form.setLabelAlignment(Qt.AlignRight)

        self.settings_elements = QLabel("—")
        self.settings_elements.setWordWrap(True)
        self.settings_excitation = QLabel("—")
        self.settings_fit = QLabel("—")
        self.settings_tube = QLabel("—")

        form.addRow("Elements", self.settings_elements)
        form.addRow("Beam", self.settings_excitation)
        form.addRow("Fit", self.settings_fit)
        form.addRow("Tube", self.settings_tube)
        layout.addLayout(form)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip(
            "Pull current elements / exp params / fit options from the Analysis tab.\n"
            "Configure those there first, then refresh here before processing."
        )
        refresh_btn.clicked.connect(self._update_settings_summary)
        layout.addWidget(refresh_btn)

        self._update_settings_summary()
        return group

    def _update_settings_summary(self):
        """Update settings summary from Analysis tab"""
        if not self.element_panel:
            self.settings_elements.setText("not ready")
            self.settings_excitation.setText("—")
            self.settings_fit.setText("—")
            self.settings_tube.setText("—")
            self.settings_elements.setStyleSheet("color: #888;")
            return

        try:
            elements = [e['symbol'] for e in self.element_panel.selected_elements]
            excitation = self.element_panel.excitation_spin.value()
            current = self.element_panel.current_spin.value()
            live_time = self.element_panel.live_time_spin.value()
            background = self.element_panel.background_combo.currentText()
            peak_shape = self.element_panel.peak_shape_combo.currentText()
            escape_peaks = self.element_panel.escape_peaks_check.isChecked()
            tube_on = self.element_panel.tube_lines_check.isChecked()
            tube_element = (
                self.element_panel.tube_element_combo.currentText() if tube_on else "off"
            )

            self.settings_elements.setStyleSheet("")
            if elements:
                shown = ", ".join(elements[:12])
                if len(elements) > 12:
                    shown += f" (+{len(elements) - 12})"
                self.settings_elements.setText(shown)
                self.settings_elements.setToolTip(", ".join(elements))
            else:
                self.settings_elements.setText("none selected")
                self.settings_elements.setStyleSheet("color: #b00020;")
                self.settings_elements.setToolTip(
                    "Select elements on the Analysis → Elements tab"
                )

            self.settings_excitation.setText(
                f"{excitation:g} keV · {current:g} mA · {live_time:g} s"
            )
            self.settings_fit.setText(
                f"{peak_shape} · {background}"
                + (" · escape" if escape_peaks else "")
            )
            self.settings_tube.setText(tube_element)

            self.config.elements = elements
            self.config.excitation_energy = excitation
            self.config.tube_current = current
            self.config.live_time = live_time
            self.config.background_method = background.lower()
            self.config.peak_shape = peak_shape.lower()
            self.config.include_escape_peaks = escape_peaks
            self.config.tube_element = tube_element if tube_on else None

        except Exception as e:
            self.settings_elements.setText("error")
            self.settings_elements.setStyleSheet("color: #b00020;")
            self.settings_elements.setToolTip(str(e))

    def _create_file_selection_group(self):
        """Create file selection group"""
        group = QGroupBox("Files")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(140)
        self.file_list.setToolTip(
            "Spectra to process with the Analysis-tab settings"
        )
        layout.addWidget(self.file_list, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        add_files_btn = QPushButton("Add…")
        add_files_btn.setToolTip("Add individual spectrum files")
        add_files_btn.clicked.connect(self._add_files)
        btn_layout.addWidget(add_files_btn)

        add_dir_btn = QPushButton("Folder…")
        add_dir_btn.setToolTip("Add all supported spectra from a directory")
        add_dir_btn.clicked.connect(self._add_directory)
        btn_layout.addWidget(add_dir_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_files)
        btn_layout.addWidget(clear_btn)

        layout.addLayout(btn_layout)

        self.file_count_label = QLabel("0 files")
        self.file_count_label.setStyleSheet("color: #666;")
        layout.addWidget(self.file_count_label)

        return group

    def _clear_files(self):
        self.file_list.clear()
        self._update_file_count()

    def _create_processing_controls_group(self):
        """Create processing controls"""
        group = QGroupBox("Process")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        self.use_calibration_check = QCheckBox("Use calibration")
        self.use_calibration_check.setToolTip(
            "Apply intensity calibration if available"
        )
        layout.addWidget(self.use_calibration_check)

        self.save_fits_check = QCheckBox("Save individual fits")
        self.save_fits_check.setChecked(True)
        layout.addWidget(self.save_fits_check)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumHeight(4)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.progress_label)

        self.process_btn = QPushButton("Process All")
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; color: #666; }
        """)
        self.process_btn.setToolTip(
            "Fit every listed spectrum with the current Analysis-tab settings"
        )
        self.process_btn.clicked.connect(self._process_batch)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)

        return group

    def _create_summary_group(self):
        """Create summary statistics group"""
        group = QGroupBox("Summary")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(88)
        self.summary_text.setPlaceholderText("No results yet")
        self.summary_text.setStyleSheet(
            "QTextEdit { background-color: #f5f5f5; font-size: 11px; "
            "border: 1px solid #ddd; }"
        )
        layout.addWidget(self.summary_text)

        return group

    def _create_spectrum_list_group(self):
        """Create spectrum list group"""
        group = QGroupBox("Spectra")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Spectrum", "OK", "R²", "χ²", "s"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.setToolTip("Select a row to show its fit on the right")
        self.results_table.itemSelectionChanged.connect(self._on_spectrum_selected)
        layout.addWidget(self.results_table)

        return group

    def _create_export_group(self):
        """Create export controls"""
        group = QGroupBox("Export")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        export_csv_btn = QPushButton("CSV")
        export_csv_btn.clicked.connect(lambda: self._export_results("csv"))
        layout.addWidget(export_csv_btn)

        export_excel_btn = QPushButton("Excel")
        export_excel_btn.clicked.connect(lambda: self._export_results("excel"))
        layout.addWidget(export_excel_btn)

        layout.addStretch()
        return group

    def _create_element_trends_selection(self):
        """Create element selection for trends plotting"""
        group = QGroupBox("Elements")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        self.element_trend_checks = {}
        self.element_checks_layout = QVBoxLayout()
        layout.addLayout(self.element_checks_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        select_all_btn = QPushButton("All")
        select_all_btn.clicked.connect(self._select_all_trends)
        btn_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("None")
        deselect_all_btn.clicked.connect(self._deselect_all_trends)
        btn_layout.addWidget(deselect_all_btn)
        layout.addLayout(btn_layout)

        update_btn = QPushButton("Update")
        update_btn.clicked.connect(self._update_trends_plots)
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        layout.addWidget(update_btn)

        layout.addStretch()
        return group

    def _create_trends_plot_widget(self):
        """Create widget for concentration trends plots"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for plots
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container for plots
        self.trends_plot_container = QWidget()
        self.trends_plot_layout = QVBoxLayout(self.trends_plot_container)
        self.trends_plot_layout.setSpacing(10)
        
        scroll.setWidget(self.trends_plot_container)
        layout.addWidget(scroll)
        
        # Store plot widgets
        self.trend_plots = {}
        
        return widget
    
    def _create_plot_widget(self):
        """Create spectrum visualization widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot with subplots
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')
        
        # Top: Spectrum with element contributions
        self.spectrum_plot = self.plot_widget.addPlot(row=0, col=0)
        self.spectrum_plot.setLabel('left', 'Counts', color='k')
        self.spectrum_plot.setLabel('bottom', 'Energy (keV)', color='k')
        self.spectrum_plot.setTitle('Spectrum fit', color='k', size='11pt')
        self.spectrum_plot.addLegend(offset=(10, 10))
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.25)
        
        # Measured spectrum
        self.measured_curve = self.spectrum_plot.plot(
            pen=pg.mkPen('#00008B', width=2), name='Measured'
        )
        
        # Fitted spectrum
        self.fitted_curve = self.spectrum_plot.plot(
            pen=pg.mkPen('r', width=2, style=Qt.DashLine), name='Total Fit'
        )
        
        # Element contribution curves (will be added dynamically)
        self.element_curves = {}
        
        # Bottom: Residuals
        self.residual_plot = self.plot_widget.addPlot(row=1, col=0)
        self.residual_plot.setLabel('left', 'Residuals (σ)', color='k')
        self.residual_plot.setLabel('bottom', 'Energy (keV)', color='k')
        self.residual_plot.showGrid(x=True, y=True, alpha=0.3)
        self.residual_plot.addLine(y=0, pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        
        self.residual_curve = self.residual_plot.plot(
            pen=None, symbol='o', symbolSize=3, symbolBrush='b'
        )
        
        layout.addWidget(self.plot_widget)
        
        return widget
    
    def _add_files(self):
        """Add spectrum files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Spectrum Files",
            "",
            "All Supported (*.txt *.csv *.mca);;Text Files (*.txt);;CSV Files (*.csv);;MCA Files (*.mca)"
        )
        
        if file_paths:
            for path in file_paths:
                self.file_list.addItem(path)
            self._update_file_count()
    
    def _add_directory(self):
        """Add all spectrum files from directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory with Spectrum Files"
        )
        
        if directory:
            # Find all spectrum files
            dir_path = Path(directory)
            for pattern in ["*.txt", "*.csv", "*.mca"]:
                for file_path in dir_path.glob(pattern):
                    self.file_list.addItem(str(file_path))
            self._update_file_count()
    
    def _update_file_count(self):
        """Update file count label"""
        count = self.file_list.count()
        self.file_count_label.setText(f"{count} file{'s' if count != 1 else ''}")
        # Update settings to check if elements are selected
        self._update_settings_summary()
        self.process_btn.setEnabled(count > 0 and len(self.config.elements) > 0)
    
    def _process_batch(self):
        """Start batch processing"""
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "No Files", "Please select spectrum files to process.")
            return
        
        if not self.config.elements:
            QMessageBox.warning(self, "No Elements", "Please select elements to fit.")
            return
        
        # Get file paths
        file_paths = []
        for i in range(self.file_list.count()):
            file_paths.append(Path(self.file_list.item(i).text()))
        
        # Create processor
        self.processor = BatchProcessor(self.config)
        
        # Create worker
        self.worker = BatchProcessingWorker(self.processor, file_paths)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_processing_complete)
        self.worker.error.connect(self._on_processing_error)
        
        # Start processing
        self.progress_bar.setVisible(True)
        self.process_btn.setEnabled(False)
        self.worker.start()
    
    def _on_progress(self, current, total, message):
        """Update progress"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(message)
    
    def _on_processing_complete(self, results):
        """Handle processing completion"""
        self.results = results
        self.progress_bar.setVisible(False)
        self.progress_label.setText(f"Processing complete! {len(results)} spectra processed.")
        self.process_btn.setEnabled(True)
        
        # Update results table
        self._populate_results_table()
        
        # Update summary
        self._update_summary()
        
        # Populate element checkboxes for trends
        self._populate_element_checkboxes()
        
        QMessageBox.information(
            self,
            "Processing Complete",
            f"Processed {len(results)} spectra.\n"
            "Open Results to review fits and Trends.",
        )
    
    def _on_processing_error(self, error_message):
        """Handle processing error"""
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self.process_btn.setEnabled(True)
        
        QMessageBox.critical(
            self,
            "Processing Error",
            f"An error occurred during processing:\n\n{error_message}"
        )
    
    def _populate_results_table(self):
        """Populate results table"""
        self.results_table.setRowCount(len(self.results))
        
        for i, result in enumerate(self.results):
            # Spectrum name
            self.results_table.setItem(i, 0, QTableWidgetItem(result.spectrum_name))
            
            # Success
            success_item = QTableWidgetItem("✓" if result.fit_success else "✗")
            success_item.setForeground(Qt.green if result.fit_success else Qt.red)
            self.results_table.setItem(i, 1, success_item)
            
            # R²
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{result.r_squared:.4f}"))
            
            # χ²
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{result.chi_squared:.4f}"))
            
            # Time
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{result.fit_time:.2f}"))
        
        self.results_table.resizeColumnsToContents()
    
    def _update_summary(self):
        """Update summary statistics"""
        if not self.processor:
            return
        
        stats = self.processor.get_summary_statistics()
        
        summary = (
            f"{stats['successful_fits']}/{stats['total_spectra']} ok  "
            f"({stats['success_rate']:.0f}%)\n"
            f"R² avg {stats['average_r_squared']:.4f}   "
            f"χ² avg {stats['average_chi_squared']:.4f}\n"
            f"Time {stats['total_processing_time']:.1f}s "
            f"({stats['average_fit_time']:.2f}s/spectrum)"
        )
        
        self.summary_text.setPlainText(summary)
    
    def _on_spectrum_selected(self):
        """Handle spectrum selection from results table"""
        selected_rows = self.results_table.selectedIndexes()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        result = self.results[row]
        
        self._display_fit_result(result)
    
    def _display_fit_result(self, result: BatchFitResult):
        """Display fit result in plot"""
        self.current_result = result
        
        if result.energy is None or result.measured_counts is None:
            return
        
        # Plot measured spectrum
        self.measured_curve.setData(x=result.energy, y=result.measured_counts)
        
        # Plot fitted spectrum
        if result.fitted_spectrum is not None:
            self.fitted_curve.setData(x=result.energy, y=result.fitted_spectrum)
        
        # Plot element contributions
        # Clear existing element curves
        for curve in self.element_curves.values():
            self.spectrum_plot.removeItem(curve)
        self.element_curves.clear()
        
        if result.element_contributions:
            colors = ['g', 'm', 'c', 'y', 'orange', 'purple']
            for i, (element, contribution) in enumerate(result.element_contributions.items()):
                color = colors[i % len(colors)]
                curve = self.spectrum_plot.plot(
                    x=result.energy,
                    y=contribution,
                    pen=pg.mkPen(color, width=1, style=Qt.DotLine),
                    name=element
                )
                self.element_curves[element] = curve
        
        # Plot residuals
        if result.residuals is not None:
            self.residual_curve.setData(x=result.energy, y=result.residuals)
        
        # Update title
        self.spectrum_plot.setTitle(
            f"{result.spectrum_name} - R²={result.r_squared:.4f}, χ²={result.chi_squared:.4f}",
            color='k'
        )
    
    def _export_results(self, format):
        """Export results to file"""
        if not self.processor or not self.results:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return
        
        # Get save path
        if format == "csv":
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Results",
                "batch_results.csv",
                "CSV Files (*.csv)"
            )
        elif format == "excel":
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Results",
                "batch_results.xlsx",
                "Excel Files (*.xlsx)"
            )
        else:
            return
        
        if file_path:
            try:
                self.processor.export_results(Path(file_path), format=format)
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Results exported to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to export results:\n{str(e)}"
                )
    
    def _populate_element_checkboxes(self):
        """Populate element checkboxes from results"""
        # Clear existing checkboxes
        for checkbox in self.element_trend_checks.values():
            checkbox.deleteLater()
        self.element_trend_checks.clear()
        
        if not self.results:
            return
        
        # Get all unique elements from results
        all_elements = set()
        for result in self.results:
            all_elements.update(result.concentrations.keys())
        
        # Create checkbox for each element
        for element in sorted(all_elements):
            checkbox = QCheckBox(element)
            checkbox.setChecked(True)  # Default to checked
            checkbox.stateChanged.connect(self._update_trends_plots)
            self.element_checks_layout.addWidget(checkbox)
            self.element_trend_checks[element] = checkbox
        
        # Initial plot update
        self._update_trends_plots()
    
    def _select_all_trends(self):
        """Select all element checkboxes"""
        for checkbox in self.element_trend_checks.values():
            checkbox.setChecked(True)
    
    def _deselect_all_trends(self):
        """Deselect all element checkboxes"""
        for checkbox in self.element_trend_checks.values():
            checkbox.setChecked(False)
    
    def _update_trends_plots(self):
        """Update concentration trends plots based on selected elements"""
        # Clear existing plots
        for plot_widget in self.trend_plots.values():
            self.trends_plot_layout.removeWidget(plot_widget)
            plot_widget.deleteLater()
        self.trend_plots.clear()
        
        if not self.results:
            return
        
        # Get selected elements
        selected_elements = [
            element for element, checkbox in self.element_trend_checks.items()
            if checkbox.isChecked()
        ]
        
        if not selected_elements:
            # Show message if no elements selected
            label = QLabel("<i>No elements selected. Check elements above to plot trends.</i>")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: gray; padding: 20px;")
            self.trends_plot_layout.addWidget(label)
            return
        
        # Create plot for each selected element
        for element in selected_elements:
            plot_widget = self._create_element_trend_plot(element)
            self.trends_plot_layout.addWidget(plot_widget)
            self.trend_plots[element] = plot_widget
        
        # Add stretch at the end
        self.trends_plot_layout.addStretch()
    
    def _create_element_trend_plot(self, element):
        """Create concentration trend plot for a single element"""
        # Create plot widget
        plot_widget = pg.GraphicsLayoutWidget()
        plot_widget.setBackground('w')
        plot_widget.setFixedHeight(250)
        
        # Create plot
        plot = plot_widget.addPlot()
        plot.setLabel('left', f'{element} Concentration', units='wt%', color='k')
        plot.setLabel('bottom', 'Spectrum Number', color='k')
        plot.setTitle(f'{element} Concentration Trend', color='k', size='12pt')
        plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Extract data
        spectrum_numbers = []
        concentrations = []
        errors = []
        
        for i, result in enumerate(self.results):
            if element in result.concentrations:
                spectrum_numbers.append(i + 1)
                concentrations.append(result.concentrations[element])
                errors.append(result.concentration_errors.get(element, 0))
        
        if not spectrum_numbers:
            # No data for this element
            plot.setTitle(f'{element} - No Data', color='k', size='12pt')
            return plot_widget
        
        # Convert to numpy arrays
        x = np.array(spectrum_numbers)
        y = np.array(concentrations)
        err = np.array(errors)
        
        # Plot data points
        plot.plot(
            x, y,
            pen=None,
            symbol='o',
            symbolSize=8,
            symbolBrush=pg.mkBrush(0, 0, 139, 200),
            symbolPen=pg.mkPen('k', width=1)
        )
        
        # Plot error bars if available
        if np.any(err > 0):
            error_bars = pg.ErrorBarItem(
                x=x, y=y,
                top=err, bottom=err,
                beam=0.5,
                pen=pg.mkPen('k', width=1)
            )
            plot.addItem(error_bars)
        
        # Add trend line if enough points
        if len(x) > 1:
            # Simple linear fit
            try:
                coeffs = np.polyfit(x, y, 1)
                trend_y = np.polyval(coeffs, x)
                plot.plot(
                    x, trend_y,
                    pen=pg.mkPen('r', width=2, style=Qt.DashLine)
                )
                
                # Add trend info to title
                slope = coeffs[0]
                if abs(slope) > 0.001:
                    plot.setTitle(
                        f'{element} Concentration Trend (slope: {slope:+.4f} wt%/spectrum)',
                        color='k', size='12pt'
                    )
            except:
                pass  # Skip trend line if fit fails
        
        return plot_widget
