"""
Batch Analysis Panel UI

This panel handles bulk spectral fitting and quantification of multiple XRF spectra.
Users can process many spectra at once and review individual fit quality.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QPushButton, QLabel, QFileDialog, QProgressBar,
                               QMessageBox, QSplitter, QTabWidget, QListWidget,
                               QListWidgetItem, QTextEdit, QTableWidget, QTableWidgetItem,
                               QHeaderView, QCheckBox, QComboBox, QScrollArea)
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
    
    def _init_ui(self):
        """Initialize the user interface with sub-tabs"""
        layout = QVBoxLayout(self)
        
        # Create splitter for controls and plot
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Tabbed interface
        left_tab_widget = QTabWidget()
        left_tab_widget.setMaximumWidth(700)
        
        # Tab 1: Setup (Files + Settings Summary)
        setup_tab = self._create_setup_tab()
        left_tab_widget.addTab(setup_tab, "Setup")
        
        # Tab 2: Results
        results_tab = self._create_results_tab()
        left_tab_widget.addTab(results_tab, "Results")
        
        splitter.addWidget(left_tab_widget)
        
        # Right side: Spectrum visualization
        plot_widget = self._create_plot_widget()
        splitter.addWidget(plot_widget)
        
        # Set initial sizes
        splitter.setSizes([600, 600])
        
        layout.addWidget(splitter)
    
    def _create_setup_tab(self):
        """Create Setup tab with files and settings summary"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)
        
        # Info banner
        info_group = self._create_info_banner()
        layout.addWidget(info_group)
        
        # Settings summary from Analysis tab
        settings_group = self._create_settings_summary_group()
        layout.addWidget(settings_group)
        
        # File selection group
        files_group = self._create_file_selection_group()
        layout.addWidget(files_group, stretch=1)
        
        # Processing controls
        controls_group = self._create_processing_controls_group()
        layout.addWidget(controls_group)
        
        return widget
    
    def _create_results_tab(self):
        """Create Results tab with sub-tabs"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create sub-tab widget
        results_subtabs = QTabWidget()
        
        # Sub-tab 1: Summary & List
        summary_tab = self._create_summary_subtab()
        results_subtabs.addTab(summary_tab, "Summary")
        
        # Sub-tab 2: Concentration Trends
        trends_tab = self._create_trends_subtab()
        results_subtabs.addTab(trends_tab, "Trends")
        
        layout.addWidget(results_subtabs)
        
        return widget
    
    def _create_summary_subtab(self):
        """Create summary sub-tab with statistics and spectrum list"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)
        
        # Summary statistics
        summary_group = self._create_summary_group()
        layout.addWidget(summary_group)
        
        # Spectrum list
        list_group = self._create_spectrum_list_group()
        layout.addWidget(list_group, stretch=2)
        
        # Export controls
        export_group = self._create_export_group()
        layout.addWidget(export_group)
        
        return widget
    
    def _create_trends_subtab(self):
        """Create concentration trends sub-tab"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(5)
        
        # Left: Element selection
        selection_group = self._create_element_trends_selection()
        layout.addWidget(selection_group)
        
        # Right: Plots
        self.trends_plot_widget = self._create_trends_plot_widget()
        layout.addWidget(self.trends_plot_widget, stretch=1)
        
        return widget
    
    def _create_info_banner(self):
        """Create info banner explaining workflow"""
        group = QGroupBox("Batch Processing Workflow")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        
        info = QLabel(
            "<b>📋 How to use Batch Analysis:</b><br>"
            "1. Go to <b>Analysis</b> tab and configure your fitting parameters<br>"
            "2. Select elements, experimental parameters, and fitting options<br>"
            "3. Return here and add your spectrum files<br>"
            "4. Click <b>Process All Spectra</b> - settings from Analysis tab will be used<br><br>"
            "<i>All spectra will be fit with the same parameters for consistency</i>"
        )
        info.setWordWrap(True)
        info.setStyleSheet("QLabel { background-color: #e3f2fd; padding: 10px; border-radius: 5px; }")
        layout.addWidget(info)
        
        return group
    
    def _create_settings_summary_group(self):
        """Create settings summary showing what will be used from Analysis tab"""
        group = QGroupBox("Current Settings (from Analysis Tab)")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        
        self.settings_summary = QTextEdit()
        self.settings_summary.setReadOnly(True)
        self.settings_summary.setMaximumHeight(120)
        self.settings_summary.setStyleSheet(
            "QTextEdit { background-color: #f5f5f5; font-family: 'Courier New', monospace; font-size: 9pt; }"
        )
        layout.addWidget(self.settings_summary)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Settings")
        refresh_btn.clicked.connect(self._update_settings_summary)
        refresh_btn.setToolTip("Update settings from Analysis tab")
        layout.addWidget(refresh_btn)
        
        # Initial update
        self._update_settings_summary()
        
        return group
    
    def _update_settings_summary(self):
        """Update settings summary from Analysis tab"""
        if not self.element_panel:
            self.settings_summary.setPlainText(
                "⚠️  Analysis tab not initialized yet.\n"
                "Configure settings in Analysis tab first."
            )
            return
        
        # Get settings from element panel
        try:
            elements = [e['symbol'] for e in self.element_panel.selected_elements]
            excitation = self.element_panel.excitation_spin.value()
            current = self.element_panel.current_spin.value()
            live_time = self.element_panel.live_time_spin.value()
            background = self.element_panel.background_combo.currentText()
            peak_shape = self.element_panel.peak_shape_combo.currentText()
            escape_peaks = self.element_panel.escape_peaks_check.isChecked()
            tube_element = self.element_panel.tube_element_combo.currentText() if self.element_panel.tube_lines_check.isChecked() else "None"
            
            summary = f"""
Elements:     {', '.join(elements) if elements else 'None selected'}
Excitation:   {excitation} keV
Tube Current: {current} mA
Live Time:    {live_time} s
Background:   {background}
Peak Shape:   {peak_shape}
Escape Peaks: {'Yes' if escape_peaks else 'No'}
Tube Lines:   {tube_element}
            """
            
            self.settings_summary.setPlainText(summary.strip())
            
            # Update config
            self.config.elements = elements
            self.config.excitation_energy = excitation
            self.config.tube_current = current
            self.config.live_time = live_time
            self.config.background_method = background.lower()
            self.config.peak_shape = peak_shape.lower()
            self.config.include_escape_peaks = escape_peaks
            self.config.tube_element = tube_element if tube_element != "None" else None
            
        except Exception as e:
            self.settings_summary.setPlainText(
                f"⚠️  Error reading settings:\n{str(e)}\n\n"
                "Make sure Analysis tab is properly configured."
            )
    
    def _create_file_selection_group(self):
        """Create file selection group"""
        group = QGroupBox("Spectrum Files")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(5)
        
        # Info
        info = QLabel(
            "<b>Select multiple spectrum files for batch processing</b><br>"
            "All spectra will be fit with the same parameters"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # File list (more space now)
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(200)
        layout.addWidget(self.file_list, stretch=1)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        add_files_btn = QPushButton("Add Files...")
        add_files_btn.clicked.connect(self._add_files)
        btn_layout.addWidget(add_files_btn)
        
        add_dir_btn = QPushButton("Add Directory...")
        add_dir_btn.clicked.connect(self._add_directory)
        btn_layout.addWidget(add_dir_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.file_list.clear)
        btn_layout.addWidget(clear_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # File count
        self.file_count_label = QLabel("0 files selected")
        layout.addWidget(self.file_count_label)
        
        return group
    
    def _create_processing_controls_group(self):
        """Create processing controls"""
        group = QGroupBox("Processing")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(3)
        
        # Options
        self.use_calibration_check = QCheckBox("Use Calibration (if available)")
        layout.addWidget(self.use_calibration_check)
        
        self.save_fits_check = QCheckBox("Save Individual Fits")
        self.save_fits_check.setChecked(True)
        layout.addWidget(self.save_fits_check)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)
        
        # Process button
        self.process_btn = QPushButton("Process All Spectra")
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.process_btn.clicked.connect(self._process_batch)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)
        
        return group
    
    def _create_summary_group(self):
        """Create summary statistics group"""
        group = QGroupBox("Summary Statistics")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(3)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(100)
        self.summary_text.setPlainText("No results yet")
        layout.addWidget(self.summary_text)
        
        return group
    
    def _create_spectrum_list_group(self):
        """Create spectrum list group"""
        group = QGroupBox("Processed Spectra")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(3)
        
        # Info
        info = QLabel("<b>Click a spectrum to view fit details</b>")
        layout.addWidget(info)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Spectrum", "Success", "R²", "χ²", "Time (s)"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.itemSelectionChanged.connect(self._on_spectrum_selected)
        layout.addWidget(self.results_table)
        
        return group
    
    def _create_export_group(self):
        """Create export controls"""
        group = QGroupBox("Export Results")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        
        export_csv_btn = QPushButton("Export CSV")
        export_csv_btn.clicked.connect(lambda: self._export_results("csv"))
        layout.addWidget(export_csv_btn)
        
        export_excel_btn = QPushButton("Export Excel")
        export_excel_btn.clicked.connect(lambda: self._export_results("excel"))
        layout.addWidget(export_excel_btn)
        
        layout.addStretch()
        
        return group
    
    def _create_element_trends_selection(self):
        """Create element selection for trends plotting"""
        group = QGroupBox("Elements to Plot")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(5, 8, 5, 5)
        layout.setSpacing(5)
        
        # Info
        info = QLabel("<b>Select elements to plot concentration trends</b>")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Checkboxes for each element (will be populated after processing)
        self.element_trend_checks = {}
        self.element_checks_layout = QVBoxLayout()
        layout.addLayout(self.element_checks_layout)
        
        # Select/Deselect all buttons
        btn_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all_trends)
        btn_layout.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all_trends)
        btn_layout.addWidget(deselect_all_btn)
        
        layout.addLayout(btn_layout)
        
        # Update button
        update_btn = QPushButton("Update Plots")
        update_btn.clicked.connect(self._update_trends_plots)
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
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
        self.spectrum_plot.setTitle('Spectrum Fit', color='k')
        self.spectrum_plot.addLegend()
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        
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
        self.file_count_label.setText(f"{count} file{'s' if count != 1 else ''} selected")
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
            f"Successfully processed {len(results)} spectra.\n\n"
            f"Click on a spectrum in the Results tab to view fit details.\n"
            f"Check the Trends sub-tab to plot concentration trends."
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
        
        summary = f"""
Batch Processing Summary
{'='*50}

Total Spectra:      {stats['total_spectra']}
Successful Fits:    {stats['successful_fits']}
Failed Fits:        {stats['failed_fits']}
Success Rate:       {stats['success_rate']:.1f}%

Average R²:         {stats['average_r_squared']:.4f}
Average χ²:         {stats['average_chi_squared']:.4f}

Average Fit Time:   {stats['average_fit_time']:.2f} s
Total Time:         {stats['total_processing_time']:.2f} s
        """
        
        self.summary_text.setPlainText(summary.strip())
    
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
