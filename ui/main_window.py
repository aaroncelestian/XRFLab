"""
Main window for XRF Fundamental Parameters Analysis Application
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QMenu, QToolBar, QStatusBar, QMessageBox, QFileDialog,
    QTabWidget, QPushButton
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence, QIcon

from ui.spectrum_widget import SpectrumWidget
from ui.element_panel import ElementPanel
from ui.results_panel import ResultsPanel
from ui.batch_analysis_panel import BatchAnalysisPanel
from ui.standards_panel import StandardsPanel
from ui.fwhm_calibration_panel import FWHMCalibrationPanel
from utils.io_handler import IOHandler
from utils.updater import check_for_updates
from core.fitting import SpectrumFitter
from core.smart_peak_id import (
    SmartIDConfig,
    analyze_fitted_peaks,
    apply_smart_id_suggestions,
)


class MainWindow(QMainWindow):
    """Main application window with menu, toolbar, and panels"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRFLab - Fundamental Parameters Analysis")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 700)  # Minimum size for laptop screens
        
        # Initialize components
        self.io_handler = IOHandler()
        self.fitter = SpectrumFitter()
        self.current_spectrum = None
        self.fit_result = None
        self.settings = QSettings()
        self._displayed_element_lines = None  # symbol currently shown on plot, or None
        
        # Setup UI (status bar before central widget — FWHM auto-load may message it)
        self._create_actions()
        self._create_menus()
        self._create_toolbar()
        self._create_status_bar()
        self._create_central_widget()
        self._load_stylesheet()
        
        # Restore window state
        self._restore_settings()
    
    def _create_actions(self):
        """Create all menu and toolbar actions"""
        # File actions
        self.open_action = QAction("&Open Spectrum...", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.setStatusTip("Open an XRF spectrum file")
        self.open_action.triggered.connect(self.open_spectrum)
        
        self.open_project_action = QAction("Open &Project...", self)
        self.open_project_action.setStatusTip("Open an XRF project file")
        self.open_project_action.triggered.connect(self.open_project)
        
        self.save_project_action = QAction("&Save Project...", self)
        self.save_project_action.setShortcut(QKeySequence.Save)
        self.save_project_action.setStatusTip("Save current project")
        self.save_project_action.triggered.connect(self.save_project)
        
        self.export_results_action = QAction("&Export Results...", self)
        self.export_results_action.setStatusTip("Export analysis results")
        self.export_results_action.triggered.connect(self.export_results)
        
        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.setStatusTip("Exit application")
        self.exit_action.triggered.connect(self.close)
        
        # Analysis actions
        self.fit_spectrum_action = QAction("&Fit Spectrum", self)
        self.fit_spectrum_action.setShortcut("Ctrl+F")
        self.fit_spectrum_action.setStatusTip("Fit the current spectrum")
        self.fit_spectrum_action.triggered.connect(self.fit_spectrum)
        
        self.quantify_action = QAction("&Quantification", self)
        self.quantify_action.setShortcut("Ctrl+Q")
        self.quantify_action.setStatusTip("Perform quantitative analysis")
        self.quantify_action.triggered.connect(self.quantify)
        
        self.background_action = QAction("&Background Settings...", self)
        self.background_action.setStatusTip("Configure background removal")
        self.background_action.triggered.connect(self.configure_background)
        
        # View actions
        self.toggle_log_action = QAction("&Logarithmic Y-axis", self)
        self.toggle_log_action.setCheckable(True)
        self.toggle_log_action.setChecked(False)
        self.toggle_log_action.setStatusTip("Toggle logarithmic Y-axis")
        self.toggle_log_action.triggered.connect(self.toggle_log_scale)
        
        self.toggle_grid_action = QAction("Show &Grid", self)
        self.toggle_grid_action.setCheckable(True)
        self.toggle_grid_action.setChecked(True)
        self.toggle_grid_action.setStatusTip("Toggle grid display")
        self.toggle_grid_action.triggered.connect(self.toggle_grid)
        
        self.toggle_theme_action = QAction("&Dark Theme", self)
        self.toggle_theme_action.setCheckable(True)
        self.toggle_theme_action.setStatusTip("Toggle dark/light theme")
        self.toggle_theme_action.triggered.connect(self.toggle_theme)
        
        # Tools actions
        self.calibration_action = QAction("Energy &Calibration...", self)
        self.calibration_action.setStatusTip("Calibrate energy axis")
        self.calibration_action.triggered.connect(self.calibrate_energy)
        
        self.fwhm_calibration_action = QAction("&FWHM Calibration...", self)
        self.fwhm_calibration_action.setStatusTip(
            "Calibrate detector resolution (FWHM vs energy)"
        )
        self.fwhm_calibration_action.triggered.connect(self.show_fwhm_calibration)
        
        self.standards_calibration_action = QAction("&Standards Calibration...", self)
        self.standards_calibration_action.setStatusTip(
            "Intensity calibration using reference standards"
        )
        self.standards_calibration_action.triggered.connect(self.show_standards_calibration)
        
        self.element_db_action = QAction("&Element Database...", self)
        self.element_db_action.setStatusTip("View element database")
        self.element_db_action.triggered.connect(self.show_element_database)
        
        # Help actions
        self.check_updates_action = QAction("Check for &Updates...", self)
        self.check_updates_action.setStatusTip(
            "Pull the latest changes from the XRFLab repository"
        )
        self.check_updates_action.triggered.connect(self.check_for_updates)
        
        self.about_action = QAction("&About", self)
        self.about_action.setStatusTip("About this application")
        self.about_action.triggered.connect(self.show_about)
    
    def _create_menus(self):
        """Create menu bar and menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.open_project_action)
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.export_results_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        analysis_menu.addAction(self.fit_spectrum_action)
        analysis_menu.addAction(self.quantify_action)
        analysis_menu.addSeparator()
        analysis_menu.addAction(self.background_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.toggle_log_action)
        view_menu.addAction(self.toggle_grid_action)
        view_menu.addSeparator()
        view_menu.addAction(self.toggle_theme_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction(self.calibration_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.fwhm_calibration_action)
        tools_menu.addAction(self.standards_calibration_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.element_db_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.check_updates_action)
        help_menu.addSeparator()
        help_menu.addAction(self.about_action)
    
    def _create_toolbar(self):
        """Create toolbar with project quick-access buttons"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("MainToolbar")  # Set object name to avoid warning
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        toolbar.addAction(self.open_project_action)
        toolbar.addAction(self.save_project_action)
    
    def _create_central_widget(self):
        """Create the main layout with primary Analysis tabs and nested Calibration"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Primary workflow tabs only — calibration lives one level down
        self.tab_widget = QTabWidget()
        
        # Analysis tab (main interface)
        analysis_tab = self._create_analysis_tab()
        self.tab_widget.addTab(analysis_tab, "Analysis")
        
        # Batch Analysis tab (bulk spectral fitting)
        self.batch_analysis_panel = BatchAnalysisPanel()
        self.tab_widget.addTab(self.batch_analysis_panel, "Batch Analysis")
        # Connect to Analysis tab's element panel for settings
        self.batch_analysis_panel.set_element_panel(self.element_panel)
        
        # Calibration tab: infrequent setup tools, grouped and out of the primary bar
        self.calibration_tab = self._create_calibration_tab()
        self.tab_widget.addTab(self.calibration_tab, "Calibration")
        
        # Auto-load emits calibration_complete during panel __init__, before this
        # connect runs — re-apply any calibration already loaded from disk.
        if self.fwhm_calibration_panel.fwhm_calibration is not None:
            self.on_fwhm_calibration_applied(self.fwhm_calibration_panel.fwhm_calibration)
        
        layout.addWidget(self.tab_widget)
    
    def _create_calibration_tab(self):
        """Nest FWHM and Standards under one Calibration tab (setup workflow order)."""
        calibration_widget = QWidget()
        layout = QVBoxLayout(calibration_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.calibration_tabs = QTabWidget()
        
        # Step 1: detector resolution
        self.fwhm_calibration_panel = FWHMCalibrationPanel()
        self.fwhm_calibration_panel.calibration_complete.connect(
            self.on_fwhm_calibration_applied
        )
        self.calibration_tabs.addTab(self.fwhm_calibration_panel, "FWHM")
        
        # Step 2: intensity/response using known concentrations
        self.standards_panel = StandardsPanel()
        self.standards_panel.calibration_complete.connect(self.on_calibration_applied)
        self.calibration_tabs.addTab(self.standards_panel, "Standards")
        
        layout.addWidget(self.calibration_tabs)
        return calibration_widget
    
    def show_fwhm_calibration(self):
        """Open Calibration → FWHM from the Tools menu"""
        self.tab_widget.setCurrentWidget(self.calibration_tab)
        self.calibration_tabs.setCurrentWidget(self.fwhm_calibration_panel)
    
    def show_standards_calibration(self):
        """Open Calibration → Standards from the Tools menu"""
        self.tab_widget.setCurrentWidget(self.calibration_tab)
        self.calibration_tabs.setCurrentWidget(self.standards_panel)
    
    def _create_analysis_tab(self):
        """Create the analysis tab with sub-tabs on left panel for laptop screens"""
        analysis_widget = QWidget()
        layout = QHBoxLayout(analysis_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create main horizontal splitter (left panel | right side)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Tabbed interface for compact layout
        self.analysis_left_tabs = QTabWidget()
        self.analysis_left_tabs.setMaximumWidth(700)  # Doubled width for better usability
        
        # Tab 1: Sample Info & Experimental Parameters
        self.element_panel = ElementPanel()
        sample_exp_tab = self._create_sample_exp_tab()
        self.analysis_left_tabs.addTab(sample_exp_tab, "Sample/Exp")
        
        # Tab 2: Element Selection
        element_tab = self._create_element_selection_tab()
        self.analysis_left_tabs.addTab(element_tab, "Elements")
        
        # Tab 3: Fitting Controls
        fitting_tab = self._create_fitting_controls_tab()
        self.analysis_left_tabs.addTab(fitting_tab, "Fitting")
        
        # Tab 4: Results & Quantification
        results_tab = self._create_results_tab()
        self.analysis_left_tabs.addTab(results_tab, "Results")
        
        main_splitter.addWidget(self.analysis_left_tabs)
        
        # Right side - Spectrum display (keep as is)
        self.spectrum_widget = SpectrumWidget()
        self.spectrum_widget.log_scale_changed.connect(self._on_plot_log_scale_changed)
        main_splitter.addWidget(self.spectrum_widget)
        
        # Set initial sizes for horizontal splitter (50% left, 50% right)
        main_splitter.setSizes([600, 600])
        
        layout.addWidget(main_splitter)
        
        # Connect signals
        self.element_panel.elements_changed.connect(self.on_elements_changed)
        self.element_panel.fit_requested.connect(self.fit_spectrum)
        self.element_panel.peak_find_requested.connect(self.preview_peak_find)
        self.element_panel.peak_list_changed.connect(self.on_peak_list_changed)
        self.element_panel.element_clicked.connect(self.on_element_clicked)
        self.results_panel.element_selected.connect(self.on_result_element_selected)
        self.results_panel.quantify_requested.connect(self.quantify)
        self.results_panel.export_button.clicked.connect(self.export_results)
        
        return analysis_widget
    
    def _create_sample_exp_tab(self):
        """Create Sample Info & Experimental Parameters tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        open_spectrum_btn = QPushButton("Open Spectrum...")
        open_spectrum_btn.setToolTip("Open an XRF spectrum file")
        open_spectrum_btn.clicked.connect(self.open_spectrum)
        layout.addWidget(open_spectrum_btn)
        
        # Sample information group
        sample_group = self.element_panel._create_sample_info_group()
        layout.addWidget(sample_group)
        
        # Experimental parameters group
        exp_params_group = self.element_panel._create_exp_params_group()
        layout.addWidget(exp_params_group)
        
        layout.addStretch()
        return widget
    
    def _create_element_selection_tab(self):
        """Create Element Selection tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Element selection group
        element_group = self.element_panel._create_element_selection_group()
        layout.addWidget(element_group, stretch=1)
        
        return widget
    
    def _create_fitting_controls_tab(self):
        """Create Fitting Controls tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Fitting controls group
        fitting_group = self.element_panel._create_fitting_controls_group()
        layout.addWidget(fitting_group)
        
        layout.addStretch()
        return widget
    
    def _create_results_tab(self):
        """Create Results & Quantification tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create results panel
        self.results_panel = ResultsPanel()
        layout.addWidget(self.results_panel)
        
        return widget
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _load_stylesheet(self):
        """Load and apply Qt stylesheet"""
        try:
            with open("resources/styles.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            # Use default styling if stylesheet not found
            pass
    
    def _restore_settings(self):
        """Restore window settings from previous session"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def _save_settings(self):
        """Save window settings"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
    
    # Action handlers
    def open_spectrum(self):
        """Open an XRF spectrum file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open XRF Spectrum",
            "",
            "All Supported (*.txt *.csv *.mca *.h5 *.hdf5);;Text Files (*.txt);;CSV Files (*.csv);;MCA Files (*.mca);;HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        
        if file_path:
            try:
                spectrum = self.io_handler.load_spectrum(file_path)
                self.current_spectrum = spectrum
                self.spectrum_widget.set_spectrum(spectrum)
                # Note: Standards panel will get spectrum when needed
                
                # Auto-populate experimental parameters from spectrum metadata
                if hasattr(spectrum, 'metadata') and spectrum.metadata:
                    self.element_panel.update_from_spectrum_metadata(spectrum.metadata)
                
                self.status_bar.showMessage(f"Loaded: {file_path}", 5000)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Spectrum",
                    f"Failed to load spectrum:\n{str(e)}"
                )
    
    def open_project(self):
        """Open an XRF project file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "XRF Project (*.xrfp);;All Files (*)"
        )
        
        if file_path:
            try:
                # TODO: Implement project loading
                self.status_bar.showMessage(f"Opened project: {file_path}", 5000)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Opening Project",
                    f"Failed to open project:\n{str(e)}"
                )
    
    def save_project(self):
        """Save current project"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "XRF Project (*.xrfp);;All Files (*)"
        )
        
        if file_path:
            try:
                # TODO: Implement project saving
                self.status_bar.showMessage(f"Saved: {file_path}", 5000)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Saving Project",
                    f"Failed to save project:\n{str(e)}"
                )
    
    def export_results(self):
        """Export analysis results"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            try:
                results = self.results_panel.get_results()
                self.io_handler.export_results(results, file_path)
                self.status_bar.showMessage(f"Exported: {file_path}", 5000)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Exporting Results",
                    f"Failed to export results:\n{str(e)}"
                )
    
    def fit_spectrum(self):
        """Fit the current spectrum"""
        if self.current_spectrum is None:
            QMessageBox.warning(
                self,
                "No Spectrum",
                "Please load a spectrum first."
            )
            return
        
        self.status_bar.showMessage("Fitting spectrum...", 0)
        
        try:
            # Get selected elements
            elements = self.element_panel.get_selected_elements()
            if not elements:
                reply = QMessageBox.warning(
                    self,
                    "No Elements Selected",
                    "No elements are selected on the Elements tab.\n\n"
                    "Without labeled sample peaks, Fit will still run, but "
                    "quantification will be empty.\n\n"
                    "Select elements (e.g. Fe, Ca, Si), then Fit again.\n\n"
                    "Continue fitting anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.status_bar.showMessage("Fit cancelled — select elements first", 5000)
                    return
            
            # Get fitting parameters
            fit_params = self.element_panel.get_fitting_params()
            background_method = fit_params['background_method'].lower()
            peak_shape = fit_params['peak_shape'].lower()
            
            # Get experimental parameters
            exp_params = self.element_panel.get_experimental_params()

            # Optional: use manually edited peak list
            peak_positions = None
            if fit_params.get('use_peak_list'):
                peak_positions = self.element_panel.get_peak_list()
            
            # Perform fitting (pass all parameters including tube lines and experimental params)
            self.fit_result = self.fitter.fit_spectrum(
                energy=self.current_spectrum.energy,
                counts=self.current_spectrum.counts,
                elements=elements,
                background_method=background_method,
                peak_shape=peak_shape,
                auto_find_peaks=fit_params.get('auto_find_peaks', True),
                tube_element=fit_params.get('tube_element', 'Rh'),
                excitation_kv=fit_params.get('excitation_kv', 50.0),
                include_tube_lines=fit_params.get('include_tube_lines', True),
                include_compton=fit_params.get('include_compton', True),
                scatter_angle_deg=fit_params.get('scatter_angle_deg', 90.0),
                compton_fwhm_kev=fit_params.get('compton_fwhm_kev', 0.250),
                experimental_params=exp_params,
                prominence_percent=fit_params.get('prominence_percent'),
                min_height=fit_params.get('min_height'),
                min_separation_ev=fit_params.get('min_separation_ev'),
                peak_positions=peak_positions,
            )
            
            # Optional post-fit smart ID (FWHM excess + Kβ / multi-line checks)
            smart_report = None
            overlap_seeds = []
            if fit_params.get('smart_id_after_fit'):
                smart_cfg = SmartIDConfig(
                    fwhm_excess_kev=float(fit_params.get('fwhm_excess_ev', 30.0)) / 1000.0,
                    apply_suggestions=bool(fit_params.get('smart_id_apply')),
                )
                cand_map = {
                    e['symbol']: e
                    for e in (elements or self.element_panel.get_selected_elements() or [])
                    if e.get('symbol')
                }
                buttons = getattr(
                    self.element_panel.periodic_table, 'element_buttons', {}
                )
                for p in self.fit_result.peaks:
                    if p.element and not p.is_tube_line and p.element not in cand_map:
                        btn = buttons.get(p.element)
                        if btn is not None:
                            cand_map[p.element] = {
                                'symbol': p.element,
                                'z': btn.atomic_number,
                                'name': getattr(btn, 'name', p.element),
                            }
                smart_report = analyze_fitted_peaks(
                    self.fit_result.peaks,
                    list(cand_map.values()),
                    smart_cfg,
                )
                if smart_cfg.apply_suggestions:
                    self.fit_result.peaks, overlap_seeds, _n = (
                        apply_smart_id_suggestions(
                            self.fit_result.peaks, smart_report
                        )
                    )
                print('\n'.join(smart_report.summary_lines))
            
            # Update spectrum display
            self.spectrum_widget.set_fitted_spectrum(self.fit_result.fitted_spectrum)
            self.spectrum_widget.set_background(self.fit_result.background)
            self.spectrum_widget.set_peak_markers(
                self.fit_result.peaks,
                show=fit_params.get('show_peak_markers', True),
            )
            self._displayed_element_lines = None

            # Refresh peak list (include any overlap seeds for a follow-up fit)
            keep_use_list = fit_params.get('use_peak_list', False) or bool(overlap_seeds)
            peak_entries = []
            for p in self.fit_result.peaks:
                peak_entries.append({
                    'energy': float(p.energy),
                    'element': p.element,
                    'line': p.line,
                    'is_tube_line': bool(p.is_tube_line),
                })
            for seed in overlap_seeds:
                e_seed = float(seed['energy'])
                if all(abs(e_seed - e['energy']) > 0.04 for e in peak_entries):
                    peak_entries.append(seed)
            self.element_panel.set_peak_list(
                peak_entries, enable_use_list=keep_use_list
            )
            
            # Update results panel
            self.results_panel.set_fit_statistics(self.fit_result.statistics)
            self.results_panel.set_peaks(self.fit_result.peaks)
            if smart_report is not None:
                extra = "\n\n--- Smart ID ---\n" + "\n".join(smart_report.summary_lines)
                current = self.results_panel.peaks_text.toPlainText()
                self.results_panel.peaks_text.setPlainText(current + extra)
            
            # Perform quantification (area-normalized; needs labeled sample peaks)
            exp_params = self.element_panel.get_experimental_params()
            concentrations = self.fitter.quantify_elements(
                self.fit_result.peaks, exp_params
            )
            self.results_panel.set_quantification(concentrations)

            # Sync identified sample elements onto Elements tab for review.
            # User can uncheck false IDs / check missing ones, then Fit again.
            identified = []
            seen = set()
            for peak in self.fit_result.peaks:
                if peak.is_tube_line or not peak.element:
                    continue
                if peak.element in seen:
                    continue
                seen.add(peak.element)
                identified.append(peak.element)
            if identified:
                self.element_panel.set_selected_elements(identified)
            
            n_quant = len(concentrations)
            fit_msg = (
                f"Fitting complete: {len(self.fit_result.peaks)} peaks fitted, "
                f"χ²ᵣ = {self.fit_result.statistics['reduced_chi_squared']:.2f}"
            )
            if smart_report is not None:
                fit_msg += (
                    f"; smart ID: {smart_report.n_overlap_suspects} overlap suspect(s), "
                    f"{smart_report.n_relabel_suggestions} suggestion(s)"
                )
                if fit_params.get('smart_id_apply') and smart_report.n_applied:
                    fit_msg += f", applied {smart_report.n_applied}"
            if identified:
                fit_msg += (
                    f"; {len(identified)} element(s) on Elements tab — review, "
                    f"then Fit again / Run Quant"
                )
                if hasattr(self, 'analysis_left_tabs'):
                    self.analysis_left_tabs.setCurrentIndex(1)
            elif n_quant:
                fit_msg += f"; quantified {n_quant} element{'s' if n_quant != 1 else ''}"
                if hasattr(self, 'analysis_left_tabs'):
                    self.analysis_left_tabs.setCurrentIndex(3)
            else:
                fit_msg += "; no labeled sample peaks to quantify"
                if hasattr(self, 'analysis_left_tabs'):
                    self.analysis_left_tabs.setCurrentIndex(3)
            self.status_bar.showMessage(fit_msg, 12000)

            if smart_report is not None and (
                smart_report.n_overlap_suspects or smart_report.n_relabel_suggestions
            ):
                QMessageBox.information(
                    self,
                    "Smart ID Results",
                    "\n".join(smart_report.summary_lines[:40]),
                )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Fitting Error",
                f"An error occurred during fitting:\n{str(e)}"
            )
            self.status_bar.showMessage("Fitting failed", 5000)
    
    def preview_peak_find(self):
        """Run peak detection only and mark found peaks on the spectrum"""
        if self.current_spectrum is None:
            QMessageBox.warning(
                self,
                "No Spectrum",
                "Please load a spectrum first."
            )
            return
        
        fit_params = self.element_panel.get_fitting_params()
        background_method = fit_params['background_method'].lower()
        
        try:
            background = self.fitter.background_modeler.estimate_background(
                self.current_spectrum.energy,
                self.current_spectrum.counts,
                method=background_method,
            )
            counts_bg = self.fitter.background_modeler.subtract_background(
                self.current_spectrum.counts, background
            )

            preview_peaks = self.fitter.build_peak_positions(
                self.current_spectrum.energy,
                counts_bg_subtracted=counts_bg,
                elements=self.element_panel.get_selected_elements(),
                auto_find_peaks=fit_params.get('auto_find_peaks', True),
                tube_element=fit_params.get('tube_element', 'Rh'),
                excitation_kv=fit_params.get('excitation_kv', 50.0),
                include_tube_lines=fit_params.get('include_tube_lines', True),
                prominence_percent=fit_params.get('prominence_percent'),
                min_height=fit_params.get('min_height'),
                min_separation_ev=fit_params.get('min_separation_ev'),
            )
            
            # Show background so detection context is clear
            self.spectrum_widget.set_background(background)
            
            self.spectrum_widget.set_peak_markers(
                preview_peaks,
                show=fit_params.get('show_peak_markers', True),
            )
            self._displayed_element_lines = None

            # Populate editable peak list; enable use-list so Fit respects deletions
            self.element_panel.set_peak_list(preview_peaks, enable_use_list=True)
            
            # Update results peak list with detection preview
            n_unknown = sum(1 for p in preview_peaks if not p.get('element'))
            n_labeled = len(preview_peaks) - n_unknown
            lines = []
            for p in preview_peaks:
                if p.get('element') and p.get('line'):
                    tag = " [tube]" if p.get('is_tube_line') else ""
                    lines.append(
                        f"{p['energy']:.3f} keV  {p['element']} {p['line']}{tag}"
                    )
                else:
                    lines.append(f"{p['energy']:.3f} keV  (unknown)")
            if lines:
                self.results_panel.peaks_text.setPlainText(
                    f"Peak find preview ({len(preview_peaks)} total: "
                    f"{n_labeled} labeled, {n_unknown} unknown):\n" + "\n".join(lines)
                )
            else:
                self.results_panel.peaks_text.setPlainText(
                    "Peak find preview: no peaks detected.\n"
                    "Try lowering Prominence or Min height, or select elements."
                )
            
            self.status_bar.showMessage(
                f"Peak find preview: {len(preview_peaks)} peaks listed "
                f"(delete unwanted peaks, then Fit Spectrum)",
                6000
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Peak Find Error",
                f"An error occurred during peak detection:\n{str(e)}"
            )
            self.status_bar.showMessage("Peak find failed", 5000)

    def on_peak_list_changed(self):
        """Sync spectrum markers when peaks are deleted/cleared in the Fitting tab."""
        peaks = self.element_panel.get_peak_list()
        fit_params = self.element_panel.get_fitting_params()
        if peaks:
            self.spectrum_widget.set_peak_markers(
                peaks,
                show=fit_params.get('show_peak_markers', True),
            )
        else:
            self.spectrum_widget.clear_peak_markers()
        self._displayed_element_lines = None
    
    def quantify(self):
        """Perform quantitative analysis from the current fit"""
        if self.fit_result is None or not getattr(self.fit_result, 'peaks', None):
            QMessageBox.warning(
                self,
                "No Fit Results",
                "Please fit a spectrum first before running quantification."
            )
            return
        
        # Show Results tab so the table update is visible
        if hasattr(self, 'analysis_left_tabs'):
            self.analysis_left_tabs.setCurrentIndex(3)
        
        self.status_bar.showMessage("Performing quantification...", 0)
        try:
            exp_params = self.element_panel.get_experimental_params()
            concentrations = self.fitter.quantify_elements(
                self.fit_result.peaks, exp_params
            )
            self.results_panel.set_quantification(concentrations)
            n = len(concentrations)
            
            if n == 0:
                peaks = self.fit_result.peaks
                n_unknown = sum(1 for p in peaks if not p.element)
                n_tube = sum(1 for p in peaks if p.is_tube_line)
                n_labeled = sum(
                    1 for p in peaks if p.element and not p.is_tube_line
                )
                QMessageBox.warning(
                    self,
                    "Nothing to Quantify",
                    f"Fitted {len(peaks)} peaks, but none are labeled sample elements.\n\n"
                    f"  Unknown (no element): {n_unknown}\n"
                    f"  Tube lines (excluded): {n_tube}\n"
                    f"  Labeled sample peaks: {n_labeled}\n\n"
                    "Select elements on the Elements tab, then Fit Spectrum again "
                    "(peak-find unknowns are skipped until they are labeled)."
                )
                self.status_bar.showMessage(
                    "Quantification: no labeled sample peaks", 5000
                )
                return
            
            self.status_bar.showMessage(
                f"Quantification complete: {n} element{'s' if n != 1 else ''}",
                5000
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Quantification Error",
                f"An error occurred during quantification:\n{str(e)}"
            )
            self.status_bar.showMessage("Quantification failed", 5000)
    
    def configure_background(self):
        """Configure background removal settings"""
        # TODO: Implement background configuration dialog
        pass
    
    def toggle_log_scale(self, checked):
        """Toggle logarithmic Y-axis from the View menu"""
        self.spectrum_widget.set_log_scale(checked)
    
    def _on_plot_log_scale_changed(self, checked):
        """Keep View menu Log Y-axis action in sync with plot controls"""
        self.toggle_log_action.blockSignals(True)
        self.toggle_log_action.setChecked(checked)
        self.toggle_log_action.blockSignals(False)
    
    def toggle_grid(self, checked):
        """Toggle grid display"""
        self.spectrum_widget.set_grid(checked)
    
    def toggle_theme(self, checked):
        """Toggle between dark and light theme"""
        # TODO: Implement theme switching
        pass
    
    def calibrate_energy(self):
        """Open energy calibration dialog"""
        # TODO: Implement calibration dialog
        pass
    
    def show_element_database(self):
        """Show element database viewer"""
        # TODO: Implement element database viewer
        pass
    
    def check_for_updates(self):
        """Pull latest changes from the git remote and report results"""
        from PySide6.QtWidgets import QApplication
        
        self.status_bar.showMessage("Checking for updates...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            result = check_for_updates()
        finally:
            QApplication.restoreOverrideCursor()
        
        if not result.success:
            detail = result.error or ""
            text = result.message
            if detail:
                text = f"{text}\n\nDetails:\n{detail}"
            QMessageBox.warning(self, "Check for Updates", text)
            self.status_bar.showMessage("Update check failed", 5000)
            return
        
        if result.updated:
            details = []
            if result.commits:
                details.append("Commits:\n" + "\n".join(f"  • {c}" for c in result.commits[:15]))
                if len(result.commits) > 15:
                    details.append(f"  … and {len(result.commits) - 15} more")
            if result.changed_files:
                details.append(
                    "Updated files:\n"
                    + "\n".join(f"  • {f}" for f in result.changed_files[:20])
                )
                if len(result.changed_files) > 20:
                    details.append(f"  … and {len(result.changed_files) - 20} more")
            
            text = result.message
            if details:
                text = f"{text}\n\n" + "\n\n".join(details)
            
            QMessageBox.information(self, "Updates Installed", text)
            self.status_bar.showMessage(
                f"Updated {len(result.changed_files)} file(s) — restart to apply",
                8000,
            )
        else:
            QMessageBox.information(self, "Check for Updates", result.message)
            self.status_bar.showMessage("Already up to date", 5000)
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About XRFLab",
            "<h3>XRFLab</h3>"
            "<p>Version 1.0.0</p>"
            "<p>A professional application for X-ray fluorescence spectroscopy analysis "
            "using fundamental parameters method.</p>"
            "<p>Built with PySide6, PyQtGraph, and xraylib.</p>"
        )
    
    def on_elements_changed(self, elements):
        """Handle element selection changes"""
        # TODO: Update spectrum display with selected elements
        pass
    
    def on_element_clicked(self, symbol, z):
        """Handle element click — show emission lines, or clear if already shown"""
        if self._displayed_element_lines == symbol:
            self.spectrum_widget.clear_peak_markers()
            self._displayed_element_lines = None
            self.status_bar.showMessage(f"Cleared emission lines for {symbol}", 3000)
            return
        
        self.spectrum_widget.clear_peak_markers()
        self.spectrum_widget.show_element_lines(symbol, z)
        self._displayed_element_lines = symbol
        self.status_bar.showMessage(f"Showing emission lines for {symbol} (Z={z})", 3000)
    
    def on_result_element_selected(self, symbol):
        """Handle element click in Results table — overlay lines on the spectrum"""
        from core.advanced_peak_fitting import get_element_z
        
        z = get_element_z(symbol)
        if not z:
            self.status_bar.showMessage(f"Unknown element: {symbol}", 3000)
            return
        
        self.on_element_clicked(symbol, z)
    
    def on_fwhm_calibration_applied(self, fwhm_calibration):
        """Handle FWHM calibration being applied"""
        from core.fwhm_calibration import apply_fwhm_calibration_to_peak_fitter
        
        # Update the Standards panel with the FWHM calibration
        self.standards_panel.update_fwhm_status(fwhm_calibration)
        
        # Apply to Analysis / Batch peak fitting (class-level PeakFitter)
        # This locks widths to FWHM(E) during LS (USE_CALIBRATED_SHAPES=True)
        apply_fwhm_calibration_to_peak_fitter(fwhm_calibration, self.fitter.peak_fitter)

        # Fitting tab status
        self.element_panel.update_fwhm_status(fwhm_calibration)
        
        # Show status message
        if fwhm_calibration.model_type == 'detector':
            fwhm_0_ev = fwhm_calibration.parameters['fwhm_0'] * 1000
            epsilon_ev = fwhm_calibration.parameters['epsilon'] * 1000
            self.status_bar.showMessage(
                f"FWHM locked for Analysis: FWHM₀={fwhm_0_ev:.1f} eV, "
                f"ε={epsilon_ev:.2f} eV/keV (R²={fwhm_calibration.r_squared:.4f})",
                5000
            )
        else:
            self.status_bar.showMessage(
                f"FWHM locked for Analysis: {fwhm_calibration.model_type} model "
                f"(R²={fwhm_calibration.r_squared:.4f})",
                5000
            )
    
    def on_calibration_applied(self, calibration_result):
        """Handle calibration being applied"""
        self.status_bar.showMessage(
            f"Calibration applied: FWHM₀={calibration_result.fwhm_0*1000:.1f} eV, "
            f"ε={calibration_result.epsilon*1000:.2f} eV",
            5000
        )
        
        # Switch back to analysis tab
        self.tab_widget.setCurrentIndex(0)
    
    def closeEvent(self, event):
        """Handle window close event"""
        self._save_settings()
        event.accept()
