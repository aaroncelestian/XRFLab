"""
Main window for XRF Fundamental Parameters Analysis Application
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QMenu, QToolBar, QStatusBar, QMessageBox, QFileDialog,
    QTabWidget, QPushButton, QLabel
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence, QIcon

from ui.spectrum_widget import SpectrumWidget
from ui.element_panel import ElementPanel
from ui.results_panel import ResultsPanel
from ui.batch_analysis_panel import BatchAnalysisPanel
from ui.composition_panel import CompositionPanel
from ui.standards_panel import StandardsPanel
from ui.fwhm_calibration_panel import FWHMCalibrationPanel
from ui.tube_profile_panel import TubeProfilePanel
from ui.mapping_panel import MappingPanel
from utils.io_handler import IOHandler
from utils.updater import check_for_updates
from utils.desktop_shortcut import install_desktop_shortcut
from utils.paths import icon_path, resource_path
from core.fitting import SpectrumFitter
from core.fp_quantification import quantify_from_peaks
from core.session import AnalysisSession
from core.smart_peak_id import (
    SmartIDConfig,
    analyze_fitted_peaks,
    apply_smart_id_suggestions,
    auto_id_peak_positions,
    candidates_at_energy,
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
        self.session = AnalysisSession()
        self.fitter = SpectrumFitter()
        self.session.apply_instrument_to_fitter(self.fitter)
        self.settings = QSettings()
        self._displayed_element_lines = None  # symbol currently shown on plot, or None
        
        # Setup UI (status bar before central widget — FWHM auto-load may message it)
        self._create_actions()
        self._create_menus()
        self._create_toolbar()
        self._create_status_bar()
        self._create_central_widget()
        self._load_stylesheet()
        self._apply_window_icon()
        
        # Restore window state
        self._restore_settings()

    @property
    def current_spectrum(self):
        return self.session.spectrum

    @current_spectrum.setter
    def current_spectrum(self, value):
        self.session.spectrum = value

    @property
    def fit_result(self):
        return self.session.fit_result

    @fit_result.setter
    def fit_result(self, value):
        self.session.fit_result = value
    
    def _create_actions(self):
        """Create all menu and toolbar actions"""
        # File actions
        self.open_action = QAction("&Open Spectrum...", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.setStatusTip("Open an XRF spectrum file")
        self.open_action.triggered.connect(self.open_spectrum)

        self.open_ipj_action = QAction("Open &IPJ Mapping Project...", self)
        self.open_ipj_action.setStatusTip(
            "Open an Oxford INCA / Horiba XGT .ipj mapping project"
        )
        self.open_ipj_action.triggered.connect(self.open_ipj_project)
        
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
        
        self.quantify_action = QAction("&Semi-Quant (Relative Intensities)", self)
        self.quantify_action.setShortcut("Ctrl+Q")
        self.quantify_action.setStatusTip(
            "Area-normalized relative intensities (not FP wt%)."
        )
        self.quantify_action.triggered.connect(self.quantify)

        self.fp_quantify_action = QAction("&FP Composition (wt%)", self)
        self.fp_quantify_action.setShortcut("Ctrl+Shift+Q")
        self.fp_quantify_action.setStatusTip(
            "Standardless fundamental-parameters wt% using the matrix model "
            "and optional H2O / OH / CO2 assumptions."
        )
        self.fp_quantify_action.triggered.connect(self.quantify_fp)
        
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
        
        # Tools actions — jump to Calibration sub-tabs
        self.fwhm_calibration_action = QAction("&FWHM Calibration...", self)
        self.fwhm_calibration_action.setStatusTip(
            "Calibrate detector resolution (FWHM vs energy)"
        )
        self.fwhm_calibration_action.triggered.connect(self.show_fwhm_calibration)

        self.tube_profile_action = QAction("&Tube Profiles...", self)
        self.tube_profile_action.setStatusTip(
            "Measure per-voltage Rh tube scatter line ratios (15/30/50 kV)"
        )
        self.tube_profile_action.triggered.connect(self.show_tube_profiles)
        
        self.standards_calibration_action = QAction("&Standards Calibration...", self)
        self.standards_calibration_action.setStatusTip(
            "Intensity calibration using reference standards"
        )
        self.standards_calibration_action.triggered.connect(self.show_standards_calibration)
        
        # Help actions
        self.check_updates_action = QAction("Check for &Updates...", self)
        self.check_updates_action.setStatusTip(
            "Pull the latest changes from the XRFLab repository"
        )
        self.check_updates_action.triggered.connect(self.check_for_updates)

        self.install_shortcut_action = QAction("Install &Desktop Shortcut...", self)
        self.install_shortcut_action.setStatusTip(
            "Create a Desktop shortcut (Mac app / Windows .lnk) to launch XRFLab"
        )
        self.install_shortcut_action.triggered.connect(self.install_desktop_shortcut)
        
        self.about_action = QAction("&About", self)
        self.about_action.setStatusTip("About this application")
        self.about_action.triggered.connect(self.show_about)
    
    def _create_menus(self):
        """Create menu bar and menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_ipj_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_results_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        analysis_menu.addAction(self.fit_spectrum_action)
        analysis_menu.addAction(self.quantify_action)
        analysis_menu.addAction(self.fp_quantify_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.toggle_log_action)
        view_menu.addAction(self.toggle_grid_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction(self.fwhm_calibration_action)
        tools_menu.addAction(self.tube_profile_action)
        tools_menu.addAction(self.standards_calibration_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.check_updates_action)
        help_menu.addAction(self.install_shortcut_action)
        help_menu.addSeparator()
        help_menu.addAction(self.about_action)
    
    def _create_toolbar(self):
        """Create toolbar with global actions only (tab-specific tools stay in-panel)."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("MainToolbar")  # Set object name to avoid warning
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.open_ipj_action)
    
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
        self.batch_analysis_panel.set_instrument_state(self.session.instrument)

        # Composition: group batch replicates and plot sample means
        self.composition_panel = CompositionPanel()
        self.composition_panel.from_batch_requested.connect(
            self.send_batch_to_composition
        )
        self.composition_panel.sample_activated.connect(
            self.on_composition_sample_activated
        )
        self.composition_panel.open_in_batch_requested.connect(
            self.on_composition_open_in_batch
        )
        self.batch_analysis_panel.results_ready.connect(
            self.send_batch_to_composition
        )
        self.batch_analysis_panel.send_to_composition_requested.connect(
            self.open_composition_from_batch
        )
        self.tab_widget.addTab(self.composition_panel, "Composition")

        # Mapping tab (IPJ element maps, line scans, correlations)
        self.mapping_panel = MappingPanel()
        self.mapping_panel.set_fitter(self.fitter)
        self.mapping_panel.set_element_panel(self.element_panel)
        self.mapping_panel.spectrum_send_requested.connect(self.on_mapping_spectrum_sent)
        self.mapping_panel.status_message.connect(
            lambda msg: self.status_bar.showMessage(msg, 5000)
        )
        self.tab_widget.addTab(self.mapping_panel, "Mapping")
        
        # Calibration tab: infrequent setup tools, grouped and out of the primary bar
        self.calibration_tab = self._create_calibration_tab()
        self.tab_widget.addTab(self.calibration_tab, "Calibration")
        
        # Auto-load emits calibration_complete during panel __init__, before this
        # connect runs — re-apply any calibration already loaded from disk.
        if self.fwhm_calibration_panel.fwhm_calibration is not None:
            self.on_fwhm_calibration_applied(self.fwhm_calibration_panel.fwhm_calibration)

        # Tube profiles (may already be loaded in panel __init__)
        if self.tube_profile_panel.get_library() is not None:
            self.on_tube_profiles_changed(self.tube_profile_panel.get_library())
        
        layout.addWidget(self.tab_widget)
    
    def _create_calibration_tab(self):
        """Nest FWHM, Tube Profiles, and Standards under Calibration."""
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

        # Step 1b: per-kV tube scatter profiles (15/30/50)
        self.tube_profile_panel = TubeProfilePanel()
        self.tube_profile_panel.library_changed.connect(self.on_tube_profiles_changed)
        self.calibration_tabs.addTab(self.tube_profile_panel, "Tube Profiles")
        
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

    def show_tube_profiles(self):
        """Open Calibration → Tube Profiles"""
        self.tab_widget.setCurrentWidget(self.calibration_tab)
        self.calibration_tabs.setCurrentWidget(self.tube_profile_panel)
    
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
        
        # Tab order: Sample → Peak Find → Elements → Fitting → Results
        self.element_panel = ElementPanel()
        sample_exp_tab = self._create_sample_exp_tab()
        self.analysis_left_tabs.addTab(sample_exp_tab, "Sample/Exp")

        peak_find_tab = self._create_peak_find_tab()
        self.analysis_left_tabs.addTab(peak_find_tab, "Peak Find")
        
        element_tab = self._create_element_selection_tab()
        self.analysis_left_tabs.addTab(element_tab, "Elements")
        
        fitting_tab = self._create_fitting_controls_tab()
        self.analysis_left_tabs.addTab(fitting_tab, "Fitting")
        
        results_tab = self._create_results_tab()
        self.analysis_left_tabs.addTab(results_tab, "Results")
        
        main_splitter.addWidget(self.analysis_left_tabs)
        
        # Right side - Spectrum display (keep as is)
        self.spectrum_widget = SpectrumWidget()
        self.spectrum_widget.log_scale_changed.connect(self._on_plot_log_scale_changed)
        self.spectrum_widget.energy_selected.connect(self.on_spectrum_energy_picked)
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
        self.element_panel.identify_on_plot_toggled.connect(self.on_identify_on_plot_toggled)
        self.element_panel.identify_add_element.connect(self.on_identify_add_element)
        self.results_panel.element_selected.connect(self.on_result_element_selected)
        self.results_panel.quantify_requested.connect(self.quantify)
        self.results_panel.fp_quantify_requested.connect(self.quantify_fp)
        self.results_panel.matrix_assumptions_changed.connect(
            lambda: self.quantify_fp(live=True)
        )
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
    
    def _create_peak_find_tab(self):
        """Create Peak Find tab (detect + auto-ID before Elements)."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        peak_find_group = self.element_panel._create_peak_find_group()
        layout.addWidget(peak_find_group)
        layout.addStretch()
        return widget
    
    def _create_element_selection_tab(self):
        """Create Element Selection tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        hint = QLabel(
            "Review auto-ID selections from Peak Find, or use "
            "“Click Spectrum to Identify” to pick peaks on the plot. "
            "Then continue to Fitting."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555; padding: 4px;")
        layout.addWidget(hint)
        
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

    # Analysis left-tab indices for navigation after actions
    TAB_SAMPLE = 0
    TAB_PEAK_FIND = 1
    TAB_ELEMENTS = 2
    TAB_FITTING = 3
    TAB_RESULTS = 4

    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _load_stylesheet(self):
        """Load and apply Qt stylesheet"""
        try:
            with open(resource_path("styles.qss"), "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            # Use default styling if stylesheet not found
            pass

    def _apply_window_icon(self):
        """Set the application window / Dock / taskbar icon."""
        png = icon_path("xrflab.png")
        if png.is_file():
            self.setWindowIcon(QIcon(str(png)))
    
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
                self.session.set_spectrum(spectrum, path=file_path)
                self.spectrum_widget.set_spectrum(spectrum)
                
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

    def open_ipj_project(self):
        """Open an INCA/XGT .ipj mapping project in the Mapping tab."""
        self.tab_widget.setCurrentWidget(self.mapping_panel)
        self.mapping_panel.open_ipj()

    def send_batch_to_composition(self):
        """Load current batch fits into the Composition tab."""
        results = self.batch_analysis_panel.results
        if not results:
            QMessageBox.information(
                self,
                "Composition",
                "Process a batch first (Batch Analysis → Process All).",
            )
            return
        self.composition_panel.load_batch_results(results)
        n = len(self.composition_panel.summaries)
        self.status_bar.showMessage(
            f"Composition: {len(results)} spectra grouped into {n} samples",
            8000,
        )

    def open_composition_from_batch(self):
        """Send to Composition and switch to that tab."""
        self.send_batch_to_composition()
        if self.batch_analysis_panel.results:
            self.tab_widget.setCurrentWidget(self.composition_panel)

    def on_composition_sample_activated(self, _sample, names):
        """Keep Batch selection in sync without switching tabs."""
        self.batch_analysis_panel.select_spectra(names)

    def on_composition_open_in_batch(self, _sample, names):
        """Double-click a sample: jump to Batch and show one of its fits."""
        self.batch_analysis_panel.select_spectra(names)
        self.tab_widget.setCurrentWidget(self.batch_analysis_panel)

    def on_mapping_spectrum_sent(self, spectrum, peak_labels=None):
        """Receive a spectrum extracted from Mapping → load into Analysis."""
        path_label = None
        if getattr(spectrum, "metadata", None):
            path_label = spectrum.metadata.get("name")
        if self.mapping_panel.project is not None:
            proj = self.mapping_panel.project.path
            name = path_label or "spectrum"
            path_label = f"{proj}::{name}"

        self.session.set_spectrum(spectrum, path=path_label)
        self.spectrum_widget.set_spectrum(spectrum)

        if hasattr(spectrum, "metadata") and spectrum.metadata:
            self.element_panel.update_from_spectrum_metadata(spectrum.metadata)

        # Seed element selection from IPJ peak labels when present
        if peak_labels:
            symbols = []
            seen = set()
            for pl in peak_labels:
                el = pl.get("element")
                if el and el not in seen:
                    seen.add(el)
                    symbols.append(el)
            if symbols:
                try:
                    self.element_panel.set_selected_elements(symbols)
                except Exception:
                    pass

        self.tab_widget.setCurrentIndex(0)  # Analysis
        self.status_bar.showMessage(
            f"Loaded mapping spectrum into Analysis: {path_label or 'spectrum'}",
            5000,
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
            # Keep fitter aligned with session instrument calibrations
            self.session.apply_instrument_to_fitter(self.fitter)

            # Get selected elements
            elements = self.element_panel.get_selected_elements()
            self.session.set_elements(elements)
            if not elements:
                reply = QMessageBox.warning(
                    self,
                    "No Elements Selected",
                    "No elements are selected.\n\n"
                    "Recommended flow:\n"
                    "  1) Peak Find → Find Peaks + Auto-ID\n"
                    "  2) Review Elements\n"
                    "  3) Fitting → Fit Spectrum\n\n"
                    "Without labeled sample peaks, Semi-Quant will be empty.\n\n"
                    "Continue fitting anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.status_bar.showMessage(
                        "Fit cancelled — run Peak Find + Auto-ID first", 5000
                    )
                    if hasattr(self, 'analysis_left_tabs'):
                        self.analysis_left_tabs.setCurrentIndex(self.TAB_PEAK_FIND)
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
            fit_result = self.fitter.fit_spectrum(
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
            self.session.set_fit_result(fit_result)
            
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
            flags = getattr(self.fit_result, 'tube_overlap_flags', None) or []
            if flags:
                self.results_panel.set_tube_overlap_flags(flags)
            notes = (self.fit_result.statistics or {}).get('tube_constraint_notes') or []
            if notes:
                self.results_panel.set_tube_constraint_notes(notes)
            if smart_report is not None:
                extra = "\n\n--- Smart ID ---\n" + "\n".join(smart_report.summary_lines)
                current = self.results_panel.peaks_text.toPlainText()
                self.results_panel.peaks_text.setPlainText(current + extra)
            
            # Semi-quant relative intensities (area-normalized; needs labeled sample peaks)
            exp_params = self.element_panel.get_experimental_params()
            concentrations = self.fitter.quantify_elements(
                self.fit_result.peaks, exp_params
            )
            self.session.set_concentrations(concentrations, method="semi_quant_area")
            self.results_panel.set_fp_live(False)
            self.results_panel.set_formula_summary("")
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
                    f"then Fit again / Semi-Quant"
                )
                if hasattr(self, 'analysis_left_tabs'):
                    self.analysis_left_tabs.setCurrentIndex(self.TAB_RESULTS)
            elif n_quant:
                fit_msg += f"; semi-quant {n_quant} element{'s' if n_quant != 1 else ''}"
                if hasattr(self, 'analysis_left_tabs'):
                    self.analysis_left_tabs.setCurrentIndex(self.TAB_RESULTS)
            else:
                fit_msg += "; no labeled sample peaks for semi-quant"
                if hasattr(self, 'analysis_left_tabs'):
                    self.analysis_left_tabs.setCurrentIndex(self.TAB_RESULTS)
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
        """Run peak detection + optional auto-ID, then open Elements for review."""
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
            self.session.apply_instrument_to_fitter(self.fitter)

            background = self.fitter.background_modeler.estimate_background(
                self.current_spectrum.energy,
                self.current_spectrum.counts,
                method=background_method,
            )
            counts_bg = self.fitter.background_modeler.subtract_background(
                self.current_spectrum.counts, background
            )

            # Peak find first — do not require Elements yet (auto-find unknowns)
            preview_peaks = self.fitter.build_peak_positions(
                self.current_spectrum.energy,
                counts_bg_subtracted=counts_bg,
                elements=None,
                auto_find_peaks=fit_params.get('auto_find_peaks', True),
                tube_element=fit_params.get('tube_element', 'Rh'),
                excitation_kv=fit_params.get('excitation_kv', 50.0),
                include_tube_lines=fit_params.get('include_tube_lines', True),
                include_compton=fit_params.get('include_compton', True),
                scatter_angle_deg=fit_params.get('scatter_angle_deg', 90.0),
                compton_fwhm_kev=fit_params.get('compton_fwhm_kev', 0.250),
                prominence_percent=fit_params.get('prominence_percent'),
                min_height=fit_params.get('min_height'),
                min_separation_ev=fit_params.get('min_separation_ev'),
            )

            id_summary = []
            identified = []
            if fit_params.get('auto_id_after_peak_find', True):
                preview_peaks, identified, id_summary = auto_id_peak_positions(
                    preview_peaks
                )
                if identified:
                    self.element_panel.set_selected_elements(identified)
                    self.session.set_elements(
                        self.element_panel.get_selected_elements()
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
            header = (
                f"Peak find ({len(preview_peaks)} total: "
                f"{n_labeled} labeled, {n_unknown} unknown)"
            )
            if id_summary:
                header += "\n" + "\n".join(id_summary[:40])
            if lines:
                self.results_panel.peaks_text.setPlainText(
                    header + ":\n" + "\n".join(lines)
                )
            else:
                self.results_panel.peaks_text.setPlainText(
                    "Peak find: no peaks detected.\n"
                    "Try lowering Prominence or Min height."
                )

            # Stay on Peak Find so the user can edit the found-peak list first
            if identified:
                msg = (
                    f"Peak find: {len(preview_peaks)} peaks; "
                    f"auto-ID selected {len(identified)} element(s). "
                    f"Review the peak list, then Elements → Fitting."
                )
            else:
                msg = (
                    f"Peak find: {len(preview_peaks)} peaks "
                    f"({n_unknown} unlabeled). "
                    f"Select elements, then Fitting → Fit Spectrum."
                )
            self.status_bar.showMessage(msg, 10000)
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
        """Semi-quantitative relative intensities from the current fit"""
        if self.fit_result is None or not getattr(self.fit_result, 'peaks', None):
            QMessageBox.warning(
                self,
                "No Fit Results",
                "Please fit a spectrum first before running semi-quant."
            )
            return
        
        # Show Results tab so the table update is visible
        if hasattr(self, 'analysis_left_tabs'):
            self.analysis_left_tabs.setCurrentIndex(self.TAB_RESULTS)
        
        self.status_bar.showMessage("Computing relative intensities...", 0)
        try:
            exp_params = self.element_panel.get_experimental_params()
            concentrations = self.fitter.quantify_elements(
                self.fit_result.peaks, exp_params
            )
            self.session.set_concentrations(concentrations, method="semi_quant_area")
            self.results_panel.set_fp_live(False)
            self.results_panel.set_formula_summary("")
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
                    "Semi-quant: no labeled sample peaks", 5000
                )
                return
            
            self.status_bar.showMessage(
                f"Semi-quant complete: {n} element{'s' if n != 1 else ''} "
                f"(relative intensity, not FP wt%)",
                5000
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Semi-Quant Error",
                f"An error occurred during semi-quantification:\n{str(e)}"
            )
            self.status_bar.showMessage("Semi-quant failed", 5000)
    
    def quantify_fp(self, live=False):
        """Standardless FP wt% using the current matrix assumptions."""
        live = live is True
        if self.fit_result is None or not getattr(self.fit_result, "peaks", None):
            if live:
                return
            QMessageBox.warning(
                self,
                "No Fit Results",
                "Please fit a spectrum first before running FP composition.",
            )
            return

        if hasattr(self, "analysis_left_tabs"):
            self.analysis_left_tabs.setCurrentIndex(self.TAB_RESULTS)

        assumptions = self.results_panel.get_matrix_assumptions()
        self.session.matrix = assumptions
        if not live:
            self.status_bar.showMessage("Computing FP composition...", 0)
        try:
            exp_params = self.element_panel.get_experimental_params()
            result = quantify_from_peaks(
                self.fit_result.peaks, assumptions, exp_params
            )
            if not result.success:
                if live:
                    return
                self.results_panel.set_fp_live(False)
                QMessageBox.warning(
                    self,
                    "FP Composition",
                    result.message or "FP quantification failed.",
                )
                self.status_bar.showMessage("FP composition failed", 5000)
                return

            self.session.set_fp_result(result)
            self.results_panel.set_fp_live(True)
            self.results_panel.set_quantification(result.concentrations)
            bits = [f"As formulas: {result.formula_summary()}"]
            if result.residual < float("inf"):
                bits.append(
                    f"intensity residual {result.residual:.4f} "
                    f"({result.iterations} iter)"
                )
            bits.append(f"measured cations {result.measured_cation_pct:.1f} %")
            self.results_panel.set_formula_summary("    |  ".join(bits))
            n = len([k for k, v in result.concentrations.items()
                     if v.get("role") == "measured"])
            self.status_bar.showMessage(
                f"FP composition: {n} measured element{'s' if n != 1 else ''} "
                f"({assumptions.kind.value}; "
                f"H2O {assumptions.h2o_wt:g}%, OH {assumptions.oh_wt:g}%, "
                f"CO2 {assumptions.co2_wt:g}%)",
                5000,
            )
        except Exception as e:
            if live:
                return
            QMessageBox.critical(
                self,
                "FP Composition Error",
                f"An error occurred during FP quantification:\n{str(e)}",
            )
            self.status_bar.showMessage("FP composition failed", 5000)

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
    
    def install_desktop_shortcut(self):
        """Create a Desktop launcher with the XRFLab icon (Mac / Windows / Linux)."""
        result = install_desktop_shortcut()
        if result.success:
            QMessageBox.information(self, "Desktop Shortcut", result.message)
            self.status_bar.showMessage(
                f"Desktop shortcut installed: {result.path}",
                8000,
            )
        else:
            QMessageBox.warning(self, "Desktop Shortcut", result.message)
            self.status_bar.showMessage("Desktop shortcut install failed", 5000)

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About XRFLab",
            "<h3>XRFLab</h3>"
            "<p>Version 1.0.0</p>"
            "<p>Desktop XRF spectrum analysis: fitting, detector/tube calibration, "
            "area-normalized semi-quant, and standardless FP composition "
            "(matrix model with optional H₂O / OH / CO₂). "
            "Standards / fisx tools are under Calibration → Standards.</p>"
            "<p>Built with PySide6, PyQtGraph, xraylib, and fisx.</p>"
            "<p>Use <b>Help → Install Desktop Shortcut</b> to add a Desktop launcher.</p>"
        )
    
    def on_elements_changed(self, elements):
        """Handle element selection changes"""
        self.session.set_elements(elements)

    def on_identify_on_plot_toggled(self, enabled):
        """Enable/disable click-to-identify on the spectrum plot."""
        self.spectrum_widget.set_energy_pick_mode(bool(enabled))
        if enabled:
            self.status_bar.showMessage(
                "Identify mode on — click the spectrum for line candidates",
                5000,
            )
        else:
            self.status_bar.showMessage("Identify mode off", 3000)

    def on_spectrum_energy_picked(self, energy_kev):
        """Show ranked emission-line candidates for a clicked energy."""
        hits = candidates_at_energy(float(energy_kev), energy_tol_kev=0.150)
        self.element_panel.set_identify_candidates(energy_kev, hits)
        if hits:
            top = hits[0]
            self.status_bar.showMessage(
                f"{energy_kev:.3f} keV → top: {top['symbol']}-{top['line']} "
                f"(Δ {abs(top['delta_ev']):.0f} eV); "
                f"{len(hits)} candidate(s)",
                8000,
            )
        else:
            self.status_bar.showMessage(
                f"{energy_kev:.3f} keV — no common-XRF lines within ±150 eV",
                5000,
            )

    def on_identify_add_element(self, symbol):
        """Add an identify-candidate element to the periodic-table selection."""
        self.element_panel.add_selected_element(symbol)
        self.session.set_elements(self.element_panel.get_selected_elements())
        self.status_bar.showMessage(f"Added {symbol} to element selection", 4000)
        # Overlay that element's lines for confirmation
        from core.advanced_peak_fitting import get_element_z
        z = get_element_z(symbol)
        if z:
            self._displayed_element_lines = None
            self.on_element_clicked(symbol, z)
    
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
        
        # Store on session instrument state
        self.session.instrument.apply_fwhm_calibration(fwhm_calibration)
        self.session.apply_instrument_to_fitter(self.fitter)
        self.batch_analysis_panel.set_instrument_state(self.session.instrument)

        # Update the Standards panel with the FWHM calibration
        self.standards_panel.update_fwhm_status(fwhm_calibration)
        
        # Apply to Analysis peak fitting
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

    def on_tube_profiles_changed(self, library):
        """Apply per-kV tube profile library to Analysis fitting."""
        self.session.instrument.tube_profile_library = library
        self.session.apply_instrument_to_fitter(self.fitter)
        self.batch_analysis_panel.set_instrument_state(self.session.instrument)
        self.element_panel.update_tube_profile_status(library)
        n_meas = sum(1 for p in library.profiles.values() if p.source == 'measured')
        self.status_bar.showMessage(
            f"Tube profiles active: {n_meas} measured / "
            f"{len(library.available_kvs)} modes "
            f"({library.tube_element})",
            5000
        )
    
    def on_calibration_applied(self, calibration_result):
        """Handle standards calibration being applied"""
        self.session.instrument.standards_calibration = calibration_result
        self.status_bar.showMessage(
            f"Standards calibration stored: FWHM₀={calibration_result.fwhm_0*1000:.1f} eV, "
            f"ε={calibration_result.epsilon*1000:.2f} eV",
            5000
        )
        
        # Switch back to analysis tab
        self.tab_widget.setCurrentIndex(0)
    
    def closeEvent(self, event):
        """Handle window close event"""
        self._save_settings()
        event.accept()
