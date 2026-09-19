"""
Main window for XRF Fundamental Parameters Analysis Application
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QToolBar, QStatusBar, QMessageBox, QFileDialog,
    QPushButton, QLabel, QApplication, QScrollArea, QDialog
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence, QIcon

from ui.spectrum_widget import SpectrumWidget
from ui.figure_window import FigureWindow
from ui.nav_rail import AnalysisSteps, IndexedStack
from ui.element_panel import ElementPanel
from ui.results_panel import ResultsPanel
from ui.batch_analysis_panel import BatchAnalysisPanel
from ui.standards_panel import StandardsPanel
from ui.fwhm_calibration_panel import FWHMCalibrationPanel
from ui.tube_profile_panel import (
    SHOW_TUBE_PROFILE_CALIBRATION,
    TubeProfilePanel,
)
from ui.mapping_panel import MappingPanel
from utils.io_handler import IOHandler
from utils.updater import check_for_updates
from utils.desktop_shortcut import install_desktop_shortcut
from utils.paths import icon_path, resource_path
from core.fitting import SpectrumFitter
from core.xray_data import DEFAULT_SCATTER_ANGLE_DEG
from core.fp_quantification import quantify_from_peaks
from core.matrix_model import empirical_formula
from core.session import AnalysisSession
from core.project_file import (
    FILE_FILTER,
    ProjectDocument,
    ProjectFileError,
    load_project as read_project_file,
    save_project as write_project_file,
)
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
        self.setGeometry(80, 60, 980, 760)
        self.setMinimumSize(720, 520)
        
        # Initialize components
        self.io_handler = IOHandler()
        self.session = AnalysisSession()
        self.fitter = SpectrumFitter()
        self.session.apply_instrument_to_fitter(self.fitter)
        self.settings = QSettings()
        self._displayed_element_lines = None  # symbol currently shown on plot, or None
        self._project_path = None
        self._ree_fit_set = None  # ReeFitSet when the last selection was REEs + overlaps
        self.analysis_splitter = None
        
        # Setup UI (status bar before central widget)
        self._create_actions()
        self._create_menus()
        self._create_toolbar()
        self._create_status_bar()
        self._create_central_widget()
        self._load_stylesheet()
        self._apply_window_icon()
        
        # Restore window state
        self._restore_settings()
        self._restore_figures()

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

        self.open_overlay_action = QAction("Open as &Overlay...", self)
        self.open_overlay_action.setStatusTip(
            "Add a spectrum on top of the current plot for comparison"
        )
        self.open_overlay_action.triggered.connect(self.open_spectrum_as_overlay)

        self.open_ipj_action = QAction("Open &IPJ Project...", self)
        self.open_ipj_action.setStatusTip(
            "Open an Oxford INCA / Horiba XGT .ipj mapping project"
        )
        self.open_ipj_action.triggered.connect(self.open_ipj_project)

        self.merge_ipj_action = QAction("Merge &IPJ Projects...", self)
        self.merge_ipj_action.setStatusTip(
            "Merge line scans / multipoint series from many .ipj files "
            "into one Mapping project"
        )
        self.merge_ipj_action.triggered.connect(self.merge_ipj_projects)

        self.open_project_action = QAction("Open &Project...", self)
        self.open_project_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.open_project_action.setStatusTip("Open a saved XRFLab project (.xrfp)")
        self.open_project_action.triggered.connect(self.open_project)

        self.save_project_action = QAction("&Save Project", self)
        self.save_project_action.setShortcut(QKeySequence.Save)
        self.save_project_action.setStatusTip("Save the entire workspace to a .xrfp file")
        self.save_project_action.triggered.connect(self.save_project)

        self.save_project_as_action = QAction("Save Project &As...", self)
        self.save_project_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.save_project_as_action.setStatusTip("Save the workspace to a new .xrfp file")
        self.save_project_as_action.triggered.connect(self.save_project_as)
        
        self.export_results_action = QAction("&Export Results...", self)
        self.export_results_action.setStatusTip(
            "Export semi-quant relative intensities from the Results tab"
        )
        self.export_results_action.triggered.connect(self.export_results)

        self.export_fp_action = QAction("Export FP Composition (&wt%)...", self)
        self.export_fp_action.setStatusTip(
            "Export fundamental-parameters wt% from Analysis → Composition"
        )
        self.export_fp_action.triggered.connect(self.export_fp_results)

        self.export_report_action = QAction("Export &Report...", self)
        self.export_report_action.setStatusTip(
            "Export a detailed HTML report of calibrations, fit, and compositions"
        )
        self.export_report_action.triggered.connect(self.export_report)
        
        self.exit_action = QAction("&Quit", self)
        self.exit_action.setMenuRole(QAction.QuitRole)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.setShortcutContext(Qt.ApplicationShortcut)
        self.exit_action.setStatusTip("Quit XRFLab")
        self.exit_action.triggered.connect(self.close)
        
        # Analysis actions
        self.fit_spectrum_action = QAction("&Fit Spectrum", self)
        self.fit_spectrum_action.setShortcut("Ctrl+F")
        self.fit_spectrum_action.setStatusTip("Fit the current spectrum")
        self.fit_spectrum_action.triggered.connect(self.fit_spectrum)
        
        self.quantify_action = QAction("&Semi-Quant (Relative Intensities)", self)
        self.quantify_action.setShortcut("Ctrl+I")
        self.quantify_action.setStatusTip(
            "Area-normalized relative intensities (not wt%)."
        )
        self.quantify_action.triggered.connect(self.quantify)

        self.standards_quantify_action = QAction("S&tandards wt% (CRM curves)", self)
        self.standards_quantify_action.setShortcut("Ctrl+Shift+S")
        self.standards_quantify_action.setStatusTip(
            "Targeted CRM-curve wt% for Report-marked elements. "
            "Independent determinations, not a closed assay."
        )
        self.standards_quantify_action.triggered.connect(self.quantify_standards)

        self.fp_quantify_action = QAction("&FP Composition (wt%)", self)
        self.fp_quantify_action.setShortcut("Ctrl+Shift+Q")
        self.fp_quantify_action.setStatusTip(
            "Standardless FP wt% from a polychromatic tube spectrum "
            "(continuum + anode lines) and the matrix model."
        )
        self.fp_quantify_action.triggered.connect(self.quantify_fp)

        self.quantify_rees_action = QAction("Quantify &REEs + overlaps", self)
        self.quantify_rees_action.setShortcut("Ctrl+Shift+R")
        self.quantify_rees_action.setStatusTip(
            "Fit only Y, La–Lu and their spectroscopic overlaps, "
            "then read wt% from the standards curves."
        )
        self.quantify_rees_action.triggered.connect(self.quantify_rees)
        
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

        self.show_spectrum_action = QAction("Show &Spectrum", self)
        self.show_spectrum_action.setStatusTip("Open the spectrum window")
        self.show_spectrum_action.triggered.connect(
            lambda: self.spectrum_window.show_view("Spectrum", focus=True)
        )
        self.show_map_action = QAction("Show &Map", self)
        self.show_map_action.setStatusTip("Open the map window")
        self.show_map_action.triggered.connect(
            lambda: self.map_window.show_window(focus=True)
        )
        self.show_charts_action = QAction("Show &Charts", self)
        self.show_charts_action.setStatusTip("Open the charts window")
        self.show_charts_action.triggered.connect(
            lambda: self.charts_window.show_window(focus=True)
        )
        
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
        file_menu.addAction(self.open_overlay_action)
        file_menu.addAction(self.open_ipj_action)
        file_menu.addAction(self.merge_ipj_action)
        file_menu.addAction(self.open_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.save_project_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_results_action)
        file_menu.addAction(self.export_fp_action)
        file_menu.addAction(self.export_report_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        analysis_menu.addAction(self.fit_spectrum_action)
        analysis_menu.addAction(self.quantify_rees_action)
        analysis_menu.addAction(self.quantify_action)
        analysis_menu.addAction(self.standards_quantify_action)
        analysis_menu.addAction(self.fp_quantify_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.toggle_log_action)
        view_menu.addAction(self.toggle_grid_action)
        view_menu.addSeparator()
        view_menu.addAction(self.show_spectrum_action)
        view_menu.addAction(self.show_map_action)
        view_menu.addAction(self.show_charts_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction(self.fwhm_calibration_action)
        if SHOW_TUBE_PROFILE_CALIBRATION:
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
        toolbar.addAction(self.open_project_action)
        toolbar.addAction(self.save_project_action)
    
    def _create_central_widget(self):
        """Console: activity rail and one task. Plots live in figure windows."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.calibration_status = QLabel()
        self.calibration_status.setObjectName("calibrationStatus")
        self.calibration_status.setWordWrap(True)
        self.calibration_status.setTextFormat(Qt.RichText)
        layout.addWidget(self.calibration_status)

        self.tab_widget = IndexedStack(rail_width=96, object_name="activityRail")

        analysis_tab = self._create_analysis_tab()
        self.tab_widget.addTab(analysis_tab, "Analyze")

        self.batch_analysis_panel = BatchAnalysisPanel()
        self.tab_widget.addTab(self.batch_analysis_panel, "Batch")
        self.batch_analysis_panel.set_element_panel(self.element_panel)
        self.batch_analysis_panel.set_results_panel(self.results_panel)
        self.batch_analysis_panel.set_instrument_state(self.session.instrument)
        self.batch_analysis_panel.spectrum_selected.connect(
            self.on_batch_composition_spectrum_selected
        )
        self.batch_analysis_panel.figure_requested.connect(self._on_batch_figure)

        self.mapping_panel = MappingPanel()
        self.mapping_panel.set_fitter(self.fitter)
        self.mapping_panel.set_element_panel(self.element_panel)
        self.mapping_panel.spectrum_send_requested.connect(self.on_mapping_spectrum_sent)
        self.mapping_panel.spectra_compare_requested.connect(self.on_mapping_spectra_compare)
        self.mapping_panel.spectra_batch_requested.connect(self.on_mapping_spectra_to_batch)
        self.mapping_panel.project_loaded.connect(self.on_mapping_project_loaded)
        self.mapping_panel.status_message.connect(
            lambda msg: self.status_bar.showMessage(msg, 5000)
        )
        self.tab_widget.addTab(self.mapping_panel, "Maps")

        self.calibration_tab = self._create_calibration_tab()
        self.tab_widget.addTab(self.calibration_tab, "Setup")
        self.tab_widget.currentChanged.connect(self._on_mode_changed)

        if self.fwhm_calibration_panel.fwhm_calibration is not None:
            self.on_fwhm_calibration_applied(self.fwhm_calibration_panel.fwhm_calibration)

        if self.tube_profile_panel.get_library() is not None:
            self.on_tube_profiles_changed(self.tube_profile_panel.get_library())

        if self.standards_panel.calibration_result is not None:
            self.on_calibration_applied(
                self.standards_panel.calibration_result, switch_tab=False
            )

        layout.addWidget(self.tab_widget)
        self._create_figure_windows()

    def _create_figure_windows(self):
        """Host the existing plot widgets in three secondary windows."""
        self.spectrum_window = FigureWindow("Spectrum", "spectrum", self, offset=0)
        self.spectrum_window.add_view("Spectrum", self.spectrum_widget)
        self.spectrum_window.add_view("Batch fit", self.batch_analysis_panel.batch_plot_page)

        self.map_window = FigureWindow("Map", "map", self, offset=36)
        self.map_window.add_view("Map", self.mapping_panel.figure_host)

        self.charts_window = FigureWindow("Charts", "charts", self, offset=72)
        self.charts_window.add_view("Trends", self.batch_analysis_panel.trends_page)
        self.charts_window.add_view("FWHM", self.fwhm_calibration_panel.detached_plot)
        self.charts_window.add_view("Standards", self.standards_panel.detached_plot)

    def _restore_figures(self):
        for window in (self.spectrum_window, self.map_window, self.charts_window):
            window.restore_geometry()
        # Spectrum is the daily view. Remember a deliberate close.
        if self.spectrum_window.was_visible() or self.settings.value("figure/spectrum/visible") is None:
            self.spectrum_window.show_view("Spectrum")
        if self.map_window.was_visible():
            self.map_window.show_window()
        if self.charts_window.was_visible():
            self.charts_window.show_window()

    def _on_mode_changed(self, index: int) -> None:
        if index == 0:
            self.spectrum_window.show_view("Spectrum")
        elif index == 1:
            self.spectrum_window.show_view("Batch fit")
        elif index == 2:
            self.map_window.show_window()
        elif index == 3:
            self._show_setup_chart()

    def _on_batch_figure(self, kind: str) -> None:
        if self.tab_widget.currentIndex() != 1:
            return
        if kind == "charts":
            self.charts_window.show_view("Trends")
        else:
            self.spectrum_window.show_view("Batch fit")

    def _show_setup_chart(self) -> None:
        widget = self.calibration_tabs.currentWidget()
        if widget is self.standards_panel:
            self.charts_window.show_view("Standards")
        elif widget is self.tube_profile_panel:
            self.charts_window.show_window()
        else:
            self.charts_window.show_view("FWHM")

    def _refresh_spectrum_title(self) -> None:
        path = self.session.spectrum_path
        if not path:
            self.spectrum_window.setWindowTitle("Spectrum")
            return
        name = Path(str(path).split("::")[-1]).name
        self.spectrum_window.setWindowTitle(f"Spectrum — {name}")
    
    def _create_calibration_tab(self):
        """FWHM, optional tube profiles, then standards. Plots open in Charts."""
        calibration_widget = QWidget()
        layout = QVBoxLayout(calibration_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.calibration_tabs = IndexedStack(rail_width=140)

        self.fwhm_calibration_panel = FWHMCalibrationPanel()
        self.fwhm_calibration_panel.calibration_complete.connect(
            self.on_fwhm_calibration_applied
        )
        self.calibration_tabs.addTab(self.fwhm_calibration_panel, "FWHM")

        self.tube_profile_panel = TubeProfilePanel()
        self.tube_profile_panel.library_changed.connect(self.on_tube_profiles_changed)
        if SHOW_TUBE_PROFILE_CALIBRATION:
            self.calibration_tabs.addTab(self.tube_profile_panel, "Tube")

        self.standards_panel = StandardsPanel()
        self.standards_panel.calibration_complete.connect(self.on_calibration_applied)
        self.calibration_tabs.addTab(self.standards_panel, "Standards")
        self.calibration_tabs.currentChanged.connect(self._on_setup_step)

        layout.addWidget(self.calibration_tabs)
        self._refresh_calibration_status()
        return calibration_widget

    def _on_setup_step(self, _index: int) -> None:
        if self.tab_widget.currentIndex() == 3:
            self._show_setup_chart()

    def _refresh_calibration_status(self):
        """Checklist at the top of Calibration: FWHM required, rest optional."""
        if not hasattr(self, "calibration_status"):
            return

        fwhm = getattr(self.fwhm_calibration_panel, "fwhm_calibration", None)
        if fwhm is not None:
            fwhm_txt = (
                f"<b>FWHM</b> in use · R² {fwhm.r_squared:.3f} · "
                f"{fwhm.n_peaks} peaks"
            )
            fwhm_color = "#1b7a3d"
        else:
            fwhm_txt = "<b>FWHM</b> not set — Calibrate &amp; Use on the FWHM tab"
            fwhm_color = "#b36b00"

        std = getattr(self.standards_panel, "calibration_result", None) if hasattr(self, "standards_panel") else None
        curves = getattr(std, "fitted_curves", None)
        if callable(curves) and curves():
            std_txt = (
                f"<b>Standards</b> {len(curves())} CRM curve(s) ready "
                f"for targeted wt%"
            )
        elif std is not None and getattr(std, "success", False):
            std_txt = "<b>Standards</b> legacy intensity calibration stored"
        else:
            std_txt = "<b>Standards</b> optional — not required for Semi-Quant"

        extra = ""
        if SHOW_TUBE_PROFILE_CALIBRATION:
            library = None
            if hasattr(self, "tube_profile_panel"):
                library = self.tube_profile_panel.get_library()
            n_meas = 0
            if library is not None:
                n_meas = sum(
                    1 for p in library.profiles.values()
                    if getattr(p, "source", None) == "measured"
                )
            if n_meas:
                tube_txt = f"<b>Tube</b> {n_meas} measured mode(s)"
            else:
                tube_txt = "<b>Tube</b> defaults (optional blanks)"
            extra = f"&nbsp;&nbsp;·&nbsp;&nbsp;{tube_txt}"

        self.calibration_status.setText(
            f"<span style='color:{fwhm_color}'>{fwhm_txt}</span>"
            f"{extra}"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;{std_txt}"
        )
    
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
        """Analyze steps. The spectrum itself is a separate window."""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.analysis_splitter = None
        self.analysis_left_tabs = AnalysisSteps()

        self.element_panel = ElementPanel()
        self.analysis_left_tabs.addTab(self._create_sample_exp_tab(), "Sample")
        self.analysis_left_tabs.addTab(self._create_peak_find_tab(), "Peaks")
        self.analysis_left_tabs.addTab(self._create_element_selection_tab(), "Elements")
        self.analysis_left_tabs.addTab(self._create_fitting_controls_tab(), "Fit")

        self.spectrum_widget = SpectrumWidget()
        self.spectrum_widget.log_scale_changed.connect(self._on_plot_log_scale_changed)
        self.spectrum_widget.energy_selected.connect(self.on_spectrum_energy_picked)

        results_scroll = self._create_results_tab()
        self.analysis_left_tabs.results_scroll = results_scroll
        self.analysis_left_tabs.addTab(results_scroll, "Results")
        layout.addWidget(self.analysis_left_tabs)

        self.element_panel.elements_changed.connect(self.on_elements_changed)
        self.element_panel.fit_requested.connect(self.fit_spectrum)
        self.element_panel.ree_select_requested.connect(self.select_rees_and_overlaps)
        self.element_panel.ree_fit_requested.connect(self.quantify_rees)
        self.element_panel.peak_find_requested.connect(self.preview_peak_find)
        self.element_panel.peak_list_changed.connect(self.on_peak_list_changed)
        self.element_panel.element_clicked.connect(self.on_element_clicked)
        self.element_panel.identify_on_plot_toggled.connect(self.on_identify_on_plot_toggled)
        self.element_panel.identify_add_element.connect(self.on_identify_add_element)
        self.element_panel.identify_preview_element.connect(self.on_identify_preview_element)
        self.element_panel.tube_guides_changed.connect(self.refresh_tube_guides)
        self.element_panel.scatter_angle_fit_requested.connect(
            self.fit_scatter_angle_from_spectrum
        )
        self.results_panel.element_selected.connect(self.on_result_element_selected)
        self.results_panel.quantify_requested.connect(self.quantify)
        self.results_panel.standards_quantify_requested.connect(self.quantify_standards)
        self.results_panel.fp_quantify_requested.connect(self.quantify_fp)
        self.results_panel.matrix_assumptions_changed.connect(
            lambda: self.quantify_fp(live=True)
        )
        self.results_panel.export_button.clicked.connect(self.export_results)
        self.results_panel.export_fp_requested.connect(self.export_fp_results)

        self.refresh_tube_guides()
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
            "Select a candidate to preview its lines. Then continue to Fitting."
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
        """Fit statistics, semi-quant, and composition on one scrolling page."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        self.results_panel = ResultsPanel()
        layout.addWidget(self.results_panel)

        heading = QLabel("Composition")
        heading.setObjectName("sectionLabel")
        layout.addWidget(heading)
        layout.addWidget(self.results_panel.composition_tab_widget())
        self.analysis_left_tabs.composition_anchor = heading

        scroll.setWidget(inner)
        return scroll

    # Analysis left-tab indices for navigation after actions
    TAB_SAMPLE = 0
    TAB_PEAK_FIND = 1
    TAB_ELEMENTS = 2
    TAB_FITTING = 3
    TAB_RESULTS = 4
    TAB_COMPOSITION = 5

    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _load_stylesheet(self):
        """Load and apply Qt stylesheet to the app, including figure windows."""
        try:
            with open(resource_path("styles.qss"), "r") as f:
                sheet = f.read()
        except FileNotFoundError:
            return
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(sheet)
        else:
            self.setStyleSheet(sheet)

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

    def _project_dialog_dir(self) -> str:
        last = self.settings.value("last_project")
        if last:
            parent = Path(str(last)).parent
            if parent.is_dir():
                return str(parent)
        return ""

    def _set_project_path(self, path) -> None:
        self._project_path = str(path) if path else None
        if self._project_path:
            self.settings.setValue("last_project", self._project_path)
            self.setWindowTitle(f"XRFLab — {Path(self._project_path).name}")
        else:
            self.setWindowTitle("XRFLab - Fundamental Parameters Analysis")

    def _workspace_has_content(self) -> bool:
        return bool(
            self.session.spectrum is not None
            or self.mapping_panel.project is not None
            or self.batch_analysis_panel.file_list.count() > 0
            or self.batch_analysis_panel.composition_panel.rows
        )

    def capture_document(self) -> ProjectDocument:
        """Collect the full workspace into a ProjectDocument."""
        analysis_ui = {
            "element_panel": self.element_panel.capture_state(),
            "results_panel": self.results_panel.capture_state(),
            "spectrum_widget": self.spectrum_widget.capture_state(),
            "left_tab": int(self.analysis_left_tabs.currentIndex()),
            "displayed_element_lines": self._displayed_element_lines,
        }
        analysis = {
            "spectrum": self.session.spectrum,
            "spectrum_path": self.session.spectrum_path,
            "elements": self.session.elements or self.element_panel.get_selected_elements(),
            "fit_result": self.session.fit_result,
            "concentrations": self.session.concentrations,
            "quantification_method": self.session.quantification_method,
            "matrix": self.results_panel.get_matrix_assumptions(),
            "fp_result": self.session.fp_result,
            "ui": analysis_ui,
        }
        batch_state = self.batch_analysis_panel.capture_state()
        batch = {
            "file_paths": batch_state.get("file_paths") or [],
            "memory_spectra": dict(self.batch_analysis_panel._memory_spectra),
            "results": list(self.batch_analysis_panel.results or []),
            "config": batch_state.get("config") or {},
            "ui": batch_state.get("ui") or {},
        }
        calibrations = {}
        fwhm = self.fwhm_calibration_panel.fwhm_calibration
        if fwhm is not None:
            calibrations["fwhm"] = fwhm.to_dict()
        tube = self.tube_profile_panel.get_library()
        if tube is not None:
            calibrations["tube"] = tube.to_dict()
        standards = self.standards_panel.calibration_result
        if standards is not None:
            calibrations["standards"] = standards.to_dict()
        window = {
            "tab": int(self.tab_widget.currentIndex()),
            "calibration_tab": int(self.calibration_tabs.currentIndex()),
            "log_y": bool(self.toggle_log_action.isChecked()),
            "grid": bool(self.toggle_grid_action.isChecked()),
            "analysis_splitter": (
                list(self.analysis_splitter.sizes())
                if self.analysis_splitter is not None
                else None
            ),
        }
        return ProjectDocument(
            analysis=analysis,
            mapping_project=self.mapping_panel.project,
            mapping_ui=self.mapping_panel.capture_ui_state(),
            drawn_line_scan=self.mapping_panel._drawn_line_scan,
            batch=batch,
            composition=self.batch_analysis_panel.composition_panel.capture_state(),
            calibrations=calibrations,
            window=window,
        )

    def apply_document(self, document: ProjectDocument) -> None:
        """Replace the workspace with a loaded ProjectDocument."""
        from core.calibration import CalibrationResult
        from core.fwhm_calibration import FWHMCalibration
        from core.tube_profile import TubeProfileLibrary

        cals = document.calibrations or {}
        if cals.get("fwhm"):
            fwhm = FWHMCalibration.from_dict(cals["fwhm"])
            self.fwhm_calibration_panel.restore_calibration(fwhm)
            self.on_fwhm_calibration_applied(fwhm)
        if cals.get("tube"):
            library = TubeProfileLibrary.from_dict(cals["tube"])
            self.tube_profile_panel.restore_library(library)
            self.on_tube_profiles_changed(library)
        if cals.get("standards"):
            from core.standards_calibration import load_any_standards_calibration

            std = load_any_standards_calibration(cals["standards"])
            self.standards_panel.restore_calibration(std)
            self.session.instrument.standards_calibration = std

        analysis = document.analysis or {}
        ui = analysis.get("ui") or {}
        self.element_panel.restore_state(ui.get("element_panel") or {})
        self.session.spectrum = analysis.get("spectrum")
        self.session.spectrum_path = analysis.get("spectrum_path")
        self.session.elements = list(
            analysis.get("elements") or self.element_panel.get_selected_elements()
        )
        self.session.fit_result = analysis.get("fit_result")
        self.session.concentrations = dict(analysis.get("concentrations") or {})
        self.session.quantification_method = str(
            analysis.get("quantification_method") or "semi_quant_area"
        )
        matrix = analysis.get("matrix")
        if matrix is not None:
            self.session.matrix = matrix
        self.session.fp_result = analysis.get("fp_result")

        self.results_panel.clear_results()
        self.spectrum_widget.clear_overlays()
        self.spectrum_widget.set_fitted_spectrum(None)
        self.spectrum_widget.set_background(None)
        self.spectrum_widget.clear_peak_markers()
        if self.session.spectrum is not None:
            self.spectrum_widget.set_spectrum(self.session.spectrum)
        else:
            self.spectrum_widget.set_spectrum(None)
        fit = self.session.fit_result
        if fit is not None:
            self.spectrum_widget.set_fitted_spectrum(fit.fitted_spectrum)
            self.spectrum_widget.set_background(fit.background)
            show = True
            ep = ui.get("element_panel") or {}
            if "show_markers" in ep:
                show = bool(ep["show_markers"])
            self.spectrum_widget.set_peak_markers(fit.peaks, show=show)
            self.results_panel.set_fit_statistics(fit.statistics or {})
            self.results_panel.set_peaks(fit.peaks)
            flags = getattr(fit, "tube_overlap_flags", None) or []
            if flags:
                self.results_panel.set_tube_overlap_flags(flags)
        self.results_panel.restore_state(ui.get("results_panel") or {})
        if self.session.concentrations and not (ui.get("results_panel") or {}).get(
            "results_data"
        ):
            self.results_panel.set_quantification(self.session.concentrations)
        fp = self.session.fp_result
        if fp is not None:
            self.results_panel.set_fp_live(True)
            self.results_panel.set_quantification(fp.concentrations)
            bits = [f"As compounds: {fp.formula_summary()}"]
            if fp.residual < float("inf"):
                bits.append(
                    f"intensity residual {fp.residual:.4f} "
                    f"({fp.iterations} iter)"
                )
            bits.append(f"measured cations {fp.measured_cation_pct:.1f} %")
            self.results_panel.set_formula_summary(
                "    |  ".join(bits),
                empirical=empirical_formula(
                    fp.element_wt, formula_wt=fp.formula_wt
                ),
            )
        self.spectrum_widget.restore_state(ui.get("spectrum_widget") or {})
        self._refresh_spectrum_title()
        if ui.get("left_tab") is not None:
            self.analysis_left_tabs.setCurrentIndex(int(ui["left_tab"]))
        self._displayed_element_lines = ui.get("displayed_element_lines")

        self.mapping_panel.load_saved_project(
            document.mapping_project,
            document.mapping_ui,
            drawn_line_scan=document.drawn_line_scan,
        )
        batch = document.batch or {}
        self.batch_analysis_panel.restore_state(
            {
                "file_paths": batch.get("file_paths") or [],
                "config": batch.get("config") or {},
                "ui": batch.get("ui") or {},
            },
            results=batch.get("results") or [],
            memory_spectra=batch.get("memory_spectra") or {},
            composition=document.composition or {},
        )

        window = document.window or {}
        if window.get("log_y") is not None:
            self.toggle_log_action.blockSignals(True)
            self.toggle_log_action.setChecked(bool(window["log_y"]))
            self.toggle_log_action.blockSignals(False)
            self.spectrum_widget.set_log_scale(bool(window["log_y"]))
        if window.get("grid") is not None:
            self.toggle_grid_action.blockSignals(True)
            self.toggle_grid_action.setChecked(bool(window["grid"]))
            self.toggle_grid_action.blockSignals(False)
            self.spectrum_widget.set_grid(bool(window["grid"]))
        if window.get("analysis_splitter") and self.analysis_splitter is not None:
            self.analysis_splitter.setSizes([int(x) for x in window["analysis_splitter"]])
        if window.get("calibration_tab") is not None:
            self.calibration_tabs.setCurrentIndex(int(window["calibration_tab"]))
        if window.get("tab") is not None:
            self.tab_widget.setCurrentIndex(int(window["tab"]))

    def save_project(self):
        """Save the workspace to the current .xrfp path, or ask for a name."""
        if not self._project_path:
            return self.save_project_as()
        return self._write_project(self._project_path)

    def save_project_as(self):
        """Save the workspace to a new .xrfp file."""
        suggested = self._project_path or str(
            Path(self._project_dialog_dir() or ".") / "project.xrfp"
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save XRFLab Project",
            suggested,
            FILE_FILTER,
        )
        if not file_path:
            return False
        if not str(file_path).lower().endswith(".xrfp"):
            file_path = str(Path(file_path).with_suffix(".xrfp"))
        return self._write_project(file_path)

    def _write_project(self, file_path: str) -> bool:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            document = self.capture_document()
            write_project_file(file_path, document)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self,
                "Save Project Failed",
                f"Could not save project:\n{exc}",
            )
            return False
        QApplication.restoreOverrideCursor()
        self._set_project_path(file_path)
        self.status_bar.showMessage(f"Saved: {file_path}", 5000)
        return True

    def open_project(self, path=None):
        """Replace the workspace with a saved .xrfp project."""
        if not isinstance(path, str) or not path:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open XRFLab Project",
                self._project_dialog_dir(),
                FILE_FILTER,
            )
            if not file_path:
                return
            path = file_path
        if self._workspace_has_content():
            reply = QMessageBox.question(
                self,
                "Open Project",
                "Open this project and replace the current workspace?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            document = read_project_file(path)
            self.apply_document(document)
        except ProjectFileError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Open Project Failed", str(exc))
            return
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self,
                "Open Project Failed",
                f"Could not open project:\n{exc}",
            )
            return
        QApplication.restoreOverrideCursor()
        self._set_project_path(path)
        self.status_bar.showMessage(f"Opened: {path}", 5000)
    
    # Action handlers
    def open_spectrum(self):
        """Open an XRF spectrum file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open XRF Spectrum",
            "",
            "All Supported (*.txt *.csv *.mca *.h5 *.hdf5);;Text Files (*.txt);;CSV Files (*.csv);;MCA Files (*.mca);;HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        
        if not file_path:
            return
        try:
            spectrum = self.io_handler.load_spectrum(file_path)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Spectrum",
                f"Failed to load spectrum:\n{str(e)}"
            )
            return

        self.session.set_spectrum(spectrum, path=file_path)
        try:
            self.spectrum_widget.set_spectrum(spectrum)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Displaying Spectrum",
                f"Loaded the file but could not plot it:\n{str(e)}"
            )
            return

        meta = getattr(spectrum, "metadata", None)
        if isinstance(meta, dict) and meta:
            try:
                self.element_panel.update_from_spectrum_metadata(meta)
            except Exception:
                pass

        try:
            self.refresh_tube_guides()
        except Exception:
            pass
        self.status_bar.showMessage(f"Loaded: {file_path}", 5000)
        self._refresh_spectrum_title()
        self.spectrum_window.show_view("Spectrum")

    def open_spectrum_as_overlay(self):
        """Add a spectrum to the plot without replacing the current one."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Spectrum as Overlay",
            "",
            "All Supported (*.txt *.csv *.mca *.h5 *.hdf5);;Text Files (*.txt);;CSV Files (*.csv);;MCA Files (*.mca);;HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        if not file_path:
            return
        try:
            spectrum = self.io_handler.load_spectrum(file_path)
            from pathlib import Path as _Path
            name = _Path(file_path).stem
            if getattr(spectrum, "metadata", None) is not None:
                spectrum.metadata.setdefault("name", name)
            self.spectrum_widget.add_overlay(spectrum, name=name)
            self.status_bar.showMessage(f"Overlay: {name}", 5000)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Overlay",
                f"Failed to load spectrum:\n{str(e)}",
            )

    def open_ipj_project(self):
        """Open an INCA/XGT .ipj mapping project in the Mapping tab."""
        self.tab_widget.setCurrentWidget(self.mapping_panel)
        self.mapping_panel.open_ipj()

    def merge_ipj_projects(self):
        """Merge many .ipj line/multipoint projects into one Mapping project."""
        self.tab_widget.setCurrentWidget(self.mapping_panel)
        self.mapping_panel.merge_ipjs()

    def on_batch_composition_spectrum_selected(self, _sample, names):
        """Keep batch fit plot in sync when a composition row is selected."""
        self.batch_analysis_panel.select_spectra(names)

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
        self._refresh_spectrum_title()

        if hasattr(spectrum, "metadata") and spectrum.metadata:
            meta = dict(spectrum.metadata)
            # Prefer attribute live_time when metadata omitted it
            if "live_time" not in meta and getattr(spectrum, "live_time", None):
                meta["live_time"] = float(spectrum.live_time)
            self.element_panel.update_from_spectrum_metadata(meta)
        elif getattr(spectrum, "live_time", None):
            self.element_panel.update_from_spectrum_metadata(
                {"live_time": float(spectrum.live_time)}
            )

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
        self.refresh_tube_guides()
        self.status_bar.showMessage(
            f"Loaded mapping spectrum into Analysis: {path_label or 'spectrum'}",
            5000,
        )

    def on_mapping_spectra_compare(self, payloads):
        """Overlay several Proj Data spectra in Analysis."""
        if not payloads:
            return
        spec, peaks, _name = payloads[0]
        self.on_mapping_spectrum_sent(spec, peaks)
        self.spectrum_widget.clear_overlays()
        for spec_i, _peaks, name_i in payloads[1:]:
            self.spectrum_widget.add_overlay(spec_i, name=name_i)
        n = len(payloads)
        self.status_bar.showMessage(
            f"Comparing {n} spectra in Analysis",
            6000,
        )

    def on_mapping_spectra_to_batch(self, pairs, *, replace=False, switch_tab=True):
        """Queue Mapping/IPJ spectra in Batch Analysis."""
        added = self.batch_analysis_panel.add_spectra(pairs, replace=replace)
        if added == 0:
            self.status_bar.showMessage("Those spectra are already in Batch Analysis", 4000)
            if switch_tab:
                self.tab_widget.setCurrentWidget(self.batch_analysis_panel)
            return
        self.batch_analysis_panel._update_settings_summary()
        if switch_tab:
            self.tab_widget.setCurrentWidget(self.batch_analysis_panel)
        n = self.batch_analysis_panel.file_list.count()
        noun = "spectrum" if added == 1 else "spectra"
        self.status_bar.showMessage(
            f"Queued {added} {noun} in Batch Analysis "
            f"({n} total). Identify elements in Analysis, then Process All.",
            8000,
        )

    def on_mapping_project_loaded(self, project):
        """After any IPJ load: push acquisition settings into Analysis.

        Spectra-only projects also jump to Analysis and queue Batch.
        """
        if project is None:
            return
        try:
            self.mapping_panel._copy_sample_to_analysis(quiet=True)
        except Exception:
            pass
        if not project.is_spectra_only():
            return
        points = project.point_spectra()
        if not points:
            points = project.all_spectra()
        if not points:
            return
        pairs = self.mapping_panel._pairs_for_batch(points)
        self.on_mapping_spectra_to_batch(pairs, replace=True, switch_tab=False)
        first = points[0]
        self.on_mapping_spectrum_sent(first.spectrum, first.peak_labels)
        n = len(points)
        self.status_bar.showMessage(
            f"Spectra-only project: {n} point{'s' if n != 1 else ''} queued for "
            "Batch. Identify elements here, then Process All in Batch Analysis.",
            12000,
        )
    
    def export_results(self):
        """Export semi-quant relative intensities from the Results tab."""
        results = self.results_panel.get_results()
        if not results:
            QMessageBox.information(
                self,
                "Export Results",
                "No semi-quant results to export. Run Semi-Quant on the Results tab.\n\n"
                "To export FP wt%, use File → Export FP Composition (wt%) "
                "or the Export wt% button on Composition.",
            )
            return
        self._export_result_rows(
            results,
            title="Export Results",
            default_name="semi_quant.csv",
        )

    def export_report(self):
        """Export a self-contained HTML analysis report."""
        from webbrowser import open as open_browser

        from core.report import write_report
        from ui.report_dialog import ReportDialog, context_from_main_window

        context = context_from_main_window(self)
        composition_state = {}
        batch = getattr(self, "batch_analysis_panel", None)
        comp = getattr(batch, "composition_panel", None) if batch is not None else None
        if comp is not None and hasattr(comp, "capture_state"):
            composition_state = comp.capture_state()
        dialog = ReportDialog(self, context=context, composition_state=composition_state)
        if dialog.exec() != QDialog.Accepted:
            return
        path = dialog.save_path()
        if not path:
            return
        if not path.lower().endswith(".html"):
            path = path + ".html"
        try:
            write_report(path, context, dialog.options())
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export Report",
                f"Failed to write the report:\n{exc}",
            )
            return
        self.status_bar.showMessage(f"Wrote report: {path}", 8000)
        if dialog.open_after():
            open_browser(Path(path).resolve().as_uri())

    def export_fp_results(self):
        """Export FP wt% from the Composition tab."""
        results = self.results_panel.get_fp_results()
        if not results:
            QMessageBox.information(
                self,
                "Export FP Composition",
                "No FP wt% results to export. Run FP Composition on the Composition tab.",
            )
            return
        rows = []
        for item in results:
            role = item.get("role")
            rows.append(
                {
                    "Element": item.get("element"),
                    "wt%": item.get("concentration"),
                    "Source": "assumed" if role == "assumed" else "measured",
                    "Line": item.get("line", ""),
                }
            )
        self._export_result_rows(
            rows,
            title="Export FP Composition (wt%)",
            default_name="fp_composition_wt.csv",
        )

    def _export_result_rows(self, rows, *, title, default_name):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            title,
            default_name,
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)",
        )
        if not file_path:
            return
        try:
            self.io_handler.export_results(rows, file_path)
            self.status_bar.showMessage(f"Exported: {file_path}", 5000)
        except Exception as e:
            QMessageBox.critical(
                self,
                title,
                f"Failed to export results:\n{str(e)}",
            )
    
    def fit_spectrum(self, checked=False, *, auto_find_peaks=None):
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
            peak_shape = fit_params['peak_shape']
            
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
                auto_find_peaks=(
                    fit_params.get('auto_find_peaks', True)
                    if auto_find_peaks is None else bool(auto_find_peaks)
                ),
                tube_element=fit_params.get('tube_element', 'Rh'),
                excitation_kv=fit_params.get('excitation_kv', 50.0),
                include_tube_lines=fit_params.get('include_tube_lines', True),
                include_compton=fit_params.get('include_compton', True),
                scatter_angle_deg=fit_params.get(
                    'scatter_angle_deg', DEFAULT_SCATTER_ANGLE_DEG
                ),
                compton_fwhm_kev=fit_params.get('compton_fwhm_kev', 0.500),
                sample_contains_tube_element=fit_params.get(
                    'sample_contains_tube_element', False
                ),
                experimental_params=exp_params,
                prominence_percent=fit_params.get('prominence_percent'),
                min_height=fit_params.get('min_height'),
                min_separation_ev=fit_params.get('min_separation_ev'),
                peak_positions=peak_positions,
                grouped_lines=fit_params.get('grouped_lines', True),
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
            
            # Relative semi-quant only. CRM curves are an explicit Standards wt% action.
            exp_params = self.element_panel.get_experimental_params()
            concentrations = self.fitter.quantify_elements(
                self.fit_result.peaks,
                exp_params,
                tube_element=fit_params.get('tube_element', 'Rh'),
                sample_contains_tube_element=fit_params.get(
                    'sample_contains_tube_element', False
                ),
            )
            method = "semi_quant_area"
            method_label = None
            concentrations, method_label = self._apply_ree_quant_roles(
                concentrations, method_label
            )
            self.session.set_concentrations(concentrations, method=method)
            self.results_panel.set_fp_live(False)
            self.results_panel.set_formula_summary("")
            self.results_panel.set_quantification(concentrations)
            if method_label:
                self.results_panel.set_method_label(method_label)

            identified = []
            seen = set()
            for peak in self.fit_result.peaks:
                if peak.is_tube_line or not peak.element:
                    continue
                if peak.element in seen:
                    continue
                seen.add(peak.element)
                identified.append(peak.element)

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
            warn = getattr(self.fitter, 'last_compton_warning', None)
            if warn:
                fit_msg += f" — {warn}"
            if identified:
                fit_msg += (
                    f"; {len(identified)} fitted element"
                    f"{'s' if len(identified) != 1 else ''}"
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
                scatter_angle_deg=fit_params.get(
                    'scatter_angle_deg', DEFAULT_SCATTER_ANGLE_DEG
                ),
                compton_fwhm_kev=fit_params.get('compton_fwhm_kev', 0.500),
                sample_contains_tube_element=fit_params.get(
                    'sample_contains_tube_element', False
                ),
                prominence_percent=fit_params.get('prominence_percent'),
                min_height=fit_params.get('min_height'),
                min_separation_ev=fit_params.get('min_separation_ev'),
            )

            id_summary = []
            identified = []
            if fit_params.get('auto_id_after_peak_find', True):
                energy = self.current_spectrum.energy
                preview_peaks, identified, id_summary = auto_id_peak_positions(
                    preview_peaks,
                    excitation_kv=fit_params.get('excitation_kv', 50.0),
                    energy_min=float(energy[0]),
                    energy_max=float(energy[-1]),
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
                    if p.get('inferred'):
                        tag += " [expected]"
                    lines.append(
                        f"{p['energy']:.3f} keV  {p['element']} {p['line']}{tag}"
                    )
                else:
                    lines.append(f"{p['energy']:.3f} keV  (unknown)")
            header = (
                f"Peak find ({len(preview_peaks)} total: "
                f"{n_labeled} labeled, {n_unknown} unknown)"
            )
            warn = getattr(self.fitter, 'last_compton_warning', None)
            if warn:
                header += f"\n{warn}"
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
            if warn:
                msg = f"{msg} — {warn}"
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
            fit_params = self.element_panel.get_fitting_params()
            concentrations = self.fitter.quantify_elements(
                self.fit_result.peaks,
                exp_params,
                tube_element=fit_params.get('tube_element', 'Rh'),
                sample_contains_tube_element=fit_params.get(
                    'sample_contains_tube_element', False
                ),
            )
            method = "semi_quant_area"
            method_label = None
            concentrations, method_label = self._apply_ree_quant_roles(
                concentrations, method_label
            )
            self.session.set_concentrations(concentrations, method=method)
            self.results_panel.set_fp_live(False)
            self.results_panel.set_formula_summary("")
            self.results_panel.set_quantification(concentrations)
            if method_label:
                self.results_panel.set_method_label(method_label)
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

    def quantify_standards(self, checked=False, *, elements=None, _retried=False):
        """Targeted CRM-curve wt% for Report-marked elements."""
        from core.standards_calibration import (
            condition_warnings,
            recipe_mismatches,
        )

        if self.fit_result is None or not getattr(self.fit_result, "peaks", None):
            QMessageBox.warning(
                self,
                "No Fit Results",
                "Please fit a spectrum first before running Standards wt%.",
            )
            return

        cal = self._active_standards_curves()
        if cal is None:
            QMessageBox.information(
                self,
                "No CRM curves",
                "No Report-marked standards curves are active.\n\n"
                "Fit and Apply a calibration on Calibration → Standards, "
                "then tick Report for the elements you want to quantify.",
            )
            return

        fit_params = self.element_panel.get_fitting_params()
        exp_params = self.element_panel.get_experimental_params()
        fit_params = {
            **fit_params,
            "tube_current": exp_params.get("tube_current"),
            "excitation_kv": (
                fit_params.get("excitation_kv")
                or exp_params.get("excitation_energy")
            ),
        }
        mismatches = recipe_mismatches(cal.fit_settings, fit_params)
        if mismatches and not _retried:
            reply = QMessageBox.question(
                self,
                "Fit recipe does not match calibration",
                "CRM intensities were extracted with a different recipe:\n\n"
                + "\n".join(f"  • {m}" for m in mismatches)
                + "\n\nRe-fit this spectrum with the calibration recipe "
                "before applying the curves?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self.element_panel.apply_fit_recipe(cal.fit_settings)
            self.fit_spectrum(auto_find_peaks=False)
            if self.fit_result is None:
                return
            self.quantify_standards(elements=elements, _retried=True)
            return

        warnings = condition_warnings(cal.fit_settings, fit_params)
        if warnings:
            QMessageBox.warning(
                self,
                "Instrument conditions differ",
                "Tube settings differ from the CRM fit. "
                "All targeted wt% will scale with intensity.\n\n"
                + "\n".join(f"  • {w}" for w in warnings),
            )

        if hasattr(self, "analysis_left_tabs"):
            self.analysis_left_tabs.setCurrentIndex(self.TAB_RESULTS)

        targets = elements
        if targets is None:
            targets = [c.element for c in cal.fitted_curves()]
        if not targets:
            QMessageBox.information(
                self,
                "Nothing to report",
                "No fitted curves are marked Report.",
            )
            return

        self.status_bar.showMessage("Computing targeted CRM wt%...", 0)
        try:
            calibrated, method_label = self._quantify_with_standards(
                self.fit_result.peaks,
                self.fit_result,
                fit_params,
                elements=targets,
            )
            if not calibrated:
                QMessageBox.warning(
                    self,
                    "No CRM wt%",
                    "None of the Report-marked elements have fitted sample "
                    "peaks that match the curve line group.",
                )
                self.status_bar.showMessage("Standards wt%: no matching peaks", 5000)
                return
            calibrated, method_label = self._apply_ree_quant_roles(
                calibrated, method_label
            )
            self.session.set_concentrations(calibrated, method="standards_curve")
            self.results_panel.set_fp_live(False)
            self.results_panel.set_formula_summary("")
            self.results_panel.set_quantification(calibrated)
            if method_label:
                self.results_panel.set_method_label(method_label)
            n = len(calibrated)
            n_nq = sum(1 for v in calibrated.values() if v.get("not_quantified"))
            msg = (
                f"CRM wt% for {n} element{'s' if n != 1 else ''} "
                f"(independent determinations, not a closed assay)"
            )
            if n_nq:
                msg += f"; {n_nq} detected, not quantified"
            self.status_bar.showMessage(msg, 8000)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Standards wt% Error",
                f"An error occurred during CRM quantification:\n{str(e)}",
            )
            self.status_bar.showMessage("Standards wt% failed", 5000)

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
            self.analysis_left_tabs.setCurrentIndex(self.TAB_COMPOSITION)

        assumptions = self.results_panel.get_matrix_assumptions()
        self.session.matrix = assumptions
        if not live:
            self.status_bar.showMessage("Computing FP composition...", 0)
        try:
            exp_params = self.element_panel.get_experimental_params()
            fit_params = self.element_panel.get_fitting_params()
            result = quantify_from_peaks(
                self.fit_result.peaks,
                assumptions,
                exp_params,
                tube_element=fit_params.get('tube_element', 'Rh'),
                sample_contains_tube_element=fit_params.get(
                    'sample_contains_tube_element', False
                ),
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
            if not live:
                self.results_panel.set_fp_line_warnings(
                    getattr(result, "line_warnings", None)
                )
            bits = [f"As compounds: {result.formula_summary()}"]
            if result.residual < float("inf"):
                bits.append(
                    f"intensity residual {result.residual:.4f} "
                    f"({result.iterations} iter)"
                )
            bits.append(f"measured cations {result.measured_cation_pct:.1f} %")
            self.results_panel.set_formula_summary(
                "    |  ".join(bits),
                empirical=empirical_formula(
                    result.element_wt, formula_wt=result.formula_wt
                ),
            )
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
            f"<p>Version {QApplication.applicationVersion()}</p>"
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
        if self._ree_fit_set is None:
            return
        selected = {
            e.get("symbol") for e in (elements or []) if e.get("symbol")
        }
        if not selected.intersection(self._ree_fit_set.ree_set):
            self._ree_fit_set = None

    def select_rees_and_overlaps(self, checked=False):
        """Select Y, La–Lu plus spectroscopic / calibration overlaps."""
        from core.ree_targets import build_ree_fit_set, clusters_from_calibration

        fit_params = self.element_panel.get_fitting_params()
        kv = float(fit_params.get("excitation_kv", 50.0) or 50.0)
        extra = []
        cal = getattr(self.session.instrument, "standards_calibration", None)
        if cal is not None:
            extra = clusters_from_calibration(cal)
        fit_set = build_ree_fit_set(excitation_kv=kv, extra_clusters=extra)
        self._ree_fit_set = fit_set
        self.element_panel.set_selected_elements(fit_set.symbols)
        if hasattr(self, "analysis_left_tabs"):
            self.analysis_left_tabs.setCurrentIndex(self.TAB_ELEMENTS)
        self.status_bar.showMessage(
            f"REE suite: {fit_set.summary()} ({kv:g} kV)", 8000
        )
        return fit_set

    def quantify_rees(self, checked=False):
        """Fit only REEs + overlaps and quantify from the standards curves."""
        self.select_rees_and_overlaps()
        if self.current_spectrum is None:
            QMessageBox.information(
                self,
                "REEs + overlaps selected",
                "Selected the REE suite and overlap partners.\n\n"
                "Load a spectrum, then Fit REEs + overlaps again to quantify.",
            )
            return
        # Unknown auto-find peaks in the L-line region steal REE area.
        self.fit_spectrum(auto_find_peaks=False)
        fit_set = self._ree_fit_set
        self.quantify_standards(
            elements=list(fit_set.symbols) if fit_set else None,
        )

    def _apply_ree_quant_roles(self, concentrations, method_label):
        """Tag REE vs overlap rows when the current selection is a REE suite."""
        from core.ree_targets import apply_ree_roles

        fit_set = self._ree_fit_set
        if not fit_set or not concentrations:
            return concentrations, method_label
        tagged = apply_ree_roles(concentrations, fit_set)
        n_ree = sum(1 for v in tagged.values() if v.get("role") == "ree")
        n_ov = sum(1 for v in tagged.values() if v.get("role") == "overlap")
        extra = f"REE suite: {n_ree} REE(s)"
        if n_ov:
            extra += f" + {n_ov} overlap partner(s)"
        if method_label:
            method_label = f"{method_label} — {extra}"
        else:
            method_label = f"Method: {extra}"
        return tagged, method_label

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
        self._show_element_emission_lines(symbol)

    def on_identify_preview_element(self, symbol, z):
        """Show emission lines for the candidate highlighted in the identify list."""
        self._show_element_emission_lines(symbol, z)

    def _show_element_emission_lines(self, symbol, z=None):
        """Overlay an element's emission lines on the spectrum (always show)."""
        if z is None:
            from core.advanced_peak_fitting import get_element_z
            z = get_element_z(symbol)
        if not z:
            return
        self.spectrum_widget.clear_peak_markers()
        self.spectrum_widget.show_element_lines(symbol, int(z))
        self._displayed_element_lines = symbol
        self.status_bar.showMessage(f"Showing emission lines for {symbol} (Z={z})", 3000)
    
    def on_element_clicked(self, symbol, z):
        """Handle element click — show emission lines, or clear if already shown"""
        if self._displayed_element_lines == symbol:
            self.spectrum_widget.clear_peak_markers()
            self._displayed_element_lines = None
            self.status_bar.showMessage(f"Cleared emission lines for {symbol}", 3000)
            return
        self._show_element_emission_lines(symbol, z)

    def fit_scatter_angle_from_spectrum(self):
        """Fit the tube→sample→detector angle from the Compton Kα hump."""
        from core.xray_data import estimate_scatter_angle

        spectrum = self.current_spectrum
        if spectrum is None:
            QMessageBox.warning(
                self, "No Spectrum",
                "Load a spectrum first — ideally a blank or low-Z sample where "
                "the Compton hump is strong.",
            )
            return

        fit = self.element_panel.get_fitting_params()
        tube = fit.get('tube_element') or 'Rh'
        kv = float(fit.get('excitation_kv') or fit.get('excitation_energy') or 50.0)
        result = estimate_scatter_angle(
            spectrum.energy, spectrum.counts,
            tube_element=tube, excitation_kv=kv,
        )
        if result is None:
            QMessageBox.information(
                self, "Compton Hump Not Found",
                f"Could not locate a credible {tube} Compton Kα hump.\n\n"
                f"Check that the tube kV ({kv:g} keV) is above the {tube} K edge, "
                f"that the spectrum reaches ~19 keV, and that the sample is a "
                f"blank / low-Z scatterer. Heavy-matrix samples and fluorescence "
                f"lines near 18–20 keV (Mo Kβ, Zn pile-up) can hide the hump.",
            )
            return

        angle = result['angle_deg']
        fwhm_kev = result['fwhm_kev']
        self.element_panel.set_compton_geometry(
            scatter_angle_deg=angle, compton_fwhm_kev=fwhm_kev
        )
        self.status_bar.showMessage(
            f"Scatter angle fitted from {tube} Compton Kα at "
            f"{result['centroid_kev']:.3f} keV: θ = {angle:.0f}°, "
            f"Compton FWHM = {fwhm_kev*1000:.0f} eV (SNR {result['snr']:.0f})",
            10000,
        )
        self.refresh_tube_guides()

    def refresh_tube_guides(self):
        """Update faint tube-line bands on the Analysis spectrum plot."""
        try:
            self._refresh_tube_guides()
        except Exception:
            return

    def _refresh_tube_guides(self):
        from core.xray_data import build_tube_guide_regions
        import numpy as np
        panel = self.element_panel
        show = True
        if hasattr(panel, "show_tube_guides_check"):
            show = bool(panel.show_tube_guides_check.isChecked())

        e_min = e_max = None
        spectrum = getattr(self.spectrum_widget, "spectrum_data", None)
        energy = getattr(spectrum, "energy", None) if spectrum is not None else None
        if energy is not None:
            arr = np.asarray(energy).reshape(-1)
            if arr.size:
                e_min = float(np.nanmin(arr))
                e_max = float(np.nanmax(arr))

        fit = panel.get_fitting_params() if hasattr(panel, "get_fitting_params") else {}
        # Guides follow the tube anode / kV even if "Include Tube Lines" is off
        # for fitting — the overlay is a reference for element ID. Compton
        # bands honor the Compton checkbox (and need Include Tube for θ/FWHM).
        include_compton = bool(fit.get("include_compton", True))
        if hasattr(panel, "compton_check") and hasattr(panel, "tube_lines_check"):
            include_compton = (
                panel.tube_lines_check.isChecked() and panel.compton_check.isChecked()
            )

        regions = build_tube_guide_regions(
            tube_element=fit.get("tube_element", "Rh"),
            excitation_kv=float(fit.get("excitation_kv", 20.0)),
            include_compton=include_compton,
            scatter_angle_deg=float(
                fit.get("scatter_angle_deg", DEFAULT_SCATTER_ANGLE_DEG)
            ),
            compton_fwhm_kev=float(fit.get("compton_fwhm_kev", 0.500)),
            energy_min=e_min,
            energy_max=e_max,
        )
        self.spectrum_widget.set_tube_guides(regions, show=show)
    
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
                f"FWHM calibration applied (core widths locked for all peak shapes): "
                f"FWHM₀={fwhm_0_ev:.1f} eV, "
                f"ε={epsilon_ev:.2f} eV/keV (R²={fwhm_calibration.r_squared:.4f}). "
                f"Tail-Gaussian / Hypermet tails stay free.",
                8000
            )
        else:
            self.status_bar.showMessage(
                f"FWHM calibration applied (core widths locked for all peak shapes): "
                f"{fwhm_calibration.model_type} model "
                f"(R²={fwhm_calibration.r_squared:.4f}). "
                f"Tail-Gaussian / Hypermet tails stay free.",
                8000
            )

        self._refresh_calibration_status()

    def on_tube_profiles_changed(self, library):
        """Apply per-kV tube profile library to Analysis fitting."""
        self.session.instrument.tube_profile_library = library
        self.session.apply_instrument_to_fitter(self.fitter)
        self.batch_analysis_panel.set_instrument_state(self.session.instrument)
        self.element_panel.update_tube_profile_status(library)
        if hasattr(self, "standards_panel"):
            self.standards_panel.set_tube_profile_library(library)
        if SHOW_TUBE_PROFILE_CALIBRATION:
            n_meas = sum(1 for p in library.profiles.values() if p.source == 'measured')
            self.status_bar.showMessage(
                f"Tube profiles active: {n_meas} measured / "
                f"{len(library.available_kvs)} modes "
                f"({library.tube_element})",
                5000
            )
        self._refresh_calibration_status()
    
    def on_calibration_applied(self, calibration_result, *, switch_tab=True):
        """Handle standards calibration being applied"""
        from core.standards_calibration import StandardsCalibration

        self.session.instrument.standards_calibration = calibration_result
        if isinstance(calibration_result, StandardsCalibration):
            curves = calibration_result.fitted_curves()
            elements = ", ".join(c.element for c in curves)
            self.status_bar.showMessage(
                f"{len(curves)} CRM curve(s) ready for targeted wt% "
                f"({elements})",
                8000,
            )
        else:
            self.status_bar.showMessage(
                f"Legacy standards calibration stored: "
                f"FWHM₀={calibration_result.fwhm_0*1000:.1f} eV, "
                f"ε={calibration_result.epsilon*1000:.2f} eV",
                5000
            )
        
        if switch_tab:
            self.tab_widget.setCurrentIndex(0)
        self._refresh_calibration_status()

    def _active_standards_curves(self):
        """Return the StandardsCalibration with usable curves, or None."""
        from core.standards_calibration import StandardsCalibration

        cal = self.session.instrument.standards_calibration
        if isinstance(cal, StandardsCalibration) and cal.fitted_curves():
            return cal
        return None

    def _quantify_with_standards(
        self, peaks, fit_result=None, fit_params=None, *, elements=None
    ):
        """
        Targeted CRM-curve wt% for Report-marked (or caller-listed) elements.

        Returns (concentrations, method_label) or (None, None) when no
        curve-based calibration is active.
        """
        from core.standards_calibration import condition_warnings, recipe_mismatches

        cal = self._active_standards_curves()
        if cal is None or self.current_spectrum is None:
            return None, None
        mismatches = recipe_mismatches(cal.fit_settings, fit_params)
        warnings = condition_warnings(cal.fit_settings, fit_params)
        spectrum = self.current_spectrum
        try:
            concentrations = cal.quantify(
                peaks,
                spectrum.live_time,
                real_time=spectrum.real_time,
                fit_result=fit_result,
                energy=spectrum.energy,
                elements=elements,
            )
        except Exception as exc:
            print(f"Standards quantification failed: {exc}")
            return None, None
        if not concentrations:
            return None, None
        names = ", ".join(concentrations.keys())
        label = (
            f"CRM curve wt% for {names} — "
            f"independent determinations, not a closed assay"
        )
        nq = [
            e for e, v in concentrations.items() if v.get("not_quantified")
        ]
        extra = [
            e for e, v in concentrations.items()
            if v.get("quant_flag") == "extrapolated"
        ]
        if nq:
            label += f"; detected, not quantified: {', '.join(nq)}"
        if extra:
            label += f"; extrapolated: {', '.join(extra)}"
        if mismatches:
            label += (
                "; ⚠ extract recipe still differs from calibration ("
                + "; ".join(mismatches)
                + ")"
            )
        if warnings:
            label += "; ⚠ " + "; ".join(warnings)
        return concentrations, label
    
    def closeEvent(self, event):
        """Quit the whole app: hidden figure windows used to keep Python alive."""
        self._stop_background_work()
        for window in (
            getattr(self, "spectrum_window", None),
            getattr(self, "map_window", None),
            getattr(self, "charts_window", None),
        ):
            if window is not None:
                window.shutdown()
        self._save_settings()
        event.accept()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _stop_background_work(self) -> None:
        """Stop QThreads so they cannot pin the process after exec() returns."""
        for panel in (
            getattr(self, "standards_panel", None),
            getattr(self, "batch_analysis_panel", None),
            getattr(self, "fwhm_calibration_panel", None),
            getattr(self, "calibration_panel", None),
        ):
            if panel is None:
                continue
            worker = getattr(panel, "worker", None)
            if worker is None:
                continue
            stop = getattr(worker, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
            try:
                worker.requestInterruption()
                if worker.isRunning() and not worker.wait(4000):
                    worker.terminate()
                    worker.wait(1000)
            except RuntimeError:
                pass
            try:
                panel.worker = None
            except Exception:
                pass
