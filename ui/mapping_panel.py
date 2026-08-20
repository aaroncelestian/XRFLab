"""
Mapping workspace: load INCA/XGT .ipj projects.

Maps tab: element maps, RGB, correlations / correlation matrix, display
enhancement, PCA / ratio / particle analysis, and drawn intensity profiles.
Line scan tab: collected line / multipoint spectra, ROI profiles, and
area-normalized semi-quant along that series.

Spectra can be sent to Analysis (one at a time) or Batch Analysis
(selected subset, whole line scan, site, or project). Spectra-only IPJ
files open in Analysis with every point queued for batch fitting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.mapping.camera import (
    StageCamera,
    camera_from_image,
    camera_from_sample_sites,
    locate_image_crop,
    locate_red_map_rect,
    locate_scaled_template,
)
from core.mapping.correlations import (
    map_correlation,
    map_correlation_matrix,
    rgb_composite,
)
from core.mapping.display import (
    colorize_map,
    difference_map,
    embed_map_on_photo,
    enhance_map,
    format_acquisition,
    overlay_alpha,
    overlay_on_photo,
    ratio_map,
    upsample_map,
)
from core.mapping.models import (
    ElementMap,
    LineScan,
    MappingFOV,
    MappingProject,
    MapSpectrum,
    coerce_element_symbols,
)
from core.mapping.multivariate import (
    find_particles,
    particle_label_map_as_element,
    pca_element_maps,
)
from core.mapping.profiles import (
    extract_cube_element_profiles,
    extract_multi_element_profiles,
)
from ui.collapsible_section import CollapsibleSection
from ui.map_canvas import MapCanvas
from ui.pixel_spectrum_popup import PixelSpectrumPopup


def _diverging_lut():
    """Blue–white–red lookup table for r in [-1, 1]."""
    lut = np.zeros((256, 3), dtype=np.ubyte)
    for i in range(128):
        t = i / 127.0
        lut[i] = (int(30 + 225 * t), int(70 + 185 * t), int(180 + 75 * t))
    for i in range(128, 256):
        t = (i - 128) / 127.0
        lut[i] = (int(255), int(255 - 180 * t), int(255 - 200 * t))
    return lut


class MappingPanel(QWidget):
    """Top-level Mapping tab widget."""

    spectrum_send_requested = Signal(object, object)  # Spectrum, peak_labels list
    spectra_batch_requested = Signal(list)  # [(display_name, Spectrum), ...]
    project_loaded = Signal(object)  # MappingProject
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[MappingProject] = None
        self.current_fov: Optional[MappingFOV] = None
        self._fitter = None
        self._element_panel = None
        self._quant_distances: Optional[np.ndarray] = None
        self._quant_table = None  # list of dicts
        self._last_ls_profiles = None  # collected line-scan ROI profiles
        self._picked_spectrum: Optional[MapSpectrum] = None
        self._last_line: Optional[tuple] = None  # (x0, y0, x1, y1)
        self._pixel_popup: Optional[PixelSpectrumPopup] = None
        self._checked_map_names: set[str] = set()
        self._checked_roi_symbols: set[str] = set()
        self._active_line_scan: Optional[LineScan] = None
        self._drawn_line_scan: Optional[LineScan] = None
        self._drawn_cache_key = None
        self._profile_source: Optional[str] = None  # "drawn" | "ipj"
        self._tree_updating = False
        self._last_pick_xy: Optional[tuple] = None
        self._sample_form_updating = False
        self._ls_hover_index: Optional[int] = None
        self._ls_cam_px: Optional[np.ndarray] = None
        self._ls_cam_py: Optional[np.ndarray] = None
        self._ls_plot_x: Optional[np.ndarray] = None
        self._ls_plot_order: Optional[np.ndarray] = None
        self._ls_camera_model: Optional[StageCamera] = None
        self._cam_dest_rect_cache: dict = {}
        self._particle_result = None  # ParticleResult | None
        self._matrix_names: list[str] = []

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        # Left column: fixed header + scrollable collapsible tools
        left = QWidget()
        left.setMinimumWidth(260)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        btn_row = QHBoxLayout()
        self.open_btn = QPushButton("Open IPJ…")
        self.open_btn.clicked.connect(self.open_ipj)
        btn_row.addWidget(self.open_btn)
        self.merge_btn = QPushButton("Merge IPJs…")
        self.merge_btn.setToolTip(
            "Merge line scans / multipoint series from many .ipj files\n"
            "into one project. Spectra are named:\n"
            "filename_sample_site_spectrum"
        )
        self.merge_btn.clicked.connect(self.merge_ipjs)
        btn_row.addWidget(self.merge_btn)
        self.sample_info_btn = QPushButton("Sample info…")
        self.sample_info_btn.setToolTip(
            "Project, sample, site, and acquisition metadata.\n"
            "Kept off the main column so map tools stay visible."
        )
        self.sample_info_btn.clicked.connect(self._show_sample_info)
        btn_row.addWidget(self.sample_info_btn)
        left_layout.addLayout(btn_row)

        self.active_site_label = QLabel("Active site: —")
        self.active_site_label.setWordWrap(True)
        left_layout.addWidget(self.active_site_label)

        self.nav_tabs = QTabWidget()
        self.nav_tabs.setMinimumHeight(240)
        self.nav_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Sites tab: Project → Sample → Site of Interest
        sites_page = QWidget()
        sites_layout = QVBoxLayout(sites_page)
        sites_layout.setContentsMargins(0, 0, 0, 0)
        self.sites_tree = QTreeWidget()
        self.sites_tree.setHeaderLabels(["Sites", "Data"])
        self.sites_tree.setColumnCount(2)
        self.sites_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sites_tree.setEditTriggers(QAbstractItemView.EditKeyPressed)
        self.sites_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sites_tree.setAllColumnsShowFocus(True)
        sites_header = self.sites_tree.header()
        sites_header.setStretchLastSection(False)
        sites_header.setSectionResizeMode(0, QHeaderView.Stretch)
        sites_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.sites_tree.customContextMenuRequested.connect(self._on_sites_context_menu)
        self.sites_tree.itemSelectionChanged.connect(self._on_sites_selection)
        self.sites_tree.itemDoubleClicked.connect(self._on_site_activated)
        self.sites_tree.itemChanged.connect(self._on_sites_tree_item_changed)
        sites_layout.addWidget(self.sites_tree)
        sites_hint = QLabel(
            "Data column: maps / line / multi. F2 rename · Double-click to activate"
        )
        sites_hint.setWordWrap(True)
        sites_hint.setStyleSheet("color: #666; font-size: 11px;")
        sites_layout.addWidget(sites_hint)
        activate_row = QHBoxLayout()
        self.activate_site_btn = QPushButton("Activate site")
        self.activate_site_btn.setToolTip(
            "Set the selected Site of Interest as active for maps and tools"
        )
        self.activate_site_btn.clicked.connect(self._activate_selected_site)
        activate_row.addWidget(self.activate_site_btn)
        self.rename_site_btn = QPushButton("Rename…")
        self.rename_site_btn.setToolTip("Rename the selected sample or site (also F2)")
        self.rename_site_btn.clicked.connect(self._rename_selected_sites_item)
        activate_row.addWidget(self.rename_site_btn)
        sites_layout.addLayout(activate_row)
        self.nav_tabs.addTab(sites_page, "Sites")

        # Data tab: contents of the active site (SmartMap, images, spectra)
        data_page = QWidget()
        data_layout = QVBoxLayout(data_page)
        data_layout.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Data"])
        self.tree.setEditTriggers(QAbstractItemView.EditKeyPressed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_data_context_menu)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        self.tree.itemChanged.connect(self._on_data_tree_item_changed)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        data_layout.addWidget(self.tree)
        data_hint = QLabel(
            "Shift/Ctrl-click to select spectra · F2 to rename · "
            "Send selected → Batch for bulk fitting"
        )
        data_hint.setWordWrap(True)
        data_hint.setStyleSheet("color: #666; font-size: 11px;")
        data_layout.addWidget(data_hint)
        data_btn_row = QHBoxLayout()
        self.rename_data_btn = QPushButton("Rename…")
        self.rename_data_btn.setToolTip("Rename selected spectrum, line scan, or site")
        self.rename_data_btn.clicked.connect(self._rename_selected_data_item)
        data_btn_row.addWidget(self.rename_data_btn)
        self.select_all_spectra_btn = QPushButton("Select all spectra")
        self.select_all_spectra_btn.setToolTip(
            "Select every spot and line-scan point in this site"
        )
        self.select_all_spectra_btn.clicked.connect(self._select_all_spectra)
        data_btn_row.addWidget(self.select_all_spectra_btn)
        self.send_selected_batch_btn = QPushButton("Send selected → Batch")
        self.send_selected_batch_btn.setToolTip(
            "Queue the selected spectra (or the whole line scan if its "
            "folder is selected) in Batch Analysis"
        )
        self.send_selected_batch_btn.clicked.connect(self._send_selected_to_batch)
        data_btn_row.addWidget(self.send_selected_batch_btn)
        data_layout.addLayout(data_btn_row)
        self.nav_tabs.addTab(data_page, "Data")

        left_layout.addWidget(self.nav_tabs, stretch=1)

        self.sample_dialog = self._build_sample_dialog()

        # Scrollable collapsible tool sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMinimumHeight(240)

        scroll_body = QWidget()
        scroll_layout = QVBoxLayout(scroll_body)
        scroll_layout.setContentsMargins(0, 0, 4, 0)
        scroll_layout.setSpacing(8)

        # ---- Display ----
        display_sec = CollapsibleSection("Display", expanded=True)
        # Hidden view state: Data tree selection drives the canvas, not a combo.
        self.map_combo = QComboBox(self)
        self.map_combo.hide()
        self.map_combo.currentIndexChanged.connect(self._on_map_combo_changed)
        hint = QLabel(
            "Tip: check maps in the Data tree to plot them. "
            "Click a photo or Trans. x-ray there to view it."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        display_sec.addWidget(hint)
        self.rgb_check = QCheckBox("RGB composite")
        self.rgb_check.setToolTip(
            "False-color composite from the R/G/B maps below. "
            "Needs at least two different maps; otherwise it looks grayscale."
        )
        self.rgb_check.toggled.connect(self._on_rgb_toggled)
        display_sec.addWidget(self.rgb_check)

        self.overlay_check = QCheckBox("Overlay on photo")
        self.overlay_check.setToolTip(
            "Blend the current element map (or RGB composite) onto a camera "
            "photo or Trans. x-ray image. Pixels with 0 counts stay fully "
            "transparent. Pick / line tools stay on the map pixel grid."
        )
        self.overlay_check.toggled.connect(self._refresh_canvas)
        display_sec.addWidget(self.overlay_check)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Photo"))
        self.overlay_target = QComboBox()
        self.overlay_target.addItem("Map area photo", "optical")
        self.overlay_target.addItem("Trans. x-ray image", "overview")
        self.overlay_target.addItem("Sample camera", "whole_image")
        self.overlay_target.setToolTip(
            "Background for the overlay. Map area is the mapped FOV photo; "
            "Trans. x-ray is the transmission image; sample camera is the "
            "whole-sample photo."
        )
        self.overlay_target.currentIndexChanged.connect(self._refresh_canvas)
        target_row.addWidget(self.overlay_target, stretch=1)
        display_sec.addLayout(target_row)

        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("Opacity"))
        self.overlay_slider = QSlider(Qt.Horizontal)
        self.overlay_slider.setRange(0, 100)
        self.overlay_slider.setValue(45)
        self.overlay_slider.setToolTip("How strongly the XRF map tints the photo")
        self.overlay_slider.valueChanged.connect(self._on_overlay_opacity)
        overlay_row.addWidget(self.overlay_slider, stretch=1)
        self.overlay_pct = QLabel("45%")
        self.overlay_pct.setMinimumWidth(36)
        overlay_row.addWidget(self.overlay_pct)
        display_sec.addLayout(overlay_row)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Overlay color"))
        self.overlay_cmap = QComboBox()
        for key, label in (
            ("hot", "Hot"),
            ("inferno", "Inferno"),
            ("cyan", "Cyan"),
        ):
            self.overlay_cmap.addItem(label, key)
        self.overlay_cmap.setToolTip(
            "Colormap for a single-element overlay (RGB uses the composite)"
        )
        self.overlay_cmap.currentIndexChanged.connect(self._refresh_canvas)
        cmap_row.addWidget(self.overlay_cmap, stretch=1)
        display_sec.addLayout(cmap_row)

        self.overlay_mask_check = QCheckBox("Transparent where counts are low")
        self.overlay_mask_check.setChecked(True)
        self.overlay_mask_check.setToolTip(
            "Fade low-count pixels. Zero-count pixels are always fully transparent."
        )
        self.overlay_mask_check.toggled.connect(self._refresh_canvas)
        display_sec.addWidget(self.overlay_mask_check)
        self.overlay_check.setEnabled(False)
        self.overlay_target.setEnabled(False)
        self.overlay_slider.setEnabled(False)
        self.overlay_cmap.setEnabled(False)
        self.overlay_mask_check.setEnabled(False)

        self.r_combo = QComboBox()
        self.g_combo = QComboBox()
        self.b_combo = QComboBox()
        for label, combo in (("R", self.r_combo), ("G", self.g_combo), ("B", self.b_combo)):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(combo)
            display_sec.addLayout(row)
            combo.currentIndexChanged.connect(self._refresh_canvas)

        self.acq_label = QLabel("")
        self.acq_label.setWordWrap(True)
        self.acq_label.setStyleSheet("color: #444; font-size: 11px;")
        display_sec.addWidget(self.acq_label)

        enhance_hint = QLabel("Map enhancement (display only — does not change the cube)")
        enhance_hint.setWordWrap(True)
        enhance_hint.setStyleSheet("color: #666; font-size: 11px;")
        display_sec.addWidget(enhance_hint)

        self.neighborhood_combo = QComboBox()
        for size, label in (
            (1, "1×1 (single pixel)"),
            (3, "3×3 neighbors"),
            (5, "5×5 neighbors"),
            (7, "7×7 neighbors"),
        ):
            self.neighborhood_combo.addItem(label, size)
        self.neighborhood_combo.setToolTip(
            "Neighborhood used for Pick pixel spectrum (sum of neighbors) "
            "and for Mean / Median / Gaussian / Bilateral map smoothing."
        )
        self.neighborhood_combo.currentIndexChanged.connect(self._on_enhance_changed)

        self.smooth_combo = QComboBox()
        for key, label in (
            ("none", "None (raw counts)"),
            ("mean", "Mean (boxcar)"),
            ("median", "Median (despeckle)"),
            ("gaussian", "Gaussian"),
            ("bilateral", "Bilateral (edge-preserving)"),
        ):
            self.smooth_combo.addItem(label, key)
        self.smooth_combo.setToolTip(
            "Spatial filter on element maps and RGB.\n"
            "Median is best for sparse photon noise; Gaussian is smoother for figures.\n"
            "Bilateral keeps grain boundaries while reducing noise."
        )
        self.smooth_combo.currentIndexChanged.connect(self._on_enhance_changed)

        self.bin_combo = QComboBox()
        for factor, label in (
            (1, "None"),
            (2, "2×2 blocks"),
            (4, "4×4 blocks"),
        ):
            self.bin_combo.addItem(label, factor)
        self.bin_combo.setToolTip(
            "Average each N×N block, then expand back so click coordinates stay aligned."
        )
        self.bin_combo.currentIndexChanged.connect(self._on_enhance_changed)

        self.scale_combo = QComboBox()
        for key, label in (
            ("linear", "Linear"),
            ("sqrt", "Square root"),
            ("asinh", "Asinh"),
            ("log", "Log"),
        ):
            self.scale_combo.addItem(label, key)
        self.scale_combo.setToolTip(
            "Compress hot pixels so weak features show up in publication figures."
        )
        self.scale_combo.currentIndexChanged.connect(self._on_enhance_changed)

        self.contrast_combo = QComboBox()
        for key, label in (
            ("none", "None"),
            ("percentile", "Percentile stretch"),
            ("clahe", "CLAHE (adaptive)"),
            ("tophat", "Top-hat (small bright)"),
        ):
            self.contrast_combo.addItem(label, key)
        self.contrast_combo.setToolTip(
            "Draw attention to features without changing stored counts.\n"
            "CLAHE boosts local contrast; top-hat highlights small bright particles."
        )
        self.contrast_combo.currentIndexChanged.connect(self._on_enhance_changed)

        self.interp_combo = QComboBox()
        for key, label in (
            ("none", "None (blocky pixels)"),
            ("nearest", "Nearest (sharp blocks)"),
            ("bilinear", "Bilinear"),
            ("cubic", "Cubic spline"),
            ("quintic", "Quintic spline"),
        ):
            self.interp_combo.addItem(label, key)
        self.interp_combo.setCurrentIndex(3)  # cubic
        self.interp_combo.setToolTip(
            "Upsample interpolation for publication-style maps.\n"
            "Cubic spline is usually the best trade-off. "
            "Pick / line / area tools stay on the original pixel grid."
        )
        self.interp_combo.currentIndexChanged.connect(self._on_enhance_changed)

        self.upsample_combo = QComboBox()
        for factor, label in (
            (1, "1× (native)"),
            (2, "2×"),
            (4, "4×"),
            (8, "8×"),
        ):
            self.upsample_combo.addItem(label, factor)
        self.upsample_combo.setCurrentIndex(1)  # 2×
        self.upsample_combo.setToolTip(
            "Render the map on a finer grid. 4× cubic looks smooth on typical XGT maps."
        )
        self.upsample_combo.currentIndexChanged.connect(self._on_enhance_changed)

        for label, widget in (
            ("Neighborhood", self.neighborhood_combo),
            ("Smooth", self.smooth_combo),
            ("Spatial bin", self.bin_combo),
            ("Intensity", self.scale_combo),
            ("Contrast", self.contrast_combo),
            ("Interpolate", self.interp_combo),
            ("Upsample", self.upsample_combo),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(widget, stretch=1)
            display_sec.addLayout(row)

        scroll_layout.addWidget(display_sec)

        # ---- Tools ----
        tools_sec = CollapsibleSection("Tools", expanded=True)
        self.line_mode_btn = QPushButton("Draw line transect")
        self.line_mode_btn.setCheckable(True)
        self.line_mode_btn.toggled.connect(self._on_line_mode)
        tools_sec.addWidget(self.line_mode_btn)
        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("Line width"))
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 51)
        self.line_width_spin.setValue(1)
        self.line_width_spin.setSuffix(" px")
        self.line_width_spin.setToolTip(
            "Average this many neighboring pixels perpendicular to the line "
            "for a smoother profile. The yellow band on the map is that width "
            "in map pixels. 1 = center line only."
        )
        self.line_width_spin.valueChanged.connect(self._on_line_width_changed)
        width_row.addWidget(self.line_width_spin)
        tools_sec.addLayout(width_row)
        self.pick_btn = QPushButton("Pick pixel spectrum")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip(
            "Click map pixels to extract spectra into a popup viewer "
            "(stays open and updates on each click).\n"
            "Neighborhood under Display sums neighboring pixels — "
            "needed on fast, low-count maps."
        )
        self.pick_btn.toggled.connect(self._on_pick_mode)
        tools_sec.addWidget(self.pick_btn)

        tools_sec.addWidget(QLabel("Area sum spectrum"))
        area_row = QHBoxLayout()
        self.rect_btn = QPushButton("Rectangle")
        self.circle_btn = QPushButton("Circle")
        self.poly_btn = QPushButton("Polygon")
        for btn, tip in (
            (self.rect_btn, "Click two opposite corners. Hold Shift for a square."),
            (self.circle_btn, "Click center, then a point on the rim."),
            (
                self.poly_btn,
                "Click vertices. Double-click, click the first point, "
                "or right-click to close (3+ points).",
            ),
        ):
            btn.setCheckable(True)
            btn.setToolTip(tip)
            area_row.addWidget(btn)
        self.rect_btn.toggled.connect(lambda on: self._on_region_mode("rect", on))
        self.circle_btn.toggled.connect(lambda on: self._on_region_mode("circle", on))
        self.poly_btn.toggled.connect(lambda on: self._on_region_mode("poly", on))
        tools_sec.addLayout(area_row)
        self.clear_region_btn = QPushButton("Clear area")
        self.clear_region_btn.setToolTip("Remove the area outline (the summed spectrum is kept)")
        self.clear_region_btn.clicked.connect(self._clear_region)
        tools_sec.addWidget(self.clear_region_btn)

        self.clear_line_btn = QPushButton("Clear line")
        self.clear_line_btn.clicked.connect(self._clear_line)
        tools_sec.addWidget(self.clear_line_btn)

        tools_sec.addWidget(QLabel("Profile / line-scan elements"))
        self.profile_map_list = QListWidget()
        self.profile_map_list.setMinimumHeight(100)
        self.profile_map_list.setMaximumHeight(160)
        self.profile_map_list.setToolTip(
            "Checked items drive the Line profile chart:\n"
            "• Map names → intensity along the yellow drawn transect\n"
            "• Map names also → energy-window counts along an XGT line scan\n"
            "• Element ROIs → counts in that energy window along the line\n\n"
            "Elements selected in Analysis are used if nothing is checked. "
            "Use From Analysis to sync this list."
        )
        self.profile_map_list.itemChanged.connect(self._on_profile_map_check_changed)
        tools_sec.addWidget(self.profile_map_list)
        profile_btn_row = QHBoxLayout()
        self.profile_all_btn = QPushButton("All")
        self.profile_all_btn.clicked.connect(lambda: self._set_all_profile_maps(True))
        self.profile_none_btn = QPushButton("None")
        self.profile_none_btn.clicked.connect(lambda: self._set_all_profile_maps(False))
        self.profile_sync_btn = QPushButton("From Analysis")
        self.profile_sync_btn.setToolTip(
            "Add/check elements selected in Analysis → Elements "
            "(for maps and for line-scan ROI curves)"
        )
        self.profile_sync_btn.clicked.connect(self._sync_profile_maps_from_analysis)
        profile_btn_row.addWidget(self.profile_all_btn)
        profile_btn_row.addWidget(self.profile_none_btn)
        profile_btn_row.addWidget(self.profile_sync_btn)
        tools_sec.addLayout(profile_btn_row)
        self.replot_line_btn = QPushButton("Replot profile")
        self.replot_line_btn.setToolTip(
            "Refresh the Line profile chart for the current drawn transect "
            "or XGT line scan using checked elements"
        )
        self.replot_line_btn.clicked.connect(self._replot_profile)
        tools_sec.addWidget(self.replot_line_btn)

        corr_row = QHBoxLayout()
        self.corr_a = QComboBox()
        self.corr_b = QComboBox()
        corr_row.addWidget(self.corr_a)
        corr_row.addWidget(QLabel("vs"))
        corr_row.addWidget(self.corr_b)
        tools_sec.addLayout(corr_row)
        self.corr_btn = QPushButton("Plot correlation")
        self.corr_btn.clicked.connect(self._plot_correlation)
        tools_sec.addWidget(self.corr_btn)
        self.corr_matrix_btn = QPushButton("Correlation matrix (checked maps)")
        self.corr_matrix_btn.setToolTip(
            "Pearson r heatmap for all element maps checked in the Data tree"
        )
        self.corr_matrix_btn.clicked.connect(self._plot_correlation_matrix)
        tools_sec.addWidget(self.corr_matrix_btn)
        scroll_layout.addWidget(tools_sec)

        # ---- Map analysis (PCA, ratios, particles) ----
        analysis_sec = CollapsibleSection("Map analysis", expanded=False)
        analysis_hint = QLabel(
            "Derived maps are added to the Data tree (display tools stay separate)."
        )
        analysis_hint.setWordWrap(True)
        analysis_hint.setStyleSheet("color: #666; font-size: 11px;")
        analysis_sec.addWidget(analysis_hint)

        analysis_sec.addWidget(QLabel("Ratio / difference map"))
        ratio_row = QHBoxLayout()
        self.ratio_num = QComboBox()
        self.ratio_den = QComboBox()
        ratio_row.addWidget(self.ratio_num)
        ratio_row.addWidget(QLabel("/"))
        ratio_row.addWidget(self.ratio_den)
        analysis_sec.addLayout(ratio_row)
        ratio_btn_row = QHBoxLayout()
        self.ratio_btn = QPushButton("Add ratio map")
        self.ratio_btn.clicked.connect(self._add_ratio_map)
        self.diff_btn = QPushButton("Add A − B map")
        self.diff_btn.clicked.connect(self._add_difference_map)
        ratio_btn_row.addWidget(self.ratio_btn)
        ratio_btn_row.addWidget(self.diff_btn)
        analysis_sec.addLayout(ratio_btn_row)

        analysis_sec.addWidget(QLabel("PCA on checked maps"))
        pca_row = QHBoxLayout()
        pca_row.addWidget(QLabel("Components"))
        self.pca_n_spin = QSpinBox()
        self.pca_n_spin.setRange(1, 12)
        self.pca_n_spin.setValue(3)
        pca_row.addWidget(self.pca_n_spin)
        analysis_sec.addLayout(pca_row)
        self.pca_btn = QPushButton("Run PCA → add PC maps")
        self.pca_btn.setToolTip(
            "PCA on checked Data-tree maps. Adds PC1…PCk score maps; "
            "optionally set RGB to the first three PCs."
        )
        self.pca_btn.clicked.connect(self._run_pca)
        analysis_sec.addWidget(self.pca_btn)
        self.pca_rgb_check = QCheckBox("Set RGB to PC1 / PC2 / PC3")
        self.pca_rgb_check.setChecked(True)
        analysis_sec.addWidget(self.pca_rgb_check)

        analysis_sec.addWidget(QLabel("Particle finding"))
        part_map_row = QHBoxLayout()
        part_map_row.addWidget(QLabel("Map"))
        self.particle_map_combo = QComboBox()
        part_map_row.addWidget(self.particle_map_combo, stretch=1)
        analysis_sec.addLayout(part_map_row)
        part_thr_row = QHBoxLayout()
        part_thr_row.addWidget(QLabel("Threshold %ile"))
        self.particle_thr_spin = QDoubleSpinBox()
        self.particle_thr_spin.setRange(50.0, 99.9)
        self.particle_thr_spin.setDecimals(1)
        self.particle_thr_spin.setValue(90.0)
        part_thr_row.addWidget(self.particle_thr_spin)
        analysis_sec.addLayout(part_thr_row)
        part_area_row = QHBoxLayout()
        part_area_row.addWidget(QLabel("Min area"))
        self.particle_min_area = QSpinBox()
        self.particle_min_area.setRange(1, 10000)
        self.particle_min_area.setValue(5)
        self.particle_min_area.setSuffix(" px")
        part_area_row.addWidget(self.particle_min_area)
        analysis_sec.addLayout(part_area_row)
        self.particle_btn = QPushButton("Find particles")
        self.particle_btn.setToolTip(
            "Threshold + watershed on the selected map. "
            "Centroids mark the canvas; double-click a row to sum spectrum from the cube."
        )
        self.particle_btn.clicked.connect(self._find_particles)
        analysis_sec.addWidget(self.particle_btn)
        self.clear_particles_btn = QPushButton("Clear particles")
        self.clear_particles_btn.clicked.connect(self._clear_particles)
        analysis_sec.addWidget(self.clear_particles_btn)
        self.particle_table = QTableWidget(0, 4)
        self.particle_table.setHorizontalHeaderLabels(
            ["#", "Area", "Mean", "Centroid"]
        )
        self.particle_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.particle_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.particle_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.particle_table.setMaximumHeight(140)
        self.particle_table.cellDoubleClicked.connect(self._on_particle_activated)
        analysis_sec.addWidget(self.particle_table)
        scroll_layout.addWidget(analysis_sec)

        # ---- Cube ROI ----
        cube_sec = CollapsibleSection("Cube ROI map", expanded=False)
        roi_row = QHBoxLayout()
        self.roi_e0 = QDoubleSpinBox()
        self.roi_e0.setRange(0.0, 40.0)
        self.roi_e0.setDecimals(2)
        self.roi_e0.setSingleStep(0.05)
        self.roi_e0.setValue(6.30)
        self.roi_e1 = QDoubleSpinBox()
        self.roi_e1.setRange(0.0, 40.0)
        self.roi_e1.setDecimals(2)
        self.roi_e1.setSingleStep(0.05)
        self.roi_e1.setValue(6.50)
        roi_row.addWidget(QLabel("keV"))
        roi_row.addWidget(self.roi_e0)
        roi_row.addWidget(QLabel("–"))
        roi_row.addWidget(self.roi_e1)
        cube_sec.addLayout(roi_row)
        self.roi_btn = QPushButton("Add ROI map from cube")
        self.roi_btn.setToolTip(
            "Sum hyperspectral cube channels in the energy window"
        )
        self.roi_btn.clicked.connect(self._add_roi_map)
        cube_sec.addWidget(self.roi_btn)
        self.roi_from_analysis_btn = QPushButton("ROI maps from Analysis elements")
        self.roi_from_analysis_btn.setToolTip(
            "For each element selected in Analysis, add a Ka ROI map from the cube "
            "and check it for transect plotting"
        )
        self.roi_from_analysis_btn.clicked.connect(self._add_roi_maps_from_analysis)
        cube_sec.addWidget(self.roi_from_analysis_btn)
        self.cube_info = QLabel("No cube loaded")
        self.cube_info.setWordWrap(True)
        cube_sec.addWidget(self.cube_info)
        scroll_layout.addWidget(cube_sec)

        # ---- Quantification ----
        quant_sec = CollapsibleSection("Quantification", expanded=True)
        self.send_sum_btn = QPushButton("Send Sum Spectrum → Analysis")
        self.send_sum_btn.setToolTip(
            "Load this site's Sum Spectrum into Analysis to identify elements of interest"
        )
        self.send_sum_btn.clicked.connect(self._send_sum_spectrum)
        quant_sec.addWidget(self.send_sum_btn)
        self.send_btn = QPushButton("Send spectrum → Analysis")
        self.send_btn.setToolTip(
            "Send the selected tree spectrum, or a picked pixel / line-mean spectrum"
        )
        self.send_btn.clicked.connect(self._send_selected_spectrum)
        quant_sec.addWidget(self.send_btn)
        self.send_batch_btn = QPushButton("Send selected → Batch")
        self.send_batch_btn.setToolTip(
            "Queue selected Data-tree spectra in Batch Analysis. "
            "Shift/Ctrl-click in Data to choose a subset."
        )
        self.send_batch_btn.clicked.connect(self._send_selected_to_batch)
        quant_sec.addWidget(self.send_batch_btn)
        self.send_site_batch_btn = QPushButton("Send site spectra → Batch")
        self.send_site_batch_btn.setToolTip(
            "Queue every spot and line/multipoint spectrum in the active site"
        )
        self.send_site_batch_btn.clicked.connect(self._send_site_to_batch)
        quant_sec.addWidget(self.send_site_batch_btn)
        self.send_project_batch_btn = QPushButton("Send all project spectra → Batch")
        self.send_project_batch_btn.setToolTip(
            "Queue every spot and line/multipoint spectrum in this IPJ "
            "(all sites). Map sum spectra are skipped."
        )
        self.send_project_batch_btn.clicked.connect(self._send_project_to_batch)
        quant_sec.addWidget(self.send_project_batch_btn)
        self.export_profile_btn = QPushButton("Export profile CSV…")
        self.export_profile_btn.setToolTip(
            "Export the current drawn-transect intensity profile"
        )
        self.export_profile_btn.clicked.connect(self._export_map_profile_csv)
        quant_sec.addWidget(self.export_profile_btn)
        ls_note = QLabel(
            "Collected line-scan semi-quant lives on the Line scan tab — "
            "not on a transect drawn here."
        )
        ls_note.setWordWrap(True)
        ls_note.setStyleSheet("color: #555; font-size: 11px;")
        quant_sec.addWidget(ls_note)
        scroll_layout.addWidget(quant_sec)

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_body)

        splitter.addWidget(left)

        self.workspace_tabs = QTabWidget()
        self.workspace_tabs.currentChanged.connect(self._on_workspace_tab)

        # ---- Maps workspace: tools | canvas | profiles ----
        maps_page = QWidget()
        maps_split = QSplitter(Qt.Horizontal)
        maps_layout = QHBoxLayout(maps_page)
        maps_layout.setContentsMargins(0, 0, 0, 0)
        maps_layout.addWidget(maps_split)
        maps_split.addWidget(scroll)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MapCanvas()
        self.canvas.line_drawn.connect(self._on_line_drawn)
        self.canvas.cursor_moved.connect(self._on_cursor)
        self.canvas.pixel_clicked.connect(self._on_pixel_clicked)
        self.canvas.region_drawn.connect(self._on_region_drawn)
        center_layout.addWidget(self.canvas)
        self.cursor_label = QLabel("Cursor: —")
        center_layout.addWidget(self.cursor_label)
        maps_split.addWidget(center)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_tabs_label = QLabel("Drawn transect & correlations")
        right_layout.addWidget(self.plot_tabs_label)

        self.map_plot_tabs = QTabWidget()

        profile_tab = QWidget()
        profile_layout = QVBoxLayout(profile_tab)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        self.profile_plot = pg.PlotWidget(title="Map line profile")
        self.profile_plot.setLabel("bottom", "Distance (pixels)")
        self.profile_plot.setLabel("left", "Intensity")
        self.profile_plot.addLegend(offset=(10, 10))
        profile_layout.addWidget(self.profile_plot)
        self.map_plot_tabs.addTab(profile_tab, "Profile")

        corr_tab = QWidget()
        corr_layout = QVBoxLayout(corr_tab)
        corr_layout.setContentsMargins(0, 0, 0, 0)
        self.corr_plot = pg.PlotWidget(title="Element correlation")
        self.corr_plot.setLabel("bottom", "Map A")
        self.corr_plot.setLabel("left", "Map B")
        corr_layout.addWidget(self.corr_plot)
        self.map_plot_tabs.addTab(corr_tab, "Correlate")

        matrix_tab = QWidget()
        matrix_layout = QVBoxLayout(matrix_tab)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        matrix_hint = QLabel(
            "Pearson r of checked maps. Click a cell to open that pair on Correlate."
        )
        matrix_hint.setStyleSheet("color: #555; font-size: 11px;")
        matrix_hint.setWordWrap(True)
        matrix_layout.addWidget(matrix_hint)
        self.matrix_plot = pg.PlotWidget()
        self.matrix_plot.setBackground("w")
        self.matrix_plot.setAspectLocked(True)
        self.matrix_plot.invertY(True)
        self.matrix_image = pg.ImageItem()
        self.matrix_image.setLookupTable(_diverging_lut())
        self.matrix_image.setLevels((-1.0, 1.0))
        self.matrix_plot.addItem(self.matrix_image)
        self.matrix_plot.scene().sigMouseClicked.connect(self._on_matrix_clicked)
        matrix_layout.addWidget(self.matrix_plot, stretch=1)
        self.map_plot_tabs.addTab(matrix_tab, "Matrix")

        right_layout.addWidget(self.map_plot_tabs, stretch=1)

        self.info_label = QLabel("Open an .ipj mapping project to begin.")
        self.info_label.setWordWrap(True)
        right_layout.addWidget(self.info_label)
        maps_split.addWidget(right)
        maps_split.setSizes([260, 520, 400])
        self.workspace_tabs.addTab(maps_page, "Maps")

        # ---- Line scan workspace: tools | ROI profile + semi-quant ----
        ls_page = QWidget()
        ls_split = QSplitter(Qt.Horizontal)
        ls_page_layout = QHBoxLayout(ls_page)
        ls_page_layout.setContentsMargins(0, 0, 0, 0)
        ls_page_layout.addWidget(ls_split)

        ls_scroll = QScrollArea()
        ls_scroll.setWidgetResizable(True)
        ls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ls_scroll.setFrameShape(QScrollArea.NoFrame)
        ls_scroll.setMinimumWidth(240)
        ls_body = QWidget()
        ls_layout = QVBoxLayout(ls_body)
        ls_layout.setContentsMargins(0, 0, 4, 0)
        ls_layout.setSpacing(8)

        ls_intro = QLabel(
            "This tab is for spectra the instrument actually collected along "
            "a path (line scan or multipoint). Check the elements you want "
            "on the profile, then Fit / semi-quant uses that same list.\n\n"
            "A line drawn on a map stays on Maps as an intensity profile. "
            "Map pixels are usually too short-count to fit at each point."
        )
        ls_intro.setWordWrap(True)
        ls_intro.setStyleSheet("color: #444; font-size: 11px;")
        ls_layout.addWidget(ls_intro)

        self.ls_status_label = QLabel("No collected line scan in this site.")
        self.ls_status_label.setWordWrap(True)
        ls_layout.addWidget(self.ls_status_label)

        ls_layout.addWidget(QLabel("Elements on this line"))
        self.ls_element_list = QListWidget()
        self.ls_element_list.setMinimumHeight(140)
        self.ls_element_list.setToolTip(
            "Checked elements drive the ROI profile and Fit / semi-quant.\n"
            "Use From Analysis to copy the Analysis → Elements selection."
        )
        self.ls_element_list.itemChanged.connect(self._on_ls_element_check_changed)
        ls_layout.addWidget(self.ls_element_list)

        ls_btn_row = QHBoxLayout()
        self.ls_all_btn = QPushButton("All")
        self.ls_all_btn.clicked.connect(lambda: self._set_all_ls_elements(True))
        self.ls_none_btn = QPushButton("None")
        self.ls_none_btn.clicked.connect(lambda: self._set_all_ls_elements(False))
        self.ls_sync_btn = QPushButton("From Analysis")
        self.ls_sync_btn.setToolTip(
            "Check elements selected in Analysis → Elements"
        )
        self.ls_sync_btn.clicked.connect(self._sync_ls_elements_from_analysis)
        ls_btn_row.addWidget(self.ls_all_btn)
        ls_btn_row.addWidget(self.ls_none_btn)
        ls_btn_row.addWidget(self.ls_sync_btn)
        ls_layout.addLayout(ls_btn_row)

        self.ls_replot_btn = QPushButton("Replot ROI profile")
        self.ls_replot_btn.setToolTip(
            "Refresh windowed counts vs distance for the checked elements"
        )
        self.ls_replot_btn.clicked.connect(self._replot_collected_line_scan)
        ls_layout.addWidget(self.ls_replot_btn)

        self.fit_line_btn = QPushButton("Fit / semi-quant along line")
        self.fit_line_btn.setToolTip(
            "Fit each collected point with the checked elements, then plot "
            "area-normalized relative intensities (not FP wt%)."
        )
        self.fit_line_btn.clicked.connect(self._fit_line_scan)
        ls_layout.addWidget(self.fit_line_btn)

        self.ls_export_btn = QPushButton("Export line-scan CSV…")
        self.ls_export_btn.setToolTip(
            "Export semi-quant table if fitted, otherwise the ROI profile"
        )
        self.ls_export_btn.clicked.connect(self._export_line_scan_csv)
        ls_layout.addWidget(self.ls_export_btn)

        self.ls_send_btn = QPushButton("Send selected point → Analysis")
        self.ls_send_btn.setToolTip(
            "Send the tree-selected line-scan point (or the first point) "
            "to Analysis for peak ID"
        )
        self.ls_send_btn.clicked.connect(self._send_selected_spectrum)
        ls_layout.addWidget(self.ls_send_btn)

        self.ls_send_batch_btn = QPushButton("Send points → Batch")
        self.ls_send_batch_btn.setToolTip(
            "Queue this line/multipoint series in Batch Analysis. "
            "If points are selected in Data, only those are sent."
        )
        self.ls_send_batch_btn.clicked.connect(self._send_line_scan_to_batch)
        ls_layout.addWidget(self.ls_send_batch_btn)

        ls_layout.addStretch(1)
        ls_scroll.setWidget(ls_body)
        ls_split.addWidget(ls_scroll)

        self.ls_content_stack = QStackedWidget()
        ls_empty = QWidget()
        ls_empty_layout = QVBoxLayout(ls_empty)
        self.ls_empty_label = QLabel(
            "This site has no collected line scan or multipoint series.\n\n"
            "Semi-quant along a transect needs spectra the instrument "
            "acquired along a path. Activate a line-scan site in Sites, "
            "or select the series in Data.\n\n"
            "Drawing a line on an element map does not create that data — "
            "use the Maps tab for intensity profiles."
        )
        self.ls_empty_label.setWordWrap(True)
        self.ls_empty_label.setStyleSheet("color: #444;")
        ls_empty_layout.addWidget(self.ls_empty_label)
        ls_empty_layout.addStretch(1)
        self.ls_content_stack.addWidget(ls_empty)

        ls_view = QSplitter(Qt.Horizontal)
        cam_wrap = QWidget()
        cam_layout = QVBoxLayout(cam_wrap)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        self.ls_camera = MapCanvas()
        self.ls_camera.set_line_mode(False)
        self.ls_camera.cursor_moved.connect(self._on_ls_camera_cursor)
        self.ls_camera.view_clicked.connect(self._on_ls_camera_clicked)
        cam_layout.addWidget(self.ls_camera, stretch=1)
        self.ls_camera_label = QLabel(
            "Hover the ROI profile to see that point on the sample camera."
        )
        self.ls_camera_label.setWordWrap(True)
        cam_layout.addWidget(self.ls_camera_label)
        ls_view.addWidget(cam_wrap)

        ls_plots = QWidget()
        ls_plots_layout = QVBoxLayout(ls_plots)
        ls_plots_layout.setContentsMargins(0, 0, 0, 0)
        self.ls_profile_plot = pg.PlotWidget(title="Line-scan ROI profile")
        self.ls_profile_plot.setLabel("bottom", "Position")
        self.ls_profile_plot.setLabel("left", "Counts / s in window")
        self.ls_profile_plot.addLegend(offset=(10, 10))
        self._ls_profile_hover = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#ffee66", width=1.5, style=Qt.DashLine),
        )
        self._ls_profile_hover.setVisible(False)
        self.ls_profile_plot.addItem(self._ls_profile_hover)
        ls_plots_layout.addWidget(self.ls_profile_plot, stretch=1)

        self.quant_plot = pg.PlotWidget(title="Line-scan semi-quant")
        self.quant_plot.setLabel("bottom", "Position")
        self.quant_plot.setLabel("left", "Relative %")
        self.quant_plot.addLegend(offset=(10, 10))
        self._ls_quant_hover = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#ffee66", width=1.5, style=Qt.DashLine),
        )
        self._ls_quant_hover.setVisible(False)
        self.quant_plot.addItem(self._ls_quant_hover)
        ls_plots_layout.addWidget(self.quant_plot, stretch=1)

        self.ls_info_label = QLabel(
            "Select a collected line scan, check elements, then Fit / semi-quant."
        )
        self.ls_info_label.setWordWrap(True)
        ls_plots_layout.addWidget(self.ls_info_label)
        ls_view.addWidget(ls_plots)
        ls_view.setSizes([440, 520])
        self.ls_content_stack.addWidget(ls_view)
        ls_split.addWidget(self.ls_content_stack)
        ls_split.setSizes([260, 900])
        self.workspace_tabs.addTab(ls_page, "Line scan")

        splitter.addWidget(self.workspace_tabs)
        splitter.setSizes([280, 920])
        root.addWidget(splitter)

        self._last_profiles = None  # dict name -> (dist, vals)
        self._ls_profile_hover_proxy = pg.SignalProxy(
            self.ls_profile_plot.scene().sigMouseMoved,
            rateLimit=40,
            slot=self._on_ls_profile_mouse,
        )
        self._ls_quant_hover_proxy = pg.SignalProxy(
            self.quant_plot.scene().sigMouseMoved,
            rateLimit=40,
            slot=self._on_ls_profile_mouse,
        )

    # -------------------------------------------------------------- wiring
    def set_fitter(self, fitter) -> None:
        self._fitter = fitter

    def set_element_panel(self, panel) -> None:
        self._element_panel = panel

    # --------------------------------------------------------------- load
    def open_ipj(self, path: Optional[str] = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open INCA / XGT Project",
                "",
                "INCA Project (*.ipj);;All Files (*)",
            )
        if not path:
            return
        try:
            from utils.ipj_loader import load_ipj

            self.project = load_ipj(path)
        except Exception as exc:
            QMessageBox.critical(self, "IPJ load failed", str(exc))
            return

        self._apply_loaded_project()

    def merge_ipjs(self, paths: Optional[list] = None) -> None:
        """Merge line/multipoint series from many .ipj files into one project."""
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Merge INCA / XGT Projects (line scan / multipoint)",
                "",
                "INCA Project (*.ipj);;All Files (*)",
            )
        if not paths:
            return
        if len(paths) < 2:
            QMessageBox.information(
                self,
                "Merge IPJs",
                "Select at least two .ipj files to merge.",
            )
            return
        try:
            from core.mapping.merge import merge_ipj_line_scans

            self.project = merge_ipj_line_scans(paths)
        except Exception as exc:
            QMessageBox.critical(self, "IPJ merge failed", str(exc))
            return
        self._apply_loaded_project()

    def _apply_loaded_project(self) -> None:
        """Refresh trees / status after open or merge."""
        if self.project is None:
            return

        self._cam_dest_rect_cache.clear()

        self._populate_trees()
        primary = self.project.primary_fov
        if primary:
            self._activate_site(primary, switch_to_data=True)
        n_samples = self.project.metadata.get("n_samples", len(self.project.samples))
        n_sites = self.project.metadata.get("n_fovs", len(self.project.fovs))
        n_cubes = self.project.metadata.get("n_cubes", 0)
        n_series = self.project.metadata.get("n_line_scans")
        n_sources = self.project.metadata.get("n_source_files")
        self.status_message.emit(f"Loaded mapping project: {self.project.name}")
        info = (
            f"{self.project.name}: {n_samples} sample(s), {n_sites} site(s), "
            f"{len(self.project.all_spectra())} spectra"
            + (f", {n_cubes} cube(s)" if n_cubes else "")
        )
        if n_sources:
            info += f" (merged from {n_sources} .ipj)"
        if n_series:
            info += f", {n_series} line/multipoint series"
        self.info_label.setText(info)
        if self.project.is_spectra_only():
            n_pts = len(self.project.point_spectra())
            self.info_label.setText(
                f"{self.project.name}: {n_pts} point spectra, no maps or line scans. "
                "Opening in Analysis so you can identify elements, then Process All "
                "in Batch Analysis (already queued)."
            )
            self.nav_tabs.setCurrentIndex(1)
        self._fill_sample_tab()
        self.project_loaded.emit(self.project)

    def _show_sample_info(self) -> None:
        self._fill_sample_tab()
        self.sample_dialog.show()
        self.sample_dialog.raise_()
        self.sample_dialog.activateWindow()

    def _build_sample_dialog(self) -> QDialog:
        """Project / sample / site metadata — opened from Sample info…"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Sample / acquisition")
        dialog.setModal(False)
        dialog.resize(420, 520)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hint = QLabel(
            "Text entered on the instrument (name, comment, type) plus "
            "acquisition settings read from the project file."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 4, 0)
        form.setSpacing(4)

        self.proj_title_edit = QLineEdit()
        self.proj_title_edit.setReadOnly(True)
        self.proj_instrument_edit = QLineEdit()
        self.proj_instrument_edit.setReadOnly(True)
        self.proj_path_edit = QLineEdit()
        self.proj_path_edit.setReadOnly(True)
        self.proj_comment_edit = QTextEdit()
        self.proj_comment_edit.setReadOnly(True)
        self.proj_comment_edit.setMaximumHeight(48)
        form.addRow("Project", self.proj_title_edit)
        form.addRow("Instrument", self.proj_instrument_edit)
        form.addRow("File", self.proj_path_edit)
        form.addRow("Project note", self.proj_comment_edit)

        self.sample_name_edit = QLineEdit()
        self.sample_name_edit.setPlaceholderText("Sample name")
        self.sample_name_edit.editingFinished.connect(self._on_sample_name_edited)
        self.sample_type_edit = QLineEdit()
        self.sample_type_edit.setPlaceholderText("Type / material")
        self.sample_type_edit.editingFinished.connect(self._on_sample_type_edited)
        self.sample_comment_edit = QTextEdit()
        self.sample_comment_edit.setPlaceholderText("Sample comment")
        self.sample_comment_edit.setMaximumHeight(72)
        self.sample_comment_edit.textChanged.connect(self._on_sample_comment_changed)
        form.addRow("Sample", self.sample_name_edit)
        form.addRow("Type", self.sample_type_edit)
        form.addRow("Comment", self.sample_comment_edit)

        self.site_name_edit = QLineEdit()
        self.site_name_edit.setPlaceholderText("Site of Interest")
        self.site_name_edit.editingFinished.connect(self._on_site_name_edited)
        self.site_comment_edit = QTextEdit()
        self.site_comment_edit.setPlaceholderText("Site comment")
        self.site_comment_edit.setMaximumHeight(72)
        self.site_comment_edit.textChanged.connect(self._on_site_comment_changed)
        form.addRow("Site", self.site_name_edit)
        form.addRow("Site comment", self.site_comment_edit)

        self.sample_acq_label = QLabel("No project loaded")
        self.sample_acq_label.setWordWrap(True)
        self.sample_acq_label.setStyleSheet("color: #333; font-size: 11px;")
        form.addRow("Acquisition", self.sample_acq_label)

        scroll.setWidget(body)
        layout.addWidget(scroll, stretch=1)

        self.copy_sample_btn = QPushButton("Copy to Analysis → Sample/Exp")
        self.copy_sample_btn.setToolTip(
            "Fill Analysis sample name, type, kV, and current from this site"
        )
        self.copy_sample_btn.clicked.connect(self._copy_sample_to_analysis)
        layout.addWidget(self.copy_sample_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.close)
        layout.addWidget(buttons)
        return dialog

    def _current_sample(self):
        if self.project is None:
            return None
        if self.current_fov is not None:
            sid = self.current_fov.metadata.get("sample_id")
            if sid:
                found = self.project.find_sample(sid)
                if found is not None:
                    return found
        if self.project.samples:
            return self.project.samples[0]
        return None

    def _fill_sample_tab(self) -> None:
        self._sample_form_updating = True
        proj = self.project
        sample = self._current_sample()
        site = self.current_fov
        if proj is None:
            for w in (
                self.proj_title_edit,
                self.proj_instrument_edit,
                self.proj_path_edit,
                self.sample_name_edit,
                self.sample_type_edit,
                self.site_name_edit,
            ):
                w.clear()
            self.proj_comment_edit.clear()
            self.sample_comment_edit.clear()
            self.site_comment_edit.clear()
            self.sample_acq_label.setText("No project loaded")
            self._sample_form_updating = False
            return

        meta = proj.metadata or {}
        self.proj_title_edit.setText(str(meta.get("project_title") or proj.name))
        self.proj_instrument_edit.setText(str(meta.get("instrument") or ""))
        self.proj_path_edit.setText(proj.path)
        self.proj_comment_edit.setPlainText(str(meta.get("comment") or ""))

        if sample is not None:
            self.sample_name_edit.setText(sample.name)
            self.sample_type_edit.setText(str(sample.metadata.get("sample_type") or ""))
            self.sample_comment_edit.setPlainText(str(sample.metadata.get("comment") or ""))
        else:
            self.sample_name_edit.clear()
            self.sample_type_edit.clear()
            self.sample_comment_edit.clear()

        if site is not None:
            self.site_name_edit.setText(site.name)
            self.site_comment_edit.setPlainText(str(site.metadata.get("comment") or ""))
            acq = format_acquisition(site.metadata)
            extra = []
            if site.width and site.height:
                extra.append(f"{site.width} × {site.height} px")
            if site.cube is not None:
                extra.append(f"{site.cube.n_channels} channels")
            block = acq
            if extra:
                block = (block + "\n" if block else "") + " · ".join(extra)
            self.sample_acq_label.setText(block or "No acquisition metadata")
        else:
            self.site_name_edit.clear()
            self.site_comment_edit.clear()
            self.sample_acq_label.setText("Activate a site to see acquisition info")
        self._sample_form_updating = False

    def _on_sample_name_edited(self) -> None:
        if self._sample_form_updating:
            return
        sample = self._current_sample()
        if sample is None:
            return
        name = self.sample_name_edit.text().strip()
        if not name or name == sample.name:
            return
        sample.name = name
        self._sync_labels_after_rename()
        self.status_message.emit(f"Renamed sample → {name}")

    def _on_sample_type_edited(self) -> None:
        if self._sample_form_updating:
            return
        sample = self._current_sample()
        if sample is None:
            return
        sample.metadata["sample_type"] = self.sample_type_edit.text().strip()

    def _on_sample_comment_changed(self) -> None:
        if self._sample_form_updating:
            return
        sample = self._current_sample()
        if sample is not None:
            sample.metadata["comment"] = self.sample_comment_edit.toPlainText()

    def _on_site_name_edited(self) -> None:
        if self._sample_form_updating:
            return
        site = self.current_fov
        if site is None:
            return
        name = self.site_name_edit.text().strip()
        if not name or name == site.name:
            return
        site.name = name
        site.metadata["site_name"] = name
        self._sync_labels_after_rename()
        self.status_message.emit(f"Renamed site → {name}")

    def _on_site_comment_changed(self) -> None:
        if self._sample_form_updating:
            return
        site = self.current_fov
        if site is not None:
            site.metadata["comment"] = self.site_comment_edit.toPlainText()

    def _copy_sample_to_analysis(self, *, quiet: bool = False) -> bool:
        """Copy sample name / type and site tube settings into Analysis.

        Returns True if Analysis was updated. With quiet=True, missing
        Analysis wiring is silent (used on IPJ open / site change).
        """
        panel = self._element_panel
        if panel is None:
            if not quiet:
                QMessageBox.information(
                    self,
                    "Analysis",
                    "Analysis sample fields are not connected.",
                )
            return False
        sample = self._current_sample()
        site = self.current_fov
        if sample is not None and hasattr(panel, "sample_name_edit"):
            panel.sample_name_edit.setText(sample.name)
        if sample is not None and hasattr(panel, "sample_type_combo"):
            typ = str(sample.metadata.get("sample_type") or "")
            idx = panel.sample_type_combo.findText(typ)
            if idx >= 0:
                panel.sample_type_combo.setCurrentIndex(idx)

        kv = ma = live = None
        if site is not None:
            kv = site.metadata.get("kv")
            ma = site.metadata.get("ma")
            live = site.metadata.get("map_live_time_s")
            # Multipoint sites: fall back to first spectrum metadata
            if (kv is None or ma is None or live is None) and site.spectra:
                for ms in site.spectra:
                    sm = getattr(ms.spectrum, "metadata", None) or {}
                    if kv is None and sm.get("kv") is not None:
                        kv = sm.get("kv")
                    if ma is None and sm.get("ma") is not None:
                        ma = sm.get("ma")
                    if live is None:
                        try:
                            lt = float(ms.spectrum.live_time)
                        except (TypeError, ValueError):
                            lt = float(sm.get("live_time") or 0)
                        if lt > 0:
                            live = lt
                    if kv is not None and ma is not None and live is not None:
                        break

        if kv is not None and hasattr(panel, "excitation_spin"):
            panel.excitation_spin.setValue(float(kv))
        if ma is not None and hasattr(panel, "current_spin"):
            # Analysis current spinner max is 10 mA; XGT often uses 15 mA
            spin = panel.current_spin
            if float(ma) > spin.maximum():
                spin.setMaximum(max(float(ma), 50.0))
            spin.setValue(float(ma))
        if live is not None and hasattr(panel, "live_time_spin"):
            spin = panel.live_time_spin
            if float(live) > spin.maximum():
                spin.setMaximum(max(float(live), 10000.0))
            spin.setValue(float(live))

        parts = []
        if kv is not None:
            parts.append(f"{float(kv):g} kV")
        if ma is not None:
            parts.append(f"{float(ma):g} mA")
        if live is not None:
            parts.append(f"{float(live):g} s live")
        detail = ", ".join(parts) if parts else "sample fields only"
        self.status_message.emit(
            f"Copied to Analysis → Sample/Exp ({detail})"
        )
        return True

    def _populate_trees(self) -> None:
        self._populate_sites_tree()
        self._populate_data_tree()

    def _populate_sites_tree(self) -> None:
        self._tree_updating = True
        self.sites_tree.blockSignals(True)
        self.sites_tree.clear()
        if not self.project:
            self.sites_tree.blockSignals(False)
            self._tree_updating = False
            return
        root = QTreeWidgetItem([self.project.name])
        root.setData(0, Qt.UserRole, ("project", None))
        root.setFlags(root.flags() & ~Qt.ItemIsEditable)
        self.sites_tree.addTopLevelItem(root)

        for sample in self.project.samples:
            sample_item = QTreeWidgetItem([sample.name])
            sample_item.setData(0, Qt.UserRole, ("sample", sample.id))
            sample_item.setFlags(
                (sample_item.flags() | Qt.ItemIsEditable | Qt.ItemIsSelectable)
                & ~Qt.ItemIsUserCheckable
            )
            sample_item.setToolTip(0, "Sample — F2 or right-click to rename")
            root.addChild(sample_item)
            if sample.whole_image is not None:
                cam = QTreeWidgetItem([sample.whole_image.name or "Sample camera"])
                cam.setData(0, Qt.UserRole, ("whole_image", sample.id))
                cam.setFlags(cam.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsUserCheckable)
                cam.setToolTip(0, "Optical camera photo of the sample")
                sample_item.addChild(cam)
            for site in sample.sites:
                site_item = QTreeWidgetItem([site.name, site.contents_label()])
                site_item.setData(0, Qt.UserRole, ("site", sample.id, site.id))
                site_item.setFlags(
                    (site_item.flags() | Qt.ItemIsEditable | Qt.ItemIsSelectable)
                    & ~Qt.ItemIsUserCheckable
                )
                tip = self._site_tooltip(site)
                site_item.setToolTip(0, tip)
                site_item.setToolTip(1, tip)
                site_item.setForeground(1, QColor("#555555"))
                sample_item.addChild(site_item)

        root.setExpanded(True)
        for i in range(root.childCount()):
            root.child(i).setExpanded(True)
        self.sites_tree.blockSignals(False)
        self._tree_updating = False

    def _populate_data_tree(self) -> None:
        """Data tab shows contents of the active Site of Interest."""
        self._tree_updating = True
        self.tree.blockSignals(True)
        self.tree.clear()
        fov = self.current_fov
        if not self.project or fov is None:
            tip = QTreeWidgetItem(["Activate a site in the Sites tab"])
            tip.setData(0, Qt.UserRole, ("hint",))
            self.tree.addTopLevelItem(tip)
            self.tree.blockSignals(False)
            self._tree_updating = False
            return

        sample_name = self._sample_name_for_site(fov)
        root = QTreeWidgetItem([fov.name])
        root.setData(0, Qt.UserRole, ("site", fov.metadata.get("sample_id"), fov.id))
        root.setFlags(
            (root.flags() | Qt.ItemIsEditable | Qt.ItemIsSelectable)
            & ~Qt.ItemIsUserCheckable
        )
        root.setToolTip(0, f"{sample_name} — rename this site (F2 / right-click)")
        self.tree.addTopLevelItem(root)

        # Match vendor order: SmartMap, Trans. x-ray, maps, spectra
        if fov.cube is not None or fov.metadata.get("has_smartmap"):
            it = QTreeWidgetItem(["SmartMap"])
            it.setData(0, Qt.UserRole, ("smartmap", fov.id))
            root.addChild(it)

        sample = self._sample_for_site(fov)
        if sample is not None and sample.whole_image is not None:
            it = QTreeWidgetItem([sample.whole_image.name or "Sample camera"])
            it.setData(0, Qt.UserRole, ("whole_image", sample.id))
            it.setToolTip(0, "Optical camera photo of the whole sample")
            root.addChild(it)

        if fov.optical is not None:
            it = QTreeWidgetItem([fov.optical.name or "Map area photo"])
            it.setData(0, Qt.UserRole, ("optical", fov.id))
            it.setToolTip(0, "Optical camera photo of the mapped region")
            root.addChild(it)

        if fov.overview is not None:
            it = QTreeWidgetItem([fov.overview.name or "Trans. x-ray image"])
            it.setData(0, Qt.UserRole, ("overview", fov.id))
            root.addChild(it)

        vendor_maps = [
            m
            for m in fov.element_maps
            if m.metadata.get("source")
            not in ("cube_total", "cube_roi", "pca", "ratio", "difference", "particles")
        ]
        cube_maps = [
            m for m in fov.element_maps if m.metadata.get("source") in ("cube_total", "cube_roi")
        ]
        derived_maps = [
            m
            for m in fov.element_maps
            if m.metadata.get("source") in ("pca", "ratio", "difference", "particles")
        ]

        # Preserve checks across rebuilds; default-check first few vendor maps
        valid_names = {m.name for m in fov.element_maps}
        self._checked_map_names = {n for n in self._checked_map_names if n in valid_names}
        if not self._checked_map_names and vendor_maps:
            self._checked_map_names = {m.name for m in vendor_maps[: min(4, len(vendor_maps))]}
        elif not self._checked_map_names and cube_maps:
            self._checked_map_names = {cube_maps[0].name}

        if vendor_maps:
            maps_item = QTreeWidgetItem(["Element maps (check to plot)"])
            maps_item.setData(0, Qt.UserRole, ("maps_folder", fov.id))
            root.addChild(maps_item)
            for m in vendor_maps:
                it = QTreeWidgetItem([m.name])
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                it.setCheckState(
                    0,
                    Qt.Checked if m.name in self._checked_map_names else Qt.Unchecked,
                )
                it.setData(0, Qt.UserRole, ("map", fov.id, m.name))
                maps_item.addChild(it)
        if cube_maps:
            cube_item = QTreeWidgetItem(["Cube maps (check to plot)"])
            cube_item.setData(0, Qt.UserRole, ("maps_folder", fov.id))
            root.addChild(cube_item)
            for m in cube_maps:
                it = QTreeWidgetItem([m.name])
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                it.setCheckState(
                    0,
                    Qt.Checked if m.name in self._checked_map_names else Qt.Unchecked,
                )
                it.setData(0, Qt.UserRole, ("map", fov.id, m.name))
                cube_item.addChild(it)
        if derived_maps:
            der_item = QTreeWidgetItem(["Derived maps (PCA / ratio / particles)"])
            der_item.setData(0, Qt.UserRole, ("maps_folder", fov.id))
            root.addChild(der_item)
            for m in derived_maps:
                label = m.name
                pct = m.metadata.get("explained_variance_pct")
                if pct is not None:
                    label = f"{m.name} ({pct:.1f}%)"
                it = QTreeWidgetItem([label])
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                it.setCheckState(
                    0,
                    Qt.Checked if m.name in self._checked_map_names else Qt.Unchecked,
                )
                it.setData(0, Qt.UserRole, ("map", fov.id, m.name))
                tip = m.metadata.get("source", "derived")
                if m.metadata.get("input_maps"):
                    tip += ": " + ", ".join(m.metadata["input_maps"][:8])
                it.setToolTip(0, tip)
                der_item.addChild(it)

        sum_spec = fov.sum_spectrum()
        other_specs = [s for s in fov.spectra if s is not sum_spec]
        if sum_spec is not None:
            it = QTreeWidgetItem([sum_spec.name])
            it.setData(0, Qt.UserRole, ("spectrum", fov.id, sum_spec.name))
            it.setFlags(
                (it.flags() | Qt.ItemIsEditable | Qt.ItemIsSelectable)
                & ~Qt.ItemIsUserCheckable
            )
            it.setToolTip(0, "Spectrum — F2 or right-click to rename")
            root.addChild(it)

        line_point_names = {
            s.name for ls in fov.line_scans for s in ls.points
        }
        for ls in fov.line_scans:
            ls_item = QTreeWidgetItem([ls.name])
            ls_item.setData(0, Qt.UserRole, ("linescan", fov.id, ls.name))
            ls_item.setFlags(
                (ls_item.flags() | Qt.ItemIsEditable | Qt.ItemIsSelectable)
                & ~Qt.ItemIsUserCheckable
            )
            kind_tip = ls.display_label()
            ls_item.setToolTip(
                0, f"{kind_tip} — F2 or right-click to rename"
            )
            root.addChild(ls_item)
            for s in ls.points:
                it = QTreeWidgetItem([s.name])
                it.setData(0, Qt.UserRole, ("spectrum", fov.id, s.name))
                it.setFlags(
                    (it.flags() | Qt.ItemIsEditable | Qt.ItemIsSelectable)
                    & ~Qt.ItemIsUserCheckable
                )
                it.setToolTip(
                    0,
                    f"{kind_tip} point — F2 or right-click to rename",
                )
                ls_item.addChild(it)

        spots = [s for s in other_specs if s.name not in line_point_names]
        for s in spots:
            it = QTreeWidgetItem([s.name])
            it.setData(0, Qt.UserRole, ("spectrum", fov.id, s.name))
            it.setFlags(
                (it.flags() | Qt.ItemIsEditable | Qt.ItemIsSelectable)
                & ~Qt.ItemIsUserCheckable
            )
            it.setToolTip(0, "Spectrum — F2 or right-click to rename")
            root.addChild(it)

        root.setExpanded(True)
        for i in range(root.childCount()):
            root.child(i).setExpanded(True)
        self.tree.blockSignals(False)
        self._tree_updating = False

    def _on_data_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._tree_updating or column != 0:
            return
        payload = item.data(0, Qt.UserRole)
        if not payload:
            return
        kind = payload[0]
        if kind == "map":
            name = payload[2]
            if item.checkState(0) == Qt.Checked:
                self._checked_map_names.add(name)
            else:
                self._checked_map_names.discard(name)
            self._refresh_canvas()
            return
        if kind in ("spectrum", "linescan", "site"):
            self._apply_rename_from_item(item, payload)

    def _on_sites_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._tree_updating:
            return
        payload = item.data(0, Qt.UserRole)
        if not payload:
            return
        if column != 0:
            if payload[0] == "site":
                site = self._find_fov(payload[2])
                if site is not None:
                    self._tree_updating = True
                    item.setText(1, site.contents_label())
                    self._tree_updating = False
            return
        if payload[0] in ("sample", "site"):
            self._apply_rename_from_item(item, payload)

    def _apply_rename_from_item(self, item: QTreeWidgetItem, payload: tuple) -> None:
        new_name = item.text(0).strip()
        kind = payload[0]
        if not new_name:
            # Revert empty names
            self._tree_updating = True
            if kind == "sample":
                sample = self.project.find_sample(payload[1]) if self.project else None
                item.setText(0, sample.name if sample else "Sample")
            elif kind == "site":
                site = self._find_fov(payload[2])
                item.setText(0, site.name if site else "Site")
            elif kind == "spectrum":
                item.setText(0, payload[2])
            elif kind == "linescan":
                item.setText(0, payload[2])
            self._tree_updating = False
            return

        if kind == "sample":
            sample = self.project.find_sample(payload[1]) if self.project else None
            if sample is None or sample.name == new_name:
                return
            sample.name = new_name
            self._sync_labels_after_rename()
            self.status_message.emit(f"Renamed sample → {new_name}")
            return

        if kind == "site":
            site = self._find_fov(payload[2])
            if site is None or site.name == new_name:
                return
            site.name = new_name
            site.metadata["site_name"] = new_name
            tip = self._site_tooltip(site)
            item.setToolTip(0, tip)
            item.setToolTip(1, tip)
            item.setText(1, site.contents_label())
            self._sync_labels_after_rename()
            self.status_message.emit(f"Renamed site → {new_name}")
            return

        if kind == "spectrum":
            fov = self._find_fov(payload[1])
            old_name = payload[2]
            if fov is None or old_name == new_name:
                return
            ms = self._find_spectrum(fov, old_name)
            if ms is None:
                return
            ms.name = new_name
            ms.spectrum.metadata["name"] = new_name
            item.setData(0, Qt.UserRole, ("spectrum", fov.id, new_name))
            self.status_message.emit(f"Renamed spectrum → {new_name}")
            return

        if kind == "linescan":
            fov = self._find_fov(payload[1])
            old_name = payload[2]
            if fov is None or old_name == new_name:
                return
            for ls in fov.line_scans:
                if ls.name == old_name:
                    ls.name = new_name
                    item.setData(0, Qt.UserRole, ("linescan", fov.id, new_name))
                    self.status_message.emit(f"Renamed line scan → {new_name}")
                    break

    @staticmethod
    def _find_spectrum(fov: MappingFOV, name: str) -> Optional[MapSpectrum]:
        for s in fov.spectra:
            if s.name == name:
                return s
        for ls in fov.line_scans:
            for s in ls.points:
                if s.name == name:
                    return s
        return None

    def _sync_labels_after_rename(self) -> None:
        """Refresh trees / active-site label after sample or site rename."""
        if self.current_fov is not None:
            self.active_site_label.setText(
                f"Active site: {self._sample_name_for_site(self.current_fov)} "
                f"→ {self.current_fov.name}"
            )
        # Rebuild sites tree to keep tooltips / structure, preserve selection id
        current_site_id = self.current_fov.id if self.current_fov else None
        self._populate_sites_tree()
        if current_site_id:
            self._select_site_in_sites_tree(current_site_id)
        # Update data tree root text if it shows the site
        if self.current_fov is not None and self.tree.topLevelItemCount():
            root = self.tree.topLevelItem(0)
            payload = root.data(0, Qt.UserRole)
            if payload and payload[0] == "site":
                self._tree_updating = True
                root.setText(0, self.current_fov.name)
                self._tree_updating = False
        self._fill_sample_tab()

    def _prompt_rename(self, item: QTreeWidgetItem, title: str) -> None:
        payload = item.data(0, Qt.UserRole)
        if not payload:
            return
        kind = payload[0]
        if kind not in ("sample", "site", "spectrum", "linescan"):
            return
        current = item.text(0)
        new_name, ok = QInputDialog.getText(
            self, title, "New name:", text=current
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == current:
            return
        item.setText(0, new_name)  # triggers itemChanged → apply

    def _rename_selected_sites_item(self) -> None:
        items = self.sites_tree.selectedItems()
        if not items:
            QMessageBox.information(self, "Rename", "Select a sample or site first.")
            return
        item = items[0]
        payload = item.data(0, Qt.UserRole)
        if not payload or payload[0] not in ("sample", "site"):
            QMessageBox.information(self, "Rename", "Select a sample or site to rename.")
            return
        self._prompt_rename(item, "Rename")

    def _rename_selected_data_item(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            QMessageBox.information(
                self, "Rename", "Select a site, spectrum, or line scan first."
            )
            return
        item = items[0]
        payload = item.data(0, Qt.UserRole)
        if not payload or payload[0] not in ("site", "spectrum", "linescan"):
            QMessageBox.information(
                self,
                "Rename",
                "Select a site, spectrum, or line-scan series to rename.\n"
                "(Element maps keep their IPJ labels.)",
            )
            return
        self._prompt_rename(item, "Rename")

    def _on_sites_context_menu(self, pos) -> None:
        item = self.sites_tree.itemAt(pos)
        if item is None:
            return
        payload = item.data(0, Qt.UserRole)
        menu = QMenu(self)
        if payload and payload[0] == "site":
            act_act = QAction("Activate site", self)
            act_act.triggered.connect(lambda: self._on_site_activated(item))
            menu.addAction(act_act)
        if payload and payload[0] in ("sample", "site"):
            act_ren = QAction("Rename…", self)
            act_ren.setShortcut(QKeySequence("F2"))
            act_ren.triggered.connect(lambda: self._prompt_rename(item, "Rename"))
            menu.addAction(act_ren)
        if not menu.actions():
            return
        menu.exec(self.sites_tree.viewport().mapToGlobal(pos))

    def _on_data_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        payload = item.data(0, Qt.UserRole)
        if not payload or payload[0] not in ("site", "spectrum", "linescan"):
            return
        if not item.isSelected():
            self.tree.clearSelection()
            item.setSelected(True)
        menu = QMenu(self)
        act_ren = QAction("Rename…", self)
        act_ren.setShortcut(QKeySequence("F2"))
        act_ren.triggered.connect(lambda: self._prompt_rename(item, "Rename"))
        menu.addAction(act_ren)
        if payload[0] == "spectrum":
            act_an = QAction("Send to Analysis", self)
            act_an.triggered.connect(self._send_selected_spectrum)
            menu.addAction(act_an)
            act_b = QAction("Send to Batch Analysis", self)
            act_b.triggered.connect(self._send_selected_to_batch)
            menu.addAction(act_b)
        elif payload[0] == "linescan":
            act_b = QAction("Send all points to Batch Analysis", self)
            act_b.triggered.connect(self._send_selected_to_batch)
            menu.addAction(act_b)
        elif payload[0] == "site":
            act_b = QAction("Send all site spectra to Batch Analysis", self)
            act_b.triggered.connect(self._send_site_to_batch)
            menu.addAction(act_b)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _checked_display_maps(self):
        """Element maps checked in the Data tree, in tree / map order."""
        fov = self.current_fov
        if fov is None or not self._checked_map_names:
            return []
        return [m for m in fov.element_maps if m.name in self._checked_map_names]

    def _sample_for_site(self, site: MappingFOV):
        if not self.project:
            return None
        sid = site.metadata.get("sample_id")
        if sid:
            return self.project.find_sample(sid)
        for sample in self.project.samples:
            if any(s.id == site.id for s in sample.sites):
                return sample
        return None

    def _sample_name_for_site(self, site: MappingFOV) -> str:
        sample = self._sample_for_site(site)
        return sample.name if sample else "Sample"

    @staticmethod
    def _site_tooltip(site: MappingFOV) -> str:
        bits = [site.name, "F2 / right-click to rename"]
        bits.extend(site.contents_tags())
        if site.optical is not None:
            bits.append("photo")
        return " · ".join(bits)

    def _find_fov(self, fov_id: str) -> Optional[MappingFOV]:
        if not self.project:
            return None
        return self.project.find_site(fov_id)

    def _on_sites_selection(self) -> None:
        items = self.sites_tree.selectedItems()
        if not items:
            return
        payload = items[0].data(0, Qt.UserRole)
        if payload and payload[0] == "whole_image":
            self._show_sample_camera(payload[1])

    def _on_site_activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        payload = item.data(0, Qt.UserRole) if item else None
        if payload and payload[0] == "site":
            site = self._find_fov(payload[2])
            if site:
                self._activate_site(site, switch_to_data=True)
        elif payload and payload[0] == "whole_image":
            self._show_sample_camera(payload[1], switch_to_data=True)

    def _show_sample_camera(self, sample_id: str, *, switch_to_data: bool = False) -> None:
        if not self.project:
            return
        sample = self.project.find_sample(sample_id)
        if sample is None or sample.whole_image is None:
            return
        if sample.sites:
            current_ids = {s.id for s in sample.sites}
            if self.current_fov is None or self.current_fov.id not in current_ids:
                self._activate_site(sample.sites[0], switch_to_data=switch_to_data)
        self._select_combo_kind("whole_image")
        self.info_label.setText(sample.whole_image.name or "Sample camera")

    def _activate_selected_site(self) -> None:
        items = self.sites_tree.selectedItems()
        if not items:
            QMessageBox.information(
                self, "Activate site", "Select a Site of Interest in the Sites tab."
            )
            return
        payload = items[0].data(0, Qt.UserRole)
        # Walk up if sample/project selected
        item = items[0]
        while item and (not payload or payload[0] != "site"):
            item = item.parent()
            payload = item.data(0, Qt.UserRole) if item else None
        if not payload or payload[0] != "site":
            QMessageBox.information(
                self, "Activate site", "Select a Site of Interest (not just the sample)."
            )
            return
        site = self._find_fov(payload[2])
        if site:
            self._activate_site(site, switch_to_data=True)

    def _activate_site(self, site: MappingFOV, *, switch_to_data: bool = False) -> None:
        self._particle_result = None
        if hasattr(self, "particle_table"):
            self.particle_table.setRowCount(0)
        self._matrix_names = []
        self._set_fov(site)
        self.active_site_label.setText(
            f"Active site: {self._sample_name_for_site(site)} → {site.name}"
        )
        self._populate_data_tree()
        self._refresh_canvas()  # after tree defaults checks
        self._select_site_in_sites_tree(site.id)
        if switch_to_data:
            self.nav_tabs.setCurrentIndex(1)
        self._fill_sample_tab()
        self.status_message.emit(f"Active site: {site.name}")
        # Keep Analysis Experimental Parameters in sync with the active site
        self._copy_sample_to_analysis(quiet=True)

    def _select_site_in_sites_tree(self, site_id: str) -> None:
        def walk(item: QTreeWidgetItem) -> Optional[QTreeWidgetItem]:
            payload = item.data(0, Qt.UserRole)
            if payload and payload[0] == "site" and payload[2] == site_id:
                return item
            for i in range(item.childCount()):
                found = walk(item.child(i))
                if found:
                    return found
            return None

        self.sites_tree.blockSignals(True)
        for i in range(self.sites_tree.topLevelItemCount()):
            found = walk(self.sites_tree.topLevelItem(i))
            if found:
                self.sites_tree.setCurrentItem(found)
                break
        self.sites_tree.blockSignals(False)

    def _set_fov(self, fov: MappingFOV) -> None:
        self.current_fov = fov
        self._picked_spectrum = None
        self._last_pick_xy = None
        self._last_line = None
        self._active_line_scan = None
        self._drawn_line_scan = None
        self._drawn_cache_key = None
        self._profile_source = None
        self._quant_table = None
        self._last_ls_profiles = None
        # Reset checked maps for new site (repopulated with defaults in data tree)
        self._checked_map_names = set()
        self.canvas.clear_line()
        self.canvas.clear_region()
        self._fill_map_combos()
        self._fill_profile_map_list()
        self._fill_ls_element_list()
        has_photo = self._sync_overlay_photo_controls()
        self.overlay_check.blockSignals(True)
        self.overlay_check.setChecked(has_photo)
        self.overlay_check.blockSignals(False)
        # RGB of a single map is just grayscale — don't leave it stuck on
        if len(fov.element_maps) < 2:
            self.rgb_check.blockSignals(True)
            self.rgb_check.setChecked(False)
            self.rgb_check.blockSignals(False)
        avail = self._photo_availability()
        if fov.optical is not None:
            self._select_combo_kind("optical")
        elif avail.get("whole_image"):
            self._select_combo_kind("whole_image")
        else:
            self._refresh_canvas()
        self._update_cube_controls()
        if fov.line_scans:
            self._plot_ipj_line_scan(fov.line_scans[0], switch_tab=False)
        self._update_workspace_for_fov(fov)

    def _fov_has_maps(self, fov: Optional[MappingFOV]) -> bool:
        if fov is None:
            return False
        return bool(
            fov.element_maps
            or fov.cube is not None
            or fov.overview is not None
            or fov.optical is not None
        )

    def _on_workspace_tab(self, index: int) -> None:
        if index == 1:
            self._update_line_scan_page()
            ls = self._collected_line_scan()
            if ls is not None and not self._last_ls_profiles:
                self._replot_collected_line_scan()
            elif ls is not None:
                self._refresh_ls_camera(ls)

    def _update_workspace_for_fov(
        self, fov: MappingFOV, *, prefer_line_scan: bool = False
    ) -> None:
        has_ls = bool(fov.line_scans)
        has_map = self._fov_has_maps(fov)
        self.workspace_tabs.blockSignals(True)
        if prefer_line_scan and has_ls:
            self.workspace_tabs.setCurrentIndex(1)
        elif has_ls and not has_map:
            self.workspace_tabs.setCurrentIndex(1)
        elif has_map and not has_ls:
            self.workspace_tabs.setCurrentIndex(0)
        self.workspace_tabs.blockSignals(False)
        self._update_line_scan_page()

    def _update_line_scan_page(self) -> None:
        ls = self._collected_line_scan()
        has = ls is not None and ls.n_points > 0
        self.ls_content_stack.setCurrentIndex(1 if has else 0)
        self.fit_line_btn.setEnabled(has)
        self.ls_replot_btn.setEnabled(has)
        self.ls_export_btn.setEnabled(has)
        self.ls_send_btn.setEnabled(has)
        self.ls_all_btn.setEnabled(has)
        self.ls_none_btn.setEnabled(has)
        self.ls_sync_btn.setEnabled(True)
        if not has:
            self.ls_status_label.setText("No collected line scan in this site.")
            return
        tag = ls.display_label()
        span = float(ls.distances().max()) if ls.n_points else 0.0
        if ls.is_multipoint:
            self.ls_status_label.setText(
                f"{tag}: {ls.name}  ·  {ls.n_points} spectra  ·  "
                f"{span:.2f} mm along stage axis (not collection-path distance)"
            )
        else:
            self.ls_status_label.setText(
                f"{tag}: {ls.name}  ·  {ls.n_points} spectra  ·  {span:.2f} mm"
            )

    def _collected_line_scan(self) -> Optional[LineScan]:
        """Instrument-collected line scan or multipoint (not a drawn map transect)."""
        items = self.tree.selectedItems()
        if items:
            payload = items[0].data(0, Qt.UserRole)
            if payload and payload[0] == "linescan":
                fov = self._find_fov(payload[1])
                if fov:
                    for ls in fov.line_scans:
                        if ls.name == payload[2] and ls.source != "drawn":
                            return ls
            if payload and payload[0] == "spectrum":
                fov = self._find_fov(payload[1])
                if fov:
                    for ls in fov.line_scans:
                        if ls.source == "drawn":
                            continue
                        if any(p.name == payload[2] for p in ls.points):
                            return ls
        if (
            self._active_line_scan is not None
            and self._active_line_scan.source != "drawn"
            and self._active_line_scan.n_points > 0
        ):
            return self._active_line_scan
        if self.current_fov:
            for ls in self.current_fov.line_scans:
                if ls.source != "drawn" and ls.n_points > 0:
                    return ls
        return None

    def _first_line_scan_on_sample(self) -> Optional[LineScan]:
        """Collected series on this site, or another site of the same sample."""
        ls = self._collected_line_scan()
        if ls is not None:
            return ls
        sample = self._sample_for_site(self.current_fov) if self.current_fov else None
        if sample is None:
            return None
        for site in sample.sites:
            for cand in site.line_scans:
                if cand.source != "drawn" and cand.n_points > 0:
                    return cand
        return None

    def _fill_profile_map_list(self) -> None:
        """Populate checklist: element maps (transect) + element ROIs (IPJ line scan)."""
        self.profile_map_list.blockSignals(True)
        self.profile_map_list.clear()
        fov = self.current_fov
        if fov is None:
            self.profile_map_list.blockSignals(False)
            return

        # --- Map intensity series (drawn transect) ---
        for m in fov.element_maps:
            item = QListWidgetItem(f"Map: {m.name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            vendor = m.metadata.get("source") not in ("cube_total", "cube_roi")
            if vendor or m.metadata.get("source") == "cube_roi":
                checked = True
            else:
                only_total = all(
                    x.metadata.get("source") == "cube_total" for x in fov.element_maps
                )
                checked = only_total
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            item.setData(Qt.UserRole, ("map", m.name))
            self.profile_map_list.addItem(item)

        # --- Spectral ROI series (drawn transect cube windows + XGT line-scan points) ---
        if fov.line_scans or fov.cube is not None:
            rois = self._line_scan_roi_candidates(fov)
            analysis_syms = {s for s, _ in self._analysis_element_energies()}
            for sym, e_kev in rois:
                item = QListWidgetItem(f"ROI: {sym} ({e_kev:.2f} keV)")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                if self._checked_roi_symbols:
                    checked = sym in self._checked_roi_symbols
                else:
                    checked = sym in analysis_syms
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                item.setData(Qt.UserRole, ("roi", sym, float(e_kev)))
                self.profile_map_list.addItem(item)

        self.profile_map_list.blockSignals(False)

    def _fill_ls_element_list(self) -> None:
        """Populate Line scan tab element checklist (ROI windows)."""
        self.ls_element_list.blockSignals(True)
        self.ls_element_list.clear()
        fov = self.current_fov
        if fov is None:
            self.ls_element_list.blockSignals(False)
            return
        rois = self._line_scan_roi_candidates(fov)
        analysis_syms = {s for s, _ in self._analysis_element_energies()}
        for sym, e_kev in rois:
            item = QListWidgetItem(f"{sym} ({e_kev:.2f} keV)")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if self._checked_roi_symbols:
                checked = sym in self._checked_roi_symbols
            else:
                checked = sym in analysis_syms
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            item.setData(Qt.UserRole, ("roi", sym, float(e_kev)))
            self.ls_element_list.addItem(item)
        self.ls_element_list.blockSignals(False)
        self._remember_checked_ls_rois()

    def _line_scan_roi_candidates(self, fov: MappingFOV):
        """Return [(symbol, energy_kev), ...] for line-scan ROI plotting."""
        out = []
        seen = set()

        def add(sym: str, e_kev: float):
            sym = str(sym).strip()
            if not sym or sym in seen or e_kev <= 0:
                return
            seen.add(sym)
            out.append((sym, float(e_kev)))

        # Peak labels on sum or first line point
        sources = []
        sum_ms = fov.sum_spectrum()
        if sum_ms is not None:
            sources.append(sum_ms)
        if fov.line_scans and fov.line_scans[0].points:
            sources.append(fov.line_scans[0].points[0])
        for ms in sources:
            for pl in ms.peak_labels or []:
                el = pl.get("element")
                e = pl.get("energy_kev")
                if el and e is not None:
                    add(el, float(e))

        # Elements from loaded maps (so Map: Ca Ka1 can drive a spectral ROI)
        for m in fov.element_maps:
            if m.metadata.get("source") == "cube_total":
                continue
            el = (m.element or "").strip()
            if not el:
                continue
            for s, e in self._analysis_element_energies(symbols=[el]):
                add(s, e)
                break

        # Analysis-selected elements
        for sym, e_kev in self._analysis_element_energies():
            add(sym, e_kev)

        # Previously checked symbols (keep them listed even if Analysis cleared)
        for sym in sorted(self._checked_roi_symbols):
            if sym in seen:
                continue
            for s, e in self._analysis_element_energies(symbols=[sym]):
                add(s, e)

        return out

    def _analysis_symbols(self, symbols=None):
        """Chemical symbols from Analysis, or from an explicit iterable."""
        if symbols is not None:
            return coerce_element_symbols(symbols)
        if self._element_panel is None:
            return []
        return coerce_element_symbols(self._element_panel.get_selected_elements())

    def _analysis_element_energies(self, symbols=None):
        """Yield (symbol, Ka-or-La energy_kev) for Analysis elements."""
        symbols = self._analysis_symbols(symbols)
        if not symbols:
            return []

        from core.xray_data import get_element_lines

        try:
            import xraylib as xrl

            def _z(sym: str) -> int:
                try:
                    return int(xrl.SymbolToAtomicNumber(sym))
                except Exception:
                    return 0
        except Exception:
            _FALLBACK_Z = {
                "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16,
                "Cl": 17, "K": 19, "Ca": 20, "Ti": 22, "Cr": 24, "Mn": 25,
                "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "As": 33,
                "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Mo": 42, "Ag": 47,
                "Ba": 56, "Pb": 82, "Rh": 45,
            }

            def _z(sym: str) -> int:
                return int(_FALLBACK_Z.get(sym, 0))

        result = []
        for sym in symbols:
            z = _z(sym)
            if z <= 0:
                continue
            lines = get_element_lines(sym, z)
            e_kev = None
            for series in ("K", "L", "M"):
                series_lines = lines.get(series) or []
                if series_lines:
                    e_kev = float(series_lines[0]["energy"])
                    break
            if e_kev and e_kev > 0:
                result.append((sym, e_kev))
        return result

    def _list_item_payload(self, item) -> tuple:
        data = item.data(Qt.UserRole) if item is not None else None
        if isinstance(data, (tuple, list)):
            return tuple(data)
        if isinstance(data, str) and data:
            return ("map", data)
        return ()

    def _list_item_checked(self, item) -> bool:
        if item is None:
            return False
        try:
            return int(item.checkState()) == int(Qt.Checked)
        except (TypeError, ValueError):
            return item.checkState() == Qt.Checked

    def _checked_profile_maps(self):
        """Return ElementMap list for checked map entries (drawn transect)."""
        fov = self.current_fov
        if fov is None:
            return []
        names = []
        for i in range(self.profile_map_list.count()):
            item = self.profile_map_list.item(i)
            if not self._list_item_checked(item):
                continue
            data = self._list_item_payload(item)
            if data and data[0] == "map":
                names.append(data[1])
            else:
                text = item.text() or ""
                if text.startswith("Map: "):
                    names.append(text[5:])
        return [m for m in fov.element_maps if m.name in names]

    def _checked_line_scan_rois(self):
        """Return [(symbol, energy_kev), ...] for checked Maps-tab ROI entries."""
        out = []
        for i in range(self.profile_map_list.count()):
            item = self.profile_map_list.item(i)
            if not self._list_item_checked(item):
                continue
            data = self._list_item_payload(item)
            if data and data[0] == "roi":
                out.append((data[1], float(data[2])))
        return out

    def _checked_ls_rois(self):
        """Return [(symbol, energy_kev), ...] for checked Line scan tab elements."""
        out = []
        for i in range(self.ls_element_list.count()):
            item = self.ls_element_list.item(i)
            if not self._list_item_checked(item):
                continue
            data = self._list_item_payload(item)
            if data and data[0] == "roi":
                out.append((data[1], float(data[2])))
        return out

    def _rois_from_checked_maps(self):
        """Element energy windows implied by checked Map: items."""
        out = []
        seen = set()
        for m in self._checked_profile_maps():
            if m.metadata.get("source") == "cube_total":
                continue
            el = (m.element or "").strip()
            if not el or el in seen:
                continue
            energies = self._analysis_element_energies(symbols=[el])
            if not energies:
                continue
            out.append(energies[0])
            seen.add(el)
        return out

    def _remember_checked_rois(self) -> None:
        syms = {sym for sym, _ in self._checked_line_scan_rois()}
        syms.update(sym for sym, _ in self._checked_ls_rois())
        self._checked_roi_symbols = syms

    def _remember_checked_ls_rois(self) -> None:
        self._remember_checked_rois()

    def _effective_element_rois(self):
        """ROIs for a drawn map transect: checked Map/ROI items, else Analysis."""
        out = []
        seen = set()
        for src in (self._checked_line_scan_rois(), self._rois_from_checked_maps()):
            for sym, e_kev in src:
                if sym in seen:
                    continue
                out.append((sym, e_kev))
                seen.add(sym)
        if out:
            return out
        for sym, e_kev in self._analysis_element_energies():
            if sym in seen:
                continue
            out.append((sym, e_kev))
            seen.add(sym)
        return out

    def _effective_ls_rois(self):
        """ROIs for a collected line scan: checked Line scan elements, else Analysis."""
        out = list(self._checked_ls_rois())
        if out:
            return out
        return list(self._analysis_element_energies())

    def _set_all_ls_elements(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        self.ls_element_list.blockSignals(True)
        for i in range(self.ls_element_list.count()):
            self.ls_element_list.item(i).setCheckState(state)
        self.ls_element_list.blockSignals(False)
        self._remember_checked_ls_rois()
        self._replot_collected_line_scan()

    def _on_ls_element_check_changed(self, _item=None) -> None:
        self._remember_checked_ls_rois()
        self._replot_collected_line_scan()

    def _sync_ls_elements_from_analysis(self) -> None:
        if self._element_panel is None:
            QMessageBox.information(
                self,
                "From Analysis",
                "Analysis element panel is not connected.",
            )
            return
        symbols = {
            str(s).strip()
            for s in (self._element_panel.get_selected_elements() or [])
            if s
        }
        if not symbols:
            QMessageBox.information(
                self,
                "From Analysis",
                "Select elements in Analysis → Elements first.\n\n"
                "Send the Sum Spectrum or one line-scan point to Analysis, "
                "pick elements, then return here.",
            )
            return
        self._checked_roi_symbols = set(symbols)
        self._fill_ls_element_list()
        self.ls_element_list.blockSignals(True)
        matched = 0
        for i in range(self.ls_element_list.count()):
            item = self.ls_element_list.item(i)
            data = self._list_item_payload(item)
            hit = bool(data) and data[0] == "roi" and data[1] in symbols
            item.setCheckState(Qt.Checked if hit else Qt.Unchecked)
            if hit:
                matched += 1
        self.ls_element_list.blockSignals(False)
        self._remember_checked_ls_rois()
        self._replot_collected_line_scan()
        self.status_message.emit(
            f"Line scan elements: {matched} checked from Analysis"
        )

    def _line_width(self) -> int:
        return max(1, int(self.line_width_spin.value()))

    def _on_line_width_changed(self, value: int) -> None:
        width = max(1, int(value))
        self.canvas.set_band_width(width)
        if self._last_line is not None:
            self.canvas.set_line(*self._last_line, width=width)
        self._drawn_line_scan = None
        self._drawn_cache_key = None
        if self._last_line is not None:
            self._replot_last_line()

    def _set_all_profile_maps(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        self.profile_map_list.blockSignals(True)
        for i in range(self.profile_map_list.count()):
            self.profile_map_list.item(i).setCheckState(state)
        self.profile_map_list.blockSignals(False)
        self._remember_checked_rois()
        self._replot_profile()

    def _sync_profile_maps_from_analysis(self) -> None:
        if self._element_panel is None:
            QMessageBox.information(
                self,
                "From Analysis",
                "Analysis element panel is not connected.",
            )
            return
        symbols = set(self._analysis_symbols())
        if not symbols:
            QMessageBox.information(
                self,
                "From Analysis",
                "Select elements in Analysis → Elements first.\n\n"
                "Tip for XGT line scans:\n"
                "1) Activate the map site and Send Sum Spectrum → Analysis, or\n"
                "2) Send one line-scan point → Analysis,\n"
                "then pick elements and return here → From Analysis.",
            )
            return

        # Remember ROI symbols to check, then rebuild list so new ROIs appear
        self._checked_roi_symbols = set(symbols)
        self._fill_profile_map_list()

        self.profile_map_list.blockSignals(True)
        matched_maps = 0
        matched_rois = 0
        for i in range(self.profile_map_list.count()):
            item = self.profile_map_list.item(i)
            data = self._list_item_payload(item)
            if data and data[0] == "roi":
                hit = data[1] in symbols
                item.setCheckState(Qt.Checked if hit else Qt.Unchecked)
                if hit:
                    matched_rois += 1
            else:
                name = data[1] if data and data[0] == "map" else item.text()
                fov = self.current_fov
                em = fov.find_map(name) if fov else None
                el = (em.element if em else "") or ""
                hit = any(
                    el == sym
                    or str(name).lower().startswith(sym.lower())
                    or f" {sym.lower()} " in f" {str(name).lower()} "
                    for sym in symbols
                )
                item.setCheckState(Qt.Checked if hit else Qt.Unchecked)
                if hit:
                    matched_maps += 1
        self.profile_map_list.blockSignals(False)
        self._remember_checked_rois()
        self._replot_profile()
        self.status_message.emit(
            f"Profile elements: {matched_rois} ROI(s), {matched_maps} map(s) from Analysis"
        )

    def _on_profile_map_check_changed(self, _item=None) -> None:
        self._remember_checked_rois()
        self._replot_profile()

    def _replot_profile(self) -> None:
        """Refresh the Maps-tab drawn-transect intensity profile."""
        if self._last_line is not None:
            self._replot_last_line()

    def _replot_collected_line_scan(self) -> None:
        ls = self._collected_line_scan()
        if ls is not None:
            self._plot_ipj_line_scan(ls, switch_tab=False)

    def _update_cube_controls(self) -> None:
        fov = self.current_fov
        has = fov is not None and fov.cube is not None
        self.roi_btn.setEnabled(has)
        self.roi_from_analysis_btn.setEnabled(has)
        self.pick_btn.setEnabled(has)
        for btn in (self.rect_btn, self.circle_btn, self.poly_btn):
            btn.setEnabled(has)
        acq = format_acquisition(fov.metadata if fov is not None else None)
        if fov is not None:
            self.acq_label.setText(acq)
        else:
            self.acq_label.setText("")
        if not has:
            self.pick_btn.setChecked(False)
            for btn in (self.rect_btn, self.circle_btn, self.poly_btn):
                btn.setChecked(False)
            self.cube_info.setText(
                "No hyperspectral cube in this FOV"
                + (f"\n{acq}" if acq else "")
            )
            return
        c = fov.cube
        cube_line = (
            f"Cube {c.n_channels} ch × {c.height}×{c.width} "
            f"({c.ev_per_channel:.0f} eV/ch)"
        )
        self.cube_info.setText(cube_line + (f"\n{acq}" if acq else ""))

    def _fill_map_combos(self) -> None:
        fov = self.current_fov
        for combo in (
            self.map_combo,
            self.r_combo,
            self.g_combo,
            self.b_combo,
            self.corr_a,
            self.corr_b,
            self.ratio_num,
            self.ratio_den,
            self.particle_map_combo,
        ):
            combo.blockSignals(True)
            combo.clear()

        if fov is None:
            for combo in (
                self.map_combo,
                self.r_combo,
                self.g_combo,
                self.b_combo,
                self.corr_a,
                self.corr_b,
                self.ratio_num,
                self.ratio_den,
                self.particle_map_combo,
            ):
                combo.blockSignals(False)
            return

        if fov.overview is not None:
            self.map_combo.addItem(f"Overview: {fov.overview.name}", ("overview",))
        sample = self._sample_for_site(fov)
        if sample is not None and sample.whole_image is not None:
            self.map_combo.addItem(
                f"Camera: {sample.whole_image.name}", ("whole_image",)
            )
        if fov.optical is not None:
            self.map_combo.addItem(f"Camera: {fov.optical.name}", ("optical",))
        for m in fov.element_maps:
            self.map_combo.addItem(m.name, ("map", m.name))
            for combo in (
                self.r_combo,
                self.g_combo,
                self.b_combo,
                self.corr_a,
                self.corr_b,
                self.ratio_num,
                self.ratio_den,
                self.particle_map_combo,
            ):
                combo.addItem(m.name, m.name)

        # Sensible RGB defaults: first three maps
        maps = fov.element_maps
        if len(maps) >= 1:
            self.r_combo.setCurrentIndex(0)
            self.ratio_num.setCurrentIndex(0)
            self.particle_map_combo.setCurrentIndex(0)
        if len(maps) >= 2:
            self.g_combo.setCurrentIndex(1)
            self.corr_b.setCurrentIndex(1)
            self.ratio_den.setCurrentIndex(1)
        if len(maps) >= 3:
            self.b_combo.setCurrentIndex(2)

        for combo in (
            self.map_combo,
            self.r_combo,
            self.g_combo,
            self.b_combo,
            self.corr_a,
            self.corr_b,
            self.ratio_num,
            self.ratio_den,
            self.particle_map_combo,
        ):
            combo.blockSignals(False)

    # ----------------------------------------------------------- display
    def _neighborhood_size(self) -> int:
        val = self.neighborhood_combo.currentData()
        return int(val) if val else 1

    def _enhance_kwargs(self) -> dict:
        return {
            "smooth": self.smooth_combo.currentData() or "none",
            "neighborhood": self._neighborhood_size(),
            "bin_factor": int(self.bin_combo.currentData() or 1),
            "scale": self.scale_combo.currentData() or "linear",
            "contrast": self.contrast_combo.currentData() or "none",
        }

    def _process_map_array(self, data: np.ndarray) -> np.ndarray:
        return enhance_map(data, **self._enhance_kwargs())

    def _display_array(self, data: np.ndarray) -> np.ndarray:
        method = self.interp_combo.currentData() or "none"
        factor = int(self.upsample_combo.currentData() or 1)
        return upsample_map(data, factor=factor, method=method)

    def _enhanced_element_map(self, em: Optional[ElementMap]) -> Optional[ElementMap]:
        if em is None:
            return None
        return ElementMap(
            name=em.name,
            data=self._process_map_array(em.data),
            line=em.line,
            element=em.element,
            metadata=dict(em.metadata),
        )

    def _on_enhance_changed(self) -> None:
        self._refresh_canvas()
        if self._last_pick_xy is not None and self.pick_btn.isChecked():
            self._on_pixel_clicked(*self._last_pick_xy)

    def _on_overlay_opacity(self, value: int) -> None:
        self.overlay_pct.setText(f"{int(value)}%")
        if self.overlay_check.isChecked():
            self._refresh_canvas()

    def _combo_kind(self) -> str:
        data = self.map_combo.currentData()
        if data and data[0]:
            return str(data[0])
        return ""

    def _on_map_combo_changed(self, _index: int = 0) -> None:
        """Honor Data-tree image selection even if RGB or map checks are on."""
        kind = self._combo_kind()
        if kind in ("optical", "whole_image", "overview"):
            self.rgb_check.blockSignals(True)
            self.rgb_check.setChecked(False)
            self.rgb_check.blockSignals(False)
            self._select_overlay_target(kind)
        if kind == "overview":
            self.overlay_check.blockSignals(True)
            self.overlay_check.setChecked(False)
            self.overlay_check.blockSignals(False)
        self._clear_map_checks()
        self._refresh_canvas()

    def _on_rgb_toggled(self, checked: bool) -> None:
        if checked:
            # RGB is a map view — jump off a photo so the composite is visible
            kind = self._combo_kind()
            if kind in ("optical", "whole_image", "overview"):
                fov = self.current_fov
                if fov is not None and fov.element_maps:
                    self.map_combo.blockSignals(True)
                    for i in range(self.map_combo.count()):
                        d = self.map_combo.itemData(i)
                        if d and d[0] == "map":
                            self.map_combo.setCurrentIndex(i)
                            break
                    self.map_combo.blockSignals(False)
        self._refresh_canvas()

    def _overlay_map_source(self):
        """Element map or None; RGB overlay is handled separately."""
        fov = self.current_fov
        if fov is None:
            return None
        checked = self._checked_display_maps()
        if len(checked) == 1:
            return checked[0]
        data = self.map_combo.currentData()
        if data and data[0] == "map":
            return fov.find_map(data[1])
        if checked:
            return checked[0]
        if fov.element_maps:
            return fov.element_maps[0]
        return None

    def _photo_availability(self) -> dict:
        """Which overlay backgrounds exist for the active site."""
        fov = self.current_fov
        if fov is None:
            return {"optical": False, "overview": False, "whole_image": False}
        sample = self._sample_for_site(fov)
        has_camera = sample is not None and sample.whole_image is not None
        return {
            "optical": fov.optical is not None,
            "overview": fov.overview is not None,
            "whole_image": has_camera,
        }

    def _preferred_overlay_target(self) -> str:
        avail = self._photo_availability()
        for key in ("optical", "overview", "whole_image"):
            if avail.get(key):
                return key
        return "optical"

    def _select_overlay_target(self, kind: str) -> None:
        for i in range(self.overlay_target.count()):
            if self.overlay_target.itemData(i) == kind:
                item = self.overlay_target.model().item(i)
                if item is not None and not item.isEnabled():
                    return
                if self.overlay_target.currentIndex() != i:
                    self.overlay_target.blockSignals(True)
                    self.overlay_target.setCurrentIndex(i)
                    self.overlay_target.blockSignals(False)
                return

    def _sync_overlay_photo_controls(self) -> bool:
        """Enable overlay widgets from available photos. Returns True if any photo exists."""
        avail = self._photo_availability()
        has_photo = any(avail.values())
        for w in (
            self.overlay_check,
            self.overlay_target,
            self.overlay_slider,
            self.overlay_cmap,
            self.overlay_mask_check,
        ):
            w.setEnabled(has_photo)
        model = self.overlay_target.model()
        for i in range(self.overlay_target.count()):
            key = self.overlay_target.itemData(i)
            item = model.item(i)
            if item is not None:
                item.setEnabled(bool(avail.get(key)))
        if has_photo:
            current = self.overlay_target.currentData()
            if not avail.get(current):
                self._select_overlay_target(self._preferred_overlay_target())
        return has_photo

    def _overlay_photo(self):
        """Return (photo array, short label) for the selected overlay target."""
        fov = self.current_fov
        if fov is None:
            return None, ""
        target = self.overlay_target.currentData() or "optical"
        if target == "whole_image":
            sample = self._sample_for_site(fov)
            if sample is not None and sample.whole_image is not None:
                return sample.whole_image.data, "sample camera"
            return None, ""
        if target == "overview":
            if fov.overview is not None:
                return fov.overview.data, "trans. x-ray"
            return None, ""
        if fov.optical is not None:
            return fov.optical.data, "map area photo"
        if fov.overview is not None:
            return fov.overview.data, "trans. x-ray"
        return None, ""

    def _show_photo_overlay(self) -> bool:
        fov = self.current_fov
        photo, photo_label = self._overlay_photo()
        if fov is None or photo is None:
            return False
        opacity = self.overlay_slider.value() / 100.0
        cmap = self.overlay_cmap.currentData() or "hot"
        mask_low = self.overlay_mask_check.isChecked()
        title = "Map on photo"
        coord = None
        overlay_rgb = None
        alpha = None

        if self.rgb_check.isChecked() and fov.element_maps:
            r_src = fov.find_map(self.r_combo.currentData() or "")
            g_src = fov.find_map(self.g_combo.currentData() or "")
            b_src = fov.find_map(self.b_combo.currentData() or "")
            r = self._enhanced_element_map(r_src)
            g = self._enhanced_element_map(g_src)
            b = self._enhanced_element_map(b_src)
            try:
                overlay_rgb = rgb_composite(r, g, b)
            except Exception as exc:
                self.status_message.emit(f"RGB overlay failed: {exc}")
                return False
            coord = overlay_rgb[:, :, 0]
            counts = None
            for src in (r_src, g_src, b_src):
                if src is None:
                    continue
                data = np.asarray(src.data, dtype=np.float64)
                counts = data if counts is None else np.maximum(counts, data)
            if counts is None:
                counts = np.max(overlay_rgb, axis=2)
            intensity = np.max(overlay_rgb, axis=2)
            alpha = overlay_alpha(counts, intensity, mask_low=mask_low)
            title = f"RGB on {photo_label}"
        else:
            em = self._overlay_map_source()
            if em is None:
                self.status_message.emit("No element map to overlay")
                return False
            processed = self._process_map_array(em.data)
            overlay_rgb, scaled = colorize_map(processed, cmap=cmap)
            coord = processed
            alpha = overlay_alpha(em.data, scaled, mask_low=mask_low)
            title = f"{em.name} on {photo_label}"

        try:
            dest_rect = None
            reg_how = None
            target = self.overlay_target.currentData() or "optical"
            if target == "whole_image":
                dest_rect, reg_how = self._map_dest_rect_on_sample_camera(photo)
            blended = overlay_on_photo(
                photo,
                overlay_rgb,
                alpha=alpha,
                opacity=opacity,
                dest_rect=dest_rect,
            )
            note = ""
            if target == "whole_image" and dest_rect is not None:
                note = f" (registered via {reg_how})" if reg_how else " (registered)"
            elif target == "whole_image":
                note = " (no stage rect — stretched; check map geometry)"
            if target == "whole_image":
                grid = embed_map_on_photo(
                    np.asarray(coord, dtype=np.float64),
                    np.asarray(photo).shape[:2],
                    dest_rect,
                )
                self.canvas.set_image(
                    grid, display=blended, rgb=True, title=title + note
                )
            else:
                self.canvas.set_image(
                    coord, display=blended, rgb=True, title=title + note
                )
        except Exception as exc:
            self.status_message.emit(f"Photo overlay failed: {exc}")
            return False
        return True

    def _stage_camera_for_photo(self, photo) -> Optional[StageCamera]:
        """Calibrated StageCamera for this sample overview (crop/red when possible)."""
        fov = self.current_fov
        sample = self._sample_for_site(fov) if fov is not None else None
        sites = list(sample.sites) if sample is not None else (
            [fov] if fov is not None else []
        )
        cam = camera_from_sample_sites(photo, sites)
        if cam is not None:
            self._ls_camera_model = cam
            return cam
        cam = camera_from_image(photo)
        if cam is not None:
            self._ls_camera_model = cam
        return cam

    def _map_dest_rect_on_sample_camera(self, photo) -> tuple:
        """Pixel rect + registration method for the active map on the sample camera.

        Returns ``(rect, method)`` where method is ``map-area photo``, ``red ROI``,
        ``stage``, or ``None`` if placement is unknown.
        """
        fov = self.current_fov
        if fov is None:
            return None, None
        photo_arr = np.asarray(photo)
        opt_shape = None
        if fov.optical is not None:
            opt_shape = tuple(np.asarray(fov.optical.data).shape)
        cache_key = (fov.id, tuple(photo_arr.shape), opt_shape)
        cached = self._cam_dest_rect_cache.get(cache_key)
        if cached is not None:
            return cached

        # 1) Exact MapAreaImage crop of the sample camera
        if fov.optical is not None:
            opt = np.asarray(fov.optical.data)
            rect = locate_image_crop(photo_arr, opt)
            if rect is not None:
                result = (rect, "map-area photo")
                self._cam_dest_rect_cache[cache_key] = result
                return result

            # 2) Magnified map-area optical resized to stage size, searched
            #    near the stage-predicted centre (common on newer XGT IPJs)
            size = fov.stage_size_mm
            if size is not None and fov.stage_center_mm is not None:
                cam0 = camera_from_image(photo_arr)
                if cam0 is not None:
                    tw = float(size[0]) / cam0.mm_per_px_x
                    th = float(size[1]) / cam0.mm_per_px_y
                    cx, cy = cam0.stage_to_pixel(*fov.stage_center_mm)
                    rect = locate_scaled_template(
                        photo_arr,
                        opt,
                        tw,
                        th,
                        center_xy=(cx, cy),
                    )
                    if rect is not None:
                        result = (rect, "map-area optical")
                        self._cam_dest_rect_cache[cache_key] = result
                        return result

        # 3) Instrument-drawn red map box on the overview BMP
        red = locate_red_map_rect(photo_arr)
        if red is not None:
            result = (red, "red ROI")
            self._cam_dest_rect_cache[cache_key] = result
            return result

        # 4) Stage mm → pixels (uses crop/optical/red-calibrated camera when available)
        bounds = fov.stage_bounds_mm()
        if bounds is None:
            return None, None
        cam = self._stage_camera_for_photo(photo_arr)
        if cam is None:
            return None, None
        rect = cam.stage_bounds_to_pixel_rect(bounds)
        result = (rect, "stage")
        self._cam_dest_rect_cache[cache_key] = result
        return result

    def _refresh_canvas(self) -> None:
        fov = self.current_fov
        if fov is None:
            return

        kind = self._combo_kind()
        if kind != "whole_image":
            self.canvas.clear_series_markers()
        photo, _label = self._overlay_photo()
        overlay_ok = self.overlay_check.isChecked() and photo is not None
        if overlay_ok:
            if self._show_photo_overlay():
                self._show_particle_markers()
                return

        # Photo / Trans. x-ray from the Data tree beat RGB and map checks
        if kind == "overview" and fov.overview is not None:
            self.canvas.set_image(
                fov.overview.data, rgb=False, title=fov.overview.name or "Overview"
            )
            return
        if kind == "optical" and fov.optical is not None:
            self._show_photo(fov.optical)
            return
        if kind == "whole_image":
            sample = self._sample_for_site(fov)
            if sample is not None and sample.whole_image is not None:
                self._show_photo(sample.whole_image)
                return

        checked = self._checked_display_maps()
        if checked and not self.rgb_check.isChecked():
            panels = []
            for m in checked:
                processed = self._process_map_array(m.data)
                panels.append((m.name, processed, self._display_array(processed)))
            try:
                self.canvas.set_images(panels)
            except Exception as exc:
                self.status_message.emit(f"Map display failed: {exc}")
            self._show_particle_markers()
            return

        if self.rgb_check.isChecked() and fov.element_maps:
            r = self._enhanced_element_map(fov.find_map(self.r_combo.currentData() or ""))
            g = self._enhanced_element_map(fov.find_map(self.g_combo.currentData() or ""))
            b = self._enhanced_element_map(fov.find_map(self.b_combo.currentData() or ""))
            try:
                rgb = rgb_composite(r, g, b)
                self.canvas.set_image(
                    rgb, display=self._display_array(rgb), rgb=True, title="RGB"
                )
            except Exception as exc:
                self.status_message.emit(f"RGB composite failed: {exc}")
            self._show_particle_markers()
            return

        data = self.map_combo.currentData()
        if not data:
            if fov.overview is not None:
                self.canvas.set_image(
                    fov.overview.data, rgb=False, title=fov.overview.name
                )
            elif fov.element_maps:
                m = fov.element_maps[0]
                processed = self._process_map_array(m.data)
                self.canvas.set_image(
                    processed,
                    display=self._display_array(processed),
                    rgb=False,
                    title=m.name,
                )
            self._show_particle_markers()
            return
        if data[0] == "map":
            m = fov.find_map(data[1])
            if m is not None:
                processed = self._process_map_array(m.data)
                self.canvas.set_image(
                    processed,
                    display=self._display_array(processed),
                    rgb=False,
                    title=m.name,
                )
            self._show_particle_markers()

    def _show_photo(self, image) -> None:
        arr = image.data
        rgb = arr.ndim == 3 and arr.shape[-1] >= 3
        try:
            self.canvas.set_image(arr, rgb=rgb, title=image.name or "Photo")
        except Exception as exc:
            self.status_message.emit(f"Photo display failed: {exc}")
            return
        kind = (getattr(image, "metadata", {}) or {}).get("kind")
        if kind == "whole_image" or self._combo_kind() == "whole_image":
            self._stage_camera_for_photo(image)
            self._overlay_line_scan_on_maps_camera()
        else:
            self.canvas.clear_series_markers()

    def _clear_map_checks(self) -> None:
        """Uncheck element maps so a photo/overview can occupy the canvas."""
        if not self._checked_map_names:
            return
        self._checked_map_names = set()
        self._tree_updating = True
        def walk(item: QTreeWidgetItem) -> None:
            payload = item.data(0, Qt.UserRole)
            if payload and payload[0] == "map":
                item.setCheckState(0, Qt.Unchecked)
            for i in range(item.childCount()):
                walk(item.child(i))
        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))
        self._tree_updating = False

    def _select_combo_kind(self, kind: str) -> None:
        self._clear_map_checks()
        for i in range(self.map_combo.count()):
            d = self.map_combo.itemData(i)
            if d and d[0] == kind:
                if self.map_combo.currentIndex() == i:
                    self._refresh_canvas()
                else:
                    self.map_combo.setCurrentIndex(i)
                return
        self._refresh_canvas()

    def _on_tree_selection(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        payload = items[0].data(0, Qt.UserRole)
        if not payload:
            return
        kind = payload[0]
        if kind == "hint":
            return
        if kind == "smartmap":
            fov = self._find_fov(payload[1])
            if fov and fov is not self.current_fov:
                self._activate_site(fov)
            # Prefer cube total / first element map
            for i in range(self.map_combo.count()):
                d = self.map_combo.itemData(i)
                if d and d[0] == "map":
                    self.map_combo.setCurrentIndex(i)
                    break
            self.info_label.setText("SmartMap hyperspectral cube")
        elif kind in ("site", "overview", "optical", "maps_folder"):
            fov_id = payload[1] if kind != "site" else payload[2]
            fov = self._find_fov(fov_id)
            if fov and fov is not self.current_fov:
                self._activate_site(fov)
            if kind == "overview":
                self._select_combo_kind("overview")
            elif kind == "optical":
                self._select_combo_kind("optical")
        elif kind == "whole_image":
            self._show_sample_camera(payload[1])
        elif kind == "map":
            fov = self._find_fov(payload[1])
            if fov and fov is not self.current_fov:
                self._activate_site(fov)
            for i in range(self.map_combo.count()):
                d = self.map_combo.itemData(i)
                if d and d[0] == "map" and d[1] == payload[2]:
                    self.map_combo.setCurrentIndex(i)
                    break
        elif kind == "spectrum":
            fov = self._find_fov(payload[1])
            if fov and fov is not self.current_fov:
                self._activate_site(fov)
            self.info_label.setText(f"Selected spectrum: {payload[2]}")
        elif kind == "linescan":
            fov = self._find_fov(payload[1])
            if fov and fov is not self.current_fov:
                self._activate_site(fov)
            if fov:
                for ls in fov.line_scans:
                    if ls.name == payload[2]:
                        self._plot_ipj_line_scan(ls, switch_tab=True)
                        self._update_line_scan_page()
                        break

    # -------------------------------------------------------- line tools
    def _uncheck_draw_buttons(self, except_btn=None) -> None:
        for btn in (
            self.line_mode_btn,
            self.pick_btn,
            self.rect_btn,
            self.circle_btn,
            self.poly_btn,
        ):
            if btn is except_btn:
                continue
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self.canvas.set_line_mode(False)
        self.canvas.set_pick_mode(False)
        if except_btn not in (self.rect_btn, self.circle_btn, self.poly_btn):
            self.canvas.set_region_mode(None)

    def _on_line_mode(self, checked: bool) -> None:
        if checked:
            self._uncheck_draw_buttons(self.line_mode_btn)
        self.canvas.set_line_mode(checked)
        if checked:
            self.status_message.emit("Line mode: click start, then end point on the map")

    def _on_pick_mode(self, checked: bool) -> None:
        if checked:
            self._uncheck_draw_buttons(self.pick_btn)
        self.canvas.set_pick_mode(checked)
        if checked:
            n = self._neighborhood_size()
            extra = f" ({n}×{n} sum)" if n > 1 else ""
            self.status_message.emit(
                f"Pick mode: click a map pixel to extract its cube spectrum{extra}"
            )

    def _on_region_mode(self, kind: str, checked: bool) -> None:
        btn = {"rect": self.rect_btn, "circle": self.circle_btn, "poly": self.poly_btn}[
            kind
        ]
        if checked:
            self._uncheck_draw_buttons(btn)
            self.canvas.set_region_mode(kind)
            tips = {
                "rect": "Rectangle: click two opposite corners (Shift = square)",
                "circle": "Circle: click center, then a point on the rim",
                "poly": "Polygon: click vertices; double-click or right-click to close",
            }
            self.status_message.emit(tips[kind])
        else:
            self.canvas.set_region_mode(None)

    def _clear_region(self) -> None:
        self.canvas.clear_region()
        for btn in (self.rect_btn, self.circle_btn, self.poly_btn):
            btn.setChecked(False)

    def _on_region_drawn(self, kind: str, params) -> None:
        fov = self.current_fov
        if fov is None or fov.cube is None:
            self.status_message.emit("No cube available for area sum")
            return
        ms = fov.spectrum_in_region(kind, params)
        for btn in (self.rect_btn, self.circle_btn, self.poly_btn):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self.canvas.set_region_mode(None)
        if ms is None:
            self.status_message.emit("Area is empty — draw a larger region")
            return
        # Keep in the site so it appears under Data
        existing = {s.name for s in fov.spectra}
        name = ms.name
        n = 2
        while name in existing:
            name = f"{ms.name} ({n})"
            n += 1
        ms.name = name
        ms.spectrum.metadata["name"] = name
        fov.spectra.append(ms)
        self._picked_spectrum = ms
        self._show_pixel_spectrum(ms)
        self._populate_data_tree()
        n_used = ms.metadata.get("n_pixels", 0)
        self.info_label.setText(
            f"{ms.name}: {ms.spectrum.total_counts:.0f} counts from {n_used} pixels "
            "— Send → Analysis, or draw another area"
        )
        self.status_message.emit(
            f"Area sum {ms.name}: {ms.spectrum.total_counts:.0f} counts"
        )

    def _clear_line(self) -> None:
        self.canvas.clear_line()
        self._reset_profile_plot()
        self._last_profiles = None
        self._last_line = None
        self._drawn_line_scan = None
        self._drawn_cache_key = None
        if self._profile_source == "drawn":
            self._profile_source = None

    def _replot_last_line(self) -> None:
        if self._last_line is None:
            return
        x0, y0, x1, y1 = self._last_line
        self._extract_and_plot_line(x0, y0, x1, y1, finish_mode=False)

    def _on_line_drawn(self, x0, y0, x1, y1) -> None:
        self._last_line = (float(x0), float(y0), float(x1), float(y1))
        self._profile_source = "drawn"
        self._drawn_line_scan = None
        self._drawn_cache_key = None
        self.canvas.set_band_width(self._line_width())
        self._extract_and_plot_line(x0, y0, x1, y1, finish_mode=True)

    def _extract_and_plot_line(
        self, x0, y0, x1, y1, *, finish_mode: bool = True
    ) -> None:
        fov = self.current_fov
        if fov is None:
            return
        start = (float(x0), float(y0))
        end = (float(x1), float(y1))
        width = self._line_width()
        maps = self._checked_profile_maps()
        element_maps = [
            m for m in maps if m.metadata.get("source") != "cube_total"
        ]
        profiles = {}
        if element_maps:
            profiles.update(
                extract_multi_element_profiles(
                    element_maps, start, end, width=width
                )
            )

        rois = self._effective_element_rois()
        if fov.cube is not None and rois:
            half = 0.10  # keV, matches ROI maps from Analysis
            have_el = {(m.element or "").strip() for m in element_maps}
            cube_rois = []
            for sym, e_kev in rois:
                if sym in have_el:
                    continue
                cube_rois.append((sym, float(e_kev) - half, float(e_kev) + half))
            if cube_rois:
                profiles.update(
                    extract_cube_element_profiles(
                        fov.cube, cube_rois, start, end, width=width
                    )
                )

        n = len(profiles)
        if profiles:
            self._last_profiles = profiles
            self._plot_profiles(profiles)
        elif maps:
            profiles = extract_multi_element_profiles(
                maps, start, end, width=width
            )
            self._last_profiles = profiles
            self._plot_profiles(profiles)
            n = len(profiles)
        elif fov.cube is not None and not rois:
            dist, _xs, _ys, counts = fov.cube.spectra_along_line(
                x0, y0, x1, y1, width=width
            )
            totals = counts.sum(axis=1)
            self._last_profiles = {"Total counts": (dist, totals)}
            self._plot_profiles(self._last_profiles)
            n = 1
        else:
            self._reset_profile_plot()
            self._last_profiles = None
            self.status_message.emit(
                "No per-element series for this transect — check maps under "
                "Tools, or select elements in Analysis"
            )
            return

        if finish_mode and fov.cube is not None:
            _, mean_counts = fov.cube.mean_spectrum_line(
                x0, y0, x1, y1, width=width
            )
            ms = fov.spectrum_at_pixel(
                int(round(0.5 * (x0 + x1))), int(round(0.5 * (y0 + y1)))
            )
            if ms is not None:
                if ms.spectrum.num_channels == mean_counts.size:
                    ms.spectrum.counts = mean_counts
                elif ms.spectrum.num_channels == mean_counts.size * 2:
                    fine = np.zeros(mean_counts.size * 2, dtype=np.float64)
                    fine[0::2] = mean_counts * 0.5
                    fine[1::2] = mean_counts * 0.5
                    ms.spectrum.counts = fine
                else:
                    ms.spectrum.counts = mean_counts
                wtag = f", {width} px wide" if width > 1 else ""
                ms.name = (
                    f"Line mean ({x0:.0f},{y0:.0f})→({x1:.0f},{y1:.0f}){wtag}"
                )
                ms.spectrum.metadata["name"] = ms.name
                ms.kind = "roi"
                self._picked_spectrum = ms

        if finish_mode:
            self.line_mode_btn.setChecked(False)
            self.canvas.set_line_mode(False)
        names = ", ".join(self._last_profiles.keys()) if self._last_profiles else ""
        extra = f", {width} px wide" if width > 1 else ""
        self.status_message.emit(
            f"Line profile ({x0:.0f},{y0:.0f}) → ({x1:.0f},{y1:.0f}): "
            f"{n} series{extra}"
            + (f" ({names})" if names else "")
        )

    def _on_pixel_clicked(self, x: float, y: float) -> None:
        fov = self.current_fov
        if fov is None or fov.cube is None:
            self.status_message.emit("No cube available for pixel spectrum")
            return
        self._last_pick_xy = (x, y)
        nhood = self._neighborhood_size()
        ms = fov.spectrum_at_pixel(int(round(x)), int(round(y)), neighborhood=nhood)
        if ms is None:
            return
        self._picked_spectrum = ms
        # Keep pick mode on so the next click updates the popup
        self.canvas.set_spot_markers([x], [y])
        self._show_pixel_spectrum(ms)
        n_used = ms.metadata.get("n_pixels", 1)
        self.info_label.setText(
            f"{ms.name}: {ms.spectrum.total_counts:.0f} counts"
            + (f" from {n_used} pixels" if nhood > 1 else "")
            + " — click another pixel to update, or Send → Analysis"
        )
        self.status_message.emit(f"Pixel spectrum {ms.name}")

    def _show_pixel_spectrum(self, ms: MapSpectrum) -> None:
        if self._pixel_popup is None:
            self._pixel_popup = PixelSpectrumPopup(self)
            self._pixel_popup.send_requested.connect(self.spectrum_send_requested.emit)
            self._pixel_popup.finished.connect(self._on_pixel_popup_closed)
        self._pixel_popup.set_spectrum(ms)

    def _on_pixel_popup_closed(self, _result: int = 0) -> None:
        # Leaving pick mode when the viewer is closed feels natural
        self.pick_btn.setChecked(False)
        self.canvas.set_pick_mode(False)

    def _add_roi_map(self) -> None:
        fov = self.current_fov
        if fov is None or fov.cube is None:
            QMessageBox.information(
                self, "ROI map", "Current FOV has no hyperspectral cube."
            )
            return
        e0 = float(self.roi_e0.value())
        e1 = float(self.roi_e1.value())
        if e1 < e0:
            e0, e1 = e1, e0
        em = fov.add_roi_map_from_cube(e0, e1)
        if em is None:
            return
        self._fill_map_combos()
        # Select new map
        for i in range(self.map_combo.count()):
            d = self.map_combo.itemData(i)
            if d and d[0] == "map" and d[1] == em.name:
                self.map_combo.setCurrentIndex(i)
                break
        self._refresh_canvas()
        self._populate_data_tree()
        self._fill_profile_map_list()
        # Ensure the new ROI is checked
        self.profile_map_list.blockSignals(True)
        for i in range(self.profile_map_list.count()):
            item = self.profile_map_list.item(i)
            data = self._list_item_payload(item)
            name = data[1] if data and data[0] == "map" else item.text()
            if name == em.name:
                item.setCheckState(Qt.Checked)
        self.profile_map_list.blockSignals(False)
        self._replot_last_line()
        self.status_message.emit(f"Added cube ROI map: {em.name}")

    def _add_roi_maps_from_analysis(self) -> None:
        fov = self.current_fov
        if fov is None or fov.cube is None:
            QMessageBox.information(
                self, "ROI maps", "Active site has no hyperspectral cube."
            )
            return
        if self._element_panel is None:
            QMessageBox.information(
                self, "ROI maps", "Analysis element panel is not connected."
            )
            return
        symbols = self._analysis_symbols()
        if not symbols:
            QMessageBox.information(
                self,
                "ROI maps",
                "Select elements in Analysis → Elements first.\n\n"
                "Workflow: Send Sum Spectrum → Analysis, find peaks / pick elements, "
                "then return here.",
            )
            return

        from core.xray_data import get_element_lines

        # Atomic number lookup for common symbols
        try:
            import xraylib as xrl

            def _z(sym: str) -> int:
                try:
                    return int(xrl.SymbolToAtomicNumber(sym))
                except Exception:
                    return 0
        except Exception:
            _FALLBACK_Z = {
                "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16,
                "Cl": 17, "K": 19, "Ca": 20, "Ti": 22, "Cr": 24, "Mn": 25,
                "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "As": 33,
                "Se": 34, "Br": 35, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
                "Mo": 42, "Ag": 47, "Cd": 48, "Sn": 50, "Ba": 56, "Pb": 82,
                "Rh": 45,
            }

            def _z(sym: str) -> int:
                return int(_FALLBACK_Z.get(sym, 0))

        half_width = 0.10  # keV
        created = []
        for sym in symbols:
            z = _z(sym)
            if z <= 0:
                continue
            lines = get_element_lines(sym, z)
            ka = None
            for series in ("K", "L", "M"):
                series_lines = lines.get(series) or []
                if series_lines:
                    ka = float(series_lines[0]["energy"])
                    break
            if ka is None or ka <= 0:
                continue
            e0, e1 = ka - half_width, ka + half_width
            name = f"{sym} Ka ROI" if lines.get("K") else f"{sym} ROI"
            em = fov.add_roi_map_from_cube(e0, e1, name=name)
            if em is not None:
                # Stash element symbol for sync matching
                em.element = sym
                created.append(em.name)

        self._fill_map_combos()
        self._fill_profile_map_list()
        self._populate_data_tree()
        # Check only the created / matching Analysis maps
        self.profile_map_list.blockSignals(True)
        for i in range(self.profile_map_list.count()):
            item = self.profile_map_list.item(i)
            data = self._list_item_payload(item)
            name = data[1] if data and data[0] == "map" else item.text()
            item.setCheckState(
                Qt.Checked if name in created else Qt.Unchecked
            )
        self.profile_map_list.blockSignals(False)
        self._replot_last_line()
        if created:
            self.status_message.emit(
                f"Added {len(created)} cube ROI map(s): {', '.join(created)}"
            )
        else:
            QMessageBox.information(
                self, "ROI maps", "Could not resolve Ka energies for selected elements."
            )

    def _send_sum_spectrum(self) -> None:
        fov = self.current_fov
        if fov is None:
            QMessageBox.information(self, "Sum Spectrum", "Activate a site first.")
            return
        ms = fov.sum_spectrum()
        if ms is None:
            QMessageBox.information(
                self,
                "Sum Spectrum",
                "This site has no Sum Spectrum in the IPJ.",
            )
            return
        self._picked_spectrum = None  # prefer explicit sum send
        # Select it in the data tree if present
        for i in range(self.tree.topLevelItemCount()):
            self._select_spectrum_in_tree(self.tree.topLevelItem(i), ms.name)
        self.spectrum_send_requested.emit(ms.spectrum, ms.peak_labels)
        self.status_message.emit(
            f"Sent Sum Spectrum “{ms.name}” to Analysis — "
            "identify elements, then use From Analysis / ROI maps from Analysis"
        )

    def _select_spectrum_in_tree(self, item: QTreeWidgetItem, name: str) -> bool:
        payload = item.data(0, Qt.UserRole)
        if payload and payload[0] == "spectrum" and payload[2] == name:
            self.tree.setCurrentItem(item)
            return True
        for i in range(item.childCount()):
            if self._select_spectrum_in_tree(item.child(i), name):
                return True
        return False

    def _reset_profile_plot(self) -> None:
        self.profile_plot.clear()
        legend = getattr(self.profile_plot.plotItem, "legend", None)
        if legend is not None:
            legend.clear()

    def _reset_ls_profile_plot(self) -> None:
        self.ls_profile_plot.clear()
        legend = getattr(self.ls_profile_plot.plotItem, "legend", None)
        if legend is not None:
            legend.clear()
        self.ls_profile_plot.addItem(self._ls_profile_hover)

    def _plot_profiles(self, profiles: dict) -> None:
        self._reset_profile_plot()
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
        for i, (name, (dist, vals)) in enumerate(profiles.items()):
            pen = pg.mkPen(colors[i % len(colors)], width=2)
            self.profile_plot.plot(dist, vals, pen=pen, name=name)

    def _line_scan_axis_labels(self, line_scan: LineScan) -> None:
        if line_scan.is_multipoint:
            bottom = "Position along transect"
        else:
            bottom = "Distance along scan"
        self.ls_profile_plot.setLabel("bottom", bottom, units="mm")
        self.ls_profile_plot.setLabel("left", "Counts / s in window")
        self.quant_plot.setLabel("bottom", bottom, units="mm")

    @staticmethod
    def _window_cps(spectrum, e_kev: float, half: float = 0.15) -> float:
        """Live-time-normalized counts in a ±half keV window."""
        e_ax = spectrum.energy
        mask = (e_ax >= e_kev - half) & (e_ax <= e_kev + half)
        counts = float(spectrum.counts[mask].sum()) if mask.any() else 0.0
        live = float(spectrum.live_time or 0.0)
        if live > 0:
            return counts / live
        return counts

    def _sample_camera_image(self):
        """Sample-level overview camera for the current site, if any."""
        fov = self.current_fov
        if fov is None:
            return None
        sample = self._sample_for_site(fov)
        if sample is None or sample.whole_image is None:
            return None
        return sample.whole_image

    def _refresh_ls_camera(self, line_scan: Optional[LineScan] = None) -> None:
        image = self._sample_camera_image()
        ls = line_scan or self._collected_line_scan()
        if image is None:
            self.ls_camera.clear_series_markers()
            self._ls_cam_px = self._ls_cam_py = None
            self._ls_camera_model = None
            self.ls_camera_label.setText(
                "No sample camera in this project — overlay needs the overview photo."
            )
            return
        sample = self._sample_for_site(self.current_fov) if self.current_fov else None
        sites = list(sample.sites) if sample is not None else []
        cam = camera_from_sample_sites(image, sites) or camera_from_image(image)
        self._ls_camera_model = cam
        rgb = image.data.ndim == 3 and image.data.shape[-1] >= 3
        try:
            self.ls_camera.set_image(
                image.data, rgb=rgb, title=image.name or "Sample camera"
            )
        except Exception as exc:
            self.status_message.emit(f"Sample camera display failed: {exc}")
            return
        if ls is None or cam is None:
            self.ls_camera.clear_series_markers()
            self._ls_cam_px = self._ls_cam_py = None
            return
        coords = ls.stage_xy()
        if coords is None:
            self.ls_camera.clear_series_markers()
            self._ls_cam_px = self._ls_cam_py = None
            self.ls_camera_label.setText(
                "This series has no stage coordinates to overlay on the camera."
            )
            return
        order = ls.plot_order()
        px, py = cam.stages_to_pixels(coords[order, 0], coords[order, 1])
        # Keep pixel arrays aligned with original point index for hover lookup
        px_all, py_all = cam.stages_to_pixels(coords[:, 0], coords[:, 1])
        self._ls_cam_px, self._ls_cam_py = px_all, py_all
        hi = self._ls_hover_index
        hi_plot = None
        if hi is not None:
            matches = np.where(order == hi)[0]
            if matches.size:
                hi_plot = int(matches[0])
        self.ls_camera.set_series_markers(
            px, py, highlight=hi_plot, connect=True
        )
        n = ls.n_points
        cal = (
            abs(float(cam.origin_x_mm)) > 1e-6
            or abs(float(cam.origin_y_mm)) > 1e-6
        )
        origin_note = (
            "origin calibrated from map/ROI"
            if cal
            else "origin at image centre (uncalibrated)"
        )
        self.ls_camera_label.setText(
            f"{n} points on the sample camera "
            f"({cam.fov_width_mm:.0f}×{cam.fov_height_mm:.0f} mm FOV, "
            f"{origin_note}). Hover the profile or click a point."
        )
        self._overlay_line_scan_on_maps_camera(ls)

    def _overlay_line_scan_on_maps_camera(self, line_scan: Optional[LineScan] = None) -> None:
        """If Maps is showing the sample camera, overlay the same points."""
        if self._combo_kind() != "whole_image":
            return
        ls = line_scan or self._first_line_scan_on_sample()
        cam = self._ls_camera_model
        if ls is None or cam is None:
            self.canvas.clear_series_markers()
            return
        coords = ls.stage_xy()
        if coords is None:
            self.canvas.clear_series_markers()
            return
        order = ls.plot_order()
        px, py = cam.stages_to_pixels(coords[order, 0], coords[order, 1])
        hi = self._ls_hover_index
        hi_plot = None
        if hi is not None:
            matches = np.where(order == hi)[0]
            if matches.size:
                hi_plot = int(matches[0])
        self.canvas.set_series_markers(px, py, highlight=hi_plot, connect=True)

    def _set_ls_hover_index(self, index: Optional[int]) -> None:
        ls = self._collected_line_scan()
        if index is not None and (ls is None or not (0 <= index < ls.n_points)):
            index = None
        if index == self._ls_hover_index and index is not None:
            self._sync_ls_hover_overlays()
            return
        self._ls_hover_index = index
        self._sync_ls_hover_overlays()
        if index is None or ls is None:
            return
        pt = ls.points[index]
        dist = ls.distances()
        x_mm = float(dist[index]) if dist.size > index else 0.0
        xy = ""
        if pt.x is not None and pt.y is not None:
            xy = f"  ·  stage ({pt.x:.3f}, {pt.y:.3f}) mm"
        self.ls_camera_label.setText(
            f"{pt.name}  ·  {x_mm:.2f} mm along transect{xy}"
        )

    def _sync_ls_hover_overlays(self) -> None:
        idx = self._ls_hover_index
        dist = None
        ls = self._collected_line_scan()
        if idx is not None and ls is not None:
            d = ls.distances()
            if d.size > idx:
                dist = float(d[idx])
        for line in (self._ls_profile_hover, self._ls_quant_hover):
            if dist is None:
                line.setVisible(False)
            else:
                line.setValue(dist)
                line.setVisible(True)
        if self._ls_cam_px is None or ls is None:
            return
        order = ls.plot_order()
        hi_plot = None
        if idx is not None:
            matches = np.where(order == idx)[0]
            if matches.size:
                hi_plot = int(matches[0])
        self.ls_camera.set_series_highlight(hi_plot)
        if self._combo_kind() == "whole_image":
            self.canvas.set_series_highlight(hi_plot)

    def _nearest_ls_point_by_distance(self, x_mm: float) -> Optional[int]:
        xs = self._ls_plot_x
        order = self._ls_plot_order
        if xs is None or order is None or xs.size == 0:
            ls = self._collected_line_scan()
            if ls is None:
                return None
            xs = ls.distances()
            order = ls.plot_order()
            if xs.size == 0:
                return None
        i = int(np.argmin(np.abs(xs - x_mm)))
        return int(order[i])

    def _nearest_ls_point_by_pixel(self, px: float, py: float) -> Optional[int]:
        if self._ls_cam_px is None or self._ls_cam_py is None:
            return None
        d2 = (self._ls_cam_px - px) ** 2 + (self._ls_cam_py - py) ** 2
        i = int(np.argmin(d2))
        # Ignore clicks far from any point (~3 mm at 26 px/mm ≈ 80 px)
        if float(np.sqrt(d2[i])) > 80.0:
            return None
        return i

    def _on_ls_profile_mouse(self, event) -> None:
        pos = event[0]
        for plot in (self.ls_profile_plot, self.quant_plot):
            if plot.sceneBoundingRect().contains(pos):
                mouse = plot.plotItem.vb.mapSceneToView(pos)
                idx = self._nearest_ls_point_by_distance(float(mouse.x()))
                self._set_ls_hover_index(idx)
                return

    def _on_ls_camera_cursor(self, x, y, _val) -> None:
        idx = self._nearest_ls_point_by_pixel(x, y)
        if idx is not None:
            self._set_ls_hover_index(idx)

    def _on_ls_camera_clicked(self, x, y) -> None:
        idx = self._nearest_ls_point_by_pixel(x, y)
        if idx is None:
            return
        self._set_ls_hover_index(idx)
        ls = self._collected_line_scan()
        if ls is None:
            return
        pt = ls.points[idx]
        self._select_spectrum_in_tree_root(pt.name)

    def _select_spectrum_in_tree_root(self, name: str) -> None:
        for i in range(self.tree.topLevelItemCount()):
            if self._select_spectrum_in_tree(self.tree.topLevelItem(i), name):
                return

    def _plot_ipj_line_scan(self, line_scan: LineScan, *, switch_tab: bool = False) -> None:
        """Plot windowed counts vs distance for a collected line / multipoint."""
        if line_scan.source == "drawn":
            return
        self._active_line_scan = line_scan
        self._profile_source = "ipj"
        if switch_tab:
            self.workspace_tabs.setCurrentIndex(1)
        self._reset_ls_profile_plot()
        if line_scan.n_points == 0:
            return
        self._line_scan_axis_labels(line_scan)
        xs = line_scan.distances()
        order = line_scan.plot_order()
        x_plot = xs[order]
        rois = self._effective_ls_rois()
        profiles = {}
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
        half = 0.15  # keV
        for ci, (sym, e_kev) in enumerate(rois):
            series = np.array(
                [self._window_cps(p.spectrum, e_kev, half) for p in line_scan.points],
                dtype=np.float64,
            )
            y_plot = series[order]
            label = f"{sym} ({e_kev:.2f} keV)"
            profiles[label] = (x_plot, y_plot)
            color = colors[ci % len(colors)]
            self.ls_profile_plot.plot(
                x_plot,
                y_plot,
                pen=pg.mkPen(color, width=2),
                symbol="o",
                symbolSize=7,
                symbolBrush=color,
                symbolPen=None,
                name=label,
            )

        if not profiles:
            totals = np.array(
                [
                    (
                        float(p.spectrum.total_counts) / float(p.spectrum.live_time)
                        if p.spectrum.live_time
                        else float(p.spectrum.total_counts)
                    )
                    for p in line_scan.points
                ],
                dtype=np.float64,
            )
            y_plot = totals[order]
            profiles = {"Total counts / s": (x_plot, y_plot)}
            self.ls_profile_plot.plot(
                x_plot,
                y_plot,
                pen=pg.mkPen("#377eb8", width=2),
                symbol="o",
                symbolSize=7,
                symbolBrush="#377eb8",
                symbolPen=None,
                name="Total counts / s",
            )

        n_roi = len(rois)
        span = float(xs.max()) if xs.size else 0.0
        where = (
            f"vs stage position ({span:.2f} mm span)"
            if line_scan.is_multipoint
            else f"vs scan distance ({span:.2f} mm)"
        )
        self.ls_info_label.setText(
            f"{line_scan.display_label()} “{line_scan.name}”: "
            + (
                f"{n_roi} element ROI(s) {where}. "
                "Fit / semi-quant uses this same list."
                if n_roi
                else f"only total counts {where}. Check elements above, or From Analysis."
            )
        )
        if n_roi == 0:
            self.status_message.emit(
                f"{line_scan.display_label()}: only total counts — "
                "check elements on the Line scan tab"
            )
        else:
            names = ", ".join(profiles.keys())
            self.status_message.emit(
                f"{line_scan.display_label()}: {n_roi} element series ({names})"
            )
        self._last_ls_profiles = profiles
        self._ls_plot_x = x_plot
        self._ls_plot_order = order
        self._refresh_ls_camera(line_scan)
        self._update_line_scan_page()

    def _on_cursor(self, x, y, val) -> None:
        self.cursor_label.setText(f"Cursor: x={x:.1f}, y={y:.1f}, value={val:.3g}")

    # ------------------------------------------------------ correlation
    def _plot_correlation(self) -> None:
        fov = self.current_fov
        if fov is None:
            return
        a = fov.find_map(self.corr_a.currentData() or "")
        b = fov.find_map(self.corr_b.currentData() or "")
        if a is None or b is None:
            QMessageBox.information(self, "Correlation", "Select two element maps.")
            return
        try:
            x, y, r, rho = map_correlation(a, b)
        except Exception as exc:
            QMessageBox.warning(self, "Correlation", str(exc))
            return
        self.corr_plot.clear()
        # Subsample for speed
        if len(x) > 8000:
            idx = np.random.default_rng(0).choice(len(x), 8000, replace=False)
            x, y = x[idx], y[idx]
        self.corr_plot.plot(
            x, y, pen=None, symbol="o", symbolSize=3,
            symbolBrush=(50, 100, 200, 80), symbolPen=None,
        )
        self.corr_plot.setLabel("bottom", a.name)
        self.corr_plot.setLabel("left", b.name)
        self.corr_plot.setTitle(f"Correlation  r={r:.3f}  ρ={rho:.3f}")
        self.map_plot_tabs.setCurrentIndex(1)
        self.status_message.emit(f"{a.name} vs {b.name}: Pearson r={r:.3f}, Spearman ρ={rho:.3f}")

    def _plot_correlation_matrix(self) -> None:
        maps = self._checked_display_maps()
        if len(maps) < 2:
            QMessageBox.information(
                self,
                "Correlation matrix",
                "Check at least two element maps in the Data tree.",
            )
            return
        try:
            matrix, names = map_correlation_matrix(maps, method="pearson")
        except Exception as exc:
            QMessageBox.warning(self, "Correlation matrix", str(exc))
            return
        self._matrix_names = list(names)
        n = len(names)
        filled = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
        self.matrix_image.setImage(filled, autoLevels=False)
        self.matrix_image.setLevels((-1.0, 1.0))
        ticks = [(i + 0.5, names[i]) for i in range(n)]
        self.matrix_plot.getAxis("bottom").setTicks([ticks])
        self.matrix_plot.getAxis("left").setTicks([ticks])
        self.matrix_plot.setTitle(f"Pearson r ({n} maps)")
        self.matrix_plot.setXRange(0, n, padding=0)
        self.matrix_plot.setYRange(0, n, padding=0)
        self.map_plot_tabs.setCurrentIndex(2)
        self.status_message.emit(f"Correlation matrix for {n} checked maps")

    def _on_matrix_clicked(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        names = self._matrix_names
        n = len(names)
        if n == 0:
            return
        view = self.matrix_plot.getViewBox()
        pos = view.mapSceneToView(event.scenePos())
        i = int(pos.x())
        j = int(pos.y())
        if not (0 <= i < n and 0 <= j < n) or i == j:
            return
        # Set corr combos to this pair and plot scatter
        self._set_combo_by_data(self.corr_a, names[i])
        self._set_combo_by_data(self.corr_b, names[j])
        self._plot_correlation()

    def _set_combo_by_data(self, combo: QComboBox, data) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    def _maps_after_upsert(self) -> None:
        """Refresh UI after adding derived maps."""
        self._fill_map_combos()
        self._populate_data_tree()
        self._fill_profile_map_list()
        self._refresh_canvas()

    def _add_ratio_map(self) -> None:
        fov = self.current_fov
        if fov is None:
            return
        a = fov.find_map(self.ratio_num.currentData() or "")
        b = fov.find_map(self.ratio_den.currentData() or "")
        if a is None or b is None:
            QMessageBox.information(self, "Ratio map", "Select two element maps.")
            return
        if a.name == b.name:
            QMessageBox.information(self, "Ratio map", "Choose different maps for A and B.")
            return
        try:
            data = ratio_map(a.data, b.data)
        except Exception as exc:
            QMessageBox.warning(self, "Ratio map", str(exc))
            return
        name = f"{a.name}/{b.name}"
        em = ElementMap(
            name=name,
            data=data,
            line=name,
            element="",
            metadata={"source": "ratio", "numerator": a.name, "denominator": b.name},
        )
        fov.upsert_map(em)
        self._checked_map_names.add(name)
        self._maps_after_upsert()
        self.status_message.emit(f"Added ratio map {name}")

    def _add_difference_map(self) -> None:
        fov = self.current_fov
        if fov is None:
            return
        a = fov.find_map(self.ratio_num.currentData() or "")
        b = fov.find_map(self.ratio_den.currentData() or "")
        if a is None or b is None:
            QMessageBox.information(self, "Difference map", "Select two element maps.")
            return
        try:
            data = difference_map(a.data, b.data)
        except Exception as exc:
            QMessageBox.warning(self, "Difference map", str(exc))
            return
        name = f"{a.name}−{b.name}"
        em = ElementMap(
            name=name,
            data=data,
            line=name,
            element="",
            metadata={"source": "difference", "a": a.name, "b": b.name},
        )
        fov.upsert_map(em)
        self._checked_map_names.add(name)
        self._maps_after_upsert()
        self.status_message.emit(f"Added difference map {name}")

    def _run_pca(self) -> None:
        fov = self.current_fov
        if fov is None:
            return
        maps = self._checked_display_maps()
        # Prefer non-derived maps for PCA input
        maps = [
            m
            for m in maps
            if m.metadata.get("source") not in ("pca", "particles")
        ]
        if len(maps) < 2:
            QMessageBox.information(
                self,
                "PCA",
                "Check at least two element maps in the Data tree "
                "(exclude previous PC maps).",
            )
            return
        n_comp = int(self.pca_n_spin.value())
        try:
            result = pca_element_maps(maps, n_components=n_comp)
        except Exception as exc:
            QMessageBox.warning(self, "PCA", str(exc))
            return
        fov.remove_maps_by_source("pca")
        for em in result.score_maps:
            fov.upsert_map(em)
            self._checked_map_names.add(em.name)
        self._maps_after_upsert()
        if self.pca_rgb_check.isChecked() and len(result.score_maps) >= 1:
            self._set_combo_by_data(self.r_combo, result.score_maps[0].name)
            if len(result.score_maps) >= 2:
                self._set_combo_by_data(self.g_combo, result.score_maps[1].name)
            if len(result.score_maps) >= 3:
                self._set_combo_by_data(self.b_combo, result.score_maps[2].name)
            self.rgb_check.blockSignals(True)
            self.rgb_check.setChecked(True)
            self.rgb_check.blockSignals(False)
            self._refresh_canvas()
        pcts = ", ".join(
            f"PC{i + 1}={p * 100:.1f}%"
            for i, p in enumerate(result.explained_variance_ratio)
        )
        self.status_message.emit(f"PCA on {len(maps)} maps: {pcts}")

    def _find_particles(self) -> None:
        fov = self.current_fov
        if fov is None:
            return
        name = self.particle_map_combo.currentData() or ""
        em = fov.find_map(name)
        if em is None:
            QMessageBox.information(self, "Particles", "Select a map.")
            return
        try:
            result = find_particles(
                em.data,
                threshold_percentile=float(self.particle_thr_spin.value()),
                min_area=int(self.particle_min_area.value()),
                source_map=em.name,
                element_maps=fov.element_maps[:12],
            )
        except Exception as exc:
            QMessageBox.warning(self, "Particles", str(exc))
            return
        self._particle_result = result
        label_em = particle_label_map_as_element(result, name="Particles")
        fov.upsert_map(label_em)
        self._checked_map_names.add(label_em.name)
        self._fill_particle_table(result)
        self._maps_after_upsert()
        self._show_particle_markers()
        self.status_message.emit(
            f"Found {len(result.particles)} particles on {em.name} "
            f"(threshold={result.threshold:.3g})"
        )

    def _fill_particle_table(self, result) -> None:
        self.particle_table.setRowCount(0)
        for p in result.particles:
            row = self.particle_table.rowCount()
            self.particle_table.insertRow(row)
            cx, cy = p.centroid_xy
            vals = [
                str(p.id),
                str(p.area_px),
                f"{p.mean_intensity:.3g}",
                f"({cx:.1f}, {cy:.1f})",
            ]
            for col, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, p.id)
                self.particle_table.setItem(row, col, item)

    def _show_particle_markers(self) -> None:
        result = self._particle_result
        if result is None or not result.particles:
            self.canvas.clear_series_markers()
            return
        xs = [p.centroid_xy[0] for p in result.particles]
        ys = [p.centroid_xy[1] for p in result.particles]
        self.canvas.set_series_markers(xs, ys, connect=False)

    def _clear_particles(self) -> None:
        self._particle_result = None
        self.particle_table.setRowCount(0)
        self.canvas.clear_series_markers()
        fov = self.current_fov
        if fov is not None:
            fov.remove_maps_by_source("particles")
            self._checked_map_names.discard("Particles")
            self._maps_after_upsert()
        self.status_message.emit("Cleared particles")

    def _on_particle_activated(self, row: int, _col: int) -> None:
        result = self._particle_result
        fov = self.current_fov
        if result is None or fov is None:
            return
        item = self.particle_table.item(row, 0)
        if item is None:
            return
        pid = item.data(Qt.UserRole)
        particle = next((p for p in result.particles if p.id == pid), None)
        if particle is None:
            return
        self.canvas.set_series_highlight(row)
        if fov.cube is None:
            self.status_message.emit(
                f"Particle {particle.id}: area={particle.area_px} px "
                f"(no cube for spectrum sum)"
            )
            return
        mask = result.label_map == particle.id
        try:
            counts, n_pix = fov.cube.spectrum_in_mask(mask)
        except Exception as exc:
            QMessageBox.warning(self, "Particle spectrum", str(exc))
            return
        if n_pix < 1:
            return
        from core.spectrum import Spectrum

        energy = fov.cube.energy_axis_kev()
        dwell = fov.estimated_dwell_s() or 1.0
        sum_ms = fov.sum_spectrum()
        ms = MapSpectrum(
            spectrum=Spectrum(
                energy=energy,
                counts=counts.astype(np.float64),
                live_time=float(dwell) * float(n_pix),
                real_time=float(dwell) * float(n_pix),
                metadata={"name": f"Particle {particle.id}"},
            ),
            name=f"Particle {particle.id}",
            x=particle.centroid_xy[0],
            y=particle.centroid_xy[1],
            kind="roi",
            peak_labels=list(sum_ms.peak_labels) if sum_ms else [],
        )
        self._picked_spectrum = ms
        self._show_pixel_spectrum(ms)
        self.status_message.emit(
            f"Particle {particle.id}: {particle.area_px} px, "
            f"mean={particle.mean_intensity:.3g}"
        )

    # ----------------------------------------------- send / fit / export
    def _selected_map_spectrum(self) -> Optional[MapSpectrum]:
        if self._picked_spectrum is not None:
            # Prefer explicit cube pick unless tree has a spectrum selected
            items = self.tree.selectedItems()
            tree_is_spectrum = False
            if items:
                payload = items[0].data(0, Qt.UserRole)
                if payload and payload[0] == "spectrum":
                    tree_is_spectrum = True
            if not tree_is_spectrum:
                return self._picked_spectrum
        items = self.tree.selectedItems()
        if items:
            payload = items[0].data(0, Qt.UserRole)
            if payload and payload[0] == "spectrum":
                fov = self._find_fov(payload[1])
                if fov:
                    for s in fov.spectra:
                        if s.name == payload[2]:
                            return s
                    for ls in fov.line_scans:
                        for s in ls.points:
                            if s.name == payload[2]:
                                return s
        # Fallback: picked pixel, then sum spectrum of current FOV
        if self._picked_spectrum is not None:
            return self._picked_spectrum
        if self.current_fov:
            s = self.current_fov.sum_spectrum()
            if s:
                return s
            if self.current_fov.spectra:
                return self.current_fov.spectra[0]
        return None

    def _send_selected_spectrum(self) -> None:
        ms = self._selected_map_spectrum()
        if ms is None:
            QMessageBox.information(
                self,
                "Send to Analysis",
                "Select a spectrum in the tree, or Pick pixel spectrum from the cube.",
            )
            return
        self.spectrum_send_requested.emit(ms.spectrum, ms.peak_labels)
        self.status_message.emit(f"Sent “{ms.name}” to Analysis")

    def _batch_display_name(self, ms: MapSpectrum, fov: Optional[MappingFOV] = None) -> str:
        site = fov or (
            self.project.find_fov_for_spectrum(ms) if self.project else None
        )
        if site is None:
            return ms.name
        sample = self._sample_name_for_site(site)
        n_sites = len(self.project.fovs) if self.project else 1
        if n_sites > 1:
            return f"{sample} / {site.name} / {ms.name}"
        return ms.name

    def _pairs_for_batch(self, spectra: list) -> list:
        pairs = []
        for ms in spectra:
            fov = (
                self.project.find_fov_for_spectrum(ms)
                if self.project
                else self.current_fov
            )
            spec = ms.spectrum
            if isinstance(spec.metadata, dict):
                spec.metadata.setdefault("name", ms.name)
            pairs.append((self._batch_display_name(ms, fov), spec))
        return pairs

    def _emit_batch_spectra(self, spectra: list, *, empty_message: str) -> None:
        if not spectra:
            QMessageBox.information(self, "Send to Batch", empty_message)
            return
        self.spectra_batch_requested.emit(self._pairs_for_batch(spectra))
        noun = "spectrum" if len(spectra) == 1 else "spectra"
        self.status_message.emit(f"Queued {len(spectra)} {noun} in Batch Analysis")

    def _spectra_from_tree_item(self, item: QTreeWidgetItem) -> list:
        payload = item.data(0, Qt.UserRole) if item else None
        if not payload:
            return []
        kind = payload[0]
        if kind == "spectrum":
            fov = self._find_fov(payload[1])
            if fov is None:
                return []
            name = payload[2]
            for s in fov.spectra:
                if s.name == name:
                    return [s]
            for ls in fov.line_scans:
                for s in ls.points:
                    if s.name == name:
                        return [s]
            return []
        if kind == "linescan":
            fov = self._find_fov(payload[1])
            if fov is None:
                return []
            for ls in fov.line_scans:
                if ls.name == payload[2]:
                    return list(ls.points)
            return []
        if kind == "site":
            fov_id = payload[2] if len(payload) > 2 else payload[1]
            fov = self._find_fov(fov_id)
            if fov is None:
                return []
            return fov.point_spectra()
        return []

    def _collect_selected_map_spectra(self) -> list:
        items = self.tree.selectedItems()
        out = []
        seen: set[int] = set()
        for item in items:
            for ms in self._spectra_from_tree_item(item):
                key = id(ms)
                if key in seen:
                    continue
                seen.add(key)
                out.append(ms)
        return out

    def _select_all_spectra(self) -> None:
        self.tree.clearSelection()
        count = 0

        def walk(item: QTreeWidgetItem) -> None:
            nonlocal count
            payload = item.data(0, Qt.UserRole)
            if payload and payload[0] == "spectrum":
                found = self._spectra_from_tree_item(item)
                if found and MappingFOV._is_sum_spectrum(found[0]):
                    pass
                else:
                    item.setSelected(True)
                    count += 1
            for i in range(item.childCount()):
                walk(item.child(i))

        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))
        self.tree.blockSignals(False)
        if count:
            self.status_message.emit(f"Selected {count} spectra in this site")
        else:
            QMessageBox.information(
                self,
                "Select spectra",
                "This site has no spot or line-scan spectra to select.",
            )

    def _send_selected_to_batch(self) -> None:
        spectra = self._collect_selected_map_spectra()
        self._emit_batch_spectra(
            spectra,
            empty_message=(
                "Select one or more spectra in Data (Shift/Ctrl-click), "
                "or a line-scan folder, then send to Batch.\n\n"
                "Use Select all spectra for every point in this site."
            ),
        )

    def _send_site_to_batch(self) -> None:
        fov = self.current_fov
        spectra = fov.point_spectra() if fov else []
        self._emit_batch_spectra(
            spectra,
            empty_message="This site has no spot or line/multipoint spectra to send.",
        )

    def _send_project_to_batch(self) -> None:
        spectra = self.project.point_spectra() if self.project else []
        if not spectra and self.project:
            spectra = self.project.all_spectra()
        self._emit_batch_spectra(
            spectra,
            empty_message="This project has no spectra to send to Batch.",
        )

    def _send_line_scan_to_batch(self) -> None:
        line_scan = self._collected_line_scan()
        if line_scan is None or line_scan.n_points == 0:
            QMessageBox.information(
                self,
                "Send to Batch",
                "No collected line scan or multipoint series in this site.",
            )
            return
        selected = self._collect_selected_map_spectra()
        point_ids = {id(p) for p in line_scan.points}
        subset = [s for s in selected if id(s) in point_ids]
        spectra = subset if subset else list(line_scan.points)
        self._emit_batch_spectra(
            spectra,
            empty_message="This line scan has no points to send.",
        )

    def _fit_line_scan(self) -> None:
        if self._fitter is None or self._element_panel is None:
            QMessageBox.warning(
                self, "Fit line scan", "Analysis fitter is not connected."
            )
            return
        self.workspace_tabs.setCurrentIndex(1)
        line_scan = self._collected_line_scan()
        if line_scan is None or line_scan.n_points == 0:
            QMessageBox.information(
                self,
                "Fit / semi-quant",
                "Semi-quant along a transect is only for collected line scans "
                "or multipoint series (spectra the instrument acquired along "
                "a path).\n\n"
                "A line drawn on a map is an intensity profile — use the Maps "
                "tab for that. Activate a line-scan site, or select the series "
                "in Data.",
            )
            return

        rois = self._checked_ls_rois()
        if rois:
            elements = coerce_element_symbols(sym for sym, _ in rois)
            auto_find = False
        else:
            elements = coerce_element_symbols(
                self._element_panel.get_selected_elements() or []
            )
            auto_find = bool(
                self._element_panel.get_fitting_params().get("auto_find_peaks", True)
            )
        if not elements:
            QMessageBox.information(
                self,
                "Fit / semi-quant",
                "Check the elements you want on this line (the same list as "
                "the ROI profile), or select them in Analysis → Elements "
                "and click From Analysis.",
            )
            return

        fit_params = self._element_panel.get_fitting_params()
        exp_params = self._element_panel.get_experimental_params()
        background_method = str(fit_params.get("background_method", "snip")).lower()
        peak_shape = str(fit_params.get("peak_shape", "gaussian")).lower()

        distances = line_scan.distances()
        rows = []
        self.quant_plot.clear()
        legend = getattr(self.quant_plot.plotItem, "legend", None)
        if legend is not None:
            legend.clear()
        self.quant_plot.addItem(self._ls_quant_hover)

        n = line_scan.n_points
        progress = QProgressDialog(
            f"Fitting {line_scan.display_label().lower()}…",
            "Cancel",
            0,
            n,
            self,
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        canceled = False
        try:
            for i, pt in enumerate(line_scan.points):
                progress.setValue(i)
                progress.setLabelText(
                    f"Fitting point {i + 1}/{n} ({pt.name})"
                )
                if progress.wasCanceled():
                    canceled = True
                    break
                sp = pt.spectrum
                result = self._fitter.fit_spectrum(
                    energy=sp.energy,
                    counts=sp.counts,
                    elements=elements,
                    background_method=background_method,
                    peak_shape=peak_shape,
                    auto_find_peaks=auto_find,
                    tube_element=fit_params.get("tube_element", "Rh"),
                    excitation_kv=fit_params.get("excitation_kv", 50.0),
                    include_tube_lines=fit_params.get("include_tube_lines", True),
                    include_compton=fit_params.get("include_compton", True),
                    scatter_angle_deg=fit_params.get("scatter_angle_deg", 90.0),
                    compton_fwhm_kev=fit_params.get("compton_fwhm_kev", 0.250),
                    experimental_params=exp_params,
                    prominence_percent=fit_params.get("prominence_percent"),
                    min_height=fit_params.get("min_height"),
                    min_separation_ev=fit_params.get("min_separation_ev"),
                )
                quant = self._fitter.quantify_elements(result.peaks, exp_params)
                row = {
                    "index": i,
                    "name": pt.name,
                    "distance": float(distances[i]),
                    "x": "" if pt.x is None else float(pt.x),
                    "y": "" if pt.y is None else float(pt.y),
                    "live_time": float(pt.spectrum.live_time or 0.0),
                }
                if isinstance(quant, dict):
                    for k, v in quant.items():
                        if isinstance(v, dict):
                            row[str(k)] = float(
                                v.get(
                                    "relative_intensity_pct",
                                    v.get("concentration", 0.0),
                                )
                            )
                        elif isinstance(v, (int, float)):
                            row[str(k)] = float(v)
                rows.append(row)
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "Fit line scan failed", str(exc))
            return
        progress.setValue(n)
        progress.close()

        if not rows:
            return

        self._quant_table = rows
        self._quant_distances = distances[: len(rows)]
        self._line_scan_axis_labels(line_scan)
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
        skip = {"index", "name", "distance", "x", "y", "live_time"}
        element_keys = [k for k in rows[0].keys() if k not in skip]
        plot_x = np.asarray([r["distance"] for r in rows], dtype=np.float64)
        order = (
            np.argsort(plot_x, kind="mergesort")
            if line_scan.is_multipoint
            else np.arange(len(rows))
        )
        x_plot = plot_x[order]
        for ci, key in enumerate(element_keys):
            ys = np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)
            color = colors[ci % len(colors)]
            self.quant_plot.plot(
                x_plot,
                ys[order],
                pen=pg.mkPen(color, width=2),
                symbol="o",
                symbolSize=7,
                symbolBrush=color,
                symbolPen=None,
                name=key,
            )
        done = (
            f"Semi-quant along {line_scan.name}: {len(rows)}/{n} points, "
            f"{len(element_keys)} elements (area-normalized relative %)"
        )
        if canceled:
            done += " — stopped early"
        self.ls_info_label.setText(done)
        self.status_message.emit(
            "Line-scan fit complete" if not canceled else "Line-scan fit canceled"
        )

    def _write_profile_csv(self, path: str, profiles: dict) -> None:
        import csv

        names = list(profiles.keys())
        dist = profiles[names[0]][0]
        cols = {"distance": dist}
        for name in names:
            cols[name] = profiles[name][1]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(cols.keys()))
            for i in range(len(dist)):
                w.writerow([cols[k][i] for k in cols])

    def _export_map_profile_csv(self) -> None:
        if not self._last_profiles:
            QMessageBox.information(
                self,
                "Export",
                "No drawn-transect profile to export yet. Draw a line on the map.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export map profile CSV", "", "CSV (*.csv)"
        )
        if not path:
            return
        self._write_profile_csv(path, self._last_profiles)
        self.status_message.emit(f"Exported map profile → {path}")

    def _export_line_scan_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export line-scan CSV", "", "CSV (*.csv)"
        )
        if not path:
            return
        if self._quant_table:
            import csv

            keys = list(self._quant_table[0].keys())
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self._quant_table)
            self.status_message.emit(f"Exported semi-quant table → {path}")
            return
        if not self._last_ls_profiles:
            QMessageBox.information(
                self,
                "Export",
                "No line-scan profile to export yet. Check elements and replot, "
                "or run Fit / semi-quant.",
            )
            return
        self._write_profile_csv(path, self._last_ls_profiles)
        self.status_message.emit(f"Exported ROI profile → {path}")
