"""
Mapping workspace: load INCA/XGT .ipj projects, visualize element maps,
draw line profiles, correlate elements, and send spectra to Analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.mapping.correlations import map_correlation, rgb_composite
from core.mapping.models import LineScan, MappingFOV, MappingProject, MapSpectrum
from core.mapping.profiles import extract_multi_element_profiles
from ui.collapsible_section import CollapsibleSection
from ui.map_canvas import MapCanvas
from ui.pixel_spectrum_popup import PixelSpectrumPopup


class MappingPanel(QWidget):
    """Top-level Mapping tab widget."""

    spectrum_send_requested = Signal(object, object)  # Spectrum, peak_labels list
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[MappingProject] = None
        self.current_fov: Optional[MappingFOV] = None
        self._fitter = None
        self._element_panel = None
        self._quant_distances: Optional[np.ndarray] = None
        self._quant_table = None  # list of dicts
        self._picked_spectrum: Optional[MapSpectrum] = None
        self._last_line: Optional[tuple] = None  # (x0, y0, x1, y1)
        self._pixel_popup: Optional[PixelSpectrumPopup] = None
        self._checked_map_names: set[str] = set()
        self._checked_roi_symbols: set[str] = set()
        self._active_line_scan: Optional[LineScan] = None
        self._tree_updating = False

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
        left_layout.addLayout(btn_row)

        self.active_site_label = QLabel("Active site: —")
        self.active_site_label.setWordWrap(True)
        left_layout.addWidget(self.active_site_label)

        self.nav_tabs = QTabWidget()
        self.nav_tabs.setMinimumHeight(160)
        self.nav_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Sites tab: Project → Sample → Site of Interest
        sites_page = QWidget()
        sites_layout = QVBoxLayout(sites_page)
        sites_layout.setContentsMargins(0, 0, 0, 0)
        self.sites_tree = QTreeWidget()
        self.sites_tree.setHeaderLabels(["Sites"])
        self.sites_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sites_tree.setEditTriggers(QAbstractItemView.EditKeyPressed)
        self.sites_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sites_tree.customContextMenuRequested.connect(self._on_sites_context_menu)
        self.sites_tree.itemSelectionChanged.connect(self._on_sites_selection)
        self.sites_tree.itemDoubleClicked.connect(self._on_site_activated)
        self.sites_tree.itemChanged.connect(self._on_sites_tree_item_changed)
        sites_layout.addWidget(self.sites_tree)
        sites_hint = QLabel("F2 or right-click → Rename · Double-click site to activate")
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
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        data_layout.addWidget(self.tree)
        data_hint = QLabel("F2 or right-click → Rename spectra / line scans / site")
        data_hint.setWordWrap(True)
        data_hint.setStyleSheet("color: #666; font-size: 11px;")
        data_layout.addWidget(data_hint)
        data_btn_row = QHBoxLayout()
        self.rename_data_btn = QPushButton("Rename…")
        self.rename_data_btn.setToolTip("Rename selected spectrum, line scan, or site")
        self.rename_data_btn.clicked.connect(self._rename_selected_data_item)
        data_btn_row.addWidget(self.rename_data_btn)
        data_btn_row.addStretch(1)
        data_layout.addLayout(data_btn_row)
        self.nav_tabs.addTab(data_page, "Data")

        left_layout.addWidget(self.nav_tabs, stretch=1)

        # Scrollable collapsible tool sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMinimumHeight(180)

        scroll_body = QWidget()
        scroll_layout = QVBoxLayout(scroll_body)
        scroll_layout.setContentsMargins(0, 0, 4, 0)
        scroll_layout.setSpacing(8)

        # ---- Display ----
        display_sec = CollapsibleSection("Display", expanded=True)
        display_sec.addWidget(QLabel("Map / overview (if none checked)"))
        self.map_combo = QComboBox()
        self.map_combo.currentIndexChanged.connect(self._refresh_canvas)
        display_sec.addWidget(self.map_combo)
        hint = QLabel("Tip: check element maps in the Data tree to show subplots.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        display_sec.addWidget(hint)
        self.rgb_check = QCheckBox("RGB composite")
        self.rgb_check.toggled.connect(self._refresh_canvas)
        display_sec.addWidget(self.rgb_check)
        self.r_combo = QComboBox()
        self.g_combo = QComboBox()
        self.b_combo = QComboBox()
        for label, combo in (("R", self.r_combo), ("G", self.g_combo), ("B", self.b_combo)):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(combo)
            display_sec.addLayout(row)
            combo.currentIndexChanged.connect(self._refresh_canvas)
        scroll_layout.addWidget(display_sec)

        # ---- Tools ----
        tools_sec = CollapsibleSection("Tools", expanded=True)
        self.line_mode_btn = QPushButton("Draw line transect")
        self.line_mode_btn.setCheckable(True)
        self.line_mode_btn.toggled.connect(self._on_line_mode)
        tools_sec.addWidget(self.line_mode_btn)
        self.pick_btn = QPushButton("Pick pixel spectrum")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip(
            "Click map pixels to extract spectra into a popup viewer "
            "(stays open and updates on each click)"
        )
        self.pick_btn.toggled.connect(self._on_pick_mode)
        tools_sec.addWidget(self.pick_btn)
        self.clear_line_btn = QPushButton("Clear line")
        self.clear_line_btn.clicked.connect(self._clear_line)
        tools_sec.addWidget(self.clear_line_btn)

        tools_sec.addWidget(QLabel("Profile / line-scan elements"))
        self.profile_map_list = QListWidget()
        self.profile_map_list.setMinimumHeight(100)
        self.profile_map_list.setMaximumHeight(160)
        self.profile_map_list.setToolTip(
            "Checked items drive the Line profile chart:\n"
            "• Map names → drawn map transect intensities\n"
            "• Element ROIs → XGT line-scan point spectra (energy windows)\n\n"
            "Use From Analysis after selecting elements in Analysis → Elements "
            "(Send Sum Spectrum or a line point first to find peaks)."
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
        scroll_layout.addWidget(tools_sec)

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
        self.fit_line_btn = QPushButton("Fit / semi-quant line scan")
        self.fit_line_btn.setToolTip(
            "Fit each point on the selected IPJ line scan using "
            "current Analysis element & fitting settings"
        )
        self.fit_line_btn.clicked.connect(self._fit_line_scan)
        quant_sec.addWidget(self.fit_line_btn)
        self.export_profile_btn = QPushButton("Export profile CSV…")
        self.export_profile_btn.clicked.connect(self._export_profile_csv)
        quant_sec.addWidget(self.export_profile_btn)
        scroll_layout.addWidget(quant_sec)

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_body)
        left_layout.addWidget(scroll, stretch=2)

        splitter.addWidget(left)

        # Center: map canvas
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MapCanvas()
        self.canvas.line_drawn.connect(self._on_line_drawn)
        self.canvas.cursor_moved.connect(self._on_cursor)
        self.canvas.pixel_clicked.connect(self._on_pixel_clicked)
        center_layout.addWidget(self.canvas)
        self.cursor_label = QLabel("Cursor: —")
        center_layout.addWidget(self.cursor_label)
        splitter.addWidget(center)

        # Right: profile / correlation / quant plots
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_tabs_label = QLabel("Profiles & correlations")
        right_layout.addWidget(self.plot_tabs_label)

        self.profile_plot = pg.PlotWidget(title="Line profile")
        self.profile_plot.setLabel("bottom", "Distance (pixels)")
        self.profile_plot.setLabel("left", "Intensity")
        self.profile_plot.addLegend(offset=(10, 10))
        right_layout.addWidget(self.profile_plot, stretch=1)

        self.corr_plot = pg.PlotWidget(title="Element correlation")
        self.corr_plot.setLabel("bottom", "Map A")
        self.corr_plot.setLabel("left", "Map B")
        right_layout.addWidget(self.corr_plot, stretch=1)

        self.quant_plot = pg.PlotWidget(title="Line-scan semi-quant")
        self.quant_plot.setLabel("bottom", "Point index / distance")
        self.quant_plot.setLabel("left", "Relative %")
        self.quant_plot.addLegend(offset=(10, 10))
        right_layout.addWidget(self.quant_plot, stretch=1)

        self.info_label = QLabel("Open an .ipj mapping project to begin.")
        self.info_label.setWordWrap(True)
        right_layout.addWidget(self.info_label)

        splitter.addWidget(right)
        splitter.setSizes([280, 520, 420])
        root.addWidget(splitter)

        self._last_profiles = None  # dict name -> (dist, vals)

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

        self._populate_trees()
        primary = self.project.primary_fov
        if primary:
            self._activate_site(primary, switch_to_data=True)
        n_samples = self.project.metadata.get("n_samples", len(self.project.samples))
        n_sites = self.project.metadata.get("n_fovs", len(self.project.fovs))
        n_cubes = self.project.metadata.get("n_cubes", 0)
        self.status_message.emit(f"Loaded mapping project: {self.project.name}")
        self.info_label.setText(
            f"{self.project.name}: {n_samples} sample(s), {n_sites} site(s), "
            f"{len(self.project.all_spectra())} spectra"
            + (f", {n_cubes} cube(s)" if n_cubes else "")
        )

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
                site_item = QTreeWidgetItem([site.name])
                site_item.setData(0, Qt.UserRole, ("site", sample.id, site.id))
                site_item.setFlags(
                    (site_item.flags() | Qt.ItemIsEditable | Qt.ItemIsSelectable)
                    & ~Qt.ItemIsUserCheckable
                )
                site_item.setToolTip(0, self._site_tooltip(site))
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
            m for m in fov.element_maps if m.metadata.get("source") not in ("cube_total", "cube_roi")
        ]
        cube_maps = [
            m for m in fov.element_maps if m.metadata.get("source") in ("cube_total", "cube_roi")
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
        if self._tree_updating or column != 0:
            return
        payload = item.data(0, Qt.UserRole)
        if not payload:
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
            item.setToolTip(0, self._site_tooltip(site))
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
        menu = QMenu(self)
        act_ren = QAction("Rename…", self)
        act_ren.setShortcut(QKeySequence("F2"))
        act_ren.triggered.connect(lambda: self._prompt_rename(item, "Rename"))
        menu.addAction(act_ren)
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
        if site.cube is not None or site.metadata.get("has_smartmap"):
            bits.append("SmartMap")
        if site.optical is not None:
            bits.append("photo")
        n_maps = len(
            [m for m in site.element_maps if m.metadata.get("source") != "cube_roi"]
        )
        if n_maps:
            bits.append(f"{n_maps} maps")
        if site.line_scans:
            ls = site.line_scans[0]
            tag = "line" if ls.is_line_scan else "multipoint"
            bits.append(f"{ls.n_points}-pt {tag}")
        elif site.spectra:
            bits.append(f"{len(site.spectra)} spectra")
        return " · ".join(bits)

    @staticmethod
    def _site_label(site: MappingFOV) -> str:
        bits = [site.name]
        if site.cube is not None:
            bits.append("SmartMap")
        elif site.metadata.get("has_smartmap"):
            bits.append("SmartMap")
        n_maps = len(
            [m for m in site.element_maps if m.metadata.get("source") != "cube_roi"]
        )
        if n_maps:
            bits.append(f"{n_maps} maps")
        if site.line_scans:
            ls = site.line_scans[0]
            tag = "line" if ls.is_line_scan else "multipoint"
            bits.append(f"{ls.n_points}-pt {tag}")
        n_spec = len(site.spectra)
        if n_spec and not site.line_scans:
            bits.append(f"{n_spec} spectra")
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
        self._set_fov(site)
        self.active_site_label.setText(
            f"Active site: {self._sample_name_for_site(site)} → {site.name}"
        )
        self._populate_data_tree()
        self._refresh_canvas()  # after tree defaults checks
        self._select_site_in_sites_tree(site.id)
        if switch_to_data:
            self.nav_tabs.setCurrentIndex(1)
        self.status_message.emit(f"Active site: {site.name}")

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
        self._last_line = None
        self._active_line_scan = None
        # Reset checked maps for new site (repopulated with defaults in data tree)
        self._checked_map_names = set()
        self._fill_map_combos()
        self._fill_profile_map_list()
        self._refresh_canvas()
        self._update_cube_controls()
        if fov.line_scans:
            self._plot_ipj_line_scan(fov.line_scans[0])
        elif self._active_line_scan is None:
            pass

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

        # --- Spectral ROI series (XGT line-scan points) ---
        if fov.line_scans:
            rois = self._line_scan_roi_candidates(fov)
            for sym, e_kev in rois:
                item = QListWidgetItem(f"ROI: {sym} ({e_kev:.2f} keV)")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                checked = sym in self._checked_roi_symbols
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                item.setData(Qt.UserRole, ("roi", sym, float(e_kev)))
                self.profile_map_list.addItem(item)

        self.profile_map_list.blockSignals(False)

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

    def _analysis_element_energies(self, symbols=None):
        """Yield (symbol, Ka-or-La energy_kev) for Analysis elements."""
        if symbols is None:
            if self._element_panel is None:
                return []
            symbols = [
                str(s).strip()
                for s in (self._element_panel.get_selected_elements() or [])
                if s
            ]
        if not symbols:
            return []

        from core.xray_data import get_element_lines

        try:
            import xraylib as xrl

            def _z(sym: str) -> int:
                return int(xrl.SymbolToAtomicNumber(sym))
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

    def _checked_profile_maps(self):
        """Return ElementMap list for checked map entries (drawn transect)."""
        fov = self.current_fov
        if fov is None:
            return []
        names = []
        for i in range(self.profile_map_list.count()):
            item = self.profile_map_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
            data = item.data(Qt.UserRole)
            if isinstance(data, tuple) and data and data[0] == "map":
                names.append(data[1])
            elif isinstance(data, str):
                names.append(data)
        return [m for m in fov.element_maps if m.name in names]

    def _checked_line_scan_rois(self):
        """Return [(symbol, energy_kev), ...] for checked ROI entries."""
        out = []
        for i in range(self.profile_map_list.count()):
            item = self.profile_map_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
            data = item.data(Qt.UserRole)
            if isinstance(data, tuple) and data and data[0] == "roi":
                out.append((data[1], float(data[2])))
        return out

    def _remember_checked_rois(self) -> None:
        self._checked_roi_symbols = {sym for sym, _ in self._checked_line_scan_rois()}

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
            data = item.data(Qt.UserRole)
            if isinstance(data, tuple) and data and data[0] == "roi":
                hit = data[1] in symbols
                item.setCheckState(Qt.Checked if hit else Qt.Unchecked)
                if hit:
                    matched_rois += 1
            else:
                name = data[1] if isinstance(data, tuple) and data else (
                    data if isinstance(data, str) else item.text()
                )
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
        """Refresh drawn-transect and/or active XGT line-scan profile chart."""
        if self._last_line is not None:
            self._replot_last_line()
        if self._active_line_scan is not None:
            self._plot_ipj_line_scan(self._active_line_scan)
    def _update_cube_controls(self) -> None:
        fov = self.current_fov
        has = fov is not None and fov.cube is not None
        self.roi_btn.setEnabled(has)
        self.roi_from_analysis_btn.setEnabled(has)
        self.pick_btn.setEnabled(has)
        if not has:
            self.pick_btn.setChecked(False)
            self.cube_info.setText("No hyperspectral cube in this FOV")
            return
        c = fov.cube
        self.cube_info.setText(
            f"Cube {c.n_channels} ch × {c.height}×{c.width} "
            f"({c.ev_per_channel:.0f} eV/ch)"
        )

    def _fill_map_combos(self) -> None:
        fov = self.current_fov
        for combo in (
            self.map_combo,
            self.r_combo,
            self.g_combo,
            self.b_combo,
            self.corr_a,
            self.corr_b,
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
            for combo in (self.r_combo, self.g_combo, self.b_combo, self.corr_a, self.corr_b):
                combo.addItem(m.name, m.name)

        # Sensible RGB defaults: first three maps
        maps = fov.element_maps
        if len(maps) >= 1:
            self.r_combo.setCurrentIndex(0)
        if len(maps) >= 2:
            self.g_combo.setCurrentIndex(1)
            self.corr_b.setCurrentIndex(1)
        if len(maps) >= 3:
            self.b_combo.setCurrentIndex(2)

        for combo in (
            self.map_combo,
            self.r_combo,
            self.g_combo,
            self.b_combo,
            self.corr_a,
            self.corr_b,
        ):
            combo.blockSignals(False)

    # ----------------------------------------------------------- display
    def _refresh_canvas(self) -> None:
        fov = self.current_fov
        if fov is None:
            return

        checked = self._checked_display_maps()
        if checked and not self.rgb_check.isChecked():
            panels = [(m.name, m.data) for m in checked]
            try:
                self.canvas.set_images(panels)
            except Exception as exc:
                self.status_message.emit(f"Map display failed: {exc}")
            return

        if self.rgb_check.isChecked() and fov.element_maps:
            r = fov.find_map(self.r_combo.currentData() or "")
            g = fov.find_map(self.g_combo.currentData() or "")
            b = fov.find_map(self.b_combo.currentData() or "")
            try:
                rgb = rgb_composite(r, g, b)
                self.canvas.set_image(rgb, rgb=True, title="RGB")
            except Exception as exc:
                self.status_message.emit(f"RGB composite failed: {exc}")
            return

        data = self.map_combo.currentData()
        if not data:
            if fov.overview is not None:
                self.canvas.set_image(
                    fov.overview.data, rgb=False, title=fov.overview.name
                )
            return
        if data[0] == "overview" and fov.overview is not None:
            self.canvas.set_image(
                fov.overview.data, rgb=False, title=fov.overview.name or "Overview"
            )
        elif data[0] == "optical" and fov.optical is not None:
            self._show_photo(fov.optical)
        elif data[0] == "whole_image":
            sample = self._sample_for_site(fov)
            if sample is not None and sample.whole_image is not None:
                self._show_photo(sample.whole_image)
        elif data[0] == "map":
            m = fov.find_map(data[1])
            if m is not None:
                self.canvas.set_image(m.data, rgb=False, title=m.name)

    def _show_photo(self, image) -> None:
        arr = image.data
        rgb = arr.ndim == 3 and arr.shape[-1] >= 3
        try:
            self.canvas.set_image(arr, rgb=rgb, title=image.name or "Photo")
        except Exception as exc:
            self.status_message.emit(f"Photo display failed: {exc}")

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
                        self._plot_ipj_line_scan(ls)
                        break

    # -------------------------------------------------------- line tools
    def _on_line_mode(self, checked: bool) -> None:
        if checked:
            self.pick_btn.blockSignals(True)
            self.pick_btn.setChecked(False)
            self.pick_btn.blockSignals(False)
            self.canvas.set_pick_mode(False)
        self.canvas.set_line_mode(checked)
        if checked:
            self.status_message.emit("Line mode: click start, then end point on the map")

    def _on_pick_mode(self, checked: bool) -> None:
        if checked:
            self.line_mode_btn.blockSignals(True)
            self.line_mode_btn.setChecked(False)
            self.line_mode_btn.blockSignals(False)
            self.canvas.set_line_mode(False)
        self.canvas.set_pick_mode(checked)
        if checked:
            self.status_message.emit(
                "Pick mode: click a map pixel to extract its cube spectrum"
            )

    def _clear_line(self) -> None:
        self.canvas.clear_line()
        self.profile_plot.clear()
        self._last_profiles = None
        self._last_line = None

    def _replot_last_line(self) -> None:
        if self._last_line is None:
            return
        x0, y0, x1, y1 = self._last_line
        self._extract_and_plot_line(x0, y0, x1, y1, finish_mode=False)

    def _on_line_drawn(self, x0, y0, x1, y1) -> None:
        self._last_line = (float(x0), float(y0), float(x1), float(y1))
        self._extract_and_plot_line(x0, y0, x1, y1, finish_mode=True)

    def _extract_and_plot_line(
        self, x0, y0, x1, y1, *, finish_mode: bool = True
    ) -> None:
        fov = self.current_fov
        if fov is None:
            return
        maps = self._checked_profile_maps()
        if maps:
            profiles = extract_multi_element_profiles(maps, (x0, y0), (x1, y1))
            self._last_profiles = profiles
            self._plot_profiles(profiles)
            n = len(profiles)
        elif fov.cube is not None:
            # Fallback: total counts along line from cube
            length = float(np.hypot(x1 - x0, y1 - y0))
            n_pts = max(2, int(np.ceil(length)) + 1)
            xs = np.linspace(x0, x1, n_pts)
            ys = np.linspace(y0, y1, n_pts)
            dist = np.linspace(0.0, length, n_pts)
            totals = np.array(
                [
                    float(fov.cube.spectrum_at(int(round(x)), int(round(y))).sum())
                    for x, y in zip(xs, ys)
                ]
            )
            self._last_profiles = {"Total counts": (dist, totals)}
            self._plot_profiles(self._last_profiles)
            n = 1
            if finish_mode:
                _, mean_counts = fov.cube.mean_spectrum_line(x0, y0, x1, y1, n_pts)
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
                    ms.name = f"Line mean ({x0:.0f},{y0:.0f})→({x1:.0f},{y1:.0f})"
                    ms.spectrum.metadata["name"] = ms.name
                    ms.kind = "roi"
                    self._picked_spectrum = ms
        else:
            self.profile_plot.clear()
            self._last_profiles = None
            self.status_message.emit(
                "No maps checked for transect — check elements above, "
                "or Send Sum Spectrum → Analysis to choose elements"
            )
            return

        if finish_mode:
            self.line_mode_btn.setChecked(False)
            self.canvas.set_line_mode(False)
        self.status_message.emit(
            f"Line profile ({x0:.0f},{y0:.0f}) → ({x1:.0f},{y1:.0f}): {n} series"
        )

    def _on_pixel_clicked(self, x: float, y: float) -> None:
        fov = self.current_fov
        if fov is None or fov.cube is None:
            self.status_message.emit("No cube available for pixel spectrum")
            return
        ms = fov.spectrum_at_pixel(int(round(x)), int(round(y)))
        if ms is None:
            return
        self._picked_spectrum = ms
        # Keep pick mode on so the next click updates the popup
        self.canvas.set_spot_markers([x], [y])
        self._show_pixel_spectrum(ms)
        self.info_label.setText(
            f"{ms.name}: {ms.spectrum.total_counts:.0f} counts — "
            f"click another pixel to update, or Send → Analysis"
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
            if (item.data(Qt.UserRole) or item.text()) == em.name:
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
        symbols = [
            str(s).strip()
            for s in (self._element_panel.get_selected_elements() or [])
            if s
        ]
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
                return int(xrl.SymbolToAtomicNumber(sym))
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
            name = item.data(Qt.UserRole) or item.text()
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

    def _plot_profiles(self, profiles: dict) -> None:
        self.profile_plot.clear()
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
        for i, (name, (dist, vals)) in enumerate(profiles.items()):
            pen = pg.mkPen(colors[i % len(colors)], width=2)
            self.profile_plot.plot(dist, vals, pen=pen, name=name)

    def _plot_ipj_line_scan(self, line_scan: LineScan) -> None:
        """Plot total counts + checked element ROIs vs point index for XGT line."""
        self._active_line_scan = line_scan
        self.profile_plot.clear()
        if line_scan.n_points == 0:
            return
        xs = line_scan.distances()
        totals = np.array([p.spectrum.total_counts for p in line_scan.points])
        profiles = {"Total counts": (xs, totals)}
        self.profile_plot.plot(
            xs, totals, pen=pg.mkPen("#377eb8", width=2), name="Total counts"
        )

        colors = ["#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
        rois = self._checked_line_scan_rois()
        half = 0.15  # keV
        for ci, (sym, e_kev) in enumerate(rois):
            series = []
            for p in line_scan.points:
                e_ax = p.spectrum.energy
                mask = (e_ax >= e_kev - half) & (e_ax <= e_kev + half)
                series.append(
                    float(p.spectrum.counts[mask].sum()) if mask.any() else 0.0
                )
            ys = np.asarray(series, dtype=np.float64)
            label = f"{sym} ({e_kev:.2f} keV)"
            profiles[label] = (xs, ys)
            self.profile_plot.plot(
                xs,
                ys,
                pen=pg.mkPen(colors[ci % len(colors)], width=2),
                name=label,
            )

        n_roi = len(rois)
        self.info_label.setText(
            f"{line_scan.name}: total"
            + (f" + {n_roi} element ROI(s)" if n_roi else "")
            + ". Check ROI elements under Tools (From Analysis), "
            "or Send a point → Analysis."
        )
        if n_roi == 0:
            self.status_message.emit(
                f"{line_scan.display_label()}: only total counts shown — "
                "select elements in Analysis, then Tools → From Analysis"
            )
        self._last_profiles = profiles

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
        self.status_message.emit(f"{a.name} vs {b.name}: Pearson r={r:.3f}, Spearman ρ={rho:.3f}")

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

    def _fit_line_scan(self) -> None:
        if self._fitter is None or self._element_panel is None:
            QMessageBox.warning(
                self, "Fit line scan", "Analysis fitter is not connected."
            )
            return
        # Prefer selected linescan, else first in current FOV / project
        line_scan = None
        items = self.tree.selectedItems()
        if items:
            payload = items[0].data(0, Qt.UserRole)
            if payload and payload[0] == "linescan":
                fov = self._find_fov(payload[1])
                if fov:
                    for ls in fov.line_scans:
                        if ls.name == payload[2]:
                            line_scan = ls
                            break
        if line_scan is None and self.current_fov and self.current_fov.line_scans:
            line_scan = self.current_fov.line_scans[0]
        if line_scan is None and self.project:
            for fov in self.project.fovs:
                if fov.line_scans:
                    line_scan = fov.line_scans[0]
                    break
        if line_scan is None or line_scan.n_points == 0:
            QMessageBox.information(
                self, "Fit line scan", "No IPJ line-scan series found in this project."
            )
            return

        elements = self._element_panel.get_selected_elements()
        if not elements:
            QMessageBox.information(
                self,
                "Fit line scan",
                "Select elements in the Analysis → Elements tab first.",
            )
            return
        fit_params = self._element_panel.get_fitting_params()
        exp_params = self._element_panel.get_experimental_params()
        background_method = str(fit_params.get("background_method", "snip")).lower()
        peak_shape = str(fit_params.get("peak_shape", "gaussian")).lower()

        distances = line_scan.distances()
        rows = []
        self.quant_plot.clear()
        try:
            for i, pt in enumerate(line_scan.points):
                sp = pt.spectrum
                result = self._fitter.fit_spectrum(
                    energy=sp.energy,
                    counts=sp.counts,
                    elements=elements,
                    background_method=background_method,
                    peak_shape=peak_shape,
                    auto_find_peaks=fit_params.get("auto_find_peaks", True),
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
                row = {"index": i, "name": pt.name, "distance": float(distances[i])}
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
                self.status_message.emit(
                    f"Fitting line scan {i + 1}/{line_scan.n_points}…"
                )
        except Exception as exc:
            QMessageBox.critical(self, "Fit line scan failed", str(exc))
            return

        self._quant_table = rows
        self._quant_distances = distances
        # Plot each element series
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
        element_keys = [
            k for k in rows[0].keys() if k not in ("index", "name", "distance")
        ] if rows else []
        for ci, key in enumerate(element_keys):
            ys = [r.get(key, np.nan) for r in rows]
            self.quant_plot.plot(
                distances,
                ys,
                pen=pg.mkPen(colors[ci % len(colors)], width=2),
                name=key,
            )
        self.info_label.setText(
            f"Semi-quant along {line_scan.name}: {len(rows)} points, "
            f"{len(element_keys)} elements"
        )
        self.status_message.emit("Line-scan fit complete")

    def _export_profile_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export profile CSV", "", "CSV (*.csv)"
        )
        if not path:
            return
        # Prefer quant table if available
        if self._quant_table:
            import csv

            keys = list(self._quant_table[0].keys())
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self._quant_table)
            self.status_message.emit(f"Exported quant table → {path}")
            return
        if not self._last_profiles:
            QMessageBox.information(self, "Export", "No profile to export yet.")
            return
        # Merge profiles on distance
        names = list(self._last_profiles.keys())
        dist = self._last_profiles[names[0]][0]
        cols = {"distance": dist}
        for name in names:
            cols[name] = self._last_profiles[name][1]
        import csv

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(cols.keys()))
            for i in range(len(dist)):
                w.writerow([cols[k][i] for k in cols])
        self.status_message.emit(f"Exported profile → {path}")
