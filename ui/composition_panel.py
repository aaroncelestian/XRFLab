"""
Composition tab: group batch replicates into samples and plot means.

Ternary, correlate, ratio, and correlation-matrix plots operate on
sample averages (e.g. 20 basalts × 4 pellet spots → 20 points).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.composition import (
    DEFAULT_GROUP_REGEX,
    GroupMode,
    SQRT3_OVER_2,
    SampleSummary,
    assign_samples,
    component_keys,
    convert_values,
    correlation,
    correlation_matrix,
    display_values,
    export_sample_means_csv,
    export_sample_means_excel,
    ratio_points,
    replicate_ratio_points,
    replicate_scatter_points,
    replicate_ternary_points,
    rows_from_batch_results,
    scatter_points,
    summarize_samples,
    ternary_points,
    ternary_xy,
)

_MEAN_BRUSH = pg.mkBrush(0, 0, 139, 220)
_MEAN_PEN = pg.mkPen("k", width=1)
_REP_BRUSH = pg.mkBrush(140, 140, 140, 140)
_REP_PEN = pg.mkPen(120, 120, 120, 80)
_SEL_BRUSH = pg.mkBrush(200, 80, 0, 230)


def _prefer(keys, candidates):
    for name in candidates:
        if name in keys:
            return name
    return keys[0] if keys else ""


def _ternary_grid(steps=(0.2, 0.4, 0.6, 0.8)):
    """Line segments for constant-A/B/C grid, as list of (x, y) pairs to plot."""
    segments = []
    for t in steps:
        # constant A
        segments.append((ternary_xy(t, 1.0 - t, 0.0), ternary_xy(t, 0.0, 1.0 - t)))
        # constant B
        segments.append((ternary_xy(1.0 - t, t, 0.0), ternary_xy(0.0, t, 1.0 - t)))
        # constant C
        segments.append((ternary_xy(1.0 - t, 0.0, t), ternary_xy(0.0, 1.0 - t, t)))
    return segments


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


class CompositionPanel(QWidget):
    """Group batch spectra into samples and plot sample-mean compositions."""

    from_batch_requested = Signal()
    sample_activated = Signal(str, list)  # sample name, member spectrum names
    open_in_batch_requested = Signal(str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        self.summaries: list[SampleSummary] = []
        self._group_mode = GroupMode.AUTO
        self._selected_sample = None
        self._filling_table = False
        self._filling_combos = False
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_plots())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 780])
        root.addWidget(splitter)

        self.status_label = QLabel(
            "Load batch results: 20 pellets × 4 spots → 20 sample means on the plots."
        )
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #555;")
        root.addWidget(self.status_label)

    def _build_left(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_group_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_table_group(), stretch=1)
        layout.addWidget(self._build_export_group())
        return widget

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Source")
        layout = QHBoxLayout(group)
        self.from_batch_btn = QPushButton("From Batch")
        self.from_batch_btn.setToolTip(
            "Pull the latest Batch Analysis fits and group them into samples"
        )
        self.from_batch_btn.clicked.connect(self.from_batch_requested.emit)
        layout.addWidget(self.from_batch_btn)
        layout.addStretch()
        return group

    def _build_group_group(self) -> QGroupBox:
        group = QGroupBox("Group replicates")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("By"))
        self.group_combo = QComboBox()
        self.group_combo.addItem("Auto (folder or prefix)", GroupMode.AUTO)
        self.group_combo.addItem("Parent folder", GroupMode.FOLDER)
        self.group_combo.addItem("Filename prefix", GroupMode.PREFIX)
        self.group_combo.addItem("Regex", GroupMode.REGEX)
        self.group_combo.addItem("None (each spectrum)", GroupMode.NONE)
        self.group_combo.setToolTip(
            "Auto uses parent folders when each pellet is a folder, "
            "otherwise strips _1 / _rep2 / _a from the filename.\n"
            "Edit the Sample column to fix leftovers."
        )
        self.group_combo.currentIndexChanged.connect(self._on_group_mode_changed)
        row.addWidget(self.group_combo, stretch=1)
        layout.addLayout(row)

        self.regex_edit = QLineEdit(DEFAULT_GROUP_REGEX)
        self.regex_edit.setPlaceholderText("(?P<sample>...)")
        self.regex_edit.setToolTip("Named group 'sample', or first capture group")
        self.regex_edit.editingFinished.connect(self._apply_grouping)
        self.regex_edit.setVisible(False)
        layout.addWidget(self.regex_edit)

        self.group_summary = QLabel("No spectra loaded")
        self.group_summary.setStyleSheet("color: #555;")
        self.group_summary.setWordWrap(True)
        layout.addWidget(self.group_summary)
        return group

    def _build_display_group(self) -> QGroupBox:
        group = QGroupBox("Display")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.oxides_check = QCheckBox("Oxides")
        self.oxides_check.setToolTip(
            "Convert element intensities with standard oxide factors "
            "(Si→SiO2, Fe→FeO, …). Still relative intensity, not FP wt%."
        )
        self.oxides_check.toggled.connect(self._on_display_changed)
        row.addWidget(self.oxides_check)

        self.fe_combo = QComboBox()
        self.fe_combo.addItems(["FeO", "Fe2O3", "Fe3O4"])
        self.fe_combo.setEnabled(False)
        self.fe_combo.currentTextChanged.connect(self._on_display_changed)
        row.addWidget(self.fe_combo)

        self.close_check = QCheckBox("Close to 100%")
        self.close_check.setToolTip(
            "Renormalize the displayed composition to 100% "
            "(ternary always closes the three axes)."
        )
        self.close_check.toggled.connect(self._on_display_changed)
        row.addWidget(self.close_check)
        row.addStretch()
        layout.addLayout(row)

        self.replicates_check = QCheckBox("Show replicate spots")
        self.replicates_check.setToolTip(
            "Draw the individual pellet spots behind each sample mean"
        )
        self.replicates_check.toggled.connect(self._refresh_plots)
        layout.addWidget(self.replicates_check)

        self.labels_check = QCheckBox("Label samples")
        self.labels_check.setChecked(True)
        self.labels_check.toggled.connect(self._refresh_plots)
        layout.addWidget(self.labels_check)
        return group

    def _build_table_group(self) -> QGroupBox:
        group = QGroupBox("Sample means")
        layout = QVBoxLayout(group)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setToolTip(
            "Each row is one sample (mean of its replicates). "
            "Edit Sample to rename or merge groups. Click a row to highlight "
            "it on the plots. Double-click to open those spectra in Batch."
        )
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.cellDoubleClicked.connect(self._on_table_double_clicked)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        layout.addWidget(self.table)
        return group

    def _build_export_group(self) -> QGroupBox:
        group = QGroupBox("Export")
        layout = QHBoxLayout(group)
        csv_btn = QPushButton("CSV")
        csv_btn.clicked.connect(lambda: self._export("csv"))
        layout.addWidget(csv_btn)
        xls_btn = QPushButton("Excel")
        xls_btn.clicked.connect(lambda: self._export("excel"))
        layout.addWidget(xls_btn)
        layout.addStretch()
        return group

    def _build_plots(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(self._build_ternary_tab(), "Ternary")
        self.plot_tabs.addTab(self._build_correlate_tab(), "Correlate")
        self.plot_tabs.addTab(self._build_ratio_tab(), "Ratios")
        self.plot_tabs.addTab(self._build_matrix_tab(), "Matrix")
        layout.addWidget(self.plot_tabs)
        return widget

    def _build_ternary_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.tern_a = QComboBox()
        self.tern_b = QComboBox()
        self.tern_c = QComboBox()
        for combo, label in (
            (self.tern_a, "A (left)"),
            (self.tern_b, "B (right)"),
            (self.tern_c, "C (top)"),
        ):
            row.addWidget(QLabel(label))
            combo.currentTextChanged.connect(self._refresh_plots)
            row.addWidget(combo)
        row.addStretch()
        layout.addLayout(row)

        self.ternary_plot = pg.PlotWidget()
        self.ternary_plot.setBackground("w")
        self.ternary_plot.hideAxis("left")
        self.ternary_plot.hideAxis("bottom")
        self.ternary_plot.setAspectLocked(True)
        self.ternary_plot.setMenuEnabled(False)
        layout.addWidget(self.ternary_plot, stretch=1)
        return tab

    def _build_correlate_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.corr_x = QComboBox()
        self.corr_y = QComboBox()
        self.corr_x.currentTextChanged.connect(self._refresh_plots)
        self.corr_y.currentTextChanged.connect(self._refresh_plots)
        row.addWidget(QLabel("X"))
        row.addWidget(self.corr_x)
        row.addWidget(QLabel("Y"))
        row.addWidget(self.corr_y)
        self.errorbar_check = QCheckBox("Error bars (replicate std)")
        self.errorbar_check.setChecked(True)
        self.errorbar_check.toggled.connect(self._refresh_plots)
        row.addWidget(self.errorbar_check)
        row.addStretch()
        layout.addLayout(row)

        self.corr_plot = pg.PlotWidget()
        self.corr_plot.setBackground("w")
        self.corr_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.corr_plot, stretch=1)
        return tab

    def _build_ratio_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.ratio_x_num = QComboBox()
        self.ratio_x_den = QComboBox()
        self.ratio_y_num = QComboBox()
        self.ratio_y_den = QComboBox()
        for combo in (
            self.ratio_x_num,
            self.ratio_x_den,
            self.ratio_y_num,
            self.ratio_y_den,
        ):
            combo.currentTextChanged.connect(self._refresh_plots)
        row.addWidget(self.ratio_x_num)
        row.addWidget(QLabel("/"))
        row.addWidget(self.ratio_x_den)
        row.addWidget(QLabel("vs"))
        row.addWidget(self.ratio_y_num)
        row.addWidget(QLabel("/"))
        row.addWidget(self.ratio_y_den)
        row.addStretch()
        layout.addLayout(row)

        self.ratio_plot = pg.PlotWidget()
        self.ratio_plot.setBackground("w")
        self.ratio_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.ratio_plot, stretch=1)
        return tab

    def _build_matrix_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        hint = QLabel("Pearson r of sample means. Click a cell to open that pair on Correlate.")
        hint.setStyleSheet("color: #555;")
        layout.addWidget(hint)
        self.matrix_plot = pg.PlotWidget()
        self.matrix_plot.setBackground("w")
        self.matrix_plot.setAspectLocked(True)
        self.matrix_plot.invertY(True)
        self.matrix_image = pg.ImageItem()
        self.matrix_image.setLookupTable(_diverging_lut())
        self.matrix_image.setLevels((-1.0, 1.0))
        self.matrix_plot.addItem(self.matrix_image)
        self.matrix_plot.scene().sigMouseClicked.connect(self._on_matrix_clicked)
        layout.addWidget(self.matrix_plot, stretch=1)
        return tab

    # ---------------------------------------------------------------- data
    def load_batch_results(self, results) -> None:
        """Replace the table from BatchFitResult objects."""
        self.rows = rows_from_batch_results(results or [])
        self._apply_grouping()

    def _current_mode(self) -> GroupMode:
        data = self.group_combo.currentData()
        return data if isinstance(data, GroupMode) else GroupMode.AUTO

    def _on_group_mode_changed(self) -> None:
        mode = self._current_mode()
        self.regex_edit.setVisible(mode == GroupMode.REGEX)
        self._apply_grouping()

    def _apply_grouping(self) -> None:
        if not self.rows:
            self.summaries = []
            self.group_summary.setText("No spectra loaded")
            self._fill_table()
            self._refresh_plots()
            return
        used = assign_samples(
            self.rows,
            self._current_mode(),
            regex=self.regex_edit.text().strip() or DEFAULT_GROUP_REGEX,
        )
        self._group_mode = used
        self.summaries = summarize_samples(self.rows)
        n_spec = sum(1 for r in self.rows if r.success)
        n_fail = sum(1 for r in self.rows if not r.success)
        n_samp = len(self.summaries)
        ns = [s.n for s in self.summaries]
        extra = f"  ({n_fail} failed fits omitted)" if n_fail else ""
        mode_label = used.value
        self.group_summary.setText(
            f"{n_spec} spectra → {n_samp} samples  "
            f"(n={min(ns) if ns else 0}–{max(ns) if ns else 0}, {mode_label})"
            f"{extra}"
        )
        self.status_label.setText(
            f"Plotting {n_samp} sample means. Click a row or point to inspect "
            "that pellet’s spectra in Batch."
        )
        self._fill_table()
        self._fill_combos()
        self._refresh_plots()

    def _as_oxides(self) -> bool:
        return self.oxides_check.isChecked()

    def _fe_as(self) -> str:
        return self.fe_combo.currentText() or "FeO"

    def _close(self) -> bool:
        return self.close_check.isChecked()

    def _on_display_changed(self) -> None:
        self.fe_combo.setEnabled(self._as_oxides())
        self._fill_table()
        self._fill_combos()
        self._refresh_plots()

    def _keys(self) -> list:
        return component_keys(
            self.summaries,
            as_oxides=self._as_oxides(),
            fe_as=self._fe_as(),
            close=self._close(),
        )

    def _fill_combos(self) -> None:
        keys = self._keys()
        self._filling_combos = True
        combos = [
            self.tern_a,
            self.tern_b,
            self.tern_c,
            self.corr_x,
            self.corr_y,
            self.ratio_x_num,
            self.ratio_x_den,
            self.ratio_y_num,
            self.ratio_y_den,
        ]
        previous = [c.currentText() for c in combos]
        for combo in combos:
            combo.clear()
            combo.addItems(keys)
        defaults = {
            self.tern_a: _prefer(keys, ("SiO2", "Si")),
            self.tern_b: _prefer(keys, ("Al2O3", "Al", "MgO", "Mg")),
            self.tern_c: _prefer(keys, ("FeO", "Fe2O3", "Fe3O4", "Fe", "CaO", "Ca")),
            self.corr_x: _prefer(keys, ("SiO2", "Si")),
            self.corr_y: _prefer(keys, ("FeO", "Fe2O3", "Fe3O4", "Fe")),
            self.ratio_x_num: _prefer(keys, ("FeO", "Fe2O3", "Fe3O4", "Fe")),
            self.ratio_x_den: _prefer(keys, ("MgO", "Mg")),
            self.ratio_y_num: _prefer(keys, ("SiO2", "Si")),
            self.ratio_y_den: _prefer(keys, ("Al2O3", "Al")),
        }
        for combo, prev in zip(combos, previous):
            target = prev if prev in keys else defaults.get(combo, "")
            if target:
                combo.setCurrentText(target)
        self._filling_combos = False

    def _fill_table(self) -> None:
        self._filling_table = True
        keys = self._keys()
        self.table.setColumnCount(2 + len(keys))
        self.table.setHorizontalHeaderLabels(["Sample", "n"] + list(keys))
        self.table.setRowCount(len(self.summaries))
        for i, summary in enumerate(self.summaries):
            name_item = QTableWidgetItem(summary.sample)
            self.table.setItem(i, 0, name_item)
            n_item = QTableWidgetItem(str(summary.n))
            n_item.setFlags(n_item.flags() & ~Qt.ItemIsEditable)
            n_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 1, n_item)
            vals = display_values(
                summary,
                as_oxides=self._as_oxides(),
                fe_as=self._fe_as(),
                close=self._close(),
            )
            std_vals = convert_values(
                summary.std,
                as_oxides=self._as_oxides(),
                fe_as=self._fe_as(),
                close=False,
            )
            for j, key in enumerate(keys):
                mean = vals.get(key, 0.0)
                std = std_vals.get(key, 0.0)
                text = f"{mean:.2f}" if summary.n < 2 else f"{mean:.2f} ± {std:.2f}"
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, 2 + j, item)
        self.table.resizeColumnsToContents()
        self._filling_table = False

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling_table or item.column() != 0:
            return
        row = item.row()
        if not (0 <= row < len(self.summaries)):
            return
        new_name = item.text().strip()
        if not new_name:
            return
        old = self.summaries[row].sample
        if new_name == old:
            return
        for member in self.summaries[row].rows:
            member.sample = new_name
        self.summaries = summarize_samples(self.rows)
        self._fill_table()
        self._refresh_plots()
        self._select_sample_in_table(new_name)

    def _on_table_selection(self) -> None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return
        index = rows[0].row()
        if not (0 <= index < len(self.summaries)):
            return
        summary = self.summaries[index]
        self._selected_sample = summary.sample
        self._refresh_plots()
        self.sample_activated.emit(summary.sample, summary.member_names())
        self.status_label.setText(
            f"{summary.sample}: {summary.n} spectra  "
            f"({', '.join(summary.member_names()[:6])}"
            f"{'…' if summary.n > 6 else ''})"
        )

    def _on_table_double_clicked(self, row: int, _column: int) -> None:
        if not (0 <= row < len(self.summaries)):
            return
        summary = self.summaries[row]
        self.open_in_batch_requested.emit(summary.sample, summary.member_names())

    def _select_sample_in_table(self, sample: str) -> None:
        self.table.blockSignals(True)
        self.table.clearSelection()
        for i, summary in enumerate(self.summaries):
            if summary.sample == sample:
                self.table.selectRow(i)
                break
        self.table.blockSignals(False)

    def _on_point_clicked(self, _item, points) -> None:
        if not points:
            return
        sample = points[0].data()
        if not sample:
            return
        self._selected_sample = sample
        self._select_sample_in_table(sample)
        summary = next((s for s in self.summaries if s.sample == sample), None)
        if summary:
            self.sample_activated.emit(summary.sample, summary.member_names())

    # ---------------------------------------------------------------- plots
    def _mean_scatter(self, xs, ys, names) -> pg.ScatterPlotItem:
        spots = [
            {
                "pos": (x, y),
                "data": name,
                "size": 11,
                "brush": _SEL_BRUSH if name == self._selected_sample else _MEAN_BRUSH,
                "pen": _MEAN_PEN,
            }
            for x, y, name in zip(xs, ys, names)
        ]
        scatter = pg.ScatterPlotItem(spots=spots, hoverable=True)
        scatter.sigClicked.connect(self._on_point_clicked)
        return scatter

    def _add_labels(self, plot, xs, ys, names) -> None:
        if not self.labels_check.isChecked():
            return
        for x, y, name in zip(xs, ys, names):
            text = pg.TextItem(name, color=(40, 40, 40), anchor=(0.5, 1.4))
            text.setFont(QFont("Arial", 8))
            text.setPos(x, y)
            plot.addItem(text)

    def _refresh_plots(self) -> None:
        if self._filling_combos:
            return
        self._draw_ternary()
        self._draw_correlate()
        self._draw_ratios()
        self._draw_matrix()

    def _kw(self) -> dict:
        return {
            "as_oxides": self._as_oxides(),
            "fe_as": self._fe_as(),
            "close": self._close(),
        }

    def _draw_ternary(self) -> None:
        plot = self.ternary_plot
        plot.clear()
        a = self.tern_a.currentText()
        b = self.tern_b.currentText()
        c = self.tern_c.currentText()
        # triangle
        corners = [ternary_xy(1, 0, 0), ternary_xy(0, 1, 0), ternary_xy(0, 0, 1)]
        xs = [p[0] for p in corners] + [corners[0][0]]
        ys = [p[1] for p in corners] + [corners[0][1]]
        plot.plot(xs, ys, pen=pg.mkPen("k", width=2))
        grid_pen = pg.mkPen((180, 180, 180), width=1)
        for (x0, y0), (x1, y1) in _ternary_grid():
            plot.plot([x0, x1], [y0, y1], pen=grid_pen)
        for text, (x, y), anchor in (
            (a or "A", corners[0], (1.15, 0.5)),
            (b or "B", corners[1], (-0.15, 0.5)),
            (c or "C", corners[2], (0.5, -0.15)),
        ):
            label = pg.TextItem(text, color="k", anchor=anchor)
            label.setFont(QFont("Arial", 10))
            label.setPos(x, y)
            plot.addItem(label)

        if not self.summaries or not (a and b and c):
            plot.setXRange(-0.15, 1.15, padding=0)
            plot.setYRange(-0.12, SQRT3_OVER_2 + 0.12, padding=0)
            return

        if self.replicates_check.isChecked():
            reps = replicate_ternary_points(self.summaries, a, b, c, **self._kw())
            if reps:
                scatter = pg.ScatterPlotItem(
                    x=[p[2] for p in reps],
                    y=[p[3] for p in reps],
                    size=6,
                    brush=_REP_BRUSH,
                    pen=_REP_PEN,
                    hoverable=False,
                )
                plot.addItem(scatter)

        pts = ternary_points(self.summaries, a, b, c, **self._kw())
        if pts:
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            names = [p[0] for p in pts]
            plot.addItem(self._mean_scatter(xs, ys, names))
            self._add_labels(plot, xs, ys, names)
        plot.setXRange(-0.15, 1.15, padding=0)
        plot.setYRange(-0.12, SQRT3_OVER_2 + 0.12, padding=0)

    def _draw_correlate(self) -> None:
        plot = self.corr_plot
        plot.clear()
        xk = self.corr_x.currentText()
        yk = self.corr_y.currentText()
        unit = "% (closed)" if self._close() else ("oxide units" if self._as_oxides() else "rel. %")
        plot.setLabel("bottom", f"{xk}  {unit}", color="k")
        plot.setLabel("left", f"{yk}  {unit}", color="k")
        if not self.summaries or not xk or not yk:
            plot.setTitle("Element correlation", color="k")
            return

        if self.replicates_check.isChecked():
            reps = replicate_scatter_points(self.summaries, xk, yk, **self._kw())
            if reps:
                plot.addItem(
                    pg.ScatterPlotItem(
                        x=[p[2] for p in reps],
                        y=[p[3] for p in reps],
                        size=6,
                        brush=_REP_BRUSH,
                        pen=_REP_PEN,
                    )
                )

        pts = scatter_points(self.summaries, xk, yk, **self._kw())
        xs = np.array([p[1] for p in pts], dtype=float)
        ys = np.array([p[2] for p in pts], dtype=float)
        names = [p[0] for p in pts]
        if self.errorbar_check.isChecked() and pts:
            xerr = np.array([p[3] for p in pts], dtype=float)
            yerr = np.array([p[4] for p in pts], dtype=float)
            if np.any(xerr > 0) or np.any(yerr > 0):
                plot.addItem(
                    pg.ErrorBarItem(
                        x=xs,
                        y=ys,
                        left=xerr,
                        right=xerr,
                        top=yerr,
                        bottom=yerr,
                        beam=0.0,
                        pen=pg.mkPen(80, 80, 80, 160),
                    )
                )
        plot.addItem(self._mean_scatter(xs, ys, names))
        self._add_labels(plot, xs, ys, names)
        r, rho = correlation(xs, ys)
        title = f"{xk} vs {yk}"
        if np.isfinite(r):
            title += f"   r={r:.3f}  ρ={rho:.3f}"
        plot.setTitle(title, color="k")

    def _draw_ratios(self) -> None:
        plot = self.ratio_plot
        plot.clear()
        xn, xd = self.ratio_x_num.currentText(), self.ratio_x_den.currentText()
        yn, yd = self.ratio_y_num.currentText(), self.ratio_y_den.currentText()
        xlab = f"{xn}/{xd}" if xn and xd else "X ratio"
        ylab = f"{yn}/{yd}" if yn and yd else "Y ratio"
        plot.setLabel("bottom", xlab, color="k")
        plot.setLabel("left", ylab, color="k")
        if not self.summaries or not all((xn, xd, yn, yd)):
            plot.setTitle("Element ratios", color="k")
            return

        kw = self._kw()
        if self.replicates_check.isChecked():
            reps = replicate_ratio_points(
                self.summaries, xn, xd, yn, yd, **kw
            )
            if reps:
                plot.addItem(
                    pg.ScatterPlotItem(
                        x=[p[2] for p in reps],
                        y=[p[3] for p in reps],
                        size=6,
                        brush=_REP_BRUSH,
                        pen=_REP_PEN,
                    )
                )
        pts = ratio_points(self.summaries, xn, xd, yn, yd, **kw)
        if not pts:
            plot.setTitle(f"{xlab} vs {ylab}  (no finite ratios)", color="k")
            return
        xs = np.array([p[1] for p in pts], dtype=float)
        ys = np.array([p[2] for p in pts], dtype=float)
        names = [p[0] for p in pts]
        plot.addItem(self._mean_scatter(xs, ys, names))
        self._add_labels(plot, xs, ys, names)
        r, rho = correlation(xs, ys)
        title = f"{xlab} vs {ylab}"
        if np.isfinite(r):
            title += f"   r={r:.3f}  ρ={rho:.3f}"
        plot.setTitle(title, color="k")

    def _draw_matrix(self) -> None:
        keys = self._keys()
        n = len(keys)
        if n == 0 or len(self.summaries) < 2:
            self.matrix_image.setImage(np.zeros((1, 1)))
            self.matrix_plot.setTitle("Need ≥2 samples", color="k")
            return
        matrix = correlation_matrix(self.summaries, keys, **self._kw())
        filled = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
        self.matrix_image.setImage(filled, autoLevels=False)
        self.matrix_image.setLevels((-1.0, 1.0))
        ticks = [(i + 0.5, keys[i]) for i in range(n)]
        self.matrix_plot.getAxis("bottom").setTicks([ticks])
        self.matrix_plot.getAxis("left").setTicks([ticks])
        self.matrix_plot.setTitle("Pearson r (sample means)", color="k")
        self.matrix_plot.setXRange(0, n, padding=0)
        self.matrix_plot.setYRange(0, n, padding=0)

    def _on_matrix_clicked(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        keys = self._keys()
        n = len(keys)
        if n == 0:
            return
        view = self.matrix_plot.getViewBox()
        pos = view.mapSceneToView(event.scenePos())
        i = int(pos.x())
        j = int(pos.y())
        if not (0 <= i < n and 0 <= j < n) or i == j:
            return
        self._filling_combos = True
        self.corr_x.setCurrentText(keys[i])
        self.corr_y.setCurrentText(keys[j])
        self._filling_combos = False
        self.plot_tabs.setCurrentIndex(1)
        self._refresh_plots()

    def _export(self, fmt: str) -> None:
        if not self.summaries:
            QMessageBox.information(self, "Export", "No sample means to export.")
            return
        if fmt == "csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export sample means", "sample_means.csv", "CSV (*.csv)"
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export sample means", "sample_means.xlsx", "Excel (*.xlsx)"
            )
        if not path:
            return
        try:
            kw = dict(self._kw(), include_std=True)
            if fmt == "csv":
                export_sample_means_csv(Path(path), self.summaries, **kw)
            else:
                export_sample_means_excel(Path(path), self.summaries, **kw)
        except Exception as exc:
            QMessageBox.critical(self, "Export", str(exc))
            return
        QMessageBox.information(self, "Export", f"Wrote {path}")
