"""
Spectrum display widget using PyQtGraph for high-performance plotting
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from core.xray_data import get_element_lines, build_tube_guide_regions
from core.peak_fitting import PeakFitter

_OVERLAY_COLORS = [
    "#d32f2f",
    "#388e3c",
    "#f57c00",
    "#7b1fa2",
    "#0097a7",
    "#c2185b",
    "#689f38",
    "#5d4037",
    "#455a64",
    "#e65100",
]


def _normalize_counts(counts, scale_to=1.0):
    y = np.asarray(counts, dtype=float)
    peak = float(np.max(y)) if y.size else 0.0
    if peak <= 0:
        return y
    return y * (float(scale_to) / peak)


def enable_box_zoom(plot, enabled=True):
    """Left-drag a rectangle to zoom; right-drag pans; wheel zooms."""
    if plot is None:
        return
    if hasattr(plot, "getViewBox"):
        vb = plot.getViewBox()
    elif hasattr(plot, "getPlotItem"):
        vb = plot.getPlotItem().getViewBox()
    else:
        vb = plot
    if vb is None:
        return
    vb.setMouseMode(pg.ViewBox.RectMode if enabled else pg.ViewBox.PanMode)


class SpectrumWidget(QWidget):
    """Widget for displaying XRF spectra with interactive features"""
    
    energy_selected = Signal(float)  # Emitted when user clicks on spectrum
    log_scale_changed = Signal(bool)  # Emitted when Log Y-axis control changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.spectrum_data = None
        self.fitted_data = None
        self.background_data = None
        self.peak_markers = []
        self._peak_marker_specs = []  # list of dicts to redraw after plot clear
        self._tube_guide_items = []
        self._tube_guide_specs = []
        self._tube_guides_visible = True
        self._energy_pick_mode = False
        self._pick_marker = None
        self._overlays = []
        self._overlay_color_i = 0
        
        self._setup_ui()
        self._configure_plot()
    
    def _setup_ui(self):
        """Setup the widget layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot widget for main spectrum
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        layout.addWidget(self.plot_widget, stretch=3)
        
        # Create plot widget for residuals
        self.residuals_widget = pg.PlotWidget()
        self.residuals_widget.setBackground('w')
        self.residuals_widget.setMaximumHeight(150)
        layout.addWidget(self.residuals_widget, stretch=1)
        
        # Compact plot controls + cursor info
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(4, 2, 4, 2)
        bottom_bar.setSpacing(6)
        
        small_btn_style = (
            "QPushButton, QCheckBox {"
            "  font-size: 11px;"
            "  padding: 1px 8px;"
            "  min-height: 18px;"
            "  max-height: 22px;"
            "}"
        )
        
        self.home_button = QPushButton("Home")
        self.home_button.setToolTip(
            "Reset plot view to fit all data.\n"
            "Left-drag a box to zoom; right-drag to pan; scroll to zoom."
        )
        self.home_button.setStyleSheet(small_btn_style)
        self.home_button.setFixedHeight(22)
        self.home_button.clicked.connect(self.reset_view)
        bottom_bar.addWidget(self.home_button)
        
        self.log_y_checkbox = QCheckBox("Log Y-axis")
        self.log_y_checkbox.setToolTip("Toggle logarithmic Y-axis")
        self.log_y_checkbox.setStyleSheet(small_btn_style)
        self.log_y_checkbox.toggled.connect(self._on_log_y_toggled)
        bottom_bar.addWidget(self.log_y_checkbox)

        self.pin_overlay_button = QPushButton("Pin overlay")
        self.pin_overlay_button.setToolTip(
            "Keep the current spectrum on the plot when you load another one"
        )
        self.pin_overlay_button.setStyleSheet(small_btn_style)
        self.pin_overlay_button.setFixedHeight(22)
        self.pin_overlay_button.clicked.connect(self.pin_current_overlay)
        bottom_bar.addWidget(self.pin_overlay_button)

        self.clear_overlays_button = QPushButton("Clear overlays")
        self.clear_overlays_button.setToolTip("Remove comparison spectra from the plot")
        self.clear_overlays_button.setStyleSheet(small_btn_style)
        self.clear_overlays_button.setFixedHeight(22)
        self.clear_overlays_button.clicked.connect(self.clear_overlays)
        bottom_bar.addWidget(self.clear_overlays_button)

        self.normalize_checkbox = QCheckBox("Normalize")
        self.normalize_checkbox.setToolTip(
            "Scale each spectrum to its own maximum for shape comparison"
        )
        self.normalize_checkbox.setStyleSheet(small_btn_style)
        self.normalize_checkbox.toggled.connect(self._on_normalize_toggled)
        bottom_bar.addWidget(self.normalize_checkbox)
        
        self.info_label = QLabel("Energy: -- keV | Counts: --")
        self.info_label.setStyleSheet("padding: 2px 5px; background-color: #f0f0f0;")
        bottom_bar.addWidget(self.info_label, stretch=1)
        
        layout.addLayout(bottom_bar)
        self._refresh_overlay_controls()
    
    def _configure_plot(self):
        """Configure plot appearance and behavior"""
        # Main plot configuration
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setLabel('left', 'Counts', units='')
        plot_item.setLabel('bottom', 'Energy', units='keV')
        plot_item.showGrid(x=True, y=True, alpha=0.3)
        plot_item.setLogMode(False, False)  # Linear Y-axis by default
        self._ensure_legend(plot_item)
        enable_box_zoom(plot_item)
        
        # Enable antialiasing for smooth lines
        self.plot_widget.setAntialiasing(True)
        
        # Add crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        plot_item.addItem(self.vLine, ignoreBounds=True)
        plot_item.addItem(self.hLine, ignoreBounds=True)

        # Click-to-identify marker (hidden until pick mode uses it)
        self._pick_marker = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen('#c62828', width=2, style=Qt.SolidLine),
        )
        self._pick_marker.setVisible(False)
        plot_item.addItem(self._pick_marker, ignoreBounds=True)
        
        # Connect mouse movement and clicks
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        
        # Residuals plot configuration
        residuals_item = self.residuals_widget.getPlotItem()
        residuals_item.setLabel('left', 'Residuals', units='')
        residuals_item.setLabel('bottom', 'Energy', units='keV')
        residuals_item.showGrid(x=True, y=True, alpha=0.3)
        residuals_item.addLine(y=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        enable_box_zoom(residuals_item)
        
        # Link X-axes
        self.residuals_widget.setXLink(self.plot_widget)
    
    def set_spectrum(self, spectrum):
        """
        Set the spectrum data to display
        
        Args:
            spectrum: Spectrum object with energy and counts arrays
        """
        self.spectrum_data = spectrum
        self._update_plot()

    def _next_overlay_color(self) -> str:
        color = _OVERLAY_COLORS[self._overlay_color_i % len(_OVERLAY_COLORS)]
        self._overlay_color_i += 1
        return color

    def _spectrum_overlay_name(self, spectrum, fallback="spectrum") -> str:
        meta = getattr(spectrum, "metadata", None) or {}
        name = meta.get("name") or meta.get("source_label") or fallback
        return str(name)

    def add_overlay(self, spectrum, name=None, color=None):
        """Keep a comparison spectrum on the plot (does not replace the primary)."""
        if spectrum is None:
            return
        energy = np.asarray(getattr(spectrum, "energy", None), dtype=float)
        counts = np.asarray(getattr(spectrum, "counts", None), dtype=float)
        if energy.size == 0 or counts.size == 0 or energy.size != counts.size:
            return
        label = str(name or self._spectrum_overlay_name(spectrum))
        existing = {ov["name"] for ov in self._overlays}
        base = label
        n = 2
        while label in existing:
            label = f"{base} ({n})"
            n += 1
        self._overlays.append(
            {
                "name": label,
                "energy": energy.copy(),
                "counts": counts.copy(),
                "color": color or self._next_overlay_color(),
            }
        )
        self._refresh_overlay_controls()
        self._update_plot()

    def pin_current_overlay(self):
        """Pin the currently displayed spectrum as a comparison overlay."""
        if self.spectrum_data is None:
            return
        self.add_overlay(
            self.spectrum_data,
            name=self._spectrum_overlay_name(self.spectrum_data, "Pinned"),
        )

    def clear_overlays(self):
        self._overlays = []
        self._overlay_color_i = 0
        self._refresh_overlay_controls()
        self._update_plot()

    def overlay_count(self) -> int:
        return len(self._overlays)

    def _refresh_overlay_controls(self):
        n = len(self._overlays)
        if hasattr(self, "pin_overlay_button"):
            self.pin_overlay_button.setText(
                "Pin overlay" if n == 0 else f"Pin overlay ({n})"
            )
        if hasattr(self, "clear_overlays_button"):
            self.clear_overlays_button.setEnabled(n > 0)

    def _on_normalize_toggled(self, _checked=False):
        self._update_plot()
    
    def set_fitted_spectrum(self, fitted_spectrum):
        """Set fitted spectrum data"""
        self.fitted_data = fitted_spectrum
        self._update_plot()
    
    def set_background(self, background):
        """Set background data"""
        self.background_data = background
        self._update_plot()
    
    def add_peak_marker(self, energy, element=None, line=None, color=None, label=None,
                        redraw=True, relative_intensity=None):
        """
        Add a peak marker at specified energy
        
        Args:
            energy: Peak energy in keV
            element: Element symbol (optional)
            line: Line designation (e.g., 'Ka', 'Kb', 'La')
            color: Optional pen/label color
            label: Optional explicit label text
            redraw: If False, defer redraw (caller should call _redraw_peak_markers)
            relative_intensity: Optional 0–1 strength (scales stick height / opacity)
        """
        if label is None:
            if element and line:
                label = f"{element}-{line}"
            elif element:
                label = str(element)
            else:
                label = f"{energy:.1f}"
        
        if color is None:
            if element and str(element).upper() == 'TUBE':
                color = '#9C27B0'
            elif element:
                color = '#E65100'
            else:
                color = '#00897B'
        
        spec = {
            'energy': float(energy),
            'label': label,
            'color': color,
            'relative_intensity': (
                None if relative_intensity is None else float(relative_intensity)
            ),
        }
        self._peak_marker_specs.append(spec)
        if redraw:
            self._redraw_peak_markers()
    
    def set_peak_markers(self, peaks, show=True):
        """
        Replace peak markers from fitted Peak objects or (energy, label) specs.

        Stick height is normalized per element: that element's strongest
        fitted line is 100%, and its other lines (Kβ, L, …) scale to their
        share of that element's area. Tube lines are a separate group.
        Peak-find seeds without a fitted area use catalog relative_intensity
        when present (so Fe Kβ still shows below the detection threshold).
        Peaks without area or catalog intensity stay as full-height dashed markers.

        Args:
            peaks: List of Peak objects, or dicts with energy/element/line/is_tube_line
            show: If False, clear markers without drawing
        """
        self.clear_peak_markers()
        if not show or not peaks:
            return

        def _attr(peak, name, default=None):
            if hasattr(peak, name):
                return getattr(peak, name, default)
            if isinstance(peak, dict):
                return peak.get(name, default)
            return default

        max_area = {}
        for peak in peaks:
            area = float(_attr(peak, "area", 0.0) or 0.0)
            if area <= 0:
                area = float(_attr(peak, "amplitude", 0.0) or 0.0)
            if area <= 0:
                continue
            element = _attr(peak, "element")
            is_tube = bool(_attr(peak, "is_tube_line", False))
            if is_tube:
                key = ("tube", element or "Tube")
            elif element:
                key = ("sample", element)
            else:
                key = ("unlabeled", None)
            max_area[key] = max(max_area.get(key, 0.0), area)

        for peak in peaks:
            if hasattr(peak, "energy"):
                energy = peak.energy
            else:
                energy = peak.get("energy")
            element = _attr(peak, "element")
            line = _attr(peak, "line")
            is_tube = bool(_attr(peak, "is_tube_line", False))
            area = float(_attr(peak, "area", 0.0) or 0.0)
            if area <= 0:
                area = float(_attr(peak, "amplitude", 0.0) or 0.0)

            if is_tube:
                color = "#9C27B0"
                label = f"{element or 'Tube'}-{line or '?'}"
                key = ("tube", element or "Tube")
            elif element and line:
                color = "#E65100"
                label = f"{element}-{line}"
                key = ("sample", element)
            elif element:
                color = "#E65100"
                label = str(element)
                key = ("sample", element)
            else:
                color = "#00897B"
                label = f"{float(energy):.1f}"
                key = ("unlabeled", None)

            denom = max_area.get(key, 0.0)
            catalog_rel = _attr(peak, "relative_intensity", None)
            if denom > 0 and area > 0:
                rel = area / denom
            elif catalog_rel is not None:
                try:
                    rel = max(0.0, min(1.0, float(catalog_rel)))
                except (TypeError, ValueError):
                    rel = None
            else:
                rel = None
            if rel is not None:
                pct = int(round(100.0 * rel))
                label = f"{label} {pct}%"

            self.add_peak_marker(
                energy,
                element=element,
                line=line,
                color=color,
                label=label,
                relative_intensity=rel,
                redraw=False,
            )

        self._redraw_peak_markers()
    
    def clear_peak_markers(self):
        """Remove all peak markers"""
        plot_item = self.plot_widget.getPlotItem()
        for marker in self.peak_markers:
            plot_item.removeItem(marker)
        self.peak_markers.clear()
        self._peak_marker_specs.clear()

    def set_tube_guides(self, regions, show=True):
        """
        Set faint vertical bands marking tube elastic / Compton locations.

        Args:
            regions: Iterable of dicts from build_tube_guide_regions
                     (energy, half_width, label, kind)
            show: If False, keep specs but hide the overlay
        """
        self._tube_guide_specs = [dict(r) for r in (regions or [])]
        self._tube_guides_visible = bool(show)
        self._redraw_tube_guides()

    def clear_tube_guides(self):
        """Remove tube-line guide bands from the plot."""
        self._tube_guide_specs.clear()
        self._tube_guides_visible = False
        self._remove_tube_guide_items()

    def set_tube_guides_visible(self, show: bool):
        """Show or hide existing tube guides without clearing their specs."""
        self._tube_guides_visible = bool(show)
        self._redraw_tube_guides()

    def _remove_tube_guide_items(self):
        plot_item = self.plot_widget.getPlotItem()
        for item in self._tube_guide_items:
            try:
                plot_item.removeItem(item)
            except Exception:
                pass
        self._tube_guide_items.clear()

    def _redraw_tube_guides(self):
        """Draw tube guide bands behind the spectrum (survives plot clear)."""
        self._remove_tube_guide_items()
        if not self._tube_guides_visible or not self._tube_guide_specs:
            return
        plot_item = self.plot_widget.getPlotItem()
        # Draw elastic first, Compton on top of those (still behind data)
        for spec in self._tube_guide_specs:
            energy = float(spec.get("energy", 0.0))
            half = max(0.03, float(spec.get("half_width", 0.08)))
            kind = str(spec.get("kind") or "elastic")
            label = str(spec.get("label") or "")
            if kind == "compton":
                brush = pg.mkBrush(156, 39, 176, 28)  # purple, very faint
                pen = pg.mkPen(156, 39, 176, 50)
            else:
                brush = pg.mkBrush(123, 31, 162, 38)
                pen = pg.mkPen(123, 31, 162, 70)
            region = pg.LinearRegionItem(
                values=(energy - half, energy + half),
                orientation="vertical",
                brush=brush,
                pen=pen,
                movable=False,
            )
            region.setZValue(-20)
            try:
                region.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            except Exception:
                pass
            plot_item.addItem(region, ignoreBounds=True)
            self._tube_guide_items.append(region)

            # Label only major features so the plot stays readable
            short = label.split()[-1] if label else ""
            if short in {"Kα1", "Lα1", "Compton Kα", "Compton Kβ", "Kβ1"}:
                text = pg.TextItem(
                    text=label,
                    color=(120, 40, 140, 160),
                    anchor=(0.5, 0.0),
                )
                # Place near the top of the view; update on view change is overkill
                y = self._intensity_stick_scale() * 1.02
                text.setPos(energy, y)
                text.setZValue(-10)
                plot_item.addItem(text)
                self._tube_guide_items.append(text)
    
    def _assign_label_positions(self):
        """Stagger peak-label heights by energy so neighbors don't pile up"""
        # Alternate high→low so adjacent peaks are less likely to collide
        positions = (0.96, 0.82, 0.68, 0.88, 0.74, 0.60)
        ordered = sorted(self._peak_marker_specs, key=lambda s: s['energy'])
        for i, spec in enumerate(ordered):
            # Intensity-scaled sticks place labels near stick tip instead
            if spec.get('relative_intensity') is not None:
                rel = max(0.015, min(1.0, float(spec['relative_intensity'])))
                spec['position'] = 0.12 + 0.82 * rel
            else:
                spec['position'] = positions[i % len(positions)]
    
    def _intensity_stick_scale(self):
        """Counts scale for relative-intensity sticks (fraction of spectrum max)."""
        if self.spectrum_data is not None and len(self.spectrum_data.counts):
            peak = float(np.nanmax(self.spectrum_data.counts))
            if peak > 0:
                return 0.85 * peak
        # No spectrum loaded — arbitrary display units
        return 1000.0

    def _draw_peak_marker(self, spec):
        """Draw one stored peak marker onto the current plot"""
        plot_item = self.plot_widget.getPlotItem()
        color = spec['color']
        rel = spec.get('relative_intensity')

        if rel is not None:
            # Vertical stick whose height tracks relative radiative intensity
            rel = max(0.015, min(1.0, float(rel)))
            y_top = rel * self._intensity_stick_scale()
            width = 1.0 + 1.5 * rel
            alpha = int(80 + 175 * rel)
            pen = pg.mkPen(color, width=width)
            # Parse color for alpha if possible
            try:
                qcolor = QColor(color)
                qcolor.setAlpha(alpha)
                pen = pg.mkPen(qcolor, width=width)
            except Exception:
                pass

            e = float(spec['energy'])
            stick = plot_item.plot(
                [e, e],
                [0.0, y_top],
                pen=pen,
            )
            self.peak_markers.append(stick)

            # Text label near stick tip with intensity percent
            text = pg.TextItem(
                text=spec['label'],
                color=color,
                anchor=(0.5, 1.0),
                fill=pg.mkBrush(255, 255, 255, 210),
            )
            text.setPos(e, y_top)
            plot_item.addItem(text)
            self.peak_markers.append(text)
            return

        line_item = pg.InfiniteLine(
            pos=spec['energy'],
            angle=90,
            pen=pg.mkPen(color, width=1.2, style=Qt.DashLine),
            label=spec['label'],
            labelOpts={
                'position': spec.get('position', 0.92),
                'color': color,
                'fill': pg.mkBrush(255, 255, 255, 210),
                'movable': False,
            }
        )
        plot_item.addItem(line_item)
        self.peak_markers.append(line_item)
    
    def _redraw_peak_markers(self):
        """Re-add peak markers after plot_item.clear()"""
        plot_item = self.plot_widget.getPlotItem()
        for marker in self.peak_markers:
            plot_item.removeItem(marker)
        self.peak_markers.clear()
        
        self._assign_label_positions()
        for spec in self._peak_marker_specs:
            self._draw_peak_marker(spec)
    
    def _ensure_legend(self, plot_item):
        """Keep a readable series legend anchored top-right"""
        if plot_item.legend is None:
            legend = plot_item.addLegend(offset=(-10, 10), labelTextSize='9pt')
        else:
            legend = plot_item.legend
            legend.clear()
        
        legend.setBrush(pg.mkBrush(255, 255, 255, 235))
        legend.setPen(pg.mkPen(160, 160, 160))
        return legend
    
    def show_element_lines(self, symbol, z):
        """
        Show emission lines for an element, scaled by relative intensity.

        Stick height is element-wide (K/L/M together, fluorescence-yield
        weighted) so Mα is much weaker than Lα. Lines outside the measured
        spectrum (or above 40 keV) are omitted so e.g. Pb Kα at ~75 keV
        does not hide Pb Lα. Diagnostic α lines are always kept.
        
        Args:
            symbol: Element symbol
            z: Atomic number
        """
        lines = get_element_lines(symbol, z)
        
        series_colors = {
            'K': 'r',
            'L': 'g',
            'M': 'b',
            'N': 'm',
        }
        diagnostic_names = {'Kα1', 'Kα', 'Lα1', 'Lα', 'Mα1', 'Mα'}

        e_min, e_max = PeakFitter.MIN_PEAK_ENERGY_KEV, 40.0
        if self.spectrum_data is not None and len(getattr(self.spectrum_data, 'energy', [])):
            e_min = max(
                PeakFitter.MIN_PEAK_ENERGY_KEV,
                float(np.nanmin(self.spectrum_data.energy)),
            )
            e_max = float(np.nanmax(self.spectrum_data.energy))

        collected = []
        for series, color in series_colors.items():
            for line_data in lines.get(series, []) or []:
                energy = float(line_data['energy'])
                if energy < e_min or energy > e_max:
                    continue
                rel = float(line_data.get('relative_intensity', 1.0) or 1.0)
                collected.append((color, line_data, rel))

        max_rel = max((rel for _, _, rel in collected), default=1.0) or 1.0

        self.clear_peak_markers()

        for color, line_data, rel in collected:
            name = line_data['name']
            rel_disp = rel / max_rel
            if rel_disp < 0.02 and name not in diagnostic_names:
                continue
            pct = int(round(100.0 * rel_disp))
            label = f"{symbol}-{name} {pct}%"
            self.add_peak_marker(
                float(line_data['energy']),
                element=symbol,
                line=name,
                color=color,
                label=label,
                relative_intensity=rel_disp,
                redraw=False,
            )
        self._redraw_peak_markers()
    
    def set_log_scale(self, enabled):
        """Enable or disable logarithmic Y-axis"""
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setLogMode(False, enabled)
        if self.log_y_checkbox.isChecked() != enabled:
            self.log_y_checkbox.blockSignals(True)
            self.log_y_checkbox.setChecked(enabled)
            self.log_y_checkbox.blockSignals(False)
    
    def _on_log_y_toggled(self, checked):
        """Handle Log Y-axis checkbox on the plot toolbar"""
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setLogMode(False, checked)
        self.log_scale_changed.emit(checked)
    
    def reset_view(self):
        """Reset plot axes to fit the current data (Home)"""
        self.plot_widget.getPlotItem().autoRange()
        self.residuals_widget.getPlotItem().autoRange()
    
    def set_grid(self, enabled):
        """Enable or disable grid"""
        plot_item = self.plot_widget.getPlotItem()
        plot_item.showGrid(x=enabled, y=enabled, alpha=0.3)
    
    def _update_plot(self):
        """Update the plot with current data"""
        plot_item = self.plot_widget.getPlotItem()
        normalize = bool(
            getattr(self, "normalize_checkbox", None)
            and self.normalize_checkbox.isChecked()
        )
        plot_item.setLabel("left", "Normalized" if normalize else "Counts", units="")
        saved_specs = list(self._peak_marker_specs)
        saved_tube = list(self._tube_guide_specs)
        tube_vis = self._tube_guides_visible
        plot_item.clear()
        self.peak_markers.clear()
        self._tube_guide_items.clear()
        self._peak_marker_specs = saved_specs
        self._tube_guide_specs = saved_tube
        self._tube_guides_visible = tube_vis
        
        # Re-add crosshair and legend after clear
        plot_item.addItem(self.vLine, ignoreBounds=True)
        plot_item.addItem(self.hLine, ignoreBounds=True)
        if self._pick_marker is not None:
            plot_item.addItem(self._pick_marker, ignoreBounds=True)
        self._ensure_legend(plot_item)

        # Tube guides behind the measured curve
        self._redraw_tube_guides()

        overlays = list(self._overlays or [])
        if self.spectrum_data is None and not overlays:
            return

        primary_peak = 1.0
        if self.spectrum_data is not None and len(self.spectrum_data.counts):
            primary_peak = float(np.max(self.spectrum_data.counts)) or 1.0

        def _y(counts, peak=None):
            y = np.asarray(counts, dtype=float)
            if not normalize:
                return y
            scale = float(peak if peak is not None else (np.max(y) if y.size else 1.0))
            if scale <= 0:
                return y
            return y / scale

        for overlay in overlays:
            plot_item.plot(
                overlay["energy"],
                _y(overlay["counts"]),
                pen=pg.mkPen(overlay["color"], width=1.5),
                name=overlay["name"],
            )

        if self.spectrum_data is None:
            self.residuals_widget.getPlotItem().clear()
            self._redraw_peak_markers()
            return

        primary_name = "Measured"
        meta = getattr(self.spectrum_data, "metadata", None) or {}
        if overlays:
            primary_name = str(meta.get("name") or meta.get("source_label") or "Measured")

        plot_item.plot(
            self.spectrum_data.energy,
            _y(self.spectrum_data.counts, primary_peak),
            pen=pg.mkPen("b", width=2),
            name=primary_name,
        )

        if self.background_data is not None and not overlays:
            plot_item.plot(
                self.spectrum_data.energy,
                _y(self.background_data, primary_peak),
                pen=pg.mkPen("g", width=1, style=Qt.DashLine),
                name="Background",
            )

        if self.fitted_data is not None and not overlays:
            plot_item.plot(
                self.spectrum_data.energy,
                _y(self.fitted_data, primary_peak),
                pen=pg.mkPen("r", width=2),
                name="Fitted",
            )
            self._update_residuals()
        elif overlays:
            self.residuals_widget.getPlotItem().clear()

        self._redraw_peak_markers()

    def _update_residuals(self):
        """Update residuals plot"""
        if self.spectrum_data is None or self.fitted_data is None:
            return
        
        residuals = self.spectrum_data.counts - self.fitted_data
        
        residuals_item = self.residuals_widget.getPlotItem()
        residuals_item.clear()
        residuals_item.addLine(y=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        residuals_item.plot(
            self.spectrum_data.energy,
            residuals,
            pen=None,
            symbol='o',
            symbolSize=3,
            symbolBrush='b'
        )
    
    def set_energy_pick_mode(self, enabled: bool):
        """Enable/disable click-to-identify energy picking on the main plot."""
        self._energy_pick_mode = bool(enabled)
        enable_box_zoom(self.plot_widget.getPlotItem(), enabled=not self._energy_pick_mode)
        if self._pick_marker is not None and not enabled:
            self._pick_marker.setVisible(False)
        if enabled:
            self.info_label.setText(
                "Identify mode: click the spectrum to list line candidates"
            )
            self.plot_widget.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.plot_widget.unsetCursor()
            if self.spectrum_data is None:
                self.info_label.setText("Energy: -- keV | Counts: --")

    def mark_picked_energy(self, energy_kev: float):
        """Show a solid marker at the last picked energy."""
        if self._pick_marker is None:
            return
        self._pick_marker.setPos(float(energy_kev))
        self._pick_marker.setVisible(True)

    def _on_mouse_clicked(self, event):
        """Emit energy_selected when identify/pick mode is active."""
        if not self._energy_pick_mode:
            return
        if getattr(event, "button", lambda: None)() not in (
            Qt.MouseButton.LeftButton,
            1,  # Qt.LeftButton numeric fallback
        ):
            return
        plot_item = self.plot_widget.getPlotItem()
        scene_pos = event.scenePos()
        if not plot_item.sceneBoundingRect().contains(scene_pos):
            return
        mouse_point = plot_item.vb.mapSceneToView(scene_pos)
        energy = float(mouse_point.x())
        if self.spectrum_data is not None and len(self.spectrum_data.energy):
            idx = int(np.argmin(np.abs(self.spectrum_data.energy - energy)))
            energy = float(self.spectrum_data.energy[idx])
        self.mark_picked_energy(energy)
        self.energy_selected.emit(energy)

    def _on_mouse_moved(self, pos):
        """Handle mouse movement for crosshair and info display"""
        plot_item = self.plot_widget.getPlotItem()
        mouse_point = plot_item.vb.mapSceneToView(pos)
        
        # Update crosshair position
        self.vLine.setPos(mouse_point.x())
        self.hLine.setPos(mouse_point.y())
        
        # Update info label
        energy = mouse_point.x()
        counts = mouse_point.y()
        prefix = "Identify · " if self._energy_pick_mode else ""
        
        if self.spectrum_data is not None:
            # Find nearest data point
            idx = np.argmin(np.abs(self.spectrum_data.energy - energy))
            if idx < len(self.spectrum_data.counts):
                actual_energy = self.spectrum_data.energy[idx]
                actual_counts = self.spectrum_data.counts[idx]
                self.info_label.setText(
                    f"{prefix}Energy: {actual_energy:.3f} keV | Counts: {actual_counts:.0f}"
                )
            else:
                self.info_label.setText(
                    f"{prefix}Energy: {energy:.3f} keV | Counts: {counts:.0f}"
                )
        else:
            self.info_label.setText(
                f"{prefix}Energy: {energy:.3f} keV | Counts: {counts:.0f}"
            )
    
    def capture_state(self) -> dict:
        plot = self.plot_widget.getPlotItem()
        resid = self.residuals_widget.getPlotItem()
        xr, yr = plot.getViewBox().viewRange()
        rx, ry = resid.getViewBox().viewRange()
        return {
            "log_y": bool(self.log_y_checkbox.isChecked()),
            "view": {
                "x": [float(xr[0]), float(xr[1])],
                "y": [float(yr[0]), float(yr[1])],
            },
            "residuals_view": {
                "x": [float(rx[0]), float(rx[1])],
                "y": [float(ry[0]), float(ry[1])],
            },
            "peak_markers": list(self._peak_marker_specs or []),
        }

    def restore_state(self, state: dict) -> None:
        if not state:
            return
        if "log_y" in state:
            self.set_log_scale(bool(state["log_y"]))
        view = state.get("view") or {}
        if view.get("x") and view.get("y"):
            self.plot_widget.getPlotItem().setXRange(*view["x"], padding=0)
            self.plot_widget.getPlotItem().setYRange(*view["y"], padding=0)
        rview = state.get("residuals_view") or {}
        if rview.get("x") and rview.get("y"):
            self.residuals_widget.getPlotItem().setXRange(*rview["x"], padding=0)
            self.residuals_widget.getPlotItem().setYRange(*rview["y"], padding=0)

    def export_plot(self, file_path):
        """
        Export plot to image file
        
        Args:
            file_path: Path to save image (supports PNG, SVG, PDF)
        """
        exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
        exporter.export(file_path)
