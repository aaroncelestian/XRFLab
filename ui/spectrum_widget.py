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
from core.xray_data import get_element_lines


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
        self.home_button.setToolTip("Reset plot view to fit all data")
        self.home_button.setStyleSheet(small_btn_style)
        self.home_button.setFixedHeight(22)
        self.home_button.clicked.connect(self.reset_view)
        bottom_bar.addWidget(self.home_button)
        
        self.log_y_checkbox = QCheckBox("Log Y-axis")
        self.log_y_checkbox.setToolTip("Toggle logarithmic Y-axis")
        self.log_y_checkbox.setStyleSheet(small_btn_style)
        self.log_y_checkbox.toggled.connect(self._on_log_y_toggled)
        bottom_bar.addWidget(self.log_y_checkbox)
        
        self.info_label = QLabel("Energy: -- keV | Counts: --")
        self.info_label.setStyleSheet("padding: 2px 5px; background-color: #f0f0f0;")
        bottom_bar.addWidget(self.info_label, stretch=1)
        
        layout.addLayout(bottom_bar)
    
    def _configure_plot(self):
        """Configure plot appearance and behavior"""
        # Main plot configuration
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setLabel('left', 'Counts', units='')
        plot_item.setLabel('bottom', 'Energy', units='keV')
        plot_item.showGrid(x=True, y=True, alpha=0.3)
        plot_item.setLogMode(False, False)  # Linear Y-axis by default
        self._ensure_legend(plot_item)
        
        # Enable antialiasing for smooth lines
        self.plot_widget.setAntialiasing(True)
        
        # Add crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        plot_item.addItem(self.vLine, ignoreBounds=True)
        plot_item.addItem(self.hLine, ignoreBounds=True)
        
        # Connect mouse movement
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        
        # Residuals plot configuration
        residuals_item = self.residuals_widget.getPlotItem()
        residuals_item.setLabel('left', 'Residuals', units='')
        residuals_item.setLabel('bottom', 'Energy', units='keV')
        residuals_item.showGrid(x=True, y=True, alpha=0.3)
        residuals_item.addLine(y=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        
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
    
    def set_fitted_spectrum(self, fitted_spectrum):
        """Set fitted spectrum data"""
        self.fitted_data = fitted_spectrum
        self._update_plot()
    
    def set_background(self, background):
        """Set background data"""
        self.background_data = background
        self._update_plot()
    
    def add_peak_marker(self, energy, element=None, line=None, color=None, label=None,
                        redraw=True):
        """
        Add a peak marker at specified energy
        
        Args:
            energy: Peak energy in keV
            element: Element symbol (optional)
            line: Line designation (e.g., 'Ka', 'Kb', 'La')
            color: Optional pen/label color
            label: Optional explicit label text
            redraw: If False, defer redraw (caller should call _redraw_peak_markers)
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
        }
        self._peak_marker_specs.append(spec)
        if redraw:
            self._redraw_peak_markers()
    
    def set_peak_markers(self, peaks, show=True):
        """
        Replace peak markers from fitted Peak objects or (energy, label) specs.
        
        Args:
            peaks: List of Peak objects, or dicts with energy/element/line/is_tube_line
            show: If False, clear markers without drawing
        """
        self.clear_peak_markers()
        if not show or not peaks:
            return
        
        for peak in peaks:
            if hasattr(peak, 'energy'):
                energy = peak.energy
                element = getattr(peak, 'element', None)
                line = getattr(peak, 'line', None)
                is_tube = getattr(peak, 'is_tube_line', False)
            else:
                energy = peak.get('energy')
                element = peak.get('element')
                line = peak.get('line')
                is_tube = peak.get('is_tube_line', False)
            
            if is_tube:
                color = '#9C27B0'
                label = f"{element or 'Tube'}-{line or '?'}"
            elif element and line:
                color = '#E65100'
                label = f"{element}-{line}"
            elif element:
                color = '#E65100'
                label = str(element)
            else:
                color = '#00897B'
                label = f"{float(energy):.1f}"
            
            self.add_peak_marker(
                energy, element=element, line=line, color=color, label=label,
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
    
    def _assign_label_positions(self):
        """Stagger peak-label heights by energy so neighbors don't pile up"""
        # Alternate high→low so adjacent peaks are less likely to collide
        positions = (0.96, 0.82, 0.68, 0.88, 0.74, 0.60)
        ordered = sorted(self._peak_marker_specs, key=lambda s: s['energy'])
        for i, spec in enumerate(ordered):
            spec['position'] = positions[i % len(positions)]
    
    def _draw_peak_marker(self, spec):
        """Draw one stored peak marker onto the current plot"""
        plot_item = self.plot_widget.getPlotItem()
        color = spec['color']
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
        Show emission lines for an element
        
        Args:
            symbol: Element symbol
            z: Atomic number
        """
        # Get emission lines
        lines = get_element_lines(symbol, z)
        
        # Define colors for different series
        series_colors = {
            'K': 'r',      # Red for K lines
            'L': 'g',      # Green for L lines
            'M': 'b',      # Blue for M lines
            'N': 'm'       # Magenta for N lines
        }
        
        self.clear_peak_markers()
        
        # Add markers for each line
        for series, color in series_colors.items():
            if lines[series]:
                for line_data in lines[series]:
                    energy = line_data['energy']
                    name = line_data['name']
                    self.add_peak_marker(
                        energy,
                        element=symbol,
                        line=name,
                        color=color,
                        label=f"{symbol}-{name}",
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
        # Preserve markers across clear/redraw
        saved_specs = list(self._peak_marker_specs)
        plot_item.clear()
        self.peak_markers.clear()
        self._peak_marker_specs = saved_specs
        
        # Re-add crosshair and legend after clear
        plot_item.addItem(self.vLine, ignoreBounds=True)
        plot_item.addItem(self.hLine, ignoreBounds=True)
        self._ensure_legend(plot_item)
        
        if self.spectrum_data is None:
            return
        
        # Plot measured spectrum
        plot_item.plot(
            self.spectrum_data.energy,
            self.spectrum_data.counts,
            pen=pg.mkPen('b', width=2),
            name='Measured'
        )
        
        # Plot background if available
        if self.background_data is not None:
            plot_item.plot(
                self.spectrum_data.energy,
                self.background_data,
                pen=pg.mkPen('g', width=1, style=Qt.DashLine),
                name='Background'
            )
        
        # Plot fitted spectrum if available
        if self.fitted_data is not None:
            plot_item.plot(
                self.spectrum_data.energy,
                self.fitted_data,
                pen=pg.mkPen('r', width=2),
                name='Fitted'
            )
            
            # Update residuals
            self._update_residuals()
        
        # Restore peak markers on top of spectra
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
        
        if self.spectrum_data is not None:
            # Find nearest data point
            idx = np.argmin(np.abs(self.spectrum_data.energy - energy))
            if idx < len(self.spectrum_data.counts):
                actual_energy = self.spectrum_data.energy[idx]
                actual_counts = self.spectrum_data.counts[idx]
                self.info_label.setText(
                    f"Energy: {actual_energy:.3f} keV | Counts: {actual_counts:.0f}"
                )
            else:
                self.info_label.setText(
                    f"Energy: {energy:.3f} keV | Counts: {counts:.0f}"
                )
        else:
            self.info_label.setText(
                f"Energy: {energy:.3f} keV | Counts: {counts:.0f}"
            )
    
    def export_plot(self, file_path):
        """
        Export plot to image file
        
        Args:
            file_path: Path to save image (supports PNG, SVG, PDF)
        """
        exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
        exporter.export(file_path)
