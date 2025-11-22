"""
Spectrum display widget using PyQtGraph for high-performance plotting
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from core.xray_data import get_element_lines


class SpectrumWidget(QWidget):
    """Widget for displaying XRF spectra with interactive features"""
    
    energy_selected = Signal(float)  # Emitted when user clicks on spectrum
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.spectrum_data = None
        self.fitted_data = None
        self.background_data = None
        self.peak_markers = []
        
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
        
        # Info label for cursor position
        self.info_label = QLabel("Energy: -- keV | Counts: --")
        self.info_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(self.info_label)
    
    def _configure_plot(self):
        """Configure plot appearance and behavior"""
        # Main plot configuration
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setLabel('left', 'Counts', units='')
        plot_item.setLabel('bottom', 'Energy', units='keV')
        plot_item.showGrid(x=True, y=True, alpha=0.3)
        plot_item.setLogMode(False, True)  # Log Y-axis by default
        plot_item.addLegend()
        
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
    
    def add_peak_marker(self, energy, element, line):
        """
        Add a peak marker at specified energy
        
        Args:
            energy: Peak energy in keV
            element: Element symbol
            line: Line designation (e.g., 'Ka', 'Kb', 'La')
        """
        plot_item = self.plot_widget.getPlotItem()
        
        # Create vertical line for peak
        line_item = pg.InfiniteLine(
            pos=energy,
            angle=90,
            pen=pg.mkPen('r', width=1, style=Qt.DashLine),
            label=f"{element}-{line}",
            labelOpts={'position': 0.95, 'color': 'r'}
        )
        plot_item.addItem(line_item)
        self.peak_markers.append(line_item)
    
    def clear_peak_markers(self):
        """Remove all peak markers"""
        plot_item = self.plot_widget.getPlotItem()
        for marker in self.peak_markers:
            plot_item.removeItem(marker)
        self.peak_markers.clear()
    
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
        
        plot_item = self.plot_widget.getPlotItem()
        
        # Add markers for each line
        for series, color in series_colors.items():
            if lines[series]:
                for line_data in lines[series]:
                    energy = line_data['energy']
                    name = line_data['name']
                    
                    # Create vertical line
                    line_item = pg.InfiniteLine(
                        pos=energy,
                        angle=90,
                        pen=pg.mkPen(color, width=1.5, style=Qt.DashLine),
                        label=f"{symbol}-{name}",
                        labelOpts={'position': 0.9, 'color': color, 'angle': 90}
                    )
                    plot_item.addItem(line_item)
                    self.peak_markers.append(line_item)
    
    def set_log_scale(self, enabled):
        """Enable or disable logarithmic Y-axis"""
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setLogMode(False, enabled)
    
    def set_grid(self, enabled):
        """Enable or disable grid"""
        plot_item = self.plot_widget.getPlotItem()
        plot_item.showGrid(x=enabled, y=enabled, alpha=0.3)
    
    def _update_plot(self):
        """Update the plot with current data"""
        plot_item = self.plot_widget.getPlotItem()
        plot_item.clear()
        
        # Re-add crosshair after clear
        plot_item.addItem(self.vLine, ignoreBounds=True)
        plot_item.addItem(self.hLine, ignoreBounds=True)
        
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
