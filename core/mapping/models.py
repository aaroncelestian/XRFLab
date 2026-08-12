"""Data models for hyperspectral / element-map XRF projects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.spectrum import Spectrum


@dataclass
class ElementMap:
    """2D intensity map for one X-ray line (e.g. Fe Ka1)."""

    name: str
    data: np.ndarray
    line: str = ""
    element: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.data = np.asarray(self.data)
        if self.data.ndim != 2:
            raise ValueError("ElementMap.data must be 2D")
        if not self.line:
            self.line = self.name
        if not self.element:
            self.element = _element_from_line_name(self.name)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape  # (height, width)

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])


@dataclass
class OverviewImage:
    """Optical photo, transmission X-ray, or other 2D/RGB overview."""

    name: str
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.data = np.asarray(self.data)
        if self.data.ndim not in (2, 3):
            raise ValueError("OverviewImage.data must be 2D or 3D")

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape


@dataclass
class MapSpectrum:
    """A Spectrum with optional spatial / line-scan context."""

    spectrum: Spectrum
    name: str = ""
    x: Optional[float] = None
    y: Optional[float] = None
    index: Optional[int] = None
    kind: str = "spot"  # spot | sum | line_point | roi
    peak_labels: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.spectrum.metadata.get("name", "Spectrum")


@dataclass
class LineScan:
    """Ordered spectra along a transect (line scan) or multipoint series."""

    name: str
    points: List[MapSpectrum] = field(default_factory=list)
    # Pixel endpoints on the map canvas (x0, y0, x1, y1) when UI-drawn
    start_xy: Optional[Tuple[float, float]] = None
    end_xy: Optional[Tuple[float, float]] = None
    source: str = "ipj"  # ipj | drawn
    # line_scan = equal step spacing; multipoint = irregular gaps
    kind: str = "line_scan"  # line_scan | multipoint
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def is_line_scan(self) -> bool:
        return self.kind == "line_scan"

    @property
    def is_multipoint(self) -> bool:
        return self.kind == "multipoint"

    def display_label(self) -> str:
        """Short UI label: 'Line scan' or 'Multipoint'."""
        if self.is_multipoint:
            return "Multipoint"
        return "Line scan"

    def distances(self) -> np.ndarray:
        """Path distance along points, or unit index spacing as fallback."""
        n = self.n_points
        if n == 0:
            return np.array([])
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        if (
            all(v is not None for v in xs)
            and all(v is not None for v in ys)
            and n > 1
        ):
            coords = np.column_stack(
                [np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)]
            )
            steps = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
            out = np.zeros(n, dtype=np.float64)
            out[1:] = np.cumsum(steps)
            return out
        if self.start_xy is not None and self.end_xy is not None and n > 1:
            length = float(
                np.hypot(
                    self.end_xy[0] - self.start_xy[0],
                    self.end_xy[1] - self.start_xy[1],
                )
            )
            return np.linspace(0.0, length, n)
        return np.arange(n, dtype=np.float64)


@dataclass
class MappingFOV:
    """One field of view: maps, overview, spectra, optional line scan."""

    id: str
    name: str = ""
    width: int = 0
    height: int = 0
    element_maps: List[ElementMap] = field(default_factory=list)
    overview: Optional[OverviewImage] = None  # SEI transmission X-ray
    optical: Optional[OverviewImage] = None  # XGT map-area camera BMP
    spectra: List[MapSpectrum] = field(default_factory=list)
    line_scans: List[LineScan] = field(default_factory=list)
    cube: Optional[Any] = None  # SpectrumCube when SmartMap ListData decoded
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.id

    @property
    def map_shape(self) -> Optional[Tuple[int, int]]:
        if self.height and self.width:
            return (self.height, self.width)
        if self.cube is not None:
            return (self.cube.height, self.cube.width)
        if self.element_maps:
            return self.element_maps[0].shape
        if self.overview is not None and self.overview.data.ndim >= 2:
            return self.overview.data.shape[:2]
        return None

    @property
    def has_cube(self) -> bool:
        return self.cube is not None

    def find_map(self, name_or_element: str) -> Optional[ElementMap]:
        key = name_or_element.lower()
        for m in self.element_maps:
            if m.name.lower() == key or m.element.lower() == key or m.line.lower() == key:
                return m
        return None

    def sum_spectrum(self) -> Optional[MapSpectrum]:
        for s in self.spectra:
            if s.kind == "sum" or "sum" in s.name.lower():
                return s
        return None

    def add_roi_map_from_cube(
        self,
        e0_kev: float,
        e1_kev: float,
        name: Optional[str] = None,
    ) -> Optional[ElementMap]:
        """Create an ElementMap by summing cube channels in [e0, e1] keV."""
        if self.cube is None:
            return None
        data = self.cube.roi_map_energy(e0_kev, e1_kev)
        label = name or f"ROI {e0_kev:.2f}–{e1_kev:.2f} keV"
        em = ElementMap(
            name=label,
            data=data,
            line=label,
            element="",
            metadata={
                "source": "cube_roi",
                "e0_kev": float(e0_kev),
                "e1_kev": float(e1_kev),
            },
        )
        # Replace existing map with same name
        self.element_maps = [m for m in self.element_maps if m.name != label]
        self.element_maps.append(em)
        self.element_maps.sort(key=lambda m: m.name)
        return em

    def spectrum_at_pixel(self, x: int, y: int) -> Optional[MapSpectrum]:
        """Extract a MapSpectrum from the hyperspectral cube at pixel (x, y)."""
        if self.cube is None:
            return None

        counts = self.cube.spectrum_at(x, y)
        energy = self.cube.energy_axis_kev()
        # Align energy scale with FOV sum spectrum when available
        sum_ms = self.sum_spectrum()
        if sum_ms is not None and sum_ms.spectrum.num_channels == counts.size:
            energy = sum_ms.spectrum.energy.copy()
        elif sum_ms is not None and sum_ms.spectrum.num_channels == counts.size * 2:
            # Legacy: fine-binned sum vs coarse cube
            fine = np.zeros(counts.size * 2, dtype=np.float64)
            fine[0::2] = counts * 0.5
            fine[1::2] = counts * 0.5
            counts = fine
            energy = sum_ms.spectrum.energy.copy()

        name = f"Pixel ({int(x)}, {int(y)})"
        sp = Spectrum(
            energy=energy,
            counts=counts,
            live_time=float(sum_ms.spectrum.live_time) if sum_ms else 100.0,
            real_time=float(sum_ms.spectrum.real_time) if sum_ms else 100.0,
            metadata={
                "name": name,
                "source": "cube_pixel",
                "x": int(x),
                "y": int(y),
            },
        )
        return MapSpectrum(
            spectrum=sp,
            name=name,
            x=float(x),
            y=float(y),
            kind="roi",
            peak_labels=list(sum_ms.peak_labels) if sum_ms else [],
            metadata={"source": "cube_pixel"},
        )


@dataclass
class MappingSample:
    """One sample containing one or more Sites of Interest (FOVs)."""

    id: str
    name: str = ""
    sites: List[MappingFOV] = field(default_factory=list)
    whole_image: Optional[OverviewImage] = None  # XGT sample-camera BMP
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.id

    @property
    def fovs(self) -> List[MappingFOV]:
        """Alias for sites (FOV = Site of Interest in INCA)."""
        return self.sites

    def find_site(self, site_id: str) -> Optional[MappingFOV]:
        for site in self.sites:
            if site.id == site_id:
                return site
        return None


@dataclass
class MappingProject:
    """Loaded mapping document (typically from one .ipj file)."""

    path: str
    name: str = ""
    samples: List[MappingSample] = field(default_factory=list)
    fovs: List[MappingFOV] = field(default_factory=list)  # flattened; kept for compat
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            from pathlib import Path

            self.name = Path(self.path).stem
        # Keep fovs in sync with samples when only samples were provided
        if self.samples and not self.fovs:
            self.fovs = [site for sample in self.samples for site in sample.sites]
        elif self.fovs and not self.samples:
            # Legacy flat list → single synthetic sample
            self.samples = [
                MappingSample(id="sample", name="Sample 1", sites=list(self.fovs))
            ]

    @property
    def primary_fov(self) -> Optional[MappingFOV]:
        """Prefer site that has maps or a cube; else first site."""
        for site in self.fovs:
            if site.element_maps or site.overview is not None or site.cube is not None:
                return site
        return self.fovs[0] if self.fovs else None

    def find_sample(self, sample_id: str) -> Optional[MappingSample]:
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        return None

    def find_site(self, site_id: str) -> Optional[MappingFOV]:
        for site in self.fovs:
            if site.id == site_id:
                return site
        return None

    def all_spectra(self) -> List[MapSpectrum]:
        out: List[MapSpectrum] = []
        for fov in self.fovs:
            out.extend(fov.spectra)
            for ls in fov.line_scans:
                out.extend(ls.points)
        return out


def _element_from_line_name(name: str) -> str:
    """Extract element symbol from labels like 'Fe Ka1' or 'Na Ka1_2'."""
    token = (name or "").strip().split()[0] if name else ""
    if not token:
        return ""
    # First capital + optional lowercase (Fe, Si, Na)
    if len(token) >= 2 and token[1].islower():
        return token[:2]
    return token[:1] if token else ""
