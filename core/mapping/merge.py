"""
Merge multiple INCA/XGT .ipj projects into one MappingProject.

Scope (v1): collected line scans and multipoint series only. Maps, cubes,
and camera images are dropped. All retained sites are flattened under a
single MappingSample.

Spectrum / series labels use:
    filename_sampleName_siteName_spectrumName
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Union

from core.mapping.models import LineScan, MappingFOV, MappingProject, MappingSample, MapSpectrum

PathLike = Union[str, Path]

_DEFAULT_SAMPLE_RE = re.compile(
    r"^(Sample|Sample of Interest)\s*\d*$",
    flags=re.IGNORECASE,
)
_DEFAULT_SITE_RE = re.compile(
    r"^(Site of Interest|Site)\s*\d*$",
    flags=re.IGNORECASE,
)


def sanitize_name_token(value: str) -> str:
    """Make a path/token-safe underscore label."""
    text = (value or "").strip()
    if not text:
        return "unnamed"
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def is_default_sample_name(name: str) -> bool:
    """True for vendor placeholders like 'Sample 1'."""
    return bool(_DEFAULT_SAMPLE_RE.match((name or "").strip()))


def is_default_site_name(name: str) -> bool:
    """True for vendor placeholders like 'Site 1' / 'Site of Interest 2'."""
    return bool(_DEFAULT_SITE_RE.match((name or "").strip()))


def composite_label(
    file_stem: str,
    sample_name: str,
    site_name: str,
    spectrum_name: str,
) -> str:
    """Build ``filename_sampleName_siteName_spectrumName``."""
    return "_".join(
        [
            sanitize_name_token(file_stem),
            sanitize_name_token(sample_name or "Sample"),
            sanitize_name_token(site_name or "Site"),
            sanitize_name_token(spectrum_name),
        ]
    )


def _site_label(file_stem: str, sample_name: str, site_name: str) -> str:
    """Site tree label: ``filename_sampleName_siteName``."""
    return "_".join(
        [
            sanitize_name_token(file_stem),
            sanitize_name_token(sample_name or "Sample"),
            sanitize_name_token(site_name or "Site"),
        ]
    )


def _copy_map_spectrum(ms: MapSpectrum, new_name: str, *, extra_meta: dict) -> MapSpectrum:
    meta = dict(ms.metadata or {})
    meta.update(extra_meta)
    meta.setdefault("original_name", ms.name)
    return MapSpectrum(
        spectrum=ms.spectrum,
        name=new_name,
        x=ms.x,
        y=ms.y,
        index=ms.index,
        kind=ms.kind,
        peak_labels=list(ms.peak_labels or []),
        metadata=meta,
    )


def _copy_line_scan(
    ls: LineScan,
    *,
    file_stem: str,
    sample_name: str,
    site_name: str,
    source_path: str,
) -> LineScan:
    extra = {
        "source_ipj": source_path,
        "source_file": file_stem,
        "source_sample": sample_name,
        "source_site": site_name,
    }
    points = [
        _copy_map_spectrum(
            pt,
            composite_label(file_stem, sample_name, site_name, pt.name),
            extra_meta=extra,
        )
        for pt in ls.points
    ]
    series_name = composite_label(file_stem, sample_name, site_name, ls.name)
    meta = dict(ls.metadata or {})
    meta.update(extra)
    meta.setdefault("original_name", ls.name)
    return LineScan(
        name=series_name,
        points=points,
        start_xy=ls.start_xy,
        end_xy=ls.end_xy,
        source=ls.source,
        kind=ls.kind,
        metadata=meta,
    )


def merge_line_scan_projects(
    projects: Sequence[MappingProject],
    *,
    name: str = "",
    path: str = "",
    sample_name: str = "Merged",
) -> MappingProject:
    """
    Flatten line/multipoint sites from several MappingProjects into one sample.

    Sites without collected line scans / multipoint series are skipped.
    """
    if not projects:
        raise ValueError("No projects to merge")

    sites: List[MappingFOV] = []
    sources: List[str] = []
    skipped_sites = 0
    site_counter = 0

    for proj in projects:
        source_path = str(proj.path or "")
        file_stem = Path(proj.path).stem if proj.path else (proj.name or "project")
        sources.append(source_path or file_stem)

        for sample in proj.samples:
            for site in sample.sites:
                series = [ls for ls in site.line_scans if ls.points]
                if not series:
                    skipped_sites += 1
                    continue
                site_counter += 1
                new_scans = [
                    _copy_line_scan(
                        ls,
                        file_stem=file_stem,
                        sample_name=sample.name,
                        site_name=site.name,
                        source_path=source_path,
                    )
                    for ls in series
                ]
                # Keep shared references: spectra list mirrors line-scan points
                spectra: List[MapSpectrum] = []
                for ls in new_scans:
                    spectra.extend(ls.points)
                unique_id = (
                    f"merged_{site_counter:04d}_"
                    f"{sanitize_name_token(file_stem)}_"
                    f"{sanitize_name_token(site.id)}"
                )
                sites.append(
                    MappingFOV(
                        id=unique_id,
                        name=_site_label(file_stem, sample.name, site.name),
                        spectra=spectra,
                        line_scans=new_scans,
                        metadata={
                            "source_ipj": source_path,
                            "source_file": file_stem,
                            "source_sample": sample.name,
                            "source_sample_id": sample.id,
                            "source_site": site.name,
                            "source_site_id": site.id,
                            "merged": True,
                            "line_scan_only": True,
                        },
                    )
                )

    if not sites:
        raise ValueError(
            "No line scans or multipoint series found in the selected projects"
        )

    merged_sample = MappingSample(
        id="merged",
        name=sample_name or "Merged",
        sites=sites,
        metadata={
            "merged": True,
            "n_source_files": len(projects),
            "sources": sources,
        },
    )
    project_name = name.strip() if name else "Merged line scans"
    out_path = path or (str(Path(projects[0].path).with_name(project_name)) if projects[0].path else project_name)
    n_series = sum(len(s.line_scans) for s in sites)
    n_points = sum(ls.n_points for s in sites for ls in s.line_scans)
    return MappingProject(
        path=out_path,
        name=project_name,
        samples=[merged_sample],
        fovs=list(sites),
        metadata={
            "format": "merged_ipj_line_scans",
            "merged_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": 1,
            "n_fovs": len(sites),
            "n_cubes": 0,
            "n_source_files": len(projects),
            "sources": sources,
            "n_line_scans": n_series,
            "n_points": n_points,
            "skipped_sites_without_series": skipped_sites,
            "project_title": project_name,
        },
    )


def merge_ipj_line_scans(
    paths: Iterable[PathLike],
    *,
    name: str = "",
    sample_name: str = "Merged",
) -> MappingProject:
    """Load many .ipj files and merge their line/multipoint series."""
    from utils.ipj_loader import load_ipj

    path_list = [Path(p) for p in paths]
    if not path_list:
        raise ValueError("No .ipj files selected")
    projects = [load_ipj(p) for p in path_list]
    project_name = name.strip() if name else f"{path_list[0].stem}_merged"
    return merge_line_scan_projects(
        projects,
        name=project_name,
        path=str(path_list[0].with_name(project_name)),
        sample_name=sample_name,
    )
