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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

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


def _is_sum_spectrum(ms: MapSpectrum) -> bool:
    if ms.kind == "sum":
        return True
    return "sum" in (ms.name or "").lower()


def _point_spectra(site: MappingFOV) -> List[MapSpectrum]:
    """Non-sum spectra on a site (spots / line points), stable order."""
    points = [s for s in site.spectra if not _is_sum_spectrum(s)]
    return sorted(
        points,
        key=lambda s: (
            0 if s.kind == "line_point" else 1,
            s.index is None,
            s.index if s.index is not None else 10**9,
            s.name or "",
        ),
    )


def series_for_site(site: MappingFOV) -> List[LineScan]:
    """
    Line/multipoint series to merge from a site.

    Prefer vendor-attached ``line_scans``. If none (common when a site has
    fewer than 3 ``Spectrum N`` points, or points are stored as spots),
    synthesize a multipoint series from all non-sum spectra.
    """
    existing = [ls for ls in site.line_scans if ls.points]
    if existing:
        return existing

    points = _point_spectra(site)
    if not points:
        return []

    try:
        from utils.ipj_loader import classify_point_series_kind

        kind = classify_point_series_kind(points)
    except Exception:
        kind = "multipoint"
    n = len(points)
    if kind == "line_scan":
        name = f"Line scan ({n} points)"
    else:
        name = f"Multipoint ({n} points)"
        kind = "multipoint"
    return [
        LineScan(
            name=name,
            points=points,
            source="ipj",
            kind=kind,
            metadata={
                "fov_id": site.id,
                "synthesized": True,
            },
        )
    ]


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
        kind=ms.kind if ms.kind != "spot" else "line_point",
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


@dataclass
class MergeReport:
    """Result of a multi-IPJ merge, including per-file outcomes."""

    project: MappingProject
    included_files: List[str] = field(default_factory=list)
    skipped_files: List[Tuple[str, str]] = field(default_factory=list)  # path, reason
    load_errors: List[Tuple[str, str]] = field(default_factory=list)

    def summary_text(self) -> str:
        n_sites = len(self.project.fovs)
        n_pts = int(self.project.metadata.get("n_points") or 0)
        lines = [
            f"Merged {n_sites} site(s), {n_pts} spectrum point(s) "
            f"from {len(self.included_files)} file(s)."
        ]
        if self.skipped_files:
            lines.append("")
            lines.append(f"Skipped {len(self.skipped_files)} file(s) with no point spectra:")
            for path, reason in self.skipped_files[:20]:
                lines.append(f"  • {Path(path).name}: {reason}")
            if len(self.skipped_files) > 20:
                lines.append(f"  … and {len(self.skipped_files) - 20} more")
        if self.load_errors:
            lines.append("")
            lines.append(f"Failed to load {len(self.load_errors)} file(s):")
            for path, err in self.load_errors[:20]:
                lines.append(f"  • {Path(path).name}: {err}")
        return "\n".join(lines)


def merge_line_scan_projects(
    projects: Sequence[MappingProject],
    *,
    name: str = "",
    path: str = "",
    sample_name: str = "Merged",
) -> MappingProject:
    """
    Flatten line/multipoint sites from several MappingProjects into one sample.

    Sites without point spectra (map-only sum spectra) are skipped.
    """
    report = merge_line_scan_projects_with_report(
        projects, name=name, path=path, sample_name=sample_name
    )
    return report.project


def merge_line_scan_projects_with_report(
    projects: Sequence[MappingProject],
    *,
    name: str = "",
    path: str = "",
    sample_name: str = "Merged",
) -> MergeReport:
    if not projects:
        raise ValueError("No projects to merge")

    sites: List[MappingFOV] = []
    sources: List[str] = []
    included_files: List[str] = []
    skipped_files: List[Tuple[str, str]] = []
    skipped_sites = 0
    site_counter = 0

    for proj in projects:
        source_path = str(proj.path or "")
        file_stem = Path(proj.path).stem if proj.path else (proj.name or "project")
        sources.append(source_path or file_stem)
        file_sites_before = site_counter

        for sample in proj.samples:
            for site in sample.sites:
                series = series_for_site(site)
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

        if site_counter > file_sites_before:
            included_files.append(source_path or file_stem)
        else:
            skipped_files.append(
                (
                    source_path or file_stem,
                    "no multipoint / line-scan / spot spectra (map-only?)",
                )
            )

    if not sites:
        raise ValueError(
            "No line scans, multipoint series, or point spectra found in the "
            "selected projects"
        )

    merged_sample = MappingSample(
        id="merged",
        name=sample_name or "Merged",
        sites=sites,
        metadata={
            "merged": True,
            "n_source_files": len(projects),
            "n_included_files": len(included_files),
            "sources": sources,
            "included_files": included_files,
            "skipped_files": [p for p, _ in skipped_files],
        },
    )
    project_name = name.strip() if name else "Merged line scans"
    out_path = path or (
        str(Path(projects[0].path).with_name(project_name))
        if projects[0].path
        else project_name
    )
    n_series = sum(len(s.line_scans) for s in sites)
    n_points = sum(ls.n_points for s in sites for ls in s.line_scans)
    project = MappingProject(
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
            "n_included_files": len(included_files),
            "sources": sources,
            "included_files": included_files,
            "skipped_files": [p for p, _ in skipped_files],
            "n_line_scans": n_series,
            "n_points": n_points,
            "skipped_sites_without_series": skipped_sites,
            "project_title": project_name,
        },
    )
    return MergeReport(
        project=project,
        included_files=included_files,
        skipped_files=skipped_files,
    )


def merge_ipj_line_scans(
    paths: Iterable[PathLike],
    *,
    name: str = "",
    sample_name: str = "Merged",
) -> MappingProject:
    """Load many .ipj files and merge their line/multipoint series."""
    return merge_ipj_line_scans_with_report(
        paths, name=name, sample_name=sample_name
    ).project


def merge_ipj_line_scans_with_report(
    paths: Iterable[PathLike],
    *,
    name: str = "",
    sample_name: str = "Merged",
) -> MergeReport:
    """Load many .ipj files and merge; return project plus skip/error report."""
    from utils.ipj_loader import load_ipj

    path_list = [Path(p) for p in paths]
    if not path_list:
        raise ValueError("No .ipj files selected")

    projects: List[MappingProject] = []
    load_errors: List[Tuple[str, str]] = []
    for p in path_list:
        try:
            projects.append(load_ipj(p))
        except Exception as exc:
            load_errors.append((str(p), str(exc)))

    if not projects:
        detail = "; ".join(f"{Path(p).name}: {e}" for p, e in load_errors[:5])
        raise ValueError(f"Could not load any .ipj files. {detail}")

    project_name = name.strip() if name else f"{path_list[0].stem}_merged"
    report = merge_line_scan_projects_with_report(
        projects,
        name=project_name,
        path=str(path_list[0].with_name(project_name)),
        sample_name=sample_name,
    )
    report.load_errors = load_errors
    return report
