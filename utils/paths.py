"""Project and resource path helpers."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository / application root directory."""
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    """Return an absolute path under resources/."""
    return project_root().joinpath("resources", *parts)


def icon_path(name: str = "xrflab.png") -> Path:
    """Return path to an application icon file under resources/icons/."""
    return resource_path("icons", name)
