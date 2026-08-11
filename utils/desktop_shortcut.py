"""
Create desktop shortcuts for launching XRFLab (macOS / Windows / Linux).
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from utils.paths import icon_path, project_root


APP_NAME = "XRFLab"


@dataclass
class ShortcutResult:
    success: bool
    path: Path | None
    message: str


def desktop_dir() -> Path:
    """Best-effort Desktop directory across platforms."""
    home = Path.home()
    candidates = [
        home / "Desktop",
        home / "desktop",
        Path(os.path.expandvars(r"%USERPROFILE%\Desktop")),
    ]
    # Linux XDG
    xdg = os.environ.get("XDG_DESKTOP_DIR")
    if xdg:
        candidates.insert(0, Path(xdg))
    for path in candidates:
        if path.is_dir():
            return path
    return home


def install_desktop_shortcut() -> ShortcutResult:
    """Install a desktop shortcut that launches this XRFLab install."""
    system = platform.system()
    if system == "Darwin":
        return _install_macos_app()
    if system == "Windows":
        return _install_windows_shortcut()
    if system == "Linux":
        return _install_linux_desktop()
    return ShortcutResult(
        False,
        None,
        f"Desktop shortcuts are not supported on {system}.",
    )


def _launch_paths() -> tuple[Path, Path, Path]:
    root = project_root()
    main_py = root / "main.py"
    python = Path(sys.executable).resolve()
    return root, main_py, python


def _pyside6_plugin_path(python: Path) -> Path | None:
    """Locate PySide6 Qt plugins for the interpreter that will run XRFLab."""
    # Prefer the live install (shortcut is created from a running XRFLab).
    if Path(sys.executable).resolve() == python.resolve():
        try:
            import PySide6

            plugins = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
            if plugins.is_dir():
                return plugins
        except Exception:
            pass
    try:
        out = subprocess.check_output(
            [
                str(python),
                "-c",
                "import PySide6; from pathlib import Path; "
                "print(Path(PySide6.__file__).resolve().parent / 'Qt' / 'plugins')",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        plugins = Path(out)
        return plugins if plugins.is_dir() else None
    except (OSError, subprocess.CalledProcessError):
        return None


def _install_macos_app() -> ShortcutResult:
    root, main_py, python = _launch_paths()
    dest = desktop_dir() / f"{APP_NAME}.app"
    contents = dest / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"

    if dest.exists():
        shutil.rmtree(dest)

    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    icns = icon_path("xrflab.icns")
    if icns.is_file():
        shutil.copy2(icns, resources / "xrflab.icns")

    # Finder launches .app with a stripped env; Qt needs an explicit plugin path.
    plugin_path = _pyside6_plugin_path(python)
    plugin_export = ""
    if plugin_path is not None:
        plugin_export = f'export QT_PLUGIN_PATH="{plugin_path}"\n'

    # Keep conda/python on PATH so nested tools resolve consistently.
    python_bin = python.parent
    launcher = macos / APP_NAME
    launcher.write_text(
        "#!/bin/bash\n"
        f'cd "{root}"\n'
        f'export PATH="{python_bin}:$PATH"\n'
        f"{plugin_export}"
        f'exec "{python}" "{main_py}" "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>{APP_NAME}</string>
    <key>CFBundleIconFile</key>
    <string>xrflab</string>
    <key>CFBundleIdentifier</key>
    <string>com.xrflab.app</string>
    <key>CFBundleName</key>
    <string>{APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>{APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
"""
    (contents / "Info.plist").write_text(plist, encoding="utf-8")

    # Refresh Finder icon cache for the new .app
    try:
        subprocess.run(
            ["touch", str(dest)],
            check=False,
            capture_output=True,
        )
    except OSError:
        pass

    return ShortcutResult(
        True,
        dest,
        f"Desktop app created:\n{dest}\n\nDouble-click to launch XRFLab.",
    )


def _install_windows_shortcut() -> ShortcutResult:
    root, main_py, python = _launch_paths()
    dest = desktop_dir() / f"{APP_NAME}.lnk"
    ico = icon_path("xrflab.ico")
    icon_location = str(ico) if ico.is_file() else str(python)

    # PowerShell CreateShortcut — no extra Python packages required
    ps = f"""
$ErrorActionPreference = 'Stop'
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{_ps_escape(dest)}')
$Shortcut.TargetPath = '{_ps_escape(python)}'
$Shortcut.Arguments = '"{_ps_escape(main_py)}"'
$Shortcut.WorkingDirectory = '{_ps_escape(root)}'
$Shortcut.IconLocation = '{_ps_escape(icon_location)}'
$Shortcut.Description = 'XRFLab - Fundamental Parameters Analysis'
$Shortcut.Save()
"""
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ShortcutResult(
            False,
            None,
            "PowerShell was not found; cannot create a Windows shortcut.",
        )

    if completed.returncode != 0 or not dest.is_file():
        err = (completed.stderr or completed.stdout or "Unknown error").strip()
        return ShortcutResult(False, None, f"Failed to create shortcut:\n{err}")

    return ShortcutResult(
        True,
        dest,
        f"Desktop shortcut created:\n{dest}\n\nDouble-click to launch XRFLab.",
    )


def _install_linux_desktop() -> ShortcutResult:
    root, main_py, python = _launch_paths()
    dest = desktop_dir() / f"{APP_NAME}.desktop"
    png = icon_path("xrflab.png")
    icon = str(png) if png.is_file() else "utilities-terminal"

    content = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        "Comment=XRF Fundamental Parameters Analysis\n"
        f'Exec="{python}" "{main_py}"\n'
        f"Path={root}\n"
        f"Icon={icon}\n"
        "Terminal=false\n"
        "Categories=Science;Education;\n"
    )
    dest.write_text(content, encoding="utf-8")
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return ShortcutResult(
        True,
        dest,
        f"Desktop launcher created:\n{dest}\n\n"
        "If your desktop blocks it, right-click → Allow Launching.",
    )


def _ps_escape(path: Path | str) -> str:
    """Escape a path for embedding inside single-quoted PowerShell strings."""
    return str(path).replace("'", "''")
