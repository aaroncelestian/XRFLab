"""
Check for application updates via git pull from the project repository.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class UpdateResult:
    """Outcome of an update check / git pull."""

    success: bool
    updated: bool
    message: str
    changed_files: List[str] = field(default_factory=list)
    commits: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _find_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk upward from start (or this file) looking for a .git directory."""
    path = (start or Path(__file__).resolve()).parent
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _run_git(repo_root: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a git command in the repository root."""
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def check_for_updates(repo_root: Optional[Path] = None) -> UpdateResult:
    """
    Fetch and fast-forward pull the latest changes from the remote.

    Returns an UpdateResult describing whether the working tree was updated,
    which files/commits changed, or any error that occurred.
    """
    root = repo_root or _find_repo_root()
    if root is None:
        return UpdateResult(
            success=False,
            updated=False,
            message="Could not find the XRFLab git repository.",
            error="No .git directory found near the application install path.",
        )

    try:
        # Confirm git is available and this is a valid repo
        rev = _run_git(root, "rev-parse", "HEAD")
        if rev.returncode != 0:
            return UpdateResult(
                success=False,
                updated=False,
                message="This install does not appear to be a valid git repository.",
                error=(rev.stderr or rev.stdout).strip() or "git rev-parse failed",
            )

        before = rev.stdout.strip()

        # Prefer a fast-forward pull so we never create unexpected merge commits
        pull = _run_git(root, "pull", "--ff-only", timeout=120)
        if pull.returncode != 0:
            detail = (pull.stderr or pull.stdout).strip() or "git pull failed"
            # Common case: local edits conflict with incoming changes
            hint = ""
            if "local changes" in detail.lower() or "would be overwritten" in detail.lower():
                hint = (
                    "\n\nYou have local file changes that conflict with the update. "
                    "Commit, stash, or discard them, then try again."
                )
            elif "Not possible to fast-forward" in detail or "diverged" in detail.lower():
                hint = (
                    "\n\nYour local branch has diverged from the remote. "
                    "Update manually with git, or reset to match origin/main."
                )
            return UpdateResult(
                success=False,
                updated=False,
                message=f"Update failed.{hint}",
                error=detail,
            )

        after_proc = _run_git(root, "rev-parse", "HEAD")
        after = after_proc.stdout.strip() if after_proc.returncode == 0 else before

        if before == after:
            return UpdateResult(
                success=True,
                updated=False,
                message="You're up to date. No new updates were found.",
            )

        # Summarize what changed
        commits: List[str] = []
        log = _run_git(root, "log", "--oneline", f"{before}..{after}")
        if log.returncode == 0 and log.stdout.strip():
            commits = [line.strip() for line in log.stdout.strip().splitlines() if line.strip()]

        changed_files: List[str] = []
        diff = _run_git(root, "diff", "--name-only", f"{before}..{after}")
        if diff.returncode == 0 and diff.stdout.strip():
            changed_files = [line.strip() for line in diff.stdout.strip().splitlines() if line.strip()]

        n_files = len(changed_files)
        n_commits = len(commits)
        file_word = "file" if n_files == 1 else "files"
        commit_word = "commit" if n_commits == 1 else "commits"

        message = (
            f"Update complete. Pulled {n_commits} {commit_word} "
            f"and updated {n_files} {file_word}.\n\n"
            "Restart XRFLab for the changes to take effect."
        )

        return UpdateResult(
            success=True,
            updated=True,
            message=message,
            changed_files=changed_files,
            commits=commits,
        )

    except FileNotFoundError:
        return UpdateResult(
            success=False,
            updated=False,
            message="Git is not installed or not available on your PATH.",
            error="git executable not found",
        )
    except subprocess.TimeoutExpired:
        return UpdateResult(
            success=False,
            updated=False,
            message="Update timed out. Check your network connection and try again.",
            error="git pull timed out",
        )
    except Exception as exc:
        return UpdateResult(
            success=False,
            updated=False,
            message="An unexpected error occurred while checking for updates.",
            error=str(exc),
        )
