"""Calculate and apply a release version from Conventional Commit history."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"
HEADER = re.compile(
    r"^\[(?P<scope>[^]]+)\] "
    r"(?P<type>feat|fix|perf|refactor|docs|test|build|ci|chore)"
    r"(?P<breaking>!)?: (?P<description>\S.*)$"
)


def commits_since_tag() -> list[tuple[str, str, str, bool]]:
    """Return hash, type, description, and breaking flag since the last tag."""
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"], cwd=ROOT, text=True
        ).strip()
        revision = f"{tag}..HEAD"
    except subprocess.CalledProcessError:
        revision = "HEAD"

    raw = subprocess.check_output(
        ["git", "log", revision, "--format=%H%x00%B%x00"], cwd=ROOT
    ).decode()
    result: list[tuple[str, str, str, bool]] = []
    records = raw.split("\0")
    for index in range(0, len(records) - 1, 2):
        commit_hash, body = records[index], records[index + 1]
        lines = body.strip().splitlines()
        if not lines:
            continue
        match = HEADER.match(lines[0])
        if match is None:
            continue
        breaking = bool(match.group("breaking")) or any(
            line.startswith("BREAKING CHANGE:") for line in lines[1:]
        )
        result.append(
            (commit_hash, match.group("type"), match.group("description"), breaking)
        )
    return result


def next_version(current: str, commits: list[tuple[str, str, str, bool]]) -> str:
    """Apply the project's SemVer bump rules to commit history."""
    major, minor, patch = (int(part) for part in current.split("."))
    if any(commit[3] for commit in commits):
        return f"{major + 1}.0.0"
    if any(commit[1] == "feat" for commit in commits):
        return f"{major}.{minor + 1}.0"
    if any(commit[1] in {"fix", "perf"} for commit in commits):
        return f"{major}.{minor}.{patch + 1}"
    return current


def main() -> None:
    """Preview or apply the next release version and changelog section."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    text = PYPROJECT.read_text()
    current_match = re.search(
        r'^version = "([0-9]+\.[0-9]+\.[0-9]+)"$', text, re.MULTILINE
    )
    if current_match is None:
        raise SystemExit("Could not find a SemVer version in pyproject.toml")

    commits = commits_since_tag()
    version = next_version(current_match.group(1), commits)
    if version == current_match.group(1):
        raise SystemExit("No feat, fix, perf, or breaking commit requires a bump.")
    if args.dry_run:
        print(version)
        return

    PYPROJECT.write_text(
        text.replace(current_match.group(0), f'version = "{version}"', 1)
    )
    grouped = {"feat": [], "fix": [], "perf": [], "other": []}
    for _, commit_type, description, _ in commits:
        grouped.get(commit_type, grouped["other"]).append(f"- {description}")
    section = [f"## {version}", ""]
    for name, entries in grouped.items():
        if entries:
            section.extend([f"### {name}", "", *entries, ""])
    existing = CHANGELOG.read_text() if CHANGELOG.exists() else "# Changelog\n\n"
    CHANGELOG.write_text(existing.rstrip() + "\n\n" + "\n".join(section))
    print(f"Bumped {current_match.group(1)} -> {version}")


if __name__ == "__main__":
    main()
