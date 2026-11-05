"""Fail closed if current kws-data Actions regain commercial publication authority.

Only live workflow entrypoints and retired importer paths are inspected here.
Archived research records, rights statements, source receipts and historic release
provenance must remain available for reproducibility; those are not approvals.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

# Avoid treating prose/comments about old publication as executable authority.
RETIRED = (
    "release.yml",
    "deployment-release.yml",
    "shipping-approval.yml",
    "model-promotion.yml",
    "product-certification.yml",
    "real-human-qualification.yml",
    "target-dut-qualification.yml",
)
PUBLISH_ACTION = re.compile(
    r"(?im)^\s*-?\s*uses:\s*(?:softprops/action-gh-release|"
    r"actions/create-release|actions/upload-release-asset)(?:@|\b)"
)
RELEASE_CLI = re.compile(
    r"(?i)\bgh\s+release\s+(?:create|upload|download|view|edit|delete)\b"
    r"|\bgh\s+api\b[^\n]*/releases(?:/|\b)"
)


def validate_research_workflow(source: str, name: str = "fixture") -> None:
    """Reject known live GitHub Release triggers, readers and publishers."""
    live = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    header = live.split("\njobs:", 1)[0]
    forbidden = (
        (r"(?m)^  release\s*:", "Release event"),
        (r"(?m)^\s+tags\s*:", "tag push"),
        (r"(?m)^\s+workflows\s*:\s*\[?\s*['\"]?Release\b",
         "Release-dependent workflow"),
    )
    for expression, reason in forbidden:
        if re.search(expression, header):
            raise AssertionError(f"{name}: {reason} forbidden in active research")
    if RELEASE_CLI.search(live):
        raise AssertionError(f"{name}: GitHub Release CLI/API cannot be research authority")
    if PUBLISH_ACTION.search(live):
        raise AssertionError(f"{name}: GitHub Release publishing action is forbidden")


class ResearchPublicationBoundaryTests(unittest.TestCase):
    def test_every_active_workflow_is_research_only(self):
        workflows = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
        self.assertTrue(workflows, "active workflow directory must be present")
        for path in workflows:
            with self.subTest(path=path.name):
                validate_research_workflow(path.read_text(encoding="utf-8"), path.name)

    def test_retired_publication_entrypoints_cannot_return(self):
        for name in RETIRED:
            with self.subTest(name=name):
                self.assertFalse((WORKFLOWS / name).exists())
        for path in (
            ROOT / "commercial",
            ROOT / "automation" / "commercial_bulk_ingest.py",
            ROOT / "automation" / "bulk_ingest.py",
        ):
            with self.subTest(path=path.as_posix()):
                self.assertFalse(path.exists())
        self.assertTrue((ROOT / "automation/research_only_bulk_ingest.py").is_file())

    def test_negative_mutations_and_historical_comment(self):
        bad = (
            "on:\n  release:\n    types: [published]\njobs:\n",
            "on:\n  push:\n    tags: [v*]\njobs:\n",
            "on:\n  workflow_run:\n    workflows: [Release]\njobs:\n",
            "on:\n  workflow_dispatch:\njobs:\n  x:\n    steps:\n"
            "      - run: gh release create v1.0.0 build/example.zip\n",
            "on:\n  workflow_dispatch:\njobs:\n  x:\n    steps:\n"
            "      - run: gh release download v1.0.0\n",
            "on:\n  workflow_dispatch:\njobs:\n  x:\n    steps:\n"
            "      - run: gh api repos/owner/repo/releases/tags/v1\n",
            "on:\n  workflow_dispatch:\njobs:\n  x:\n    steps:\n"
            "      - uses: softprops/action-gh-release@v2\n",
        )
        for source in bad:
            with self.subTest(source=source[:60]):
                with self.assertRaises(AssertionError):
                    validate_research_workflow(source)
        validate_research_workflow(
            "name: research\non:\n  pull_request:\njobs:\n  test:\n"
            "    steps:\n      - run: echo research-only\n"
            "      # old gh release create was retired\n",
            "historical-comment-only",
        )


if __name__ == "__main__":
    unittest.main()
