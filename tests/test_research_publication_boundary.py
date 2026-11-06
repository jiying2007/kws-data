"""Fail closed if current kws-data Actions regain commercial publication authority.

Live workflow/importer entrypoints and current guidance are inspected here.
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



CURRENT_GUIDANCE = {
    "README.md": (
        ("具体版本许可", "实际非商业研究用途", "声音克隆/使用权限",
         "不要求与本研究无关的商用授权", "未知或不允许的实际用途不得自动放行",
         "## 当前研究资产规范", "均以技术研究为范围",
         "不把产品放行通道作为数据准备、训练或算法比较的前置条件",
         "原始音频、来源/标签、切分和实际测量", "缺失证据不能由 catalog 或 CI 成功补充",
         "它不是当前执行规范，其中原产品资格通道已经退役"),
        ("须先核实实际版本的商用和声音使用范围",
         "详见[资产架构与固定消费](docs/ASSET_ARCHITECTURE.md)"),
    ),
}


def validate_current_guidance(source: str, name: str) -> None:
    required, retired = CURRENT_GUIDANCE[name]
    for marker in required:
        if marker not in source:
            raise AssertionError(name + ": missing research/source-rights boundary: " + marker)
    for marker in retired:
        if marker in source:
            raise AssertionError(name + ": retired commercial prerequisite: " + marker)


class ResearchPublicationBoundaryTests(unittest.TestCase):
    def test_current_guidance_requires_research_rights_not_commercial_approval(self):
        for name in CURRENT_GUIDANCE:
            with self.subTest(path=name):
                validate_current_guidance((ROOT / name).read_text(encoding="utf-8"), name)

    def test_current_guidance_rejects_missing_rights_and_restored_product_gates(self):
        for name, (required, retired) in CURRENT_GUIDANCE.items():
            text = (ROOT / name).read_text(encoding="utf-8")
            for marker in required:
                with self.subTest(path=name, missing=marker):
                    with self.assertRaises(AssertionError):
                        validate_current_guidance(text.replace(marker, ""), name)
            for marker in retired:
                with self.subTest(path=name, retired=marker):
                    with self.assertRaises(AssertionError):
                        validate_current_guidance(text + "\n" + marker, name)

    def test_historical_architecture_bytes_and_rights_are_not_rewritten(self):
        # The old document remains at its original path for byte-exact replay.
        # Current navigation explicitly demotes it; no retention exception or
        # commercial license grant is introduced by a prose-only migration.
        import hashlib
        import json
        name = "docs/ASSET_ARCHITECTURE.md"
        raw = (ROOT / name).read_bytes()
        original = "9b0f94a6cf3ce1e63112fabea0e2b467b30230ab"
        digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        self.assertEqual(digest, original)
        manifest = json.loads((ROOT / "docs/BRANCH_RETENTION_20261007.json").read_bytes())
        row = next(row for row in manifest["retained_files"] if row["path"] == name)
        self.assertEqual(row["git_blob_sha1"], original)
        self.assertEqual(row["bytes"], len(raw))
        self.assertIn("当前输出商业许可状态为 `not-established`", raw.decode())

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
