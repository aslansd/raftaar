"""Use a published field audit instead of guessing at metadata.

Raftaar locates the gripper channel by matching feature names against a list of
hints. That works where `features[k].names` is semantic, and a hub-wide audit
puts that at roughly a quarter of datasets: 25% semantic, 40% placeholder
(`motor_0`, `motor_1`, …), 30% absent.

    https://huggingface.co/datasets/laa1991/lerobot-dataset-field-audit

That audit resolves the gripper index for the datasets where it is determinable
and marks the rest as *not determinable* rather than guessing. Where it has an
answer, reading it is strictly better than pattern-matching: it is someone
else's checked work, and it covers cases the hint list does not.

This module is deliberately tolerant about the file's shape. The audit is not
Raftaar's to version, column names may change, and a reader that hard-fails on a
renamed column would be worse than no reader at all. Anything it cannot parse it
reports, and the caller falls back to inference.

    from raftaar.field_audit import load_audit
    audit = load_audit("names-audit.csv")
    audit.gripper_index("lerobot/aloha_sim_transfer_cube_human", "state")
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["FieldAudit", "load_audit"]

#: Column names this reader will accept for each thing it needs. The audit is
#: not ours, so more than one spelling is allowed for each.
_DATASET_COLS = ("dataset", "repo_id", "dataset_id", "name")
_GRIPPER_COLS = {
    "action": ("gripper_indices_action", "gripper_index_action",
               "gripper_action", "action_gripper_indices"),
    "state": ("gripper_indices_state", "gripper_index_state",
              "gripper_state", "state_gripper_indices"),
}
_TIER_COLS = ("quality_tier", "names_quality", "tier", "quality")

#: Values that mean "we looked and could not tell", as opposed to a real index.
_NOT_DETERMINABLE = {
    "", "na", "n/a", "none", "null", "nan", "not determinable",
    "not_determinable", "undetermined", "unknown", "-",
}


def _first_present(row: dict, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in row and row[name] is not None:
            return name
    return None


def _parse_indices(raw: Any) -> list[int] | None:
    """Parse an index cell. Returns None for 'not determinable'.

    Accepts a bare integer, a bracketed list, or a separated list -- the audit
    may write any of these and the distinction does not matter here.
    """
    if raw is None:
        return None
    text = str(raw).strip().strip('"').strip("'")
    if text.lower() in _NOT_DETERMINABLE:
        return None
    found = re.findall(r"-?\d+", text)
    if not found:
        return None
    return [int(v) for v in found]


@dataclass
class FieldAudit:
    """Per-dataset readings from a published audit."""

    rows: dict[str, dict] = field(default_factory=dict)
    source: str = ""
    #: Columns the reader looked for and did not find, so a silently empty
    #: audit is distinguishable from one that genuinely says nothing.
    missing_columns: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    def _row(self, dataset: str) -> dict | None:
        if dataset in self.rows:
            return self.rows[dataset]
        # Tolerate `org/name` against a bare `name` and vice versa.
        tail = dataset.split("/")[-1]
        for key, row in self.rows.items():
            if key.split("/")[-1] == tail:
                return row
        return None

    def gripper_index(self, dataset: str, which: str = "state") -> int | None:
        """The gripper channel index, or None if the audit could not determine it.

        None means "no answer available", never "there is no gripper" -- those
        are different, and the caller should fall back to inference rather than
        conclude anything.
        """
        row = self._row(dataset)
        if row is None:
            return None
        indices = _parse_indices(row.get(f"_gripper_{which}"))
        if not indices:
            return None
        return indices[0]

    def quality_tier(self, dataset: str) -> str | None:
        """`semantic`, `placeholder`, `absent`, … as the audit recorded it."""
        row = self._row(dataset)
        return None if row is None else row.get("_tier")

    def describe(self) -> dict:
        tiers: dict[str, int] = {}
        determinable = 0
        for row in self.rows.values():
            tier = row.get("_tier") or "unknown"
            tiers[tier] = tiers.get(tier, 0) + 1
            if _parse_indices(row.get("_gripper_state")):
                determinable += 1
        return {
            "source": self.source,
            "n_datasets": len(self.rows),
            "n_gripper_determinable": determinable,
            "tiers": dict(sorted(tiers.items())),
            "missing_columns": self.missing_columns,
        }


def load_audit(path: str | Path) -> FieldAudit:
    """Read ``names-audit.csv`` from the published field audit.

    Never raises for a shape it does not recognise: an audit that cannot be
    parsed is reported through ``missing_columns`` and behaves as though it
    knows nothing, so calling code falls back to its own inference.
    """
    path = Path(path).expanduser()
    audit = FieldAudit(source=str(path))
    if not path.is_file():
        audit.missing_columns.append(f"file not found: {path}")
        return audit

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        audit.missing_columns.append("file is empty")
        return audit

    sample = rows[0]
    key_col = _first_present(sample, _DATASET_COLS)
    if key_col is None:
        audit.missing_columns.append(
            f"no dataset-name column; looked for {_DATASET_COLS}")
        return audit

    action_col = _first_present(sample, _GRIPPER_COLS["action"])
    state_col = _first_present(sample, _GRIPPER_COLS["state"])
    tier_col = _first_present(sample, _TIER_COLS)
    for label, col, candidates in (
        ("gripper (action)", action_col, _GRIPPER_COLS["action"]),
        ("gripper (state)", state_col, _GRIPPER_COLS["state"]),
        ("quality tier", tier_col, _TIER_COLS),
    ):
        if col is None:
            audit.missing_columns.append(f"{label}: looked for {candidates}")

    for row in rows:
        name = (row.get(key_col) or "").strip()
        if not name:
            continue
        audit.rows[name] = {
            **row,
            "_gripper_action": row.get(action_col) if action_col else None,
            "_gripper_state": row.get(state_col) if state_col else None,
            "_tier": (row.get(tier_col) or "").strip().lower() or None
            if tier_col else None,
        }
    return audit
