from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChangeEntry:
    path: str
    before: str
    after: str


@dataclass
class DiffTracker:
    changes: list[ChangeEntry] = field(default_factory=list)

    def record(self, path: str, before: str, after: str) -> None:
        if before == after:
            return
        self.changes.append(ChangeEntry(path=path, before=before, after=after))

    def to_json(self) -> list[dict]:
        return [{"path": c.path, "before": c.before, "after": c.after} for c in self.changes]
