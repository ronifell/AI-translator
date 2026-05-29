from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ChangeEntry:
    path: str
    before: str
    after: str


@dataclass
class DiffTracker:
    changes: list[ChangeEntry] = field(default_factory=list)
    # Defensive lock: while the async pipeline runs in a single thread,
    # progress callbacks can be invoked from a worker thread that owns the
    # event loop. Append/read consistency under such cross-thread access is
    # cheap to guarantee with a tiny lock around mutation.
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record(self, path: str, before: str, after: str) -> None:
        if before == after:
            return
        with self._lock:
            self.changes.append(ChangeEntry(path=path, before=before, after=after))

    def to_json(self) -> list[dict]:
        with self._lock:
            return [{"path": c.path, "before": c.before, "after": c.after} for c in self.changes]
