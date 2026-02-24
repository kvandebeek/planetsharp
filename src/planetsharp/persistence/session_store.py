from __future__ import annotations

import json
from pathlib import Path

from planetsharp.core.models import Session


class SessionStore:
    @staticmethod
    def save(path: str, session: Session) -> None:
        output = Path(path)
        output.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str) -> Session:
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        session = Session.from_dict(data)
        session.resolve_paths(source.parent)
        return session
