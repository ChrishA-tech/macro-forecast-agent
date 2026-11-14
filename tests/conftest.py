import pytest

import macroagent.graph as graph


@pytest.fixture(autouse=True)
def rules_only_guard(monkeypatch):
    """Tests are deterministic: they do not depend on whichever trained classifier is vendored."""
    monkeypatch.setattr(graph, "default_classifier", lambda: None)
