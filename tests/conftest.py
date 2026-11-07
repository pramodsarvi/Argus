import pytest


@pytest.fixture(autouse=True)
def _grammar_mode(monkeypatch):
    """Most fakes key their answers on response_format; production default is "prompt" mode
    (see vlmpipe.bundle.STRUCTURED_MODE). Prompt-mode tests pass structured="prompt" explicitly."""
    import vlmpipe.bundle as b
    monkeypatch.setattr(b, "STRUCTURED_MODE", "grammar")
