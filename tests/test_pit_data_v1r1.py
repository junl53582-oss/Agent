from pit_data_v1r1 import source
from pit_data_v1r1.core import ObservationSettings


def test_flow_pagination_uses_provider_hard_cap(monkeypatch):
    calls = []

    def fake_get(session, url, params):
        page = int(params["pn"])
        calls.append((page, int(params["pz"])))
        count = 100 if page < 3 else 50
        body = {"data": {"total": 250, "diff": [{"f12": str(i)} for i in range(count)]}}
        return f"page-{page}".encode(), body

    monkeypatch.setattr(source.parent, "_get_json", fake_get)
    pages = source.fetch_flow_pages(session=object())
    assert len(pages) == 3
    assert calls == [(1, 100), (2, 100), (3, 100)]


def test_repair_uses_independent_dynamic_roots():
    settings = ObservationSettings()
    assert settings.data_root.as_posix().endswith("pit_observations_v1r1")
    assert settings.artifact_root.as_posix().endswith("pit_data_v1r1")
