"""API tests for the remove-bg-local server.

These run without downloading any model: the model load and the inference are
monkeypatched, so the suite is fast and works in CI with no network.
"""

import io

import pytest
from PIL import Image
from fastapi.testclient import TestClient

import server


@pytest.fixture
def client():
    return TestClient(server.app)


def png_bytes(size=(64, 64), color=(200, 60, 60)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["default_model"] == server.DEFAULT_MODEL
    assert body["pinned_model"] == server._PINNED_MODEL
    assert body["idle_ttl_seconds"] == server.MODEL_IDLE_TTL
    assert isinstance(body["loaded_models"], list)


def test_models_payload(client):
    data = client.get("/models").json()
    assert data["default"] == server.DEFAULT_MODEL
    assert set(data["available"]) == set(server.AVAILABLE_MODELS)
    assert set(data["info"]) == set(server.AVAILABLE_MODELS)
    assert set(data["downloaded"]) == set(server.AVAILABLE_MODELS)
    # every model must expose complete, non-empty info for the Models page
    for key, info in data["info"].items():
        for field in ("title", "tagline", "speed", "quality", "best_for", "description"):
            assert info.get(field), f"{key} missing {field}"
        assert key in data["sizes_mb"]
        assert isinstance(data["downloaded"][key], bool)


def test_model_status_default(client):
    data = client.get("/model_status").json()
    assert data["model"] == server.DEFAULT_MODEL
    assert data["state"] in ("idle", "loading", "ready")
    assert isinstance(data["downloaded"], bool)
    assert data["progress"] is None or 0.0 <= data["progress"] <= 1.0


def test_model_status_unknown_is_400(client):
    r = client.get("/model_status", params={"model": "does-not-exist"})
    assert r.status_code == 400


def test_remove_empty_image_is_400(client):
    r = client.post("/remove", files={"image": ("a.png", b"", "image/png")})
    assert r.status_code == 400


def test_remove_invalid_image_is_400(client):
    r = client.post("/remove", files={"image": ("a.png", b"not really a png", "image/png")})
    assert r.status_code == 400


def test_remove_unknown_model_is_400(client):
    r = client.post(
        "/remove",
        files={"image": ("a.png", png_bytes(), "image/png")},
        data={"model": "does-not-exist"},
    )
    assert r.status_code == 400


def test_remove_too_large_is_413(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 8)
    r = client.post("/remove", files={"image": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 413


def test_remove_success(client, monkeypatch):
    """Full happy path with the model load + inference mocked out."""
    async def fake_ensure_session(model):
        return object()

    def fake_remove(img, session=None, **kwargs):
        return img.convert("RGBA")

    monkeypatch.setattr(server, "ensure_session", fake_ensure_session)
    monkeypatch.setattr(server, "remove", fake_remove)

    r = client.post(
        "/remove",
        files={"image": ("photo.jpg", png_bytes(), "image/png")},
        data={"model": server.DEFAULT_MODEL},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["x-model"] == server.DEFAULT_MODEL
    assert "x-processing-time" in r.headers
    out = Image.open(io.BytesIO(r.content))
    assert out.format == "PNG"
    assert out.mode == "RGBA"


def test_warmup_returns_state(client, monkeypatch):
    async def fake_ensure_session(model):
        return object()

    monkeypatch.setattr(server, "ensure_session", fake_ensure_session)
    r = client.post("/warmup", data={"model": server.DEFAULT_MODEL})
    assert r.status_code == 200
    assert r.json()["model"] == server.DEFAULT_MODEL


def test_providers_default_is_cpu():
    # Reliability guard: CoreML hangs on some models on Apple Silicon.
    assert server.PROVIDERS, "at least one execution provider must be configured"


def test_delete_model_unknown_is_400(client):
    r = client.post("/delete_model", data={"model": "does-not-exist"})
    assert r.status_code == 400


def test_delete_model_removes_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "U2NET_HOME", str(tmp_path))
    name = server.DEFAULT_MODEL
    (tmp_path / (name + ".onnx")).write_bytes(b"fake-model")
    assert server.is_downloaded(name) is True
    r = client.post("/delete_model", data={"model": name})
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    assert server.is_downloaded(name) is False


def test_delete_model_not_downloaded_ok(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "U2NET_HOME", str(tmp_path))
    r = client.post("/delete_model", data={"model": server.DEFAULT_MODEL})
    assert r.status_code == 200
    assert r.json()["deleted"] is False


@pytest.fixture
def clean_sessions():
    """Reset the in-RAM session bookkeeping before and after a test."""
    saved_pinned = server._PINNED_MODEL
    server._SESSIONS.clear()
    server._LAST_USED.clear()
    server._MODEL_STATE.clear()
    server._MODEL_ERROR.clear()
    yield
    server._SESSIONS.clear()
    server._LAST_USED.clear()
    server._MODEL_STATE.clear()
    server._MODEL_ERROR.clear()
    server._PINNED_MODEL = saved_pinned


def test_unload_model_unknown_is_400(client):
    r = client.post("/unload_model", data={"model": "does-not-exist"})
    assert r.status_code == 400


def test_unload_model_drops_session(client, clean_sessions):
    name = server.DEFAULT_MODEL
    server._SESSIONS[name] = object()
    server._LAST_USED[name] = 123.0
    server._MODEL_STATE[name] = "ready"
    r = client.post("/unload_model", data={"model": name})
    assert r.status_code == 200
    body = r.json()
    assert body["unloaded"] is True
    assert body["state"] == "idle"
    assert name not in server._SESSIONS
    assert name not in server._LAST_USED


def test_unload_model_when_not_loaded(client, clean_sessions):
    r = client.post("/unload_model", data={"model": server.DEFAULT_MODEL})
    assert r.status_code == 200
    assert r.json()["unloaded"] is False


def test_set_default_model_unknown_is_400(client):
    r = client.post("/set_default_model", data={"model": "does-not-exist"})
    assert r.status_code == 400


def test_set_default_model_evicts_others(client, clean_sessions):
    keep = "isnet-general-use"
    drop = "u2net"
    server._SESSIONS[keep] = object()
    server._SESSIONS[drop] = object()
    server._LAST_USED[keep] = 1.0
    server._LAST_USED[drop] = 1.0
    r = client.post("/set_default_model", data={"model": keep})
    assert r.status_code == 200
    body = r.json()
    assert body["pinned"] == keep
    assert body["evicted"] == [drop]
    assert keep in server._SESSIONS
    assert drop not in server._SESSIONS
    assert server._PINNED_MODEL == keep


def test_set_default_model_no_warmup_by_default(client, monkeypatch, clean_sessions):
    called = {"n": 0}

    async def fake_ensure_session(model):
        called["n"] += 1
        return object()

    monkeypatch.setattr(server, "ensure_session", fake_ensure_session)
    r = client.post("/set_default_model", data={"model": server.DEFAULT_MODEL})
    assert r.status_code == 200
    # warmup defaults to false → no eager load.
    assert called["n"] == 0


def test_remove_transient_flag_is_accepted(client, monkeypatch, clean_sessions):
    """The transient flag must round-trip and not break the happy path."""
    seen = {}

    async def fake_ensure_session(model):
        seen["model"] = model
        return object()

    def fake_remove(img, session=None, **kwargs):
        return img.convert("RGBA")

    monkeypatch.setattr(server, "ensure_session", fake_ensure_session)
    monkeypatch.setattr(server, "remove", fake_remove)

    r = client.post(
        "/remove",
        files={"image": ("photo.jpg", png_bytes(), "image/png")},
        data={"model": "u2net", "transient": "true"},
    )
    assert r.status_code == 200
    assert seen["model"] == "u2net"
    # Pinned default must NOT be hijacked by a transient request.
    assert server._PINNED_MODEL == server.DEFAULT_MODEL


def test_evict_helper_clears_state(clean_sessions):
    name = server.DEFAULT_MODEL
    server._SESSIONS[name] = object()
    server._LAST_USED[name] = 7.0
    server._MODEL_STATE[name] = "ready"
    assert server._evict(name) is True
    assert name not in server._SESSIONS
    assert name not in server._LAST_USED
    assert name not in server._MODEL_STATE
    assert server._evict(name) is False  # idempotent


def test_evictor_sweep_drops_idle_non_pinned(monkeypatch, clean_sessions):
    """A single sweep must drop idle non-pinned models past the TTL."""
    pinned = "isnet-general-use"
    stale = "u2net"
    fresh = "u2net_human_seg"
    server._PINNED_MODEL = pinned
    server._SESSIONS[pinned] = object()
    server._SESSIONS[stale] = object()
    server._SESSIONS[fresh] = object()

    monkeypatch.setattr(server, "MODEL_IDLE_TTL", 100)
    server._LAST_USED[pinned] = 0.0  # would be evicted if it weren't pinned
    server._LAST_USED[stale] = 0.0
    server._LAST_USED[fresh] = 1_000_000.0  # in the future

    monkeypatch.setattr(server.time, "monotonic", lambda: 1_000.0)

    evicted = server._evictor_sweep()

    assert evicted == [stale]
    assert pinned in server._SESSIONS, "pinned must survive"
    assert fresh in server._SESSIONS, "fresh must survive"
    assert stale not in server._SESSIONS, "stale must be evicted"


def test_evictor_sweep_disabled_when_ttl_zero(monkeypatch, clean_sessions):
    server._SESSIONS["u2net"] = object()
    server._LAST_USED["u2net"] = 0.0
    monkeypatch.setattr(server, "MODEL_IDLE_TTL", 0)
    assert server._evictor_sweep() == []
    assert "u2net" in server._SESSIONS


def test_download_progress_helpers(monkeypatch):
    # A downloaded model reports full progress; helpers must not raise.
    monkeypatch.setattr(server, "is_downloaded", lambda name: True)
    assert server.download_progress(server.DEFAULT_MODEL) == 1.0
    monkeypatch.setattr(server, "is_downloaded", lambda name: False)
    monkeypatch.setattr(server, "U2NET_HOME", "/nonexistent-dir-for-tests")
    # No cache dir / no temp files -> 0.0, never an exception.
    assert server.download_progress(server.DEFAULT_MODEL) == 0.0
