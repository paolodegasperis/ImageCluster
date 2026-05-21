from fastapi.testclient import TestClient

from backend.main import app


def test_diagnostics_endpoint_returns_runtime_fields():
    client = TestClient(app)
    response = client.get("/api/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert "version" in payload
    assert "python_version" in payload
    assert "torch" in payload
    assert "jobs" in payload
    assert "package_versions" in payload
