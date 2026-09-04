import json

from fastapi.testclient import TestClient

from setmap_audio.app import create_app
from setmap_audio.config import Settings


def test_initial_profile_export_exposes_only_valid_profile_records(tmp_path):
    export_path = tmp_path / "profiles.json"
    export_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "label": "Established set",
                        "profile": {"aggregation": {"vectors": [{"role": "global"}]}},
                    },
                    {"label": "Incomplete record"},
                ]
            }
        )
    )
    settings = Settings(
        embedder_backend="fake",
        data_dir=tmp_path,
        initial_profile_export=export_path,
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/initial-profiles")

    assert response.status_code == 200
    assert [profile["label"] for profile in response.json()["profiles"]] == ["Established set"]


def test_initial_profile_export_defaults_to_empty(tmp_path):
    settings = Settings(embedder_backend="fake", data_dir=tmp_path, initial_profile_export=None)

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/initial-profiles")

    assert response.json() == {"profiles": []}
