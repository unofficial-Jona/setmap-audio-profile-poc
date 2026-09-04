import numpy as np

from setmap_audio.musical_map import build_musical_map, fit_global_projection


def _profile(profile_id: str, angle: float) -> dict:
    def point(offset: float) -> list[float]:
        return [float(np.cos(angle + offset)), float(np.sin(angle + offset)), 0.0]

    return {
        "id": profile_id,
        "label": profile_id.upper(),
        "vectors": [
            {"role": "global", "weight": 1.0, "vector": point(0.0)},
            {"role": "cluster_1", "weight": 0.6, "vector": point(-0.08)},
            {"role": "cluster_2", "weight": 0.4, "vector": point(0.08)},
        ],
    }


def test_map_finds_the_nearest_cosine_neighbour_and_projects_global_points():
    result = build_musical_map([_profile("a", 0.0), _profile("b", 0.1), _profile("c", 2.0)])
    by_id = {profile["id"]: profile for profile in result["profiles"]}

    assert by_id["a"]["nearest"]["id"] == "b"
    assert by_id["b"]["nearest"]["id"] == "a"
    assert set(by_id["a"]["points"]) == {"global"}
    assert all(abs(point["x"]) <= 1.15 for point in by_id["a"]["points"].values())
    assert result["method"] == "collection_pca_v2"
    assert result["projection_is_stable"] is False
    assert result["similarity_method"] == "quarter_global_three_quarter_weighted_cluster_v1"
    assert set(by_id["a"]["nearest"]) == {
        "id",
        "label",
        "similarity",
        "global_similarity",
        "cluster_similarity",
    }


def test_map_is_valid_for_one_profile():
    result = build_musical_map([_profile("solo", 0.0)])
    assert result["profiles"][0]["nearest"] is None


def test_frozen_projection_keeps_existing_positions_when_a_profile_is_added():
    reference_profiles = [_profile("a", 0.0), _profile("b", 0.5), _profile("c", 1.4)]
    globals_ = [profile["vectors"][0]["vector"] for profile in reference_profiles]
    projection = fit_global_projection(globals_)

    before = build_musical_map(reference_profiles, projection=projection)
    after = build_musical_map(reference_profiles + [_profile("d", 2.4)], projection=projection)
    before_by_id = {profile["id"]: profile for profile in before["profiles"]}
    after_by_id = {profile["id"]: profile for profile in after["profiles"]}

    assert before_by_id["a"]["points"] == after_by_id["a"]["points"]
    assert before_by_id["b"]["points"] == after_by_id["b"]["points"]
    assert after["method"] == "frozen_reference_pca_v1"
    assert after["projection_is_stable"] is True
    assert after["projection_version"] == before["projection_version"]
