import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from setmap_audio.schemas import AudioProfileExport


def test_documented_export_contract_is_versioned_and_reproducible():
    readme = Path("README.md").read_text()
    example_start = readme.index("{\n  \"schema_version\"")
    example_end = readme.index("\n```", example_start)
    payload = json.loads(readme[example_start:example_end])
    payload["model"]["dimensions"] = 2

    export = AudioProfileExport.model_validate(payload)
    assert export.schema_version == "1.0"
    assert export.model.id
    assert export.model.revision
    assert [item.role for item in export.aggregation.vectors] == [
        "global",
        "cluster_1",
        "cluster_2",
    ]


def test_export_schema_rejects_mismatched_vector_dimensions():
    readme = Path("README.md").read_text()
    example_start = readme.index("{\n  \"schema_version\"")
    example_end = readme.index("\n```", example_start)
    payload = json.loads(readme[example_start:example_end])
    with pytest.raises(ValidationError, match="vector length"):
        AudioProfileExport.model_validate(payload)
