from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelInfo(BaseModel):
    id: str
    revision: str
    dimensions: int = Field(gt=0)
    device: Literal["cpu"]
    backend: Literal["clap", "fake"]
    load_duration_ms: int = Field(ge=0)


class SourceInfo(BaseModel):
    duration_seconds: float = Field(gt=0)
    sample_interval_seconds: Literal[20, 25, 30]
    window_seconds: float = Field(gt=0)
    windows_considered: int = Field(gt=0)
    windows_retained: int = Field(gt=0)
    windows_filtered: int = Field(ge=0)
    filter_counts: dict[str, int]


class RepresentativeVector(BaseModel):
    role: Literal["global", "cluster_1", "cluster_2"]
    weight: float = Field(ge=0, le=1)
    vector: list[float] = Field(min_length=1)


class AggregationInfo(BaseModel):
    method: Literal["robust_spherical_kmeans"]
    version: Literal["3"]
    cluster_strategy: Literal[
        "spherical_kmeans", "minimum_support_spherical_kmeans", "single_embedding_duplicate"
    ]
    outlier_method: Literal["disabled", "knn_density_mad_3.5"]
    max_outlier_fraction: float = Field(ge=0, le=0.1)
    input_clip_count: int = Field(gt=0)
    core_clip_count: int = Field(gt=0)
    embedding_outlier_count: int = Field(ge=0)
    local_density_threshold: float | None
    vectors: list[RepresentativeVector] = Field(min_length=3, max_length=3)


class PcaExtreme(BaseModel):
    side: Literal["low", "high"]
    retained_clip_index: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    timecode: str


class DiagnosticsInfo(BaseModel):
    pca_explained_variance_ratio: list[float]
    pca_extreme_snippets: list[PcaExtreme] = Field(min_length=2, max_length=2)
    diversity: float = Field(ge=0)


class AudioProfileExport(BaseModel):
    schema_version: Literal["1.0"]
    model: ModelInfo
    source: SourceInfo
    aggregation: AggregationInfo
    diagnostics: DiagnosticsInfo
    timing_ms: dict[str, int]

    @model_validator(mode="after")
    def validate_vector_contract(self):
        roles = [item.role for item in self.aggregation.vectors]
        if roles != ["global", "cluster_1", "cluster_2"]:
            raise ValueError("representative vectors must be ordered global, cluster_1, cluster_2")
        if any(len(item.vector) != self.model.dimensions for item in self.aggregation.vectors):
            raise ValueError("every vector length must equal model.dimensions")
        cluster_weight = sum(item.weight for item in self.aggregation.vectors[1:])
        if abs(cluster_weight - 1.0) > 1e-5:
            raise ValueError("cluster weights must sum to one")
        if (
            self.aggregation.core_clip_count + self.aggregation.embedding_outlier_count
            != self.aggregation.input_clip_count
        ):
            raise ValueError("aggregation clip counts must reconcile")
        return self


class MusicalMapProfile(BaseModel):
    id: str
    label: str = Field(min_length=1, max_length=200)
    vectors: list[RepresentativeVector] = Field(min_length=3, max_length=3)


class MusicalMapRequest(BaseModel):
    profiles: list[MusicalMapProfile] = Field(min_length=1, max_length=100)
