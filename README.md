# SetMap

A CPU-first proof of concept that turns a DJ set or mix into three compact, searchable music
vectors for PostgreSQL/Supabase.

The default model is [`laion/larger_clap_music`](https://huggingface.co/laion/larger_clap_music),
a music-tuned CLAP checkpoint that emits text-compatible 512-dimensional embeddings. The
application streams decoding through FFmpeg, rejects unusable windows before inference, and
exports a global vector plus two minimum-support spherical k-means centroids.

## Pipeline

1. Accept MP3, WAV, FLAC, M4A, AAC, OGG, or Opus up to the configurable duration limit. The
   default is four hours.
2. Stream mono 48 kHz decoding through one FFmpeg process. The complete recording never lives
   in memory.
3. Select 10-second windows every 20, 25, or 30 seconds.
4. Deterministically reject silence, mostly silent windows, severe clipping, DC offset, and
   obvious broadband noise.
5. Lazily load CLAP on CPU and embed retained windows in small batches.
6. L2-normalize every clip vector. Optional experimental outlier treatment estimates local cosine
   density and removes only robustly isolated embeddings, capped at 10%. It is off by default
   because it reduced retrieval quality in the six-set benchmark.
7. Create:
   - `global`: normalized mean of the embedding-space core;
   - `cluster_1`: largest spherical k-means centroid;
   - `cluster_2`: second spherical k-means centroid.
8. Export each cluster's core-window proportion as its weight. A 10% minimum-support constraint
   prevents a single transition or unusual clip from becoming a nominal second style.
9. Use PCA only for explained variance, diversity, and timecodes at the dominant-axis extremes.

The frontend accepts multiple recordings, processes them through the single CPU worker, and builds
a **musical map**. Every set appears as one global position; the cluster vectors stay hidden and are
used only for high-dimensional reranking. Nearest-neighbour labels use the original weighted 512D
profile rather than visually compressed 2D distance. Completed profiles are retained in the
browser's local storage and appear under **Mapped sets** on the next visit. Audio is never retained.
Use the per-set export button or **Download JSON** to make a portable backup. **Clear** empties this
local profile cache so recordings can be processed again, but deliberately preserves downloaded
model weights. Existing batch/benchmark JSON can be loaded with **Import profiles**.

The exact Hugging Face model ID and immutable Git revision are stored with each result. Raw PCA
directions are never exported as music embeddings: they are centered, signed directions rather
than points in CLAP's cosine-search space.

## Near-immediate setup with `uv`

Requirements:

- Python 3.11 or 3.12 (`uv` can install it)
- `uv`
- FFmpeg, including `ffprobe`
- approximately 1 GB of disk space for the initial model download
- realistically 2–4 GB of free RAM while the full model is running

Install FFmpeg and `uv`:

```bash
# macOS (Homebrew)
brew install uv ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell with winget)
winget install --id Gyan.FFmpeg --exact
winget install --id astral-sh.uv --exact
```

Then:

```bash
git clone <repository-url>
cd setmap-audio-profile-poc
cp .env.example .env
uv sync --extra dev --python 3.12
uv run setmap-audio
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The page and `/health` start immediately;
model weights are downloaded and loaded only when the first real job reaches inference.

## Fast UI demonstration without model weights

Set this in `.env` before starting the server:

```dotenv
AS_EMBEDDER_BACKEND=fake
```

The deterministic fake embedder makes the upload, filtering, progress, clustering, results, and
JSON-export flow testable without a model download. The UI displays **DEMO MODE · FAKE VECTORS**.
These vectors are stable and normalized, but they are not CLAP embeddings and must never be
stored as production artist profiles.

## Configuration

```dotenv
AS_MODEL_ID=laion/larger_clap_music
AS_MODEL_REVISION=a0b4534a14f58e20944452dff00a22a06ce629d1
AS_EMBEDDER_BACKEND=clap
AS_TORCH_THREADS=4
AS_BATCH_SIZE=4
AS_SAMPLE_INTERVAL_SECONDS=25
AS_MAX_DURATION_SECONDS=14400
AS_MAX_UPLOAD_MIB=1024
AS_MIN_RMS_DBFS=-48
AS_MAX_CLIPPED_RATIO=0.02
AS_EMBEDDING_OUTLIER_MAX_FRACTION=0
AS_TELEMETRY_ENABLED=0
AS_DATA_DIR=./data
# AS_INITIAL_PROFILE_EXPORT=/absolute/path/to/profiles.json
```

Keep `AS_MODEL_REVISION` immutable for reproducible profiles. The backend is defined behind a
small `AudioEmbedder` protocol, so MERT or dedicated music-similarity encoders can be added for
evaluation later. Do not mix vectors from different models or revisions in the same index.

`AS_MAX_DURATION_SECONDS` is the hard per-recording duration limit. It defaults to `14400`
(four hours). Increase it deliberately if longer recordings are required; decoding remains
streamed, while inference time grows approximately with the number of sampled windows.

`AS_INITIAL_PROFILE_EXPORT` can point at an existing batch or benchmark JSON file. On startup,
those real profiles are merged into the browser's **Mapped sets** library without copying the
source audio. When the file contains at least three profiles, their global vectors also fit the
frozen, versioned map projection. Leave it unset for an empty library and a session-relative map;
imported and newly processed profiles still persist in browser storage. Using **Clear** records a
local dismissal so the initial file is not automatically restored on the next page load.

## CPU expectations

At the default 25-second interval, a four-hour set considers about 577 windows. Filtering may
reduce that number before model inference. The model checkpoint is roughly 780 MB, but Python,
PyTorch, model activations, and batched audio raise total memory use; budget 2–4 GB of free RAM.

Processing time varies dramatically by CPU and PyTorch build. A four-hour set can take tens of
minutes or longer on a laptop CPU. Benchmark a five-minute recording on the target machine,
then extrapolate from the exported `timing_ms.embed` value. Reduce batch size if memory is tight;
increase thread count only until the target CPU stops getting faster.

### Measured local benchmark

Measured on 2026-09-04 on an Apple M4 Pro (14 CPU cores), with PyTorch limited to four CPU
threads and **MPS disabled**. The run used six local MP3 sets, 10-second windows every 30 seconds,
batch size four, and the pinned CLAP revision above.

| Per-set statistic | Source duration | Windows | Pipeline time |
|---|---:|---:|---:|
| Minimum | 81.8 min | 164 | 10.46 s |
| Median | 96.9 min | 184 | 13.12 s |
| Mean | 101.0 min | 203 | 13.11 s |
| Maximum | 120.0 min | 241 | 15.26 s |

Total: **10.10 hours**, **1,217 windows**, **78.66 seconds**, or **64.6 ms per window**. Peak
resident memory was **0.93 GiB**. A warm cached model load measured about **1.60 seconds**. The
first observed load took about **66.8 seconds**, including the initial checkpoint fetch; download
time depends on the network and is not part of the warm pipeline total.

The global mean alone produced 4/6 same-group nearest neighbours. Symmetric weighted-cluster
matching and the prototype's 25% global / 75% cluster reranker both produced 6/6. This is a tiny
directional test, not production accuracy. See [`BENCHMARK.md`](BENCHMARK.md) for matrices,
aggregation ablations, interpretation, and the evaluation plan.

#### Outlier and medoid ablation

We explicitly tested the proposed embedding-space outlier treatment. A robust local-density trim
removed 7–10% of clips per set. On these six sets it removed useful distinctive passages:

| Aggregation variant | Same-group nearest neighbours |
|---|---:|
| Untrimmed spherical centroids (default) | 6 / 6 |
| Density-trimmed spherical centroids | 4 / 6 |
| Density-trimmed spherical medoids | 3 / 6 |

Therefore trimming remains an **experimental frontend and environment control**, off by default.
Choose 5% or 10% only for evaluation; the exact method, cap, core count, and removed count are
recorded in every JSON export. Medoids are not exported because one observed clip retained too much
track/transition-specific variance compared with a centroid.

## API and versioned JSON

Create a job:

```bash
curl -F audio=@set.mp3 -F interval_seconds=25 http://127.0.0.1:8000/api/jobs
```

Poll the returned `status_url`, then download `/api/jobs/{job_id}/export`.

```json
{
  "schema_version": "1.0",
  "model": {
    "id": "laion/larger_clap_music",
    "revision": "a0b4534a14f58e20944452dff00a22a06ce629d1",
    "dimensions": 512,
    "device": "cpu",
    "backend": "clap",
    "load_duration_ms": 1842
  },
  "source": {
    "duration_seconds": 3598.2,
    "sample_interval_seconds": 25,
    "window_seconds": 10.0,
    "windows_considered": 145,
    "windows_retained": 139,
    "windows_filtered": 6,
    "filter_counts": {"low_energy": 4, "clipped": 2}
  },
  "aggregation": {
    "method": "robust_spherical_kmeans",
    "version": "3",
    "cluster_strategy": "spherical_kmeans",
    "outlier_method": "disabled",
    "max_outlier_fraction": 0.0,
    "input_clip_count": 139,
    "core_clip_count": 139,
    "embedding_outlier_count": 0,
    "local_density_threshold": null,
    "vectors": [
      {"role": "global", "weight": 1.0, "vector": [0.012, -0.034]},
      {"role": "cluster_1", "weight": 0.647482, "vector": [0.021, -0.018]},
      {"role": "cluster_2", "weight": 0.352518, "vector": [-0.008, 0.041]}
    ]
  },
  "diagnostics": {
    "pca_explained_variance_ratio": [0.184, 0.103, 0.071],
    "pca_extreme_snippets": [
      {"side": "low", "retained_clip_index": 9, "start_seconds": 225.0, "timecode": "00:03:45"},
      {"side": "high", "retained_clip_index": 91, "start_seconds": 2275.0, "timecode": "00:37:55"}
    ],
    "diversity": 0.118
  },
  "timing_ms": {"probe": 62, "decode_and_filter": 576, "filter_compute": 411, "embed": 42031, "aggregate": 19, "total": 42698}
}
```

Vectors are shortened only in this documentation; live exports contain all 512 values.

## Supabase / PostgreSQL with pgvector

Run [`sql/001_audio_profiles.sql`](sql/001_audio_profiles.sql) in the Supabase SQL editor. It
enables pgvector, creates profile and representative-vector tables, adds an HNSW cosine index,
and defines a similarity RPC.

Each representative-vector row stores the profile ID, role, weight, embedding, model ID,
revision, aggregation method/version, window size, sampling interval, source duration, and
metadata JSONB. JavaScript Supabase clients can pass the 512-number array directly.

Start retrieval with `role = 'global'` to get candidates. For artists with varied sets, rerank
the shortlist using a blend of global cosine and symmetric weighted cluster coverage:

```text
cluster_similarity = 0.5 × Σ(query_cluster_weight × best_candidate_cluster_cosine)
                   + 0.5 × Σ(candidate_cluster_weight × best_query_cluster_cosine)
score = 0.25 × global_similarity + 0.75 × cluster_similarity
```

This prototype default improved creator-group nearest-neighbour accuracy from 4/6 to 6/6 on the
included six-set directional benchmark. It is still an experimental, versioned reranker—not a
production constant—and needs tuning on more labelled examples. The migration includes a concrete
global-vector similarity query.

## Musical-map semantics and production stability

`POST /api/musical-map` displays only normalized global vectors. Cluster centroids never become
extra dots; they remain part of the original 512-dimensional similarity reranker. When
`AS_INITIAL_PROFILE_EXPORT` provides at least three reference profiles, the service fits PCA once
at startup, fixes axis signs and scale, hashes the transform into a projection version, and applies
that frozen transform to every request. Adding a profile therefore does not move existing points.
The API reports the projection method, version, reference count, and stability state.

Without a reference export, the prototype falls back to collection-relative PCA and clearly labels
the map **SESSION PCA / NOT FROZEN**. Those temporary coordinates can move when the collection
changes and should not be persisted.

### Why not UMAP by default?

UMAP often makes more visually distinct local neighbourhoods than PCA, and `umap-learn` can
transform new vectors through an already fitted model. It does not remove the production versioning
problem: refitting changes the coordinate system, transformed out-of-distribution styles can be
placed poorly, and the dependency/runtime cost is considerably larger. The production-safe pattern
is the same for either algorithm—fit on a representative catalogue, freeze and version the fitted
artifact, then transform new profiles without refitting.

PCA is the prototype default because its transform is small, deterministic, fast, auditable, and
easy to serialize alongside a catalogue version. Once the reference catalogue is large enough,
benchmark frozen PCA against frozen UMAP using neighbourhood trustworthiness, labelled-neighbour
recall, coordinate drift, transform latency, and out-of-distribution behaviour. UMAP should replace
PCA only if that evaluation shows a meaningful visual-neighbourhood improvement.

## Privacy-conscious telemetry

Telemetry is disabled by default. When `AS_TELEMETRY_ENABLED=1`, it writes only to
`data/telemetry.sqlite3` on the local machine. Events include total and per-stage timing,
model-load time, sampling/batch/thread configuration, considered/retained/filtered window counts,
filter reasons, and sanitized exception type.

It never records audio, filenames, embeddings, IP addresses, user agents, account data, or other
personal identifiers. Temporary uploads are deleted after success or failure.

The interface uses smooth CSS gradients without a stretched grain texture. The decorative
processing radar shares one clock between its beam and detections: a contact lights only at
the beam crossing, fades over one revolution, and disappears at the next crossing. New random
contacts are scheduled each revolution. Animation pauses in background tabs and respects
reduced-motion preferences; radar contacts do not represent actual embedding progress.

## Tests and development

```bash
uv run pytest -q
uv run ruff check .
node --test tests/radar.test.mjs
```

The unit suite covers sampling, normalization, spherical clustering, relevance filtering,
deterministic fake embeddings, musical-map projection/neighbours, the documented export contract,
and failure cleanup. Tests do not download CLAP weights.

Full-model verification requires the 780 MB download and should be run separately with a short,
licensed audio fixture. Confirm that the export reports the pinned revision, 512 dimensions,
unit vector norms, and reasonable text-to-audio ranking before changing the pinned checkpoint.

## Model evaluation plan

Before considering a model switch:

1. Create 30–50 human-labelled query groups. Each contains one source artist/set,
   2–4 musically similar positives, and 5–10 deliberately dissimilar negatives.
2. Freeze audio windows, filter thresholds, and model revisions.
3. Measure Recall@5, mean reciprocal rank, positive-vs-negative cosine margin, CPU seconds per
   retained window, and peak resident memory.
4. Score global-only retrieval first, then the weighted cluster reranker.
5. Review failure cases with genre/location experts, especially multi-style and transition-heavy
   sets.
6. Compare the pinned CLAP checkpoint with MERT or a dedicated similarity encoder only under the
   same test set and compute budget. Switch only if retrieval quality improves meaningfully without
   violating the CPU deployment target.

## Prototype boundaries

- Jobs and results live in memory; restarting clears them.
- One worker processes jobs sequentially to cap CPU and RAM contention.
- Conservative signal filtering is not semantic music detection and should be calibrated against
  real sets before production use.
- Static windows can miss short transitions. A later benchmark can compare deterministic novelty
  selection.
- The server binds to localhost and has no authentication or rate limiting.

See [`AGENTS.md`](AGENTS.md) for the repository map and Codex working contract.
The measured six-set CPU run and its separation caveats are documented in
[`BENCHMARK.md`](BENCHMARK.md).

## License

Project code is MIT. The pinned LAION model is Apache-2.0; review the model card and any replacement
checkpoint's license before production use.
