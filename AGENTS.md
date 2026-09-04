# AGENTS.md

## Project contract

This repository is a CPU-first proof of concept for turning a DJ set or music mix into a
small set of CLAP vectors. Keep the default workflow local, privacy-conscious, and easy to
run. User-uploaded audio is temporary and must be deleted after processing. Never add audio,
model weights, telemetry databases, or generated vectors to Git.

## Architecture

- `src/setmap_audio/app.py`: FastAPI routes and static UI hosting.
- `src/setmap_audio/jobs.py`: single-worker background job lifecycle.
- `src/setmap_audio/audio.py`: ffprobe/ffmpeg validation and window decoding.
- `src/setmap_audio/embedder.py`: lazy CPU-only CLAP inference.
- `src/setmap_audio/filtering.py`: deterministic signal-quality filtering.
- `src/setmap_audio/aggregation.py`: global vector, spherical k-means, PCA diagnostics.
- `src/setmap_audio/musical_map.py`: collection-relative 2D projection and neighbours.
- `src/setmap_audio/telemetry.py`: local metadata-only SQLite events.
- `sql/001_audio_profiles.sql`: Supabase/PostgreSQL + pgvector schema.

## Working rules

- Preserve the public JSON export shape unless a versioned migration is documented.
- Keep all exported search vectors L2-normalized and 512-dimensional for the default model.
- Export `global`, `cluster_1`, and `cluster_2`. Use PCA only for diagnostics.
- Record the exact model ID and immutable revision in every export and database row.
- Keep model loading lazy so `/health` and the frontend start before weights are downloaded.
- Musical-map axes are relative to the supplied reference collection. Never label them as fixed
  genre, mood, or energy dimensions without a separate supervised calibration.
- Never load a complete recording into RAM. Decode only each sampled window, including for the
  configurable four-hour maximum.
- Telemetry stays disabled or local by default. Never collect filenames, raw audio, vectors,
  IP addresses, user-agent strings, or other identifying data.
- Add or update focused tests for aggregation, sampling, and changed API contracts.

## Commands

```bash
make install
make test
make lint
make run
```

The project expects Python 3.11 or 3.12 and `ffmpeg`/`ffprobe` on `PATH`.
