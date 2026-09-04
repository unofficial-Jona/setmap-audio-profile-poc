-- Supabase/PostgreSQL schema for SetMap profiles.
-- Run once in the Supabase SQL editor.
create extension if not exists vector with schema extensions;

create table if not exists public.artist_audio_profiles (
  id uuid primary key default gen_random_uuid(),
  artist_id uuid,
  schema_version text not null default '1.0',
  model_id text not null,
  model_revision text not null,
  dimensions integer not null check (dimensions = 512),
  duration_seconds double precision not null check (duration_seconds > 0),
  windows_considered integer not null check (windows_considered > 0),
  windows_retained integer not null check (windows_retained > 0),
  windows_filtered integer not null check (windows_filtered >= 0),
  sample_interval_seconds integer not null,
  window_seconds real not null,
  aggregation_method text not null,
  aggregation_metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.artist_audio_vectors (
  id bigint generated always as identity primary key,
  profile_id uuid not null references public.artist_audio_profiles(id) on delete cascade,
  role text not null check (role in ('global', 'cluster_1', 'cluster_2')),
  weight real not null check (weight >= 0 and weight <= 1),
  embedding extensions.vector(512) not null,
  model_id text not null,
  model_revision text not null,
  aggregation_method text not null,
  aggregation_version text not null,
  window_seconds real not null,
  sample_interval_seconds integer not null,
  source_duration_seconds double precision not null,
  metadata jsonb not null default '{}'::jsonb,
  unique (profile_id, role)
);

create index if not exists artist_audio_vectors_embedding_hnsw
  on public.artist_audio_vectors
  using hnsw (embedding extensions.vector_cosine_ops);

create or replace function public.match_artist_audio(
  query_embedding extensions.vector(512),
  match_count integer default 20
)
returns table (profile_id uuid, role text, similarity double precision)
language sql stable
as $$
  select
    v.profile_id,
    v.role,
    1 - (v.embedding <=> query_embedding) as similarity
  from public.artist_audio_vectors v
  order by v.embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;

-- Global-first retrieval (replace the shortened vector with all 512 values):
-- select v.profile_id, 1 - (v.embedding <=> '[0.1,-0.2,...]'::extensions.vector(512)) similarity
-- from public.artist_audio_vectors v
-- where v.role = 'global'
-- order by v.embedding <=> '[0.1,-0.2,...]'::extensions.vector(512)
-- limit 20;
