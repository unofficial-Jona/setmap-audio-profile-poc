import { mountRadar } from "./radar.mjs";

const form = document.querySelector("#upload-form");
const input = document.querySelector("#audio-file");
const dropzone = document.querySelector("#dropzone");
const fileLabel = document.querySelector("#file-label");
const fileMeta = document.querySelector("#file-meta");
const errorBox = document.querySelector("#form-error");
const statusPanel = document.querySelector("#status-panel");
const statusStage = document.querySelector("#status-stage");
const statusPercent = document.querySelector("#status-percent");
const statusMessage = document.querySelector("#status-message");
const progressBar = document.querySelector("#progress-bar");
const resultPanel = document.querySelector("#result-panel");
const mapPanel = document.querySelector("#map-panel");
const profileImport = document.querySelector("#profile-import");
const viewMapButton = document.querySelector("#view-map");
const libraryList = document.querySelector("#library-list");
const libraryEmpty = document.querySelector("#library-empty");
const libraryCount = document.querySelector("#library-count");
const libraryMapButton = document.querySelector("#library-map");
const clearLibraryButton = document.querySelector("#clear-library");
const submitButton = form.querySelector("button[type=submit]");
const palette = ["#63e89b", "#c9ff63", "#6ed9ff", "#ff9d66", "#caa4ff", "#ff7f9d", "#f5d76e", "#78f0db"];
const storageKey = "setmap.profiles.v1";
const seedDismissedKey = "setmap.initial-profiles.dismissed.v1";
const maxStoredProfiles = 80;

const profileId = () => globalThis.crypto?.randomUUID?.() || `profile-${Date.now()}-${Math.random().toString(16).slice(2)}`;
const profileName = (value) => String(value || "Untitled set").replace(/\.(mp3|wav|flac|m4a|aac|ogg|opus)$/i, "");
const prepareProfile = (profile, index = 0) => ({
  id: profile.id || profileId(),
  label: profile.label || `Set ${index + 1}`,
  created_at: profile.created_at || new Date().toISOString(),
  result: profile.result || profile.profile,
});
const profileSignature = (profile) => [
  profileName(profile.label).toLowerCase(),
  Number(profile.result?.source?.duration_seconds || 0).toFixed(2),
  profile.result?.model?.revision || "unknown",
  profile.result?.source?.sample_interval_seconds || "unknown",
  profile.result?.aggregation?.max_outlier_fraction || 0,
].join("::");

function mergeProfiles(records) {
  const known = new Set(completedProfiles.map(profileSignature));
  const added = [];
  records.map(prepareProfile).forEach((profile) => {
    const signature = profileSignature(profile);
    if (!known.has(signature)) {
      completedProfiles.push(profile);
      added.push(profile);
      known.add(signature);
    }
  });
  return added;
}

function loadProfiles() {
  try {
    const stored = JSON.parse(localStorage.getItem(storageKey) || "[]");
    return Array.isArray(stored) ? stored.map(prepareProfile).filter((item) => item.result?.aggregation?.vectors) : [];
  } catch (_) {
    return [];
  }
}

let completedProfiles = loadProfiles();

const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", "\"": "&quot;",
})[character]);

const formatBytes = (bytes) => {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
  return `${value.toFixed(index ? 1 : 0)} ${units[index]}`;
};

const formatDuration = (seconds) => {
  const minutes = Math.round(Number(seconds || 0) / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const remainder = minutes % 60;
  return `${hours}h${remainder ? ` ${remainder}m` : ""}`;
};

function saveProfiles() {
  try {
    completedProfiles = completedProfiles.slice(-maxStoredProfiles);
    localStorage.setItem(storageKey, JSON.stringify(completedProfiles));
    return true;
  } catch (_) {
    errorBox.textContent = "The profile is ready, but browser storage is full. Export the collection JSON to keep it.";
    return false;
  }
}

function downloadProfiles(profiles, filename) {
  const blob = new Blob([JSON.stringify({ schema_version: "batch-1.0", profiles }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function renderLibrary() {
  libraryCount.textContent = String(completedProfiles.length).padStart(2, "0");
  libraryEmpty.hidden = completedProfiles.length > 0;
  libraryMapButton.hidden = completedProfiles.length === 0;
  viewMapButton.hidden = completedProfiles.length === 0;
  clearLibraryButton.hidden = completedProfiles.length === 0;
  libraryList.innerHTML = completedProfiles.map((profile, index) => {
    const source = profile.result.source || {};
    const date = new Date(profile.created_at);
    const dateText = Number.isNaN(date.valueOf()) ? "saved locally" : date.toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" });
    return `<article class="library-card" data-profile-id="${escapeHtml(profile.id)}" style="--card-color:${palette[index % palette.length]}">
      <span class="library-index">${String(index + 1).padStart(2, "0")}</span>
      <button class="library-copy focus-profile" type="button" title="Locate on musical map">
        <b>${escapeHtml(profileName(profile.label))}</b>
        <small>${formatDuration(source.duration_seconds)} · ${source.windows_retained || 0} clips · ${escapeHtml(dateText)}</small>
      </button>
      <span class="library-actions">
        <button class="export-profile" type="button" title="Export profile" aria-label="Export ${escapeHtml(profileName(profile.label))}">↓</button>
        <button class="remove-profile" type="button" title="Remove from this device" aria-label="Remove ${escapeHtml(profileName(profile.label))}">×</button>
      </span>
    </article>`;
  }).join("");
}

const showFiles = () => {
  const files = [...(input.files || [])];
  if (!files.length) return;
  if (files.length === 1) {
    fileLabel.textContent = files[0].name;
    fileMeta.textContent = `${formatBytes(files[0].size)} · ready to profile`;
  } else {
    fileLabel.textContent = `${files.length} sets selected`;
    fileMeta.textContent = `${formatBytes(files.reduce((sum, file) => sum + file.size, 0))} total · processed one at a time`;
  }
};

input.addEventListener("change", showFiles);
["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault(); dropzone.classList.add("dragover");
}));
["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault(); dropzone.classList.remove("dragover");
}));
dropzone.addEventListener("drop", (event) => {
  if (event.dataTransfer.files.length) { input.files = event.dataTransfer.files; showFiles(); }
});

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

mountRadar(document.querySelector(".radar"), statusPanel);

function renderResult(profiles) {
  const retained = profiles.reduce((sum, profile) => sum + profile.result.source.windows_retained, 0);
  const considered = profiles.reduce((sum, profile) => sum + profile.result.source.windows_considered, 0);
  const duration = profiles.reduce((sum, profile) => sum + profile.result.source.duration_seconds, 0);
  const core = profiles.reduce((sum, profile) => sum + profile.result.aggregation.core_clip_count, 0);
  const processingSeconds = profiles.reduce((sum, profile) => sum + profile.result.timing_ms.total, 0) / 1000;
  document.querySelector("#stats").innerHTML = `
    <div class="stat"><strong>${retained}/${considered}</strong><small>windows retained</small></div>
    <div class="stat"><strong>${core}/${retained}</strong><small>aggregation core</small></div>
    <div class="stat"><strong>${(duration / 60).toFixed(1)}m</strong><small>collection duration</small></div>
    <div class="stat"><strong>${processingSeconds.toFixed(1)}s</strong><small>measured CPU pipeline</small></div>`;

  const latest = profiles[profiles.length - 1].result;
  document.querySelector("#vectors").innerHTML = latest.aggregation.vectors.map((item) => {
    const bars = item.vector.slice(0, 64).map((value) => {
      const height = Math.max(2, Math.min(20, Math.abs(value) * 130));
      return `<i style="height:${height}px;opacity:${value < 0 ? 0.35 : 0.9}"></i>`;
    }).join("");
    const sourceText = item.role === "global" ? "all clips" : `${Math.round(item.weight * 100)}%`;
    return `<div class="vector-row"><b>${item.role.replace("_", " ")}</b><div class="vector-preview">${bars}</div><span>${sourceText}</span></div>`;
  }).join("");
  document.querySelector(".result-heading h2").textContent = profiles.length === 1
    ? "Your sound has coordinates."
    : `${profiles.length} new sets joined the map.`;
  resultPanel.hidden = false;
}

const mapPosition = (point) => ({ x: 500 + point.x * 385, y: 280 - point.y * 215 });

async function renderMusicalMap(profiles) {
  const response = await fetch("/api/musical-map", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      profiles: profiles.map((profile, index) => ({
        id: profile.id || String(index),
        label: profile.label,
        vectors: profile.result.aggregation.vectors,
      })),
    }),
  });
  if (!response.ok) throw new Error("The musical map could not be built.");
  const map = await response.json();
  const svg = document.querySelector("#musical-map");
  const axis = `<line class="map-axis" x1="500" y1="20" x2="500" y2="540"/><line class="map-axis" x1="20" y1="280" x2="980" y2="280"/>`;
  const marks = map.profiles.map((profile, index) => {
    const color = palette[index % palette.length];
    const global = mapPosition(profile.points.global);
    const fullLabel = profileName(profile.label);
    const label = escapeHtml(fullLabel.length > 32 ? `${fullLabel.slice(0, 31)}…` : fullLabel);
    const labelOnLeft = global.x > 735;
    const labelX = global.x + (labelOnLeft ? -17 : 17);
    const anchor = labelOnLeft ? "end" : "start";
    return `
      <g class="map-profile" data-profile-id="${escapeHtml(profiles[index].id)}" style="color:${color}">
        <title>${escapeHtml(fullLabel)}</title>
        <circle class="map-halo" stroke="${color}" cx="${global.x}" cy="${global.y}" r="19"/>
        <circle class="map-global" fill="${color}" cx="${global.x}" cy="${global.y}" r="11"/>
        <text class="map-label" text-anchor="${anchor}" x="${labelX}" y="${global.y - 4}">${label}</text>
        <text class="map-weight" text-anchor="${anchor}" x="${labelX}" y="${global.y + 12}">${profile.nearest ? `${Math.round(profile.nearest.similarity * 100)}% nearest` : "reference needed"}</text>
      </g>`;
  }).join("");
  svg.innerHTML = axis + marks;
  document.querySelector("#map-legend").innerHTML = map.profiles.map((profile, index) => {
    const nearest = profile.nearest
      ? `nearest: ${escapeHtml(profile.nearest.label)} · ${(profile.nearest.similarity * 100).toFixed(1)}%`
      : "add another set to find neighbours";
    return `<div class="map-legend-item" data-profile-id="${escapeHtml(profiles[index].id)}"><i class="map-swatch" style="background:${palette[index % palette.length]}"></i><div><b>${escapeHtml(profile.label)}</b><small>${nearest}</small></div></div>`;
  }).join("");
  const projectionKey = document.querySelector("#projection-key");
  const projectionNote = document.querySelector("#projection-note");
  if (map.projection_is_stable) {
    projectionKey.innerHTML = `FROZEN PCA<br />${escapeHtml(map.projection_version)}`;
    projectionNote.textContent = `Positions use ${map.projection_version}, fitted once on ${map.reference_profile_count} reference profiles. Adding a set does not move existing points.`;
  } else {
    projectionKey.innerHTML = "SESSION PCA<br />NOT FROZEN";
    projectionNote.textContent = "No reference projection is configured, so positions are relative to this collection and can move when it changes.";
  }
  mapPanel.hidden = false;
  viewMapButton.hidden = false;
}

function focusProfile(id) {
  document.querySelectorAll("[data-profile-id]").forEach((element) => {
    element.classList.toggle("is-focused", element.dataset.profileId === id);
  });
}

async function openCollectionMap(focusId) {
  if (!completedProfiles.length) return;
  try {
    await renderMusicalMap(completedProfiles);
    if (focusId) focusProfile(focusId);
    mapPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    errorBox.textContent = error.message;
  }
}

viewMapButton.addEventListener("click", () => openCollectionMap());
libraryMapButton.addEventListener("click", () => openCollectionMap());
clearLibraryButton.addEventListener("click", () => {
  if (!window.confirm("Clear every locally saved profile? Model weights will stay cached.")) return;
  completedProfiles = [];
  localStorage.removeItem(storageKey);
  localStorage.setItem(seedDismissedKey, "1");
  renderLibrary();
  mapPanel.hidden = true;
  resultPanel.hidden = true;
  errorBox.textContent = "Local profiles cleared. You can process the source audio again now.";
});
libraryList.addEventListener("click", async (event) => {
  const card = event.target.closest(".library-card");
  if (!card) return;
  const profile = completedProfiles.find((item) => item.id === card.dataset.profileId);
  if (!profile) return;
  if (event.target.closest(".remove-profile")) {
    completedProfiles = completedProfiles.filter((item) => item.id !== profile.id);
    localStorage.setItem(seedDismissedKey, "1");
    saveProfiles();
    renderLibrary();
    if (completedProfiles.length) await renderMusicalMap(completedProfiles);
    else mapPanel.hidden = true;
    return;
  }
  if (event.target.closest(".export-profile")) {
    downloadProfiles([profile], `setmap-${profileName(profile.label).replace(/[^a-z0-9]+/gi, "-").toLowerCase() || "profile"}.json`);
    return;
  }
  await openCollectionMap(profile.id);
});

async function pollJob(statusUrl, profileIndex, profileCount, label) {
  while (true) {
    const response = await fetch(statusUrl);
    if (!response.ok) throw new Error("Could not read the processing status.");
    const job = await response.json();
    const overallProgress = (profileIndex + job.progress) / profileCount;
    statusStage.textContent = `${job.stage.toUpperCase()} · ${profileIndex + 1}/${profileCount}`;
    statusMessage.textContent = `${label} · ${job.message}`;
    statusPercent.textContent = `${Math.round(overallProgress * 100)}%`;
    progressBar.style.width = `${overallProgress * 100}%`;
    if (job.status === "complete") return job.result;
    if (job.status === "failed") throw new Error(`${label}: ${job.error || "Processing failed."}`);
    await sleep(1000);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.textContent = "";
  resultPanel.hidden = true;
  mapPanel.hidden = true;
  const files = [...(input.files || [])];
  if (!files.length) { errorBox.textContent = "Choose at least one audio file first."; return; }
  submitButton.disabled = true;
  statusPanel.hidden = false;
  const interval = form.querySelector("input[name=interval_seconds]:checked").value;
  const outlierFraction = form.querySelector("input[name=embedding_outlier_max_fraction]:checked").value;
  const newProfiles = [];
  try {
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      statusStage.textContent = `UPLOADING · ${index + 1}/${files.length}`;
      statusMessage.textContent = `${file.name} · copying to a temporary local file…`;
      const data = new FormData();
      data.append("audio", file);
      data.append("interval_seconds", interval);
      data.append("embedding_outlier_max_fraction", outlierFraction);
      const response = await fetch("/api/jobs", { method: "POST", body: data });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `${file.name}: upload failed.`);
      const result = await pollJob(payload.status_url, index, files.length, file.name);
      newProfiles.push(prepareProfile({ label: file.name, result }));
    }
    mergeProfiles(newProfiles);
    saveProfiles();
    renderLibrary();
    renderResult(newProfiles);
    await renderMusicalMap(completedProfiles);
    statusPanel.hidden = true;
    mapPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    errorBox.textContent = error.message;
    statusPanel.hidden = true;
  } finally {
    submitButton.disabled = false;
  }
});

document.querySelector("#download-json").addEventListener("click", () => {
  if (!completedProfiles.length) return;
  downloadProfiles(completedProfiles, `setmap-profiles-${new Date().toISOString().slice(0, 10)}.json`);
});

profileImport.addEventListener("change", async () => {
  const file = profileImport.files?.[0];
  if (!file) return;
  errorBox.textContent = "";
  try {
    const payload = JSON.parse(await file.text());
    const records = payload.profiles || [];
    const importedProfiles = records.map(prepareProfile);
    if (!importedProfiles.length || importedProfiles.some((record) => !record.result?.aggregation?.vectors)) {
      throw new Error("This JSON does not contain exported music profiles.");
    }
    mergeProfiles(importedProfiles);
    saveProfiles();
    renderLibrary();
    renderResult(importedProfiles);
    await renderMusicalMap(completedProfiles);
    mapPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    errorBox.textContent = `Import failed: ${error.message}`;
  } finally {
    profileImport.value = "";
  }
});

fetch("/health").then((response) => response.json()).then((health) => {
  if (health.backend === "fake") {
    const pill = document.querySelector("#runtime-pill");
    pill.innerHTML = "<i></i> DEMO MODE · FAKE VECTORS";
    pill.classList.add("demo");
  }
}).catch(() => {});

async function hydrateInitialProfiles() {
  try {
    if (localStorage.getItem(seedDismissedKey) === "1") return;
    const response = await fetch("/api/initial-profiles");
    if (!response.ok) return;
    const payload = await response.json();
    if (mergeProfiles(payload.profiles || []).length) saveProfiles();
  } catch (_) {
    // The optional local seed is non-critical; manual import remains available.
  } finally {
    renderLibrary();
  }
}

renderLibrary();
hydrateInitialProfiles();
