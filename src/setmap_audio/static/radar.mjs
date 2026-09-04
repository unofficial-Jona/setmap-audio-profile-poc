const REVOLUTION_MS = 2800;
const TAU = Math.PI * 2;

// Phase zero is twelve o'clock, increasing clockwise, matching the CSS beam.
export function planSweep(turn, random = Math.random) {
  return Array.from({ length: 2 + Math.floor(random() * 4) }, () => {
    const phase = 0.025 + random() * 0.95;
    const radius = Math.sqrt(0.08 + random() * 0.72) * 46;
    return {
      detectedAt: turn + phase,
      left: 50 + Math.sin(phase * TAU) * radius,
      top: 50 - Math.cos(phase * TAU) * radius,
      size: 3.5 + random() * 2,
    };
  });
}

export function detectionAt(target, turns) {
  const age = turns - target.detectedAt;
  if (age < 0 || age >= 1) return { opacity: 0, glow: 0, expired: age >= 1 };
  const flash = Math.exp(-age * 14);
  return { opacity: 0.12 + 0.88 * Math.exp(-age * 5), glow: 2 + 10 * flash, expired: false };
}

export function mountRadar(radar, panel) {
  const beam = radar.querySelector(".radar-sweep");
  const layer = radar.querySelector(".radar-detections");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let frame = null;
  let previousTime = null;
  let elapsed = 0;
  let generatedTurn = -1;
  let targets = [];

  function render(turns) {
    const turn = Math.floor(turns);
    // Keep the previous revolution's detections until their own next beam crossing.
    while (generatedTurn < turn) {
      generatedTurn += 1;
      for (const target of planSweep(generatedTurn)) {
        const dot = document.createElement("b");
        dot.className = "radar-dot";
        dot.style.left = `${target.left}%`;
        dot.style.top = `${target.top}%`;
        dot.style.width = dot.style.height = `${target.size}px`;
        layer.append(dot);
        targets.push({ ...target, dot });
      }
    }
    beam.style.transform = `rotate(${(turns % 1) * 360}deg)`;
    targets = targets.filter((target) => {
      const state = detectionAt(target, turns);
      if (state.expired) { target.dot.remove(); return false; }
      target.dot.style.opacity = String(state.opacity);
      target.dot.style.boxShadow = `0 0 ${state.glow}px var(--mint)`;
      return true;
    });
  }

  function tick(now) {
    if (previousTime !== null) elapsed += now - previousTime;
    previousTime = now;
    render(elapsed / REVOLUTION_MS);
    frame = requestAnimationFrame(tick);
  }

  function sync() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    previousTime = null;
    if (panel.hidden) {
      elapsed = 0;
      generatedTurn = -1;
      targets = [];
      layer.replaceChildren();
      beam.style.transform = "rotate(0deg)";
      return;
    }
    if (reducedMotion.matches) {
      // A stationary instrument when reduced motion is requested.
      layer.replaceChildren();
      targets = [];
      elapsed = 0;
      generatedTurn = -1;
      beam.style.transform = "rotate(0deg)";
      return;
    }
    if (!document.hidden) frame = requestAnimationFrame(tick);
  }

  const observer = new MutationObserver(sync);
  observer.observe(panel, { attributes: true, attributeFilter: ["hidden"] });
  document.addEventListener("visibilitychange", sync);
  reducedMotion.addEventListener("change", sync);
  sync();
}
