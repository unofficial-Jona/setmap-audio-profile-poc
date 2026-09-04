import test from "node:test";
import assert from "node:assert/strict";
import { planSweep, detectionAt } from "../src/setmap_audio/static/radar.mjs";

test("a detection lights on beam contact, fades, and disappears on the next contact", () => {
  const target = { detectedAt: 0.25 };
  assert.equal(detectionAt(target, 0.249).opacity, 0);
  assert.equal(detectionAt(target, 0.25).opacity, 1);
  assert.ok(detectionAt(target, 0.5).opacity < detectionAt(target, 0.3).opacity);
  assert.ok(detectionAt(target, 1.249).opacity > 0);
  assert.equal(detectionAt(target, 1.249).expired, false);
  assert.deepEqual(detectionAt(target, 1.25), { opacity: 0, glow: 0, expired: true });
  assert.equal(detectionAt(target, 2.25).opacity, 0);
});

test("detections survive the twelve-o'clock boundary and expire at their own angle", () => {
  const old = { detectedAt: 0.8 };
  const fresh = { detectedAt: 1.2 };
  assert.ok(detectionAt(old, 1.1).opacity > 0);
  assert.equal(detectionAt(fresh, 1.1).opacity, 0);
  assert.ok(detectionAt(fresh, 1.3).opacity > 0);
  assert.equal(detectionAt(old, 1.8).expired, true);
  assert.ok(detectionAt(fresh, 1.8).opacity > 0);
});

test("random target locations agree with clockwise beam phase and stay inside the dial", () => {
  let seed = 17;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const first = planSweep(0, random);
  const second = planSweep(1, random);
  assert.notDeepEqual(first.map(({ left, top }) => [left, top]), second.map(({ left, top }) => [left, top]));
  for (const target of [...first, ...second]) {
    const radius = Math.hypot(target.left - 50, target.top - 50);
    const angle = (Math.atan2(target.left - 50, 50 - target.top) / (Math.PI * 2) + 1) % 1;
    assert.ok(radius < 46);
    assert.ok(Math.abs(angle - target.detectedAt % 1) < 1e-12);
  }
});
