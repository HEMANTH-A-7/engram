// Dashboard frontend: fetch /api/results + /api/stats and render five cards.
// Chart.js is loaded locally (see index.html). Every card degrades gracefully
// to a "metric not generated yet" placeholder if its backing JSON is missing,
// so a fresh checkout (no benchmark run yet) still renders without errors.

const COLORS = {
  accent: "#4dabf7",
  accent2: "#51cf66",
  warn: "#ffa94d",
  danger: "#ff6b6b",
  grid: "rgba(139, 152, 165, 0.15)",
  text: "#8b98a5",
};

Chart.defaults.color = COLORS.text;
Chart.defaults.font.family =
  "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

function markMissing(cardId, msg) {
  const card = document.getElementById(cardId);
  if (!card) return;
  card.classList.add("missing");
  const p = document.createElement("p");
  p.className = "placeholder";
  p.textContent = msg || "Metric not generated yet — run the benchmark harness.";
  card.appendChild(p);
}

function pct(x) {
  return (x * 100).toFixed(1) + "%";
}

function renderTokens(tokens) {
  if (!tokens) return markMissing("card-tokens");
  document.getElementById("tokens-headline").textContent = pct(tokens.pct_saved);
  document.getElementById("tokens-caption").textContent =
    `${tokens.tokens_saved} of ${tokens.raw_history_tokens} estimated tokens saved (est. ~4 chars/token)`;

  new Chart(document.getElementById("tokens-chart"), {
    type: "bar",
    data: {
      labels: ["Raw history", "Memory layer"],
      datasets: [
        {
          label: "Estimated tokens",
          data: [tokens.raw_history_tokens, tokens.extended_tokens],
          backgroundColor: [COLORS.danger, COLORS.accent2],
          borderRadius: 4,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: COLORS.grid } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderConflict(conflict) {
  if (!conflict) return markMissing("card-conflict");
  new Chart(document.getElementById("conflict-chart"), {
    type: "bar",
    data: {
      labels: ["Accuracy", "Stale-leak rate"],
      datasets: [
        {
          label: "Resolver (extended)",
          data: [conflict.resolver.accuracy, conflict.resolver.stale_leak_rate],
          backgroundColor: COLORS.accent2,
          borderRadius: 4,
        },
        {
          label: "Naive overwrite (baseline)",
          data: [conflict.naive_overwrite.accuracy, conflict.naive_overwrite.stale_leak_rate],
          backgroundColor: COLORS.warn,
          borderRadius: 4,
        },
      ],
    },
    options: {
      plugins: { legend: { position: "bottom" } },
      scales: {
        y: { beginAtZero: true, max: 1, grid: { color: COLORS.grid } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderCost(cost) {
  if (!cost) return markMissing("card-cost");
  const report = cost.cost_report || {};
  const perK = report.usd_per_1000_memories;
  document.getElementById("cost-headline").textContent =
    perK == null ? "—" : "$" + perK.toFixed(4);
  document.getElementById("cost-caption").textContent =
    `Cascade savings vs. large-only: $${(cost.cascade_savings_usd ?? 0).toFixed(6)} · ` +
    `escalated to large: ${pct(cost.escalated_to_large_rate ?? 0)}`;
  document.getElementById("cost-note").textContent = report.pricing_note || "";
}

function renderRegret(forgetting) {
  if (!forgetting) return markMissing("card-regret");
  const el = document.getElementById("regret-headline");
  el.textContent = pct(forgetting.regret_rate);
  // High regret here is by construction (see note), so warn rather than danger.
  el.style.color = forgetting.regret_rate > 0.25 ? COLORS.warn : COLORS.accent2;
  const n = forgetting.evicted_count ?? forgetting.n_cases ?? 0;
  const note = document.getElementById("regret-note");
  if (note) {
    note.textContent =
      `Synthetic stress test (n=${n}): every evicted fact is deliberately ` +
      `re-requested, so 100% confirms the revival path fires — not a ` +
      `production regret signal.`;
  }
}

function renderStorage(storage) {
  if (!storage || !storage.series) return markMissing("card-storage");
  const series = storage.series;
  new Chart(document.getElementById("storage-chart"), {
    type: "line",
    data: {
      labels: series.map((p) => p.facts_ingested),
      datasets: [
        {
          label: "Raw history (no memory layer)",
          data: series.map((p) => p.without),
          borderColor: COLORS.danger,
          backgroundColor: "transparent",
          tension: 0.2,
        },
        {
          label: "Memory layer (versioned)",
          data: series.map((p) => p.with_),
          borderColor: COLORS.accent2,
          backgroundColor: "transparent",
          tension: 0.2,
        },
      ],
    },
    options: {
      plugins: { legend: { position: "bottom" } },
      scales: {
        y: { beginAtZero: true, grid: { color: COLORS.grid }, title: { display: true, text: "facts stored" } },
        x: { grid: { display: false }, title: { display: true, text: "facts ingested" } },
      },
    },
  });
}

function renderLive(stats) {
  if (!stats || !stats.facts) return;
  const f = stats.facts;
  document.getElementById("live-total").textContent = f.total_current;
  document.getElementById("live-hot").textContent = f.by_tier.hot;
  document.getElementById("live-warm").textContent = f.by_tier.warm;
  document.getElementById("live-cold").textContent = f.by_tier.cold;
}

async function load() {
  try {
    const [results, stats] = await Promise.all([
      fetch("/api/results").then((r) => r.json()),
      fetch("/api/stats").then((r) => r.json()).catch(() => null),
    ]);

    renderTokens(results.tokens);
    renderConflict(results.conflict);
    renderCost(results.cost);
    renderRegret(results.forgetting);
    renderStorage(results.storage);
    renderLive(stats);
  } catch (err) {
    document.getElementById("footer-note").textContent =
      "Failed to load dashboard data: " + err;
  }
}

load();
