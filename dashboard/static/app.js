// Dashboard frontend: poll /api/live and render three cards, all computed from
// the real dataset (no benchmark JSONs). Chart.js is loaded locally (see
// index.html). Charts are created once and then updated in place on each poll,
// so a 4s refresh never leaks canvases. Every card degrades to a "no data yet"
// placeholder if its slice is missing, so an empty dataset still renders.

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

// Chart instances, created lazily on first data and reused thereafter.
const charts = {};

function pct(x) {
  return (x * 100).toFixed(1) + "%";
}

function markMissing(cardId, msg) {
  const card = document.getElementById(cardId);
  if (!card || card.querySelector(".placeholder")) return;
  card.classList.add("missing");
  const p = document.createElement("p");
  p.className = "placeholder";
  p.textContent = msg || "No data yet — write a memory to populate this.";
  card.appendChild(p);
}

// Create the chart on first call, then just swap data + redraw on later calls.
function upsertChart(key, canvasId, config) {
  if (charts[key]) {
    charts[key].data = config.data;
    charts[key].update();
    return charts[key];
  }
  charts[key] = new Chart(document.getElementById(canvasId), config);
  return charts[key];
}

function renderLiveHeader(facts) {
  if (!facts) return;
  document.getElementById("live-total").textContent = facts.total_current;
  document.getElementById("live-hot").textContent = facts.by_tier.hot;
  document.getElementById("live-warm").textContent = facts.by_tier.warm;
  document.getElementById("live-cold").textContent = facts.by_tier.cold;
}

function renderTokens(tokens) {
  if (!tokens) return markMissing("card-tokens");
  document.getElementById("tokens-headline").textContent = pct(tokens.pct_saved);
  document.getElementById("tokens-caption").textContent =
    `${tokens.tokens_saved} of ${tokens.raw_history_tokens} estimated tokens saved ` +
    `across ${tokens.n_writes} writes (est. ~4 chars/token)`;

  upsertChart("tokens", "tokens-chart", {
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

function renderRevisions(rev) {
  if (!rev) return markMissing("card-revisions");
  document.getElementById("revisions-headline").textContent = rev.revisions;
  document.getElementById("revisions-caption").textContent =
    `in-place updates the resolver made · ${rev.evicted} evicted · ${rev.current} current facts`;

  const t = rev.by_tier || { hot: 0, warm: 0, cold: 0 };
  upsertChart("tiers", "tiers-chart", {
    type: "bar",
    data: {
      labels: ["Hot", "Warm", "Cold"],
      datasets: [
        {
          label: "Current facts by tier",
          data: [t.hot, t.warm, t.cold],
          backgroundColor: [COLORS.danger, COLORS.warn, COLORS.accent],
          borderRadius: 4,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: COLORS.grid } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderStorage(storage) {
  if (!storage || !storage.series || !storage.series.length) {
    return markMissing("card-storage");
  }
  const series = storage.series;
  upsertChart("storage", "storage-chart", {
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
        y: {
          beginAtZero: true,
          ticks: { precision: 0 },
          grid: { color: COLORS.grid },
          title: { display: true, text: "facts stored" },
        },
        x: {
          grid: { display: false },
          title: { display: true, text: "writes over time" },
        },
      },
    },
  });
}

// Poll interval for the whole dashboard (ms). Everything here is live now.
const LIVE_POLL_MS = 4000;

// Fetch the live payload and refresh every card. Silently keeps the
// last-known-good render on a transient fetch error so a blip doesn't clear
// the charts.
async function loadLive() {
  let data;
  try {
    data = await fetch("/api/live").then((r) => r.json());
  } catch (err) {
    return; // keep last-known-good on a transient failure
  }
  renderLiveHeader(data.facts);
  renderTokens(data.tokens);
  renderRevisions(data.revisions);
  renderStorage(data.storage);
}

loadLive();
setInterval(loadLive, LIVE_POLL_MS);
