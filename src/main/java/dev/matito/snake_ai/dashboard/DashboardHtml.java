package dev.matito.snake_ai.dashboard;

public final class DashboardHtml {
	private DashboardHtml() {}

	public static final String PAGE = """
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Snake AI Dashboard</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 16px; }
    .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
    .k { color: #555; }
    pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td { border: 1px solid #eee; padding: 4px 6px; }
    th { background: #fafafa; text-align: left; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    button, select { padding: 6px 10px; border-radius: 8px; border: 1px solid #ddd; background: #fff; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: default; }
    .badge { display: inline-block; padding: 1px 7px; border: 1px solid #ddd; border-radius: 999px; margin: 0 4px 4px 0; font-size: 11px; }
    .badge.on { background: #111; color: #fff; border-color: #111; }
    .small { font-size: 12px; }
    .full { grid-column: 1 / -1; }
    .canvasWrap { width: 100%; overflow: auto; }
    canvas { border: 1px solid #eee; border-radius: 10px; }
  </style>
</head>
<body>
  <div class="grid">
    <div class="card">
      <div><b>Status</b></div>
      <div class="k">Highscore: <span id="hs">0</span></div>
      <div class="k">Score: <span id="cs">0</span></div>
      <div class="k">Runden pro Sekunde: <span id="eps">0</span></div>
      <div class="k">Steps pro Sekunde: <span id="sps">0</span></div>
      <div class="k">Runden: <span id="epc">0</span></div>
      <div class="k">Steps: <span id="stc">0</span></div>
    </div>

    <div class="card">
      <div><b>Config</b></div>
      <pre id="cfg">...</pre>
    </div>

    <div class="card">
      <div><b>Agent</b> <span id="atype"></span></div>
      <div class="k">alpha: <span id="alpha"></span></div>
      <div class="k">gamma: <span id="gamma"></span></div>
      <div class="k">epsilon: <span id="e"></span></div>
      <div class="k">epsilonMin: <span id="emin"></span></div>
      <div class="k">Replay: <span id="replay"></span></div>
      <div class="k">Last Q: <span id="lastq"></span></div>
      <div class="k">Last State: <span id="lasts"></span></div>
      <div class="k">Last Action: <span id="lasta"></span></div>
      <div class="k">Q Row: <span id="qrow"></span></div>
    </div>

    <div class="card">
      <div><b>QTable</b> (nur QL)</div>
      <div class="row">
        <button id="prev">Prev</button>
        <button id="next">Next</button>
        <span class="k">Start: <span id="qstart">0</span></span>
      </div>
      <div style="max-height: 420px; overflow: auto; margin-top: 8px;">
        <table id="qt"></table>
      </div>
      <div class="k small" style="margin-top: 8px;">
        State Bits: danger S L R, food U R D L, dir U R D L
      </div>
    </div>

    <div class="card full">
      <div class="row">
        <div><b>NN Online</b> (nur DQN)</div>
        <select id="layerSel">
          <option value="full">Full</option>
          <option value="w1">W1</option>
          <option value="w2">W2</option>
          <option value="w3">W3</option>
        </select>
        <button id="nnPrev">Prev</button>
        <button id="nnNext">Next</button>
        <span class="k small">H1 <span id="h1pos"></span>, H2 <span id="h2pos"></span></span>
      </div>
      <div class="canvasWrap" style="margin-top: 8px;">
        <canvas id="nnOnline" width="1400" height="620"></canvas>
      </div>
      <div class="k small" style="margin-top: 8px;">
        Slice default: Inputs 0..5, H1 0..7, H2 0..7. Weights stehen auf den Linien, Bias in den Nodes.
      </div>
    </div>

    <div class="card full">
      <div><b>NN Target</b> (nur DQN)</div>
      <div class="canvasWrap" style="margin-top: 8px;">
        <canvas id="nnTarget" width="1400" height="620"></canvas>
      </div>
    </div>
  </div>

<script>
const $ = id => document.getElementById(id);

let qStart = 0;
const qLimit = 128;

const nn = {
  inStart: 0, inCount: 6,
  h1Start: 0, h1Count: 8,
  h2Start: 0, h2Count: 8,
  layer: "full"
};

const inLabels = [
  "dangerS","dangerL","dangerR",
  "dirU","dirR","dirD","dirL",
  "foodU","foodR","foodD","foodL"
];
const outLabels = ["straight","left","right"];

async function j(url) {
  const r = await fetch(url, { cache: "no-store" });
  return await r.json();
}

function pretty(obj) { return JSON.stringify(obj, null, 2); }

async function refreshStatus() {
  const s = await j("/api/status");

  $("hs").textContent = s.metrics?.highscore ?? 0;
  $("cs").textContent = s.metrics?.currentScore ?? 0;
  $("eps").textContent = (s.metrics?.episodesPerSec ?? 0).toFixed(2);
  $("sps").textContent = (s.metrics?.stepsPerSec ?? 0).toFixed(1);
  $("epc").textContent = s.metrics?.episodes ?? 0;
  $("stc").textContent = s.metrics?.steps ?? 0;

  $("cfg").textContent = pretty(s.config ?? {});
  $("atype").textContent = s.agent?.type ?? "unknown";
  $("alpha").textContent = s.agent?.alpha ?? "";
  $("gamma").textContent = s.agent?.gamma ?? "";
  $("e").textContent = s.agent?.epsilon ?? "";
  $("emin").textContent = s.agent?.epsilonMin ?? "";

  $("replay").textContent = s.agent?.replaySize ?? "";
  $("lastq").textContent = JSON.stringify(s.agent?.lastQ ?? []);
  $("lasts").textContent = s.agent?.lastStateId ?? "";
  $("lasta").textContent = s.agent?.lastAction ?? "";
  $("qrow").textContent = JSON.stringify(s.agent?.qRow ?? []);
}

function bit(x, i) { return ((x >> i) & 1) === 1; }
function badge(on, label) { return "<span class='badge" + (on ? " on" : "") + "'>" + label + "</span>"; }

function decodeQLState(id) {
  return {
    dS: bit(id,0), dL: bit(id,1), dR: bit(id,2),
    fU: bit(id,3), fR: bit(id,4), fD: bit(id,5), fL: bit(id,6),
    dirU: bit(id,7), dirR: bit(id,8), dirD: bit(id,9), dirL: bit(id,10)
  };
}

function renderQTable(rows, start) {
  const t = $("qt");
  t.innerHTML = "";
  const head = document.createElement("tr");
  head.innerHTML =
    "<th>state</th><th>danger</th><th>food</th><th>dir</th>" +
    "<th class='num'>straight</th><th class='num'>left</th><th class='num'>right</th>";
  t.appendChild(head);

  for (let i = 0; i < rows.length; i++) {
    const id = start + i;
    const s = decodeQLState(id);

    const danger = badge(s.dS,"S") + badge(s.dL,"L") + badge(s.dR,"R");
    const food = badge(s.fU,"U") + badge(s.fR,"R") + badge(s.fD,"D") + badge(s.fL,"L");
    const dir = badge(s.dirU,"U") + badge(s.dirR,"R") + badge(s.dirD,"D") + badge(s.dirL,"L");

    const tr = document.createElement("tr");
    tr.innerHTML =
      "<td class='num k'>" + id + "</td>" +
      "<td>" + danger + "</td>" +
      "<td>" + food + "</td>" +
      "<td>" + dir + "</td>" +
      "<td class='num'>" + rows[i][0].toFixed(6) + "</td>" +
      "<td class='num'>" + rows[i][1].toFixed(6) + "</td>" +
      "<td class='num'>" + rows[i][2].toFixed(6) + "</td>";
    t.appendChild(tr);
  }
}

async function refreshQTable() {
  $("qstart").textContent = qStart;
  try {
    const r = await j(`/api/qtable?start=${qStart}&limit=${qLimit}`);
    renderQTable(r.rows ?? [], r.start ?? 0);
    $("prev").disabled = qStart <= 0;
    $("next").disabled = (qStart + qLimit) >= (r.states ?? 2048);
  } catch (e) {
    $("qt").innerHTML = "<tr><td class='k'>QTable nicht verfuegbar</td></tr>";
  }
}

async function fetchSlice(net) {
  const url =
    `/api/nn/slice?net=${encodeURIComponent(net)}` +
    `&inStart=${nn.inStart}&inCount=${nn.inCount}` +
    `&h1Start=${nn.h1Start}&h1Count=${nn.h1Count}` +
    `&h2Start=${nn.h2Start}&h2Count=${nn.h2Count}`;
  return await j(url);
}

function drawNet(canvas, data, layerMode) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!data || data.error) {
    ctx.font = "14px system-ui";
    ctx.fillText("NN nicht verfuegbar", 16, 24);
    return;
  }

  const s = data.slice;
  const inN = s.inCount, h1N = s.h1Count, h2N = s.h2Count, outN = 3;

  const w1 = data.w1 || [];
  const w2 = data.w2 || [];
  const w3 = data.w3 || [];
  const b1 = data.b1 || [];
  const b2 = data.b2 || [];
  const b3 = data.b3 || [];

    const W = canvas.width;
	const H = canvas.height;
	
	const padX = Math.max(90, Math.floor(W * 0.06));
	const padY = 32;
	
	const xs = [
	  padX,
	  Math.floor(W * 0.33),
	  Math.floor(W * 0.66),
	  W - padX
	];
	
	const top = padY;
	const bot = H - padY;
			

  function yPos(i, n) {
    if (n <= 1) return (top + bot) / 2;
    return top + (bot - top) * (i / (n - 1));
  }

  function node(x, y, label, bias) {
    ctx.beginPath();
    ctx.arc(x, y, 12, 0, Math.PI * 2);
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.font = "11px system-ui";
    ctx.fillStyle = "#111";
    ctx.fillText(label, x + 16, y + 4);

    if (bias !== null && bias !== undefined) {
      ctx.fillStyle = "#555";
      ctx.font = "10px system-ui";
      ctx.fillText("b=" + bias.toFixed(3), x + 16, y + 16);
    }
  }

  function edge(x1,y1,x2,y2,w,showNum) {
    const a = Math.abs(w);
    const width = Math.max(0.6, Math.min(4.0, a * 3.0));
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = w >= 0 ? "rgba(0,90,200,0.55)" : "rgba(200,40,40,0.55)";
    ctx.lineWidth = width;
    ctx.stroke();

    if (showNum) {
      const mx = (x1 + x2) / 2;
      const my = (y1 + y2) / 2;
      ctx.fillStyle = "#111";
      ctx.font = "9px system-ui";
      ctx.fillText(w.toFixed(2), mx + 2, my + 2);
    }
  }

  const showW1 = (layerMode === "full" || layerMode === "w1");
  const showW2 = (layerMode === "full" || layerMode === "w2");
  const showW3 = (layerMode === "full" || layerMode === "w3");

  // Weights auf Linien: hier immer anzeigen, aber nur fuer die sichtbare Slice (klein gehalten).
  const showNumsW1 = showW1;
  const showNumsW2 = showW2;
  const showNumsW3 = showW3;

  if (showW1) {
    for (let i = 0; i < h1N; i++) {
      const y2 = yPos(i, h1N);
      for (let j = 0; j < inN; j++) {
        const y1 = yPos(j, inN);
        edge(xs[0], y1, xs[1], y2, (w1[i]?.[j] ?? 0), showNumsW1);
      }
    }
  }
  if (showW2) {
    for (let i = 0; i < h2N; i++) {
      const y2 = yPos(i, h2N);
      for (let j = 0; j < h1N; j++) {
        const y1 = yPos(j, h1N);
        edge(xs[1], y1, xs[2], y2, (w2[i]?.[j] ?? 0), showNumsW2);
      }
    }
  }
  if (showW3) {
    for (let i = 0; i < outN; i++) {
      const y2 = yPos(i, outN);
      for (let j = 0; j < h2N; j++) {
        const y1 = yPos(j, h2N);
        edge(xs[2], y1, xs[3], y2, (w3[i]?.[j] ?? 0), showNumsW3);
      }
    }
  }

  for (let j = 0; j < inN; j++) {
    const idx = s.inStart + j;
    const label = (idx >= 0 && idx < 11) ? inLabels[idx] : ("in" + idx);
    node(xs[0], yPos(j, inN), label, null);
  }
  for (let i = 0; i < h1N; i++) node(xs[1], yPos(i, h1N), "h1" + (s.h1Start + i), b1[i] ?? 0);
  for (let i = 0; i < h2N; i++) node(xs[2], yPos(i, h2N), "h2" + (s.h2Start + i), b2[i] ?? 0);
  for (let i = 0; i < outN; i++) node(xs[3], yPos(i, outN), outLabels[i], b3[i] ?? 0);

  ctx.fillStyle = "#555";
  ctx.font = "11px system-ui";
  ctx.fillText("net=" + (data.net || "?") + "   layer=" + layerMode, 16, canvas.height - 10);
}

async function refreshNN() {
  $("h1pos").textContent = nn.h1Start + ".." + (nn.h1Start + nn.h1Count - 1);
  $("h2pos").textContent = nn.h2Start + ".." + (nn.h2Start + nn.h2Count - 1);

  try {
    const online = await fetchSlice("online");
    drawNet($("nnOnline"), online, nn.layer);
  } catch (_) {}

  try {
    const target = await fetchSlice("target");
    drawNet($("nnTarget"), target, nn.layer);
  } catch (_) {}
}

$("prev").onclick = () => { qStart = Math.max(0, qStart - qLimit); refreshQTable(); };
$("next").onclick = () => { qStart = qStart + qLimit; refreshQTable(); };

$("layerSel").onchange = (e) => { nn.layer = e.target.value; refreshNN(); };

$("nnPrev").onclick = () => {
  nn.h1Start = Math.max(0, nn.h1Start - nn.h1Count);
  nn.h2Start = Math.max(0, nn.h2Start - nn.h2Count);
  refreshNN();
};
$("nnNext").onclick = () => {
  nn.h1Start = nn.h1Start + nn.h1Count;
  nn.h2Start = nn.h2Start + nn.h2Count;
  refreshNN();
};

setInterval(refreshStatus, 500);
// Live genug, ohne das Training zu nerven:
setInterval(refreshNN, 2000);

refreshStatus();
refreshQTable();
refreshNN();
</script>
</body>
</html>
""";
}
