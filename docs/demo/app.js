/* CruxRNN's offline viewer. Measured values come only from generated data.js. */
(() => {
  "use strict";

  const source = window.CRUX_DATA || {};
  const data = {
    main: Array.isArray(source.main) ? source.main : [],
    variants: Array.isArray(source.variants) ? source.variants : [],
    curves: Array.isArray(source.curves) ? source.curves : [],
    examples: Array.isArray(source.examples) ? source.examples : [],
    provenance: source.provenance || {},
  };
  const colors = { blocked: "#006493", interleaved: "#b94a08" };
  const names = { rnn: "Elman RNN", gru: "GRU", lstm: "LSTM" };
  const byId = (id) => document.getElementById(id);
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const accuracy = (value) => finite(value) && value >= 0 && value <= 100;
  const sampleSize = (value) => Number.isInteger(value) && value > 0;
  const format = (value) => finite(value) ? value.toFixed(2) : "Not loaded";
  const signed = (value) => finite(value) ? `${value > 0 ? "+" : ""}${value.toFixed(2)}` : "Not loaded";
  const condition = () => ({
    distance: Number(byId("distance").value),
    architecture: byId("architecture").value,
    hidden: Number(byId("hidden-size").value),
  });

  function element(tag, attributes = {}, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  function empty(container, detail) {
    const box = document.createElement("div");
    box.className = "empty";
    const heading = document.createElement("strong");
    heading.textContent = "Not loaded";
    const message = document.createElement("span");
    message.textContent = detail;
    box.append(heading, message);
    container.replaceChildren(box);
  }

  function frame(container, label, minimumX, maximumX, xLabel, ticks) {
    const width = 520, height = 270;
    const margin = { left: 43, right: 18, top: 16, bottom: 43 };
    const svg = element("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label });
    const x = (value) => margin.left + (value - minimumX) / (maximumX - minimumX || 1) * (width - margin.left - margin.right);
    const y = (value) => height - margin.bottom - value / 100 * (height - margin.top - margin.bottom);
    [0, 25, 50, 75, 100].forEach((value) => {
      svg.append(element("line", { x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value), stroke: "#dce2dc", "stroke-width": 1 }));
      svg.append(element("text", { x: margin.left - 8, y: y(value) + 3.5, "text-anchor": "end" }, value));
    });
    ticks.forEach((value) => {
      svg.append(element("text", { x: x(value), y: height - 24, "text-anchor": "middle" }, value));
    });
    svg.append(element("text", { x: (margin.left + width - margin.right) / 2, y: height - 3, "text-anchor": "middle" }, xLabel));
    svg.append(element("text", { transform: `translate(11 ${height / 2 - 10}) rotate(-90)`, "text-anchor": "middle" }, "Action accuracy (%)"));
    container.replaceChildren(svg);
    return { svg, x, y, width, height, margin };
  }

  function marker(svg, x, y, schedule, radius = 3.6) {
    if (schedule === "interleaved") {
      svg.append(element("polygon", { points: `${x},${y - radius - 1} ${x - radius - 1},${y + radius} ${x + radius + 1},${y + radius}`, fill: colors[schedule] }));
    } else {
      svg.append(element("circle", { cx: x, cy: y, r: radius, fill: colors[schedule] }));
    }
  }

  function finalRows(architecture, hidden) {
    if (architecture === "rnn" && hidden === 7 && data.main.length) {
      return data.main.filter((row) => accuracy(row.blocked) && accuracy(row.interleaved) && sampleSize(row.n)).flatMap((row) => [
        { distance: row.distance, schedule: "blocked", mean_accuracy: row.blocked, n: row.n },
        { distance: row.distance, schedule: "interleaved", mean_accuracy: row.interleaved, n: row.n },
      ]);
    }
    return data.variants.filter((row) => row.architecture === architecture && row.hidden_size === hidden && accuracy(row.mean_accuracy) && sampleSize(row.n) && Object.keys(colors).includes(row.schedule));
  }

  function drawAccuracy(selected) {
    const rows = finalRows(selected.architecture, selected.hidden).filter((row) => Number.isInteger(row.distance) && row.distance >= 1 && row.distance <= 7);
    const container = byId("accuracy-chart");
    if (!rows.length) {
      empty(container, "No recorded final accuracy for this architecture and hidden size.");
    } else {
      const { svg, x, y } = frame(container, `Recorded final action accuracy for ${names[selected.architecture]}, hidden size ${selected.hidden}`, 1, 7, "Pairwise Hamming distance", [1, 2, 3, 4, 5, 6, 7]);
      svg.append(element("line", { x1: x(selected.distance), x2: x(selected.distance), y1: y(0), y2: y(100), stroke: "#9faeaa", "stroke-dasharray": "3 4" }));
      Object.keys(colors).forEach((schedule) => {
        const series = rows.filter((row) => row.schedule === schedule).sort((a, b) => a.distance - b.distance);
        svg.append(element("polyline", { points: series.map((row) => `${x(row.distance)},${y(row.mean_accuracy)}`).join(" "), fill: "none", stroke: colors[schedule], "stroke-width": 2.2, "stroke-dasharray": schedule === "interleaved" ? "6 3" : "none" }));
        series.forEach((row) => marker(svg, x(row.distance), y(row.mean_accuracy), schedule));
      });
    }
    const body = byId("accuracy-table");
    body.replaceChildren();
    rows.slice().sort((a, b) => a.distance - b.distance || a.schedule.localeCompare(b.schedule)).forEach((row) => {
      const tr = document.createElement("tr");
      [row.distance, row.schedule === "blocked" ? "Blocked" : "Interleaved", format(row.mean_accuracy), row.n].forEach((value) => {
        const td = document.createElement("td"); td.textContent = value; tr.append(td);
      });
      body.append(tr);
    });
    byId("accuracy-caption").textContent = selected.architecture === "rnn" && selected.hidden === 7
      ? "Expanded-main means for the Elman RNN (h=7). Noisy final cues are independent of the development probes."
      : "Exploratory means for the selected architecture and capacity. Missing conditions remain empty.";
    const current = rows.filter((row) => row.distance === selected.distance);
    const blocked = current.find((row) => row.schedule === "blocked");
    const interleaved = current.find((row) => row.schedule === "interleaved");
    byId("blocked-score").textContent = blocked ? `${format(blocked.mean_accuracy)}%` : "Not loaded";
    byId("interleaved-score").textContent = interleaved ? `${format(interleaved.mean_accuracy)}%` : "Not loaded";
    const primary = selected.architecture === "rnn" && selected.hidden === 7
      ? data.main.find((row) => row.distance === selected.distance && finite(row.difference) && finite(row.ci_low) && finite(row.ci_high)) : null;
    byId("paired-score").textContent = primary ? signed(primary.difference) : blocked && interleaved ? signed(interleaved.mean_accuracy - blocked.mean_accuracy) : "Not loaded";
    byId("paired-ci").textContent = primary ? `[${format(primary.ci_low)}, ${format(primary.ci_high)}]` : "Not loaded";
    byId("ci-interpretation").textContent = primary
      ? primary.ci_low <= 0 && primary.ci_high >= 0 ? "The interval includes zero; direction is uncertain." : "Uncertainty concerns this model and generator."
      : "No paired interval is inferred from marginal means.";
    byId("result-note").textContent = primary
      ? `n=${primary.n} independent repetitions at this distance. Main intervals resample paired repetitions; individual predictions are not treated as independent samples. Read the report for multiplicity corrections and practical interpretation.`
      : blocked && interleaved ? `Descriptive difference between recorded means (blocked n=${blocked.n}, interleaved n=${interleaved.n}). Paired uncertainty and matched comparison details are in the report.`
      : "This selection has no recorded result in the bundled data. It is not shown as zero performance.";
  }

  function drawLearning(selected) {
    const rows = data.curves.filter((row) => row.distance === selected.distance && row.architecture === selected.architecture && row.hidden_size === selected.hidden && Object.keys(colors).includes(row.schedule) && finite(row.update) && row.update >= 0 && accuracy(row.mean));
    const container = byId("learning-chart");
    if (!rows.length) { empty(container, "No recorded learning curves for this selection."); return; }
    const max = Math.max(...rows.map((row) => row.update), 1);
    const ticks = [0, max / 3, 2 * max / 3, max].map((value) => Math.round(value));
    const { svg, x, y } = frame(container, `Recorded learning curves at distance ${selected.distance}, ${names[selected.architecture]}, hidden size ${selected.hidden}`, 0, max, "Main SGD updates", [...new Set(ticks)]);
    Object.keys(colors).forEach((schedule) => {
      const series = rows.filter((row) => row.schedule === schedule).sort((a, b) => a.update - b.update);
      if (!series.length) return;
      if (series.every((row) => finite(row.se) && row.se >= 0)) {
        const upper = series.map((row) => `${x(row.update)},${y(Math.min(100, row.mean + row.se))}`);
        const lower = series.slice().reverse().map((row) => `${x(row.update)},${y(Math.max(0, row.mean - row.se))}`);
        svg.append(element("polygon", { points: upper.concat(lower).join(" "), fill: colors[schedule], "fill-opacity": .12 }));
      }
      svg.append(element("polyline", { points: series.map((row) => `${x(row.update)},${y(row.mean)}`).join(" "), fill: "none", stroke: colors[schedule], "stroke-width": 2.2, "stroke-dasharray": schedule === "interleaved" ? "6 3" : "none" }));
      const last = series[series.length - 1]; marker(svg, x(last.update), y(last.mean), schedule);
    });
  }

  function drawArchitectures(selected) {
    const rows = data.variants.filter((row) => row.distance === selected.distance && row.hidden_size === selected.hidden && Object.keys(names).includes(row.architecture) && Object.keys(colors).includes(row.schedule) && accuracy(row.mean_accuracy) && sampleSize(row.n));
    const container = byId("architecture-chart");
    if (!rows.length) { empty(container, "Architecture comparison data are not loaded for this distance and hidden size."); return; }
    const architectures = Object.keys(names).filter((architecture) => rows.some((row) => row.architecture === architecture));
    const { svg, x, y } = frame(container, `Recorded architecture comparison, distance ${selected.distance}, hidden size ${selected.hidden}`, -.5, Math.max(.5, architectures.length - .5), "", []);
    architectures.forEach((architecture, index) => {
      svg.append(element("text", { x: x(index), y: 246, "text-anchor": "middle" }, `${names[architecture]} · h=${selected.hidden}`));
      Object.keys(colors).forEach((schedule, scheduleIndex) => {
        const row = rows.find((item) => item.architecture === architecture && item.schedule === schedule);
        if (!row) return;
        const center = x(index) + (scheduleIndex === 0 ? -18 : 18);
        svg.append(element("rect", { x: center - 14, y: y(row.mean_accuracy), width: 28, height: y(0) - y(row.mean_accuracy), fill: colors[schedule], rx: 1 }));
        svg.append(element("text", { x: center, y: y(row.mean_accuracy) - 6, "text-anchor": "middle", style: "font-size:9px" }, row.mean_accuracy.toFixed(1)));
        svg.append(element("text", { x: center, y: 261, "text-anchor": "middle", style: "font-size:8px" }, `n=${row.n}`));
      });
    });
  }

  function renderSequences(distance) {
    const example = data.examples.find((row) => row.distance === distance && Array.isArray(row.bases) && row.bases.length === 3 && row.bases.every((base) => Array.isArray(base) && base.length === 7 && base.every((action) => Number.isInteger(action) && action >= 0 && action <= 3)));
    const hamming = (a, b) => a.reduce((sum, action, position) => sum + (action !== b[position] ? 1 : 0), 0);
    const valid = example && [[0, 1], [0, 2], [1, 2]].every(([a, b]) => hamming(example.bases[a], example.bases[b]) === distance);
    const a = [0, 1, 2, 3, 0, 1, 2];
    const bases = valid ? example.bases : [a, a.map((action, i) => (action + (i < distance ? 1 : 0)) % 4), a.map((action, i) => (action + (i < distance ? 2 : 0)) % 4)];
    const container = byId("sequence-family"); container.replaceChildren();
    bases.forEach((base, index) => {
      const row = document.createElement("div"); row.className = "sequence-row";
      const label = document.createElement("span"); label.className = "sequence-label"; label.textContent = "ABC"[index]; row.append(label);
      base.forEach((action, position) => {
        const changed = bases.some((other) => other[position] !== bases[0][position]);
        const cell = document.createElement("span"); cell.className = `action-cell${changed ? " changed" : ""}`; cell.textContent = action;
        cell.title = `Sequence ${"ABC"[index]}, position ${position + 1}, action ${action}${changed ? ", differing position" : ", shared position"}`;
        row.append(cell);
      }); container.append(row);
    });
    byId("sequence-source").textContent = `${valid ? "One saved example family; this layout is explanatory." : "Synthetic illustration only; not an experimental sample."} A–B, A–C and B–C each differ at ${distance} of 7 positions.${distance === 7 ? " There are no shared positions; shared accuracy is undefined (NaN), not zero." : " Highlighted cells mark differing positions."}`;
  }

  function renderSchedule() {
    const schedule = byId("schedule").value;
    const chunks = { blocked: 30, large_blocks: 10, small_blocks: 3, cyclic: 1 };
    const ids = [];
    if (schedule === "interleaved") {
      for (let label = 0; label < 3; label++) for (let count = 0; count < 30; count++) ids.push(label);
      let state = 1729;
      for (let index = ids.length - 1; index > 0; index--) {
        state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
        const target = (state >>> 0) % (index + 1); [ids[index], ids[target]] = [ids[target], ids[index]];
      }
    } else {
      const chunk = chunks[schedule] || 30;
      for (let start = 0; start < 30; start += chunk) for (let label = 0; label < 3; label++) for (let count = start; count < Math.min(start + chunk, 30); count++) ids.push(label);
    }
    const strip = byId("schedule-strip"); strip.replaceChildren();
    ids.forEach((label, index) => { const cell = document.createElement("span"); cell.className = `sequence-${"abc"[label]}`; cell.title = `Illustrative example ${index + 1}: ${"ABC"[label]}`; strip.append(cell); });
    const descriptions = {
      blocked: "Finish all A examples, then B, then C. The actual experiment counterbalances block order.",
      large_blocks: "Alternate A, B and C in relatively large chunks, without changing total exposures.",
      small_blocks: "Switch between sequences more often, while retaining short consecutive runs.",
      cyclic: "Cycle A → B → C at the example level. The real experiment still uses minibatch SGD; this is not batch-size-one training.",
      interleaved: "Shuffle the exact same example IDs. This schematic shuffle is fixed for readability and is not an experimental random stream.",
    };
    byId("schedule-description").textContent = descriptions[schedule];
    strip.setAttribute("aria-label", `${schedule.replaceAll("_", " ")} illustration: 30 examples each of A, B and C`);
  }

  function render() {
    const selected = condition();
    byId("distance-value").textContent = selected.distance;
    byId("distance-help").textContent = selected.distance === 7 ? "All seven positions differ; there are no shared positions." : `${7 - selected.distance} shared positions; ${selected.distance} differing positions.`;
    byId("selected-condition").textContent = `k=${selected.distance} · ${names[selected.architecture]} · h=${selected.hidden}`;
    renderSequences(selected.distance); drawAccuracy(selected); drawLearning(selected); drawArchitectures(selected);
  }

  function provenance() {
    const container = byId("provenance");
    Object.entries(data.provenance).forEach(([key, value]) => {
      const term = document.createElement("dt"); term.textContent = key.replaceAll("_", " ");
      const description = document.createElement("dd"); description.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
      container.append(term, description);
    });
    const loaded = data.main.length || data.variants.length || data.curves.length;
    byId("data-status").textContent = loaded ? "Recorded results" : "Data not loaded";
    byId("data-warning").textContent = loaded
      ? "Every numeric result in the viewer comes from the bundled data. Consult the corresponding report and manifests for producing commits, exact settings, failures, exclusions and inference."
      : "No measured outcomes are bundled yet. The sequence and schedule illustrations remain usable. Run the repository's publication command to generate data.js from verified outputs; the viewer never substitutes demonstration scores.";
    if (!Object.keys(data.provenance).length) {
      const term = document.createElement("dt"); term.textContent = "Recorded provenance";
      const description = document.createElement("dd"); description.textContent = "Not loaded"; container.append(term, description);
    }
  }

  byId("distance").addEventListener("input", render);
  byId("architecture").addEventListener("change", render);
  byId("hidden-size").addEventListener("change", render);
  byId("schedule").addEventListener("change", renderSchedule);
  provenance(); renderSchedule(); render();
})();
