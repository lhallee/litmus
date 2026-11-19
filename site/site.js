// The Litmus site's one script: the theme toggle, and the numbers, addresses and figures the pages read from
// assets/results.json, assets/links.json and assets/figures/. Every request it makes is to this site's own address.
// A value that is missing or empty is never filled in: the page keeps its "being finalized" wording instead.

(function () {
  "use strict";

  const KEY = "litmus.site.theme";
  const root = document.documentElement;

  // The theme is applied before the page paints. Dark is the look of every visit, whatever the system asks for; the
  // light look appears only when the visitor picked it with the toggle, which this one key remembers.
  function savedTheme() {
    try {
      const value = localStorage.getItem(KEY);
      return value === "light" || value === "dark" ? value : null;
    } catch (error) {
      return null;
    }
  }
  // Parts the script draws (tables, figures, filled-in facts) are hidden without it: see .needs-js in style.css.
  root.classList.add("js");
  const saved = savedTheme();
  if (saved) root.setAttribute("data-theme", saved);

  function theme() {
    return root.getAttribute("data-theme") === "light" ? "light" : "dark";
  }

  function setupToggle() {
    for (const button of document.querySelectorAll(".theme-toggle")) {
      button.hidden = false;
      const label = () => button.setAttribute("aria-label", theme() === "dark" ? "Switch to the light look" : "Switch to the dark look");
      label();
      button.addEventListener("click", () => {
        const next = theme() === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        try { localStorage.setItem(KEY, next); } catch (error) { /* the choice lasts for this page only */ }
        label();
      });
    }
  }

  // The in-browser app opens in the site's look; #dark or #light in its address picks the app's theme for the visit.
  function setupApp() {
    const frame = document.querySelector("iframe[data-app]");
    if (!frame) return;
    frame.src = frame.getAttribute("data-app") + "#" + theme();
    frame.hidden = false;
  }

  // The film starts, muted, once half of it is in view, unless the visitor asked for less motion. A browser that
  // refuses to play it leaves the poster and the controls.
  function setupFilm() {
    const films = document.querySelectorAll("video.film");
    if (!films.length || !("IntersectionObserver" in window)) return;
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const watcher = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        watcher.unobserve(entry.target);
        const started = entry.target.play();
        if (started) started.catch(() => { /* autoplay refused: the poster and the controls stay */ });
      }
    }, { threshold: 0.5 });
    for (const film of films) watcher.observe(film);
  }

  async function json(path) {
    try {
      const response = await fetch(path, { credentials: "same-origin" });
      return response.ok ? await response.json() : null;
    } catch (error) {
      return null;
    }
  }

  function lookup(data, path) {
    let value = data;
    for (const part of path.split(".")) {
      if (value == null || typeof value !== "object") return null;
      value = value[part];
    }
    return value === "" || value === undefined ? null : value;
  }

  const has = (value) => typeof value === "number" ? Number.isFinite(value) : value != null && value !== "";

  const FORMATS = {
    mb: (v) => (v / 1e6).toFixed(1) + " MB",
    int: (v) => Math.round(v).toLocaleString("en-US"),
    k: (v) => (Math.round(v / 100) * 100).toLocaleString("en-US"),
    num: (v) => Number(v).toFixed(3),
    f1: (v) => Number(v).toFixed(1),
    pct: (v) => (100 * v).toFixed(1) + "%",
    text: (v) => String(v),
  };

  function format(value, kind) {
    return (FORMATS[kind] || FORMATS.text)(value);
  }

  // data-value="results.release.app_exe_bytes" data-format="mb": the element's text becomes the value, if there is one.
  // data-show="links.contribute_site" shows an element only when the value exists; data-hide hides it then.
  // data-link="links.contribute_site" points a link at the value.
  function bind(data) {
    for (const element of document.querySelectorAll("[data-value]")) {
      const value = lookup(data, element.getAttribute("data-value"));
      if (!has(value)) continue;
      element.textContent = format(value, element.getAttribute("data-format"));
      element.classList.remove("tbd");
    }
    for (const element of document.querySelectorAll("[data-show]")) {
      element.hidden = !has(lookup(data, element.getAttribute("data-show")));
    }
    for (const element of document.querySelectorAll("[data-hide]")) {
      element.hidden = has(lookup(data, element.getAttribute("data-hide")));
    }
    for (const element of document.querySelectorAll("[data-link]")) {
      const value = lookup(data, element.getAttribute("data-link"));
      if (has(value)) element.href = value;
    }
  }

  // Figures: <div class="figure-slot" data-figure="raid">, filled with assets/figures/raid_dark.png and _light.png
  // when they exist, else left with its "being finalized" box.
  function figures() {
    for (const slot of document.querySelectorAll("[data-figure]")) {
      const name = slot.getAttribute("data-figure");
      const alt = slot.getAttribute("data-alt") || "";
      const probe = new Image();
      probe.onload = () => {
        const frame = document.createElement("div");
        frame.className = "frame";
        for (const look of ["dark", "light"]) {
          const image = document.createElement("img");
          image.src = "assets/figures/" + name + "_" + look + ".png";
          image.alt = alt;
          image.loading = "lazy";
          image.decoding = "async";
          image.className = "on-" + look;
          frame.appendChild(image);
        }
        slot.replaceChildren(frame);
        const caption = slot.nextElementSibling;
        if (caption && caption.classList.contains("figure-caption")) caption.hidden = false;
      };
      probe.src = "assets/figures/" + name + "_" + theme() + ".png";
    }
  }

  // Results: the metrics of one row, as "value" with its interval.
  const METRICS = [
    ["pr_auc", "PR AUC", "num"],
    ["roc_auc", "ROC AUC", "num"],
    ["balanced_mcc", "Balanced MCC", "num"],
    ["tpr_at_fpr_1", "TPR at 1% FPR", "pct"],
    ["tpr_at_fpr_5", "TPR at 5% FPR", "pct"],
  ];
  const FLAGS = [
    ["flagged_at_fpr_1", "Flagged at the 1% FPR threshold", "pct"],
    ["flagged_at_fpr_5", "Flagged at the 5% FPR threshold", "pct"],
  ];

  function cell(row, key, kind) {
    const td = document.createElement("td");
    td.className = "n";
    if (!row || !has(row[key])) {
      td.textContent = "n/a";
      td.classList.add("soft");
      return td;
    }
    td.textContent = format(row[key], kind);
    if (has(row[key + "_low"]) && has(row[key + "_high"])) {
      const interval = document.createElement("span");
      interval.className = "iv";
      interval.textContent = format(row[key + "_low"], kind) + " to " + format(row[key + "_high"], kind);
      td.appendChild(interval);
    }
    return td;
  }

  function textCell(text, className) {
    const td = document.createElement("td");
    td.textContent = text;
    if (className) td.className = className;
    return td;
  }

  // A name cell, marked when the detector trained on this benchmark (results.json's `in_distribution`: its training
  // split, or for another detector possibly its test texts; the legend under each table says which).
  function nameCell(text, row) {
    const td = textCell(text);
    if (row && row.in_distribution === true) {
      const mark = document.createElement("span");
      mark.className = "seen";
      mark.title = "Trained on this benchmark; the legend below the table says how";
      mark.textContent = "T";
      const words = document.createElement("span");
      words.className = "sr-heading";
      words.textContent = " (trained on this benchmark)";
      td.append(" ", mark, words);
    }
    return td;
  }

  function emptyRow(tbody, columns, message) {
    const tr = document.createElement("tr");
    const td = textCell(message, "empty");
    td.colSpan = columns;
    tr.appendChild(td);
    tbody.replaceChildren(tr);
  }

  function texts(row) {
    const counts = [row.positives, row.negatives].filter(has);
    return counts.length ? counts.reduce((a, b) => a + b, 0) : null;
  }

  function results(data) {
    const results = data.results || {};
    const rows = Array.isArray(results.rows) ? results.rows : [];
    const ours = results.model && results.model.name ? results.model.name : "Litmus";
    const task = results.primary_task || "ai_vs_human";
    const benchmarks = Array.isArray(results.benchmarks) ? results.benchmarks : [];
    const named = (id) => (benchmarks.find((b) => b.id === id) || { name: id }).name;
    const isFlag = (id) => (benchmarks.find((b) => b.id === id) || {}).kind === "flag_rate";

    // Headline tiles: Litmus on its own held-out test split.
    const tiles = document.getElementById("test-tiles");
    const test = rows.find((r) => r.benchmark === "test" && r.detector === ours && r.task === task);
    if (tiles && test && METRICS.some(([key]) => has(test[key]))) {
      const fragment = document.createDocumentFragment();
      for (const [key, label, kind] of METRICS.filter(([k]) => k !== "tpr_at_fpr_5")) {
        if (!has(test[key])) continue;
        const tile = document.createElement("div");
        tile.className = "stat" + (key === "roc_auc" ? " accent" : "");
        const value = document.createElement("span");
        value.className = "value";
        value.textContent = format(test[key], kind);
        const name = document.createElement("span");
        name.className = "label";
        name.textContent = label;
        tile.append(value, name);
        if (has(test[key + "_low"]) && has(test[key + "_high"])) {
          const interval = document.createElement("span");
          interval.className = "interval";
          interval.textContent = "95% interval " + format(test[key + "_low"], kind) + " to " + format(test[key + "_high"], kind);
          tile.appendChild(interval);
        }
        fragment.appendChild(tile);
      }
      tiles.replaceChildren(fragment);
      for (const note of document.querySelectorAll("[data-when-results]")) note.hidden = false;
    }
    for (const note of document.querySelectorAll("[data-until-results]")) note.hidden = rows.length > 0;

    // Litmus on every benchmark, in the order results.json lists them.
    const summary = document.getElementById("summary-body");
    if (summary) {
      const ordered = benchmarks.map((b) => b.id).filter((id) => id !== "test");
      const lines = ordered.map((id) => rows.find((r) => r.benchmark === id && r.detector === ours && (isFlag(id) || r.task === task))).filter(Boolean);
      if (!lines.length) {
        emptyRow(summary, 7, "Results are being finalized.");
      } else {
        summary.replaceChildren(...lines.map((row) => {
          const tr = document.createElement("tr");
          tr.appendChild(nameCell(named(row.benchmark), row));
          if (isFlag(row.benchmark)) {
            const parts = FLAGS.filter(([key]) => has(row[key])).map(([key, , kind]) => format(row[key], kind));
            const note = textCell(parts.length === 2
              ? "Flagged: " + parts[0] + " at the 1% FPR threshold, " + parts[1] + " at the 5% threshold"
              : "Flag rates are being finalized", "soft small");
            note.colSpan = METRICS.length;
            tr.appendChild(note);
          } else {
            for (const [key, , kind] of METRICS) tr.appendChild(cell(row, key, kind));
          }
          const n = texts(row);
          tr.appendChild(textCell(has(n) ? format(n, "int") : (has(row.texts) ? format(row.texts, "int") : "n/a"), "n"));
          return tr;
        }));
      }
    }

    // Every detector on one benchmark, chosen from a list.
    const pick = document.getElementById("compare-benchmark");
    const pickTask = document.getElementById("compare-task");
    const compare = document.getElementById("compare-body");
    if (pick && compare) {
      const ids = benchmarks.map((b) => b.id).filter((id) => rows.some((r) => r.benchmark === id));
      if (!ids.length) {
        emptyRow(compare, 8, "Results are being finalized.");
        pick.disabled = true;
        if (pickTask) pickTask.disabled = true;
      } else {
        pick.replaceChildren(...ids.map((id) => new Option(named(id), id)));
        const tasks = [...new Set(rows.map((r) => r.task).filter(has))];
        if (pickTask) {
          const words = { ai_vs_human: "AI against human", any_ai_vs_human: "AI or mixed against human", mixed_vs_human: "Mixed against human" };
          pickTask.replaceChildren(...tasks.map((t) => new Option(words[t] || t, t)));
          pickTask.value = tasks.includes(task) ? task : tasks[0];
        }
        const draw = () => {
          const id = pick.value;
          const chosenTask = pickTask ? pickTask.value : task;
          const flag = isFlag(id);
          const head = document.getElementById("compare-head");
          const columns = flag ? FLAGS : METRICS;
          if (head) {
            const tr = document.createElement("tr");
            const first = document.createElement("th");
            first.textContent = "Detector";
            tr.appendChild(first);
            for (const [, label] of columns) {
              const th = document.createElement("th");
              th.className = "n";
              th.textContent = label;
              tr.appendChild(th);
            }
            const last = document.createElement("th");
            last.className = "n";
            last.textContent = "Texts";
            tr.appendChild(last);
            head.replaceChildren(tr);
          }
          const sortKey = flag ? "flagged_at_fpr_1" : "roc_auc";
          const chosen = rows
            .filter((r) => r.benchmark === id && (flag || r.task === chosenTask))
            .sort((a, b) => (has(b[sortKey]) ? b[sortKey] : -1) - (has(a[sortKey]) ? a[sortKey] : -1));
          if (!chosen.length) {
            emptyRow(compare, columns.length + 2, "No results for this task on this benchmark.");
            return;
          }
          compare.replaceChildren(...chosen.map((row) => {
            const tr = document.createElement("tr");
            if (row.detector === ours) tr.className = "ours";
            tr.appendChild(nameCell(row.detector, row));
            for (const [key, , kind] of columns) tr.appendChild(cell(row, key, kind));
            const n = texts(row);
            tr.appendChild(textCell(has(n) ? format(n, "int") : (has(row.texts) ? format(row.texts, "int") : "n/a"), "n"));
            return tr;
          }));
        };
        pick.addEventListener("change", draw);
        if (pickTask) pickTask.addEventListener("change", draw);
        draw();
      }
    }

    // RAID: every detector, by TPR at 5% FPR.
    const raid = document.getElementById("raid-body");
    if (raid) {
      const chosen = rows
        .filter((r) => r.benchmark === "raid" && r.task === task)
        .sort((a, b) => (has(b.tpr_at_fpr_5) ? b.tpr_at_fpr_5 : -1) - (has(a.tpr_at_fpr_5) ? a.tpr_at_fpr_5 : -1));
      if (!chosen.length) {
        emptyRow(raid, 5, "Results are being finalized.");
      } else {
        raid.replaceChildren(...chosen.map((row) => {
          const tr = document.createElement("tr");
          if (row.detector === ours) tr.className = "ours";
          tr.append(nameCell(row.detector, row), cell(row, "tpr_at_fpr_5", "pct"), cell(row, "tpr_at_fpr_1", "pct"), cell(row, "roc_auc", "num"), cell(row, "pr_auc", "num"));
          return tr;
        }));
      }
    }

    // Speed on one processor core.
    const cost = document.getElementById("cost-body");
    if (cost) {
      const costs = (Array.isArray(results.costs) ? results.costs : []).filter((c) => has(c.tokens_per_second))
        .sort((a, b) => b.tokens_per_second - a.tokens_per_second);
      if (!costs.length) {
        emptyRow(cost, 3, "Results are being finalized.");
      } else {
        cost.replaceChildren(...costs.map((c) => {
          const tr = document.createElement("tr");
          if (c.detector === ours) tr.className = "ours";
          tr.append(textCell(c.detector), textCell(format(c.tokens_per_second, "int"), "n"), textCell(c.measured ? "Measured" : "Estimated", "soft"));
          return tr;
        }));
      }
    }

    // The legend of the "trained on this benchmark" mark, once there are rows to mark.
    if (rows.length) {
      for (const legend of document.querySelectorAll(".seen-legend")) legend.hidden = false;
    }

    // The detectors compared, named from the rows themselves.
    const list = document.getElementById("detector-list");
    if (list) {
      const names = [...new Set(rows.map((r) => r.detector).filter((d) => has(d) && d !== ours))];
      if (names.length) {
        list.textContent = names.length === 1 ? names[0] : names.slice(0, -1).join(", ") + (names.length > 2 ? "," : "") + " and " + names[names.length - 1];
        list.classList.remove("tbd");
      }
    }
  }

  async function start() {
    setupToggle();
    setupApp();
    setupFilm();
    figures();
    const needs = document.querySelector("[data-value], [data-show], [data-hide], [data-link], #test-tiles, #summary-body, #compare-body, #raid-body, #cost-body");
    if (!needs) return;
    const [resultsData, linksData] = await Promise.all([json("assets/results.json"), json("assets/links.json")]);
    const data = { results: resultsData || {}, links: linksData || {} };
    bind(data);
    results(data);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
