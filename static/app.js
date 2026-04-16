let appMetrics = null;
let csvChartInstance = null;
let metricsChartInstance = null;
let simChartInstance = null;
let lastPredictionData = null;

document.addEventListener("DOMContentLoaded", function () {
    initTabs();
    fetchStatus();
    initDropZone();
    loadGeminiKey();
});

function initTabs() {
    var tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach(function (tab) {
        tab.addEventListener("click", function () {
            var target = tab.dataset.tab;
            tabs.forEach(function (t) { t.classList.remove("active"); });
            tab.classList.add("active");
            document.querySelectorAll(".tab-content").forEach(function (p) { p.classList.remove("active"); });
            document.getElementById("panel-" + target).classList.add("active");
        });
    });
}

function fetchStatus() {
    fetch("/api/status")
        .then(function (r) { return r.json(); })
        .then(function (data) {
            var badge = document.getElementById("statusBadge");
            var text = document.getElementById("statusText");
            if (data.ready) {
                text.textContent = "SYSTEM ONLINE";
                badge.classList.remove("offline");
            } else {
                text.textContent = "OFFLINE — TRAIN MODEL FIRST";
                badge.classList.add("offline");
            }
            if (data.metrics) {
                appMetrics = data.metrics;
                renderDashboard(data.metrics);
            }
        })
        .catch(function () {
            document.getElementById("statusText").textContent = "CONNECTION ERROR";
            document.getElementById("statusBadge").classList.add("offline");
        });
}

function renderDashboard(m) {
    var grid = document.getElementById("metricsGrid");
    var cards = [
        { label: "Accuracy", value: (m.accuracy * 100).toFixed(1) + "%", color: "blue", detail: "Direction classification" },
        { label: "F1 Macro", value: m.f1_macro.toFixed(4), color: "purple", detail: "Balanced class performance" },
        { label: "ROC-AUC", value: m.roc_auc.toFixed(4), color: "cyan", detail: "Discrimination ability" },
        { label: "Vol MAE", value: m.vol_mae.toFixed(4), color: "green", detail: "Mean absolute error" },
        { label: "Vol RMSE", value: m.vol_rmse.toFixed(4), color: "green", detail: "Root mean squared error" },
    ];
    grid.innerHTML = "";
    cards.forEach(function (c) {
        var el = document.createElement("div");
        el.className = "metric-card " + c.color;
        el.innerHTML =
            '<div class="metric-label">' + c.label + "</div>" +
            '<div class="metric-value">' + c.value + "</div>" +
            '<div class="metric-detail">' + c.detail + "</div>";
        grid.appendChild(el);
    });
    renderConfusion(m.confusion_matrix);
    renderGauge(m.accuracy);
    renderMetricsChart(m);
}

function renderConfusion(cm) {
    if (!cm) return;
    var html =
        '<table class="confusion-matrix">' +
        "<thead><tr><th></th><th>Pred DOWN</th><th>Pred UP</th></tr></thead>" +
        "<tbody>" +
        '<tr><th>Actual DOWN</th><td class="correct">' + cm[0][0] + '</td><td class="incorrect">' + cm[0][1] + "</td></tr>" +
        '<tr><th>Actual UP</th><td class="incorrect">' + cm[1][0] + '</td><td class="correct">' + cm[1][1] + "</td></tr>" +
        "</tbody></table>";
    document.getElementById("confusionContent").innerHTML = html;
}

function renderGauge(accuracy) {
    var pct = accuracy * 100;
    var svg = document.querySelector("#accuracyGauge svg");
    var angle = (pct / 100) * 180;
    var startRad = Math.PI;
    var endRad = Math.PI + (angle * Math.PI) / 180;
    var cx = 90, cy = 90, r = 70;
    var x1 = cx + r * Math.cos(startRad);
    var y1 = cy + r * Math.sin(startRad);
    var x2 = cx + r * Math.cos(endRad);
    var y2 = cy + r * Math.sin(endRad);
    var largeArc = angle > 180 ? 1 : 0;
    svg.innerHTML =
        '<path d="M ' + (cx - r) + " " + cy + " A " + r + " " + r + ' 0 0 1 ' + (cx + r) + " " + cy + '" fill="none" stroke="#e2e8f0" stroke-width="12" stroke-linecap="round"/>' +
        '<path d="M ' + x1 + " " + y1 + " A " + r + " " + r + " 0 " + largeArc + " 1 " + x2 + " " + y2 + '" fill="none" stroke="url(#gaugeGrad)" stroke-width="12" stroke-linecap="round"/>' +
        '<defs><linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">' +
        '<stop offset="0%" stop-color="#4f46e5"/><stop offset="100%" stop-color="#7c3aed"/>' +
        "</linearGradient></defs>";
    document.getElementById("gaugeValue").textContent = pct.toFixed(1) + "%";
}

function renderMetricsChart(m) {
    var ctx = document.getElementById("metricsChart").getContext("2d");
    if (metricsChartInstance) metricsChartInstance.destroy();
    metricsChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Accuracy", "F1 Macro", "F1 Weighted", "ROC-AUC"],
            datasets: [{
                data: [m.accuracy, m.f1_macro, m.f1_weighted, m.roc_auc],
                backgroundColor: ["rgba(79,70,229,0.15)", "rgba(124,58,237,0.15)", "rgba(168,85,247,0.15)", "rgba(8,145,178,0.15)"],
                borderColor: ["#4f46e5", "#7c3aed", "#a855f7", "#0891b2"],
                borderWidth: 2,
                borderRadius: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#0f172a",
                    titleColor: "#f1f5f9",
                    bodyColor: "#94a3b8",
                    borderColor: "#334155",
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: { label: function (ctx) { return (ctx.raw * 100).toFixed(2) + "%"; } }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: "#94a3b8", font: { size: 11, weight: 600 } } },
                y: { min: 0, max: 1, grid: { color: "#f1f5f9" }, ticks: { color: "#94a3b8", callback: function (v) { return (v * 100) + "%"; } } }
            }
        }
    });
}

function initDropZone() {
    var zone = document.getElementById("dropZone");
    var input = document.getElementById("csvFileInput");
    zone.addEventListener("dragover", function (e) { e.preventDefault(); zone.classList.add("dragover"); });
    zone.addEventListener("dragleave", function () { zone.classList.remove("dragover"); });
    zone.addEventListener("drop", function (e) {
        e.preventDefault(); zone.classList.remove("dragover");
        if (e.dataTransfer.files.length) uploadCSV(e.dataTransfer.files[0]);
    });
    input.addEventListener("change", function () { if (input.files.length) uploadCSV(input.files[0]); });
}

function loadGeminiKey() {
    var saved = localStorage.getItem("nexus_gemini_key");
    if (saved) document.getElementById("input-gemini-key").value = saved;
}

function toggleKeyVisibility() {
    var inp = document.getElementById("input-gemini-key");
    inp.type = inp.type === "password" ? "text" : "password";
}

function getGeminiKey() {
    var key = document.getElementById("input-gemini-key").value.trim();
    if (key) localStorage.setItem("nexus_gemini_key", key);
    return key;
}

function uploadCSV(file) {
    var spinner = document.getElementById("csvSpinner");
    var results = document.getElementById("csvResults");
    var zone = document.getElementById("dropZone");
    zone.style.display = "none";
    spinner.classList.add("visible");
    results.classList.remove("visible");

    var formData = new FormData();
    formData.append("file", file);
    formData.append("profit_target", document.getElementById("input-profit-target").value || "");
    formData.append("target_companies", document.getElementById("input-target-companies").value || "");
    formData.append("time_horizon", document.getElementById("input-time-horizon").value || "1 week");

    fetch("/api/predict/csv", { method: "POST", body: formData })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            spinner.classList.remove("visible");
            if (data.error) { alert("Error: " + data.error); zone.style.display = ""; return; }
            lastPredictionData = data;
            renderCSVResults(data);
            renderMomentumBars(data);
            renderRiskHeatmap(data);
            triggerAdvisory(data);
        })
        .catch(function (err) {
            spinner.classList.remove("visible");
            zone.style.display = "";
            alert("Upload failed: " + err.message);
        });
}

function renderCSVResults(data) {
    var results = document.getElementById("csvResults");
    var summary = document.getElementById("csvSummary");
    var tbody = document.getElementById("csvTableBody");
    var s = data.summary;
    summary.innerHTML =
        buildCSVStat(s.total, "Total Entries", "var(--accent-blue)") +
        buildCSVStat(s.up_count, "UP Signals", "var(--accent-green)") +
        buildCSVStat(s.down_count, "DOWN Signals", "var(--accent-red)") +
        buildCSVStat((s.avg_confidence * 100).toFixed(1) + "%", "Avg Confidence", "var(--accent-purple)");
    tbody.innerHTML = "";
    data.results.forEach(function (r) {
        var dirClass = r.direction === "UP" ? "dir-up" : "dir-down";
        var arrow = r.direction === "UP" ? "▲" : "▼";
        var signal = getSignal(r.confidence);
        var tr = document.createElement("tr");
        tr.innerHTML =
            "<td>" + r.date + "</td>" +
            '<td class="' + dirClass + '">' + arrow + " " + r.direction + "</td>" +
            '<td class="mono">' + (r.confidence * 100).toFixed(1) + "%</td>" +
            '<td class="mono">' + (r.vol_delta >= 0 ? "+" : "") + r.vol_delta.toFixed(4) + "</td>" +
            "<td>" + signal + "</td>";
        tbody.appendChild(tr);
    });
    results.classList.add("visible");
    renderCSVChart(s);
}

function buildCSVStat(value, label, color) {
    return '<div class="csv-stat"><div class="csv-stat-value" style="color:' + color + '">' + value + '</div><div class="csv-stat-label">' + label + "</div></div>";
}

function getSignal(conf) {
    if (conf >= 0.70) return '<span class="signal-badge high">HIGH YIELD</span>';
    if (conf >= 0.55) return '<span class="signal-badge moderate">MODERATE</span>';
    return '<span class="signal-badge uncertain">UNCERTAIN</span>';
}

function renderCSVChart(s) {
    var ctx = document.getElementById("csvChart").getContext("2d");
    if (csvChartInstance) csvChartInstance.destroy();
    csvChartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["UP Signals", "DOWN Signals"],
            datasets: [{ data: [s.up_count, s.down_count], backgroundColor: ["rgba(5,150,105,0.2)", "rgba(220,38,38,0.2)"], borderColor: ["#059669", "#dc2626"], borderWidth: 2, hoverOffset: 8 }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, cutout: "65%",
            plugins: {
                legend: { position: "bottom", labels: { color: "#475569", font: { size: 13, weight: 600 }, padding: 20 } },
                tooltip: { backgroundColor: "#0f172a", titleColor: "#f1f5f9", bodyColor: "#94a3b8", borderColor: "#334155", borderWidth: 1, cornerRadius: 8, padding: 12 }
            }
        }
    });
}

/* ===== MOMENTUM BARS ===== */
function renderMomentumBars(data) {
    var container = document.getElementById("momentumBars");
    var s = data.summary;
    var upPct = (s.up_count / s.total * 100).toFixed(1);
    var downPct = (s.down_count / s.total * 100).toFixed(1);
    var confPct = (s.avg_confidence * 100).toFixed(1);
    var volAbs = Math.min(Math.abs(s.avg_vol_delta || 0) * 100, 100).toFixed(1);

    var bars = [
        { label: "Bullish", pct: upPct, cls: "bullish" },
        { label: "Bearish", pct: downPct, cls: "bearish" },
        { label: "Confidence", pct: confPct, cls: parseFloat(confPct) > 60 ? "bullish" : "neutral" },
        { label: "Volatility", pct: volAbs, cls: parseFloat(volAbs) > 50 ? "bearish" : "neutral" },
    ];

    container.innerHTML = "";
    bars.forEach(function (b) {
        container.innerHTML +=
            '<div class="momentum-bar-row">' +
            '<div class="momentum-label">' + b.label + '</div>' +
            '<div class="momentum-track"><div class="momentum-fill ' + b.cls + '" style="width: ' + b.pct + '%">' + b.pct + '%</div></div>' +
            '</div>';
    });
}

/* ===== RISK HEATMAP ===== */
function renderRiskHeatmap(data) {
    var container = document.getElementById("riskHeatmap");
    container.innerHTML = "";
    var results = data.results;
    var cols = Math.min(results.length, 50);
    var rowHtml = '<div class="heatmap-row">';

    for (var i = 0; i < cols; i++) {
        var r = results[i];
        var risk = (1 - r.confidence) + Math.abs(r.vol_delta) * 2;
        risk = Math.min(risk, 1);
        var color;
        if (risk < 0.3) color = "#059669";
        else if (risk < 0.5) color = "#34d399";
        else if (risk < 0.65) color = "#fbbf24";
        else if (risk < 0.8) color = "#f97316";
        else color = "#dc2626";

        var opacity = 0.5 + risk * 0.5;
        rowHtml += '<div class="heatmap-cell" style="background:' + color + ';opacity:' + opacity.toFixed(2) + '">' +
            '<div class="hm-tooltip">' + r.date + ' | ' + r.direction + ' | Risk: ' + (risk * 100).toFixed(0) + '%</div></div>';

        if ((i + 1) % 25 === 0 && i < cols - 1) {
            rowHtml += '</div><div class="heatmap-row">';
        }
    }
    rowHtml += '</div>';
    container.innerHTML = rowHtml;
}

/* ===== ADVISORY (DEMO + GEMINI) ===== */
function triggerAdvisory(data) {
    var card = document.getElementById("advisoryCard");
    var loading = document.getElementById("advisoryLoading");
    var content = document.getElementById("advisoryContent");
    var modeBadge = document.getElementById("advisoryModeBadge");
    card.style.display = "block";
    content.innerHTML = "";
    loading.style.display = "block";

    var apiKey = getGeminiKey();

    if (!apiKey) {
        modeBadge.style.display = "inline-flex";
        setTimeout(function () {
            loading.style.display = "none";
            content.innerHTML = renderMarkdown(generateDemoAdvisory(data));
        }, 1200);
        return;
    }

    modeBadge.style.display = "none";
    var s = data.summary;
    var intent = data.intent || {};

    var prompt = "You are an expert financial market analyst. Analyze these ML model predictions and user intent to provide actionable advisory.\n\n";
    prompt += "## ML Predictions\n- Entries: " + s.total + "\n- UP: " + s.up_count + " (" + ((s.up_count / s.total) * 100).toFixed(1) + "%)\n- DOWN: " + s.down_count + "\n- Dominant: " + s.dominant_direction + "\n- Avg confidence: " + (s.avg_confidence * 100).toFixed(1) + "%\n- Avg vol delta: " + (s.avg_vol_delta || 0).toFixed(4) + "\n- Model accuracy: " + (appMetrics ? (appMetrics.accuracy * 100).toFixed(1) + "%" : "N/A") + "\n\n";
    prompt += "## User Intent\n- Weekly profit target: ₹" + (intent.profit_target || "N/A") + "\n- Targets: " + (intent.target_companies || "N/A") + "\n- Horizon: " + (intent.time_horizon || "1 week") + "\n\n";

    var topR = data.results.slice(0, 10);
    prompt += "## Sample (first 10)\n";
    topR.forEach(function (r) { prompt += "- " + r.date + ": " + r.direction + " (conf:" + (r.confidence * 100).toFixed(1) + "% vol:" + r.vol_delta.toFixed(4) + ")\n"; });
    prompt += "\nProvide: 1.Executive Summary 2.Focus Areas 3.Risk Assessment 4.Projected Gain/Loss for the ₹" + (intent.profit_target || "target") + " target 5.Recommended Actions 6.What Could Go Wrong. Use bullet points, be concise.";

    fetch("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + apiKey, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.7, maxOutputTokens: 2048 } })
    })
        .then(function (r) { return r.json(); })
        .then(function (response) {
            loading.style.display = "none";
            if (response.error) { content.innerHTML = '<div class="advisory-error">❌ API Error: ' + (response.error.message || "Unknown") + '. Falling back to demo mode.</div>' + renderMarkdown(generateDemoAdvisory(data)); modeBadge.style.display = "inline-flex"; return; }
            try { content.innerHTML = renderMarkdown(response.candidates[0].content.parts[0].text); } catch (e) { content.innerHTML = renderMarkdown(generateDemoAdvisory(data)); modeBadge.style.display = "inline-flex"; }
        })
        .catch(function (err) {
            loading.style.display = "none";
            content.innerHTML = '<div class="advisory-error">❌ Connection failed: ' + err.message + '. Using demo advisory.</div>' + renderMarkdown(generateDemoAdvisory(data));
            modeBadge.style.display = "inline-flex";
        });
}

function generateDemoAdvisory(data) {
    var s = data.summary;
    var intent = data.intent || {};
    var upPct = ((s.up_count / s.total) * 100).toFixed(1);
    var downPct = ((s.down_count / s.total) * 100).toFixed(1);
    var conf = (s.avg_confidence * 100).toFixed(1);
    var dir = s.dominant_direction;
    var target = intent.profit_target || "50000";
    var companies = intent.target_companies || "NIFTY, BANKNIFTY";
    var horizon = intent.time_horizon || "1 week";
    var avgVol = Math.abs(s.avg_vol_delta || 0);
    var riskLevel = avgVol > 0.5 ? "HIGH" : avgVol > 0.2 ? "MODERATE" : "LOW";
    var achievable = parseFloat(conf) > 60 && dir === "UP" ? "ACHIEVABLE" : parseFloat(conf) > 55 ? "PARTIALLY ACHIEVABLE" : "CHALLENGING";

    var text = "## Executive Summary\n";
    text += "Based on **" + s.total + " analyzed entries**, the model detects a **" + dir + " bias** with **" + conf + "% average confidence**. ";
    text += upPct + "% of signals point UP while " + downPct + "% point DOWN. ";
    text += "The overall market sentiment appears **" + (dir === "UP" ? "bullish" : "bearish") + "** for the analyzed period.\n\n";

    text += "## Focus Areas\n";
    text += "- **Primary focus**: " + companies + " — " + dir.toLowerCase() + " directional trades\n";
    text += "- **High-confidence entries**: Filter for signals with >70% confidence for best execution\n";
    text += "- **Volatility plays**: " + (avgVol > 0.3 ? "Elevated volatility suggests straddle/strangle opportunities" : "Low volatility favors directional option buying") + "\n";
    text += "- **Position timing**: Enter during high-confidence windows within the " + horizon + " period\n\n";

    text += "## Risk Assessment\n";
    text += "- **Overall risk level**: " + riskLevel + "\n";
    text += "- **Confidence spread**: Avg " + conf + "% — " + (parseFloat(conf) > 65 ? "strong signal reliability" : "moderate signal reliability, use tight stop-losses") + "\n";
    text += "- **Volatility delta**: " + (s.avg_vol_delta || 0).toFixed(4) + " — " + (avgVol > 0.3 ? "expect significant price swings, hedge accordingly" : "stable conditions, favorable for directional bets") + "\n";
    text += "- **Model accuracy**: " + (appMetrics ? (appMetrics.accuracy * 100).toFixed(1) + "%" : "~95%") + " on test data, providing high baseline reliability\n\n";

    text += "## Projected Gain/Loss\n";
    text += "- **Target**: ₹" + target + " weekly profit — **" + achievable + "**\n";
    if (achievable === "ACHIEVABLE") {
        text += "- With " + conf + "% confidence and " + dir + " bias, allocating 2-3% risk per trade across 5-8 high-confidence signals should reach this target\n";
        text += "- **Estimated range**: ₹" + (parseInt(target) * 0.7).toLocaleString() + " to ₹" + (parseInt(target) * 1.3).toLocaleString() + " given current signal strength\n";
    } else {
        text += "- Current signal confidence suggests conservative position sizing — target may need 2-3 " + horizon + " periods\n";
        text += "- **Estimated realistic range**: ₹" + (parseInt(target) * 0.3).toLocaleString() + " to ₹" + (parseInt(target) * 0.8).toLocaleString() + "\n";
    }
    text += "- **Worst case**: -₹" + (parseInt(target) * 0.4).toLocaleString() + " if signals reverse (use strict stop-losses)\n\n";

    text += "## Recommended Actions\n";
    text += "- " + (dir === "UP" ? "**Buy CE (Call options)** on " + companies + " during confirmed UP signals" : "**Buy PE (Put options)** on " + companies + " during confirmed DOWN signals") + "\n";
    text += "- Set stop-loss at 30% of premium paid per trade\n";
    text += "- **Position size**: Risk max 2% of capital per trade\n";
    text += "- Enter only on signals with confidence ≥65%\n";
    text += "- Book partial profits at 50% target, trail the rest\n\n";

    text += "## What Could Go Wrong\n";
    text += "- **Sudden news events** (RBI policy, global cues) can override technical signals\n";
    text += "- **Model limitation**: Trained on historical data — black swan events are not captured\n";
    text += "- **Volatility crush**: Post-event IV drops can erode option premiums even if direction is correct\n";
    text += "- **Overtrading**: Acting on low-confidence signals dilutes overall returns\n";

    return text;
}

function renderMarkdown(text) {
    var html = text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/^### (.+)$/gm, '<h4 class="adv-h4">$1</h4>')
        .replace(/^## (.+)$/gm, '<h3 class="adv-h3">$1</h3>')
        .replace(/^# (.+)$/gm, '<h2 class="adv-h2">$1</h2>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/^(\d+)\. (.+)$/gm, '<li><strong>$1.</strong> $2</li>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
    html = html.replace(/(<li>.*?<\/li>(\s*<br>)*)+/g, function (match) {
        return '<ul class="adv-list">' + match.replace(/<br>/g, '') + '</ul>';
    });
    return '<div class="advisory-text"><p>' + html + '</p></div>';
}

/* ===== PORTFOLIO SIMULATOR ===== */
function runSimulator() {
    var capital = parseFloat(document.getElementById("sim-capital").value) || 500000;
    var riskPct = parseFloat(document.getElementById("sim-risk-pct").value) || 2;
    var lotSize = parseInt(document.getElementById("sim-lot-size").value) || 50;
    var targetProfit = parseFloat(document.getElementById("sim-target-profit").value) || 50000;

    var riskPerTrade = capital * (riskPct / 100);
    var acc = appMetrics ? appMetrics.accuracy : 0.956;
    var avgPremium = 200;
    var estWinAmount = avgPremium * lotSize * 0.5;
    var estLossAmount = avgPremium * lotSize * 0.3;
    var expectedPerTrade = (acc * estWinAmount) - ((1 - acc) * estLossAmount);
    var tradesNeeded = Math.ceil(targetProfit / expectedPerTrade);
    var maxLots = Math.floor(riskPerTrade / (avgPremium * lotSize));
    var weeklyProjected = expectedPerTrade * Math.min(tradesNeeded, 10);
    var winRate = acc * 100;

    var bestCase = weeklyProjected * 1.4;
    var worstCase = -riskPerTrade * 3;

    var resultsDiv = document.getElementById("simResults");
    var grid = document.getElementById("simResultsGrid");
    resultsDiv.style.display = "block";

    grid.innerHTML =
        buildSimCard("₹" + riskPerTrade.toLocaleString(), "Risk per Trade", "var(--accent-red)") +
        buildSimCard(maxLots + " lots", "Max Position Size", "var(--accent-blue)") +
        buildSimCard(tradesNeeded + "", "Trades Needed", "var(--accent-purple)") +
        buildSimCard("₹" + expectedPerTrade.toFixed(0), "Expected per Trade", "var(--accent-green)") +
        buildSimCard("₹" + weeklyProjected.toFixed(0), "Projected Weekly", "var(--accent-cyan)") +
        buildSimCard(winRate.toFixed(1) + "%", "Model Win Rate", "var(--accent-blue)");

    renderSimChart(bestCase, weeklyProjected, worstCase, targetProfit);
    resultsDiv.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function buildSimCard(value, label, color) {
    return '<div class="sim-result-card"><div class="sim-result-value" style="color:' + color + '">' + value + '</div><div class="sim-result-label">' + label + '</div></div>';
}

function renderSimChart(best, expected, worst, target) {
    var ctx = document.getElementById("simChart").getContext("2d");
    if (simChartInstance) simChartInstance.destroy();
    simChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Best Case", "Expected", "Worst Case", "Your Target"],
            datasets: [{
                data: [best, expected, worst, target],
                backgroundColor: [
                    "rgba(5,150,105,0.15)", "rgba(79,70,229,0.15)",
                    "rgba(220,38,38,0.15)", "rgba(217,119,6,0.15)"
                ],
                borderColor: ["#059669", "#4f46e5", "#dc2626", "#d97706"],
                borderWidth: 2,
                borderRadius: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#0f172a", titleColor: "#f1f5f9", bodyColor: "#94a3b8",
                    callbacks: { label: function (ctx) { return "₹" + ctx.raw.toLocaleString(); } }
                }
            },
            scales: {
                x: { grid: { color: "#f1f5f9" }, ticks: { color: "#94a3b8", callback: function (v) { return "₹" + (v / 1000).toFixed(0) + "k"; } } },
                y: { grid: { display: false }, ticks: { color: "#475569", font: { size: 12, weight: 600 } } }
            }
        }
    });
}

/* ===== EXPORT FUNCTIONS ===== */
function exportReport() {
    if (!lastPredictionData) { alert("No analysis data to export. Upload a CSV first."); return; }
    var s = lastPredictionData.summary;
    var intent = lastPredictionData.intent || {};
    var text = "NEXUS Analysis Report\n" + "=".repeat(50) + "\n\n";
    text += "Generated: " + new Date().toLocaleString() + "\n\n";
    text += "INTENT\n";
    text += "  Profit Target: ₹" + (intent.profit_target || "N/A") + "\n";
    text += "  Target Companies: " + (intent.target_companies || "N/A") + "\n";
    text += "  Time Horizon: " + (intent.time_horizon || "N/A") + "\n\n";
    text += "SUMMARY\n";
    text += "  Total Entries: " + s.total + "\n";
    text += "  UP Signals: " + s.up_count + "\n";
    text += "  DOWN Signals: " + s.down_count + "\n";
    text += "  Avg Confidence: " + (s.avg_confidence * 100).toFixed(1) + "%\n";
    text += "  Dominant Direction: " + s.dominant_direction + "\n\n";
    text += "PREDICTIONS\n";
    text += "Date | Direction | Confidence | Vol Delta\n";
    text += "-".repeat(55) + "\n";
    lastPredictionData.results.forEach(function (r) {
        text += r.date + " | " + r.direction + " | " + (r.confidence * 100).toFixed(1) + "% | " + r.vol_delta.toFixed(4) + "\n";
    });

    var advisory = document.getElementById("advisoryContent").innerText;
    if (advisory) { text += "\nAI ADVISORY\n" + "=".repeat(50) + "\n" + advisory + "\n"; }

    var blob = new Blob([text], { type: "text/plain" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "nexus_report_" + new Date().toISOString().slice(0, 10) + ".txt";
    a.click();
}

function exportCSVResults() {
    if (!lastPredictionData) { alert("No data to export."); return; }
    var csv = "Date,Direction,Confidence,Vol_Delta,Prob_UP,Prob_DOWN\n";
    lastPredictionData.results.forEach(function (r) {
        csv += r.date + "," + r.direction + "," + r.confidence.toFixed(4) + "," + r.vol_delta.toFixed(4) + "," + r.prob_up.toFixed(4) + "," + r.prob_down.toFixed(4) + "\n";
    });
    var blob = new Blob([csv], { type: "text/csv" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "nexus_predictions_" + new Date().toISOString().slice(0, 10) + ".csv";
    a.click();
}

/* ===== MANUAL PREDICT ===== */
function handleManualPredict() {
    var btn = document.getElementById("predictBtn");
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner" style="width:20px;height:20px;border-width:2px;margin:0"></div> Analyzing...';
    var payload = {
        call_oi: parseFloat(document.getElementById("input-call_oi").value) || 0,
        put_oi: parseFloat(document.getElementById("input-put_oi").value) || 0,
        pcr: parseFloat(document.getElementById("input-pcr").value) || 0,
        straddle: parseFloat(document.getElementById("input-straddle").value) || 0,
        spot_return: parseFloat(document.getElementById("input-spot_return").value) || 0,
    };
    fetch("/api/predict/manual", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            btn.disabled = false;
            btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/></svg> Generate Prediction';
            if (data.error) { alert("Error: " + data.error); return; }
            renderManualResult(data);
        })
        .catch(function (err) { btn.disabled = false; btn.innerHTML = "Generate Prediction"; alert("Failed: " + err.message); });
}

function renderManualResult(data) {
    var container = document.getElementById("manualResult");
    var hero = document.getElementById("resultHero");
    var dir = data.direction.toUpperCase();
    hero.className = "result-hero " + dir.toLowerCase();
    document.getElementById("resultDirection").textContent = dir;
    var signal = data.confidence >= 0.70 ? "HIGH YIELD" : data.confidence >= 0.55 ? "MODERATE" : "UNCERTAIN";
    var sigClass = data.confidence >= 0.70 ? "high" : data.confidence >= 0.55 ? "moderate" : "uncertain";
    document.getElementById("resultSignal").innerHTML = '<span class="signal-badge ' + sigClass + '">' + signal + "</span>";
    document.getElementById("resultConfidence").textContent = (data.confidence * 100).toFixed(1) + "%";
    document.getElementById("resultVolDelta").textContent = (data.iv_change >= 0 ? "+" : "") + data.iv_change.toFixed(4);
    document.getElementById("resultProbUp").textContent = (data.probabilities.UP * 100).toFixed(2) + "%";
    document.getElementById("resultProbDown").textContent = (data.probabilities.DOWN * 100).toFixed(2) + "%";
    document.getElementById("probFillDown").style.width = (data.probabilities.DOWN * 100) + "%";
    document.getElementById("probFillUp").style.width = (data.probabilities.UP * 100) + "%";
    container.classList.add("visible");
    container.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
