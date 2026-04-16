let appMetrics = null;
let csvChartInstance = null;
let metricsChartInstance = null;
let lastPredictionData = null;

document.addEventListener("DOMContentLoaded", function () {
    initTabs();
    fetchStatus();
    initDropZone();
    loadGeminiKey();
});

function initTabs() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach(function (tab) {
        tab.addEventListener("click", function () {
            const target = tab.dataset.tab;
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
    var startAngle = 180;
    var endAngle = 180 + angle;
    var startRad = (startAngle * Math.PI) / 180;
    var endRad = (endAngle * Math.PI) / 180;
    var cx = 90, cy = 90, r = 70;
    var x1 = cx + r * Math.cos(startRad);
    var y1 = cy + r * Math.sin(startRad);
    var x2 = cx + r * Math.cos(endRad);
    var y2 = cy + r * Math.sin(endRad);
    var largeArc = angle > 180 ? 1 : 0;

    svg.innerHTML =
        '<path d="M ' + (cx - r) + " " + cy + " A " + r + " " + r + ' 0 0 1 ' + (cx + r) + " " + cy + '" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="12" stroke-linecap="round"/>' +
        '<path d="M ' + x1 + " " + y1 + " A " + r + " " + r + " 0 " + largeArc + " 1 " + x2 + " " + y2 + '" fill="none" stroke="url(#gaugeGrad)" stroke-width="12" stroke-linecap="round"/>' +
        '<defs><linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">' +
        '<stop offset="0%" stop-color="#3b82f6"/><stop offset="100%" stop-color="#8b5cf6"/>' +
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
                backgroundColor: [
                    "rgba(59, 130, 246, 0.6)",
                    "rgba(139, 92, 246, 0.6)",
                    "rgba(168, 85, 247, 0.6)",
                    "rgba(6, 182, 212, 0.6)"
                ],
                borderColor: [
                    "rgba(59, 130, 246, 1)",
                    "rgba(139, 92, 246, 1)",
                    "rgba(168, 85, 247, 1)",
                    "rgba(6, 182, 212, 1)"
                ],
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
                    backgroundColor: "rgba(17, 24, 39, 0.95)",
                    titleColor: "#f1f5f9",
                    bodyColor: "#94a3b8",
                    borderColor: "rgba(255,255,255,0.1)",
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: function (ctx) { return (ctx.raw * 100).toFixed(2) + "%"; }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: "#64748b", font: { size: 11, weight: 600 } }
                },
                y: {
                    min: 0, max: 1,
                    grid: { color: "rgba(255,255,255,0.04)" },
                    ticks: {
                        color: "#64748b",
                        callback: function (v) { return (v * 100) + "%"; }
                    }
                }
            }
        }
    });
}

function initDropZone() {
    var zone = document.getElementById("dropZone");
    var input = document.getElementById("csvFileInput");

    zone.addEventListener("dragover", function (e) {
        e.preventDefault();
        zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", function () {
        zone.classList.remove("dragover");
    });
    zone.addEventListener("drop", function (e) {
        e.preventDefault();
        zone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            uploadCSV(e.dataTransfer.files[0]);
        }
    });
    input.addEventListener("change", function () {
        if (input.files.length) {
            uploadCSV(input.files[0]);
        }
    });
}

function loadGeminiKey() {
    var saved = localStorage.getItem("nexus_gemini_key");
    if (saved) {
        document.getElementById("input-gemini-key").value = saved;
    }
}

function toggleKeyVisibility() {
    var inp = document.getElementById("input-gemini-key");
    inp.type = inp.type === "password" ? "text" : "password";
}

function getGeminiKey() {
    var key = document.getElementById("input-gemini-key").value.trim();
    if (key) {
        localStorage.setItem("nexus_gemini_key", key);
    }
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
            if (data.error) {
                alert("Error: " + data.error);
                zone.style.display = "";
                return;
            }
            lastPredictionData = data;
            renderCSVResults(data);
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
            datasets: [{
                data: [s.up_count, s.down_count],
                backgroundColor: ["rgba(16, 185, 129, 0.7)", "rgba(239, 68, 68, 0.7)"],
                borderColor: ["#10b981", "#ef4444"],
                borderWidth: 2,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "65%",
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { color: "#94a3b8", font: { size: 13, weight: 600 }, padding: 20 }
                },
                tooltip: {
                    backgroundColor: "rgba(17, 24, 39, 0.95)",
                    titleColor: "#f1f5f9",
                    bodyColor: "#94a3b8",
                    borderColor: "rgba(255,255,255,0.1)",
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                }
            }
        }
    });
}

function triggerAdvisory(data) {
    var card = document.getElementById("advisoryCard");
    var loading = document.getElementById("advisoryLoading");
    var content = document.getElementById("advisoryContent");
    var nokey = document.getElementById("advisoryNoKey");

    card.style.display = "block";
    content.innerHTML = "";
    nokey.style.display = "none";

    var apiKey = getGeminiKey();
    if (!apiKey) {
        loading.style.display = "none";
        nokey.style.display = "block";
        return;
    }

    loading.style.display = "block";

    var s = data.summary;
    var intent = data.intent || {};

    var prompt = "You are an expert financial market analyst and investment advisor. Analyze the following machine learning model output and the user's investment intent to provide actionable advisory.\n\n";
    prompt += "## ML Model Prediction Results\n";
    prompt += "- Total entries analyzed: " + s.total + "\n";
    prompt += "- UP signals: " + s.up_count + " (" + ((s.up_count / s.total) * 100).toFixed(1) + "%)\n";
    prompt += "- DOWN signals: " + s.down_count + " (" + ((s.down_count / s.total) * 100).toFixed(1) + "%)\n";
    prompt += "- Dominant direction: " + s.dominant_direction + "\n";
    prompt += "- Average confidence: " + (s.avg_confidence * 100).toFixed(1) + "%\n";
    prompt += "- Average volatility delta: " + (s.avg_vol_delta || 0).toFixed(4) + "\n";
    prompt += "- Model accuracy: " + (appMetrics ? (appMetrics.accuracy * 100).toFixed(1) + "%" : "N/A") + "\n\n";

    prompt += "## User's Investment Intent\n";
    prompt += "- Weekly profit target: ₹" + (intent.profit_target || "Not specified") + "\n";
    prompt += "- Target companies/PSUs: " + (intent.target_companies || "Not specified") + "\n";
    prompt += "- Time horizon: " + (intent.time_horizon || "1 week") + "\n\n";

    var topResults = data.results.slice(0, 10);
    prompt += "## Sample Predictions (first 10 entries)\n";
    topResults.forEach(function (r) {
        prompt += "- " + r.date + ": " + r.direction + " (conf: " + (r.confidence * 100).toFixed(1) + "%, vol_delta: " + r.vol_delta.toFixed(4) + ")\n";
    });

    prompt += "\n## Your Task\n";
    prompt += "Provide a structured advisory with these sections:\n";
    prompt += "1. **Executive Summary** - 2-3 sentence overview of market outlook based on the data\n";
    prompt += "2. **Focus Areas** - Which specific instruments/companies to prioritize given the user's targets\n";
    prompt += "3. **Risk Assessment** - Analyze volatility trends and confidence levels\n";
    prompt += "4. **Projected Gain/Loss** - Estimated achievability of the ₹" + (intent.profit_target || "target") + " weekly target with rationale\n";
    prompt += "5. **Recommended Actions** - Concrete steps (buy/sell signals, hedging, position sizing)\n";
    prompt += "6. **What Could Go Wrong** - Key risks and how to mitigate them\n\n";
    prompt += "Keep response concise but actionable. Use bullet points. Include specific numbers where possible.";

    var requestBody = {
        contents: [{
            parts: [{ text: prompt }]
        }],
        generationConfig: {
            temperature: 0.7,
            maxOutputTokens: 2048,
        }
    };

    fetch("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + apiKey, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody)
    })
        .then(function (r) { return r.json(); })
        .then(function (response) {
            loading.style.display = "none";
            if (response.error) {
                content.innerHTML = '<div class="advisory-error">❌ Gemini API Error: ' + (response.error.message || "Unknown error") + '</div>';
                return;
            }
            try {
                var text = response.candidates[0].content.parts[0].text;
                content.innerHTML = renderMarkdown(text);
            } catch (e) {
                content.innerHTML = '<div class="advisory-error">❌ Failed to parse advisory response.</div>';
            }
        })
        .catch(function (err) {
            loading.style.display = "none";
            content.innerHTML = '<div class="advisory-error">❌ Failed to connect to Gemini: ' + err.message + '</div>';
        });
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

    html = html.replace(/(<li>.*?<\/li>(\s*<br>)*)+/g, function(match) {
        return '<ul class="adv-list">' + match.replace(/<br>/g, '') + '</ul>';
    });

    return '<div class="advisory-text"><p>' + html + '</p></div>';
}

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

    fetch("/api/predict/manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            btn.disabled = false;
            btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/></svg> Generate Prediction';

            if (data.error) {
                alert("Error: " + data.error);
                return;
            }
            renderManualResult(data);
        })
        .catch(function (err) {
            btn.disabled = false;
            btn.innerHTML = "Generate Prediction";
            alert("Prediction failed: " + err.message);
        });
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

    var pDown = data.probabilities.DOWN * 100;
    var pUp = data.probabilities.UP * 100;
    document.getElementById("probFillDown").style.width = pDown + "%";
    document.getElementById("probFillUp").style.width = pUp + "%";

    container.classList.add("visible");
    container.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
