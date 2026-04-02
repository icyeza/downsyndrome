// API Base URL
// When opened as file:// (local dev), call the Flask dev server directly.
// When served via nginx (Docker), use relative URLs so nginx proxies requests.
// const API_BASE = window.location.protocol === "file:" ? "http://localhost:5000" : "";
const API_BASE = "http://localhost:5000";

let selectedFile = null;
let selectedBatchFiles = [];
let charts = {};
let predictionHistory = [];

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    setupEventListeners();
    loadModelStatus();
    loadStats();
    loadPredictionHistory();
    loadModels();
    initCharts();

    // Auto-refresh stats every 5 seconds
    setInterval(loadModelStatus, 5000);
    setInterval(loadStats, 10000);
});

function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page-container');
    const pageTitle = document.getElementById('pageTitle');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Update Active Nav
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            // Update Page Title
            pageTitle.textContent = item.querySelector('span').textContent;

            // Show target page
            const targetId = item.getAttribute('data-target');
            pages.forEach(p => p.classList.remove('active'));
            document.getElementById(targetId).classList.add('active');
        });
    });
}

function setupEventListeners() {
    // Single image upload
    const uploadArea = document.getElementById("uploadArea");
    const fileInput = document.getElementById("fileInput");

    uploadArea.addEventListener("click", () => fileInput.click());
    uploadArea.addEventListener("dragover", (e) => { e.preventDefault(); uploadArea.classList.add("dragover"); });
    uploadArea.addEventListener("dragleave", () => { uploadArea.classList.remove("dragover"); });
    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        if(e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", (e) => {
        if(e.target.files.length) handleFileSelect(e.target.files[0]);
    });

    // Batch upload
    const batchUploadArea = document.getElementById("batchUploadArea");
    const batchFileInput = document.getElementById("batchFileInput");

    batchUploadArea.addEventListener("click", () => batchFileInput.click());
    batchUploadArea.addEventListener("dragover", (e) => { e.preventDefault(); batchUploadArea.classList.add("dragover"); });
    batchUploadArea.addEventListener("dragleave", () => { batchUploadArea.classList.remove("dragover"); });
    batchUploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        batchUploadArea.classList.remove("dragover");
        if(e.dataTransfer.files.length) handleBatchFileSelect(e.dataTransfer.files);
    });
    batchFileInput.addEventListener("change", (e) => {
        if(e.target.files.length) handleBatchFileSelect(e.target.files);
    });

    // Buttons
    document.getElementById("predictBtn").addEventListener("click", (e) => { e.preventDefault(); predictImage(); });
    document.getElementById("batchUploadBtn").addEventListener("click", (e) => { e.preventDefault(); uploadBatchFiles(); });
    document.getElementById("retrainBtn").addEventListener("click", (e) => { e.preventDefault(); triggerRetrain(); });
    document.getElementById("changeImageBtn").addEventListener("click", (e) => { e.preventDefault(); resetImageSelection(); });
}

function handleFileSelect(file) {
    if (!file.type.startsWith("image/")) {
        showAlert("Please select a valid image file.", "error");
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById("previewImage");
        preview.src = e.target.result;
        preview.style.display = "block";
        document.getElementById("uploadArea").style.display = "none";
        document.getElementById("changeImageBtn").style.display = "inline-block";
        document.getElementById("predictionResults").innerHTML = "";
    };
    reader.readAsDataURL(file);
}

function resetImageSelection() {
    selectedFile = null;
    const preview = document.getElementById("previewImage");
    preview.src = "";
    preview.style.display = "none";
    document.getElementById("uploadArea").style.display = "";
    document.getElementById("changeImageBtn").style.display = "none";
    document.getElementById("predictionResults").innerHTML = "";
    document.getElementById("fileInput").value = "";
}

function handleBatchFileSelect(files) {
    selectedBatchFiles = Array.from(files).filter(f => f.type.startsWith("image/"));
    const area = document.getElementById("batchUploadArea");
    area.querySelector('p').textContent = `${selectedBatchFiles.length} file(s) selected`;
    area.style.borderColor = "var(--success)";
    const icon = area.querySelector('.upload-icon-wrap i') || area.querySelector('i');
    icon.className = "fa-solid fa-check-double";
    icon.style.color = "var(--success)";
}

async function predictImage() {
    if (!selectedFile) {
        showAlert("Please select an image first", "warning");
        return;
    }

    const btn = document.getElementById("predictBtn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Predicting...';

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);
        const response = await axios.post(`${API_BASE}/predict`, formData, {
            headers: { "Content-Type": "multipart/form-data" }
        });
        const data = response.data;
        displayPredictionResults(data);
        addToPredictionHistory(data);
        showAlert("Prediction completed!", "success");
    } catch (error) {
        let msg = error.message;
        if (error.response && error.response.data && error.response.data.error) {
            msg = error.response.data.error;
        }
        showAlert("Prediction failed: " + msg, "error");
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Run Prediction';
    }
}

function displayPredictionResults(data) {
    const resultsDiv = document.getElementById("predictionResults");
    let confPercent = 0;
    if (data.confidence !== undefined) {
        confPercent = (data.confidence * 100).toFixed(1);
    }
    
    let html = `
    <div style="background: var(--surface-2); padding: 18px; border-radius: var(--radius); border: 1px solid var(--border);">
        <div style="color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; font-weight:700; letter-spacing:0.8px; margin-bottom: 6px;">Top Result</div>
        <div style="font-size: 1.6rem; font-weight: 700; color: var(--primary-blue);">${data.prediction || 'Unknown'}</div>

        <div class="confidence-container" style="margin-top:16px;">
            <div class="confidence-label">
                <span>Confidence Level</span>
                <span style="font-weight:700; color:var(--secondary-blue);">${confPercent}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confPercent}%;"></div>
            </div>
        </div>
    `;

    if (data.all_predictions && data.all_predictions.length > 1) {
        html += `<div style="margin-top: 18px;">
            <div style="color: var(--text-muted); margin-bottom: 10px; font-weight:700; font-size: 0.72rem; text-transform:uppercase; letter-spacing:0.8px;">Class Distribution</div>`;
        data.all_predictions.forEach(pred => {
            html += `<div style="display:flex; justify-content:space-between; align-items:center; padding: 7px 0; border-top: 1px solid var(--border); font-size:0.87rem;">
                <span style="color:var(--text-dark); font-weight:500;">${pred.class}</span>
                <span style="font-weight:700; color:var(--secondary-blue);">${pred.percentage.toFixed(1)}%</span>
            </div>`;
        });
        html += `</div>`;
    }
    html += `</div>`;
    resultsDiv.innerHTML = html;
}

async function uploadBatchFiles() {
    if (selectedBatchFiles.length === 0) {
        showAlert("No files selected for batch upload.", "warning");
        return;
    }

    const labelValue = document.getElementById("dataLabel").value;
    const isPredictMode = labelValue === "predict";

    const btn = document.getElementById("batchUploadBtn");
    btn.disabled = true;
    btn.innerHTML = isPredictMode
        ? '<span class="spinner dark"></span> Predicting...'
        : '<span class="spinner dark"></span> Uploading...';

    try {
        const formData = new FormData();
        selectedBatchFiles.forEach(file => formData.append("files", file));
        formData.append("label", labelValue);

        const response = await axios.post(`${API_BASE}/upload-training-data`, formData);
        const data = response.data;

        if (isPredictMode && data.prediction_results !== undefined) {
            const results = data.prediction_results;
            const uploadResultsDiv = document.getElementById("uploadResults");
            let html = `<div style="margin-bottom:10px; font-weight:600; color:var(--text-dark);">${data.total_predicted} image(s) predicted</div>`;
            if (results.length) {
                html += `<table style="width:100%; border-collapse:collapse; font-size:0.86rem;">
                    <thead>
                        <tr style="border-bottom:2px solid var(--border);">
                            <th style="text-align:left; padding:8px 10px; color:var(--text-muted); font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.6px;">Filename</th>
                            <th style="text-align:left; padding:8px 10px; color:var(--text-muted); font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.6px;">Prediction</th>
                            <th style="text-align:left; padding:8px 10px; color:var(--text-muted); font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.6px;">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>`;
                results.forEach(r => {
                    const conf = r.confidence_percentage != null ? r.confidence_percentage.toFixed(1) + '%' : '—';
                    const pred = r.success ? r.prediction : `<span style="color:var(--danger)">Error</span>`;
                    html += `<tr style="border-bottom:1px solid var(--border);">
                        <td style="padding:8px 10px; color:var(--text-muted); word-break:break-all;">${r.filename}</td>
                        <td style="padding:8px 10px; color:var(--text-dark); font-weight:500;">${pred}</td>
                        <td style="padding:8px 10px; color:var(--text-dark);">${conf}</td>
                    </tr>`;
                });
                html += `</tbody></table>`;
            }
            uploadResultsDiv.innerHTML = html;
        } else {
            showAlert(`Successfully uploaded ${data.uploaded_files} files.`, "success");
            if (data.retraining_needed) {
                showAlert(`Enough new samples accumulated — go to Retraining to start.`, "info");
            }
            document.getElementById("uploadResults").innerHTML = "";
        }

        // Reset file selection
        selectedBatchFiles = [];
        const area = document.getElementById("batchUploadArea");
        area.querySelector('p').textContent = "Select multiple images";
        area.style.borderColor = "";
        const areaIcon = area.querySelector('.upload-icon-wrap i') || area.querySelector('i');
        areaIcon.className = "fa-solid fa-images";
        areaIcon.style.color = "";
        if (!isPredictMode) loadStats();
    } catch (error) {
        showAlert(isPredictMode ? "Prediction failed." : "Upload failed.", "error");
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-file-arrow-up"></i> Upload Batch';
    }
}

async function triggerRetrain() {
    const lr = document.getElementById("hpLearningRate").value;
    const batchSize = document.getElementById("hpBatchSize").value;
    const epochs = document.getElementById("hpEpochs").value;
    const optimizer = document.getElementById("hpOptimizer").value;

    if (!confirm(`Start model retraining with:\n- Epochs: ${epochs}\n- Batch Size: ${batchSize}\n- Learning Rate: ${lr}\n- Optimizer: ${optimizer}`)) {
        return;
    }

    const btn = document.getElementById("retrainBtn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner dark"></span> Initializing...';

    try {
        await axios.post(`${API_BASE}/retrain`, {
            learning_rate: parseFloat(lr),
            batch_size: parseInt(batchSize),
            epochs: parseInt(epochs),
            optimizer: optimizer
        });
        showAlert("Retraining started!", "success");
        pollRetrainStatus(btn);
    } catch (error) {
        showAlert("Failed to start retraining.", "error");
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-rotate"></i> Start Retraining Process';
    }
}

function pollRetrainStatus(btn) {
    const resultsDiv = document.getElementById("retrainResults");
    let dots = 0;

    resultsDiv.innerHTML = `<div class="alert alert-info" style="margin-top:20px;">
        <i class="fa-solid fa-microchip"></i>
        <span id="retrainStatusMsg">Training in progress...</span>
    </div>`;

    const interval = setInterval(async () => {
        try {
            const res = await axios.get(`${API_BASE}/retrain-status`);
            const s = res.data;

            dots = (dots + 1) % 4;
            const ellipsis = '.'.repeat(dots + 1);

            if (s.in_progress) {
                btn.innerHTML = `<span class="spinner dark"></span> Training${ellipsis}`;
                document.getElementById("retrainStatusMsg").textContent = `Training in progress${ellipsis}`;
            } else {
                clearInterval(interval);
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-rotate"></i> Start Retraining Process';

                if (s.last_result === 'success') {
                    resultsDiv.innerHTML = `<div class="alert alert-success" style="margin-top:20px;">
                        <i class="fa-solid fa-circle-check"></i>
                        <span>Retraining complete! New model is now active.</span>
                    </div>`;
                    showAlert("Model retrained successfully!", "success");
                    loadModels();
                    loadStats();
                    loadModelStatus();
                } else if (s.last_result === 'failed') {
                    resultsDiv.innerHTML = `<div class="alert alert-error" style="margin-top:20px;">
                        <i class="fa-solid fa-triangle-exclamation"></i>
                        <span>Retraining failed. Check API logs for details.</span>
                    </div>`;
                    showAlert("Retraining failed.", "error");
                }
            }
        } catch (e) {
            clearInterval(interval);
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-rotate"></i> Start Retraining Process';
        }
    }, 3000);
}

async function loadModelStatus() {
    try {
        const response = await axios.get(`${API_BASE}/health`);
        const data = response.data;
        const box = document.getElementById("statusHealthBox");
        const txt = document.getElementById("statusHealthText");
        const sub = document.getElementById("statusHealthSub");

        if (data.status === "healthy" && data.binary_model_ready) {
            box.className = "status-box healthy";
            txt.textContent = "Ready";
            sub.textContent = "Classes: " + (data.class_labels ? data.class_labels.join(", ") : (data.class_count||0));
        } else if (data.status === "healthy" && !data.binary_model_ready) {
            box.className = "status-box warning";
            txt.textContent = "Incomplete State";
            sub.textContent = "Needs balanced classes";
        } else {
            box.className = "status-box warning";
            txt.textContent = "Error";
            sub.textContent = "Service offline/unavailable";
        }

        if (data.uptime) {
            const uptime = new Date(data.uptime);
            const diff = Math.floor((new Date() - uptime) / 1000);
            const h = Math.floor(diff / 3600);
            const m = Math.floor((diff % 3600) / 60);
            document.getElementById("statusUptime").textContent = `${h}h ${m}m`;
        }
    } catch (error) {
        // fail silently or show issue in bar
    }
}

async function loadPredictionHistory() {
    try {
        const response = await axios.get(`${API_BASE}/prediction-history`);
        const data = response.data;
        if (data.recent_predictions && data.recent_predictions.length) {
            predictionHistory = data.recent_predictions.map(p => ({
                class: p.predicted_class || 'Unknown',
                confidence: p.confidence || 0,
                timestamp: p.timestamp || new Date().toISOString()
            }));
            updatePredictionHistory();
        }
    } catch (error) {}
}

async function loadStats() {
    try {
        const response = await axios.get(`${API_BASE}/stats`);
        const data = response.data;
        document.getElementById("statusPredictions").textContent = data.predictions_total || 0;

        if (data.retrain_report) {
            document.getElementById("newSamplesCount").textContent = data.retrain_report.new_samples_accumulated || 0;
        }

        // Class distribution chart — real counts per class folder
        if (data.class_distribution && charts.classDistribution) {
            const labels = Object.keys(data.class_distribution);
            const values = Object.values(data.class_distribution);
            if (labels.length) {
                charts.classDistribution.data.labels = labels;
                charts.classDistribution.data.datasets[0].data = values;
                charts.classDistribution.update();
            }
        }

        // Confidence distribution chart — real bucketed history
        if (data.confidence_buckets && charts.confidence) {
            charts.confidence.data.datasets[0].data = data.confidence_buckets;
            charts.confidence.update();
        }
    } catch (error) {}
}

function addToPredictionHistory(prediction) {
    predictionHistory.unshift({
        class: prediction.prediction || 'Unknown',
        confidence: prediction.confidence || 0,
        timestamp: prediction.timestamp || new Date().toISOString()
    });
    if (predictionHistory.length > 30) predictionHistory.pop();
    updatePredictionHistory();
}

function updatePredictionHistory() {
    const list = document.getElementById("predictionHistory");
    if (!predictionHistory.length) {
        list.innerHTML = `<p style="text-align: center; color: var(--text-muted); padding: 40px 0;">No predictions recorded yet.</p>`;
        return;
    }

    let html = "";
    predictionHistory.forEach(pred => {
        const timeStr = new Date(pred.timestamp).toLocaleTimeString();
        const confObj = (pred.confidence * 100).toFixed(1);
        let iconClass = "fa-check";
        let iconColor = "var(--success)";
        
        if(pred.confidence < 0.6) {
             iconClass = "fa-question";
             iconColor = "var(--warning)";
        }
        
        html += `
        <div class="history-item">
            <div class="h-info">
                <div class="h-icon" style="color:${iconColor}; background:rgba(0,0,0,0.03);"><i class="fa-solid ${iconClass}"></i></div>
                <div class="h-details">
                    <h4>${pred.class}</h4>
                    <p><i class="fa-regular fa-clock"></i> ${timeStr}</p>
                </div>
            </div>
            <div class="h-score">${confObj}%</div>
        </div>`;
    });
    list.innerHTML = html;
}

function showAlert(message, type) {
    const container = document.getElementById('alertContainer');
    const alert = document.createElement('div');
    const icon = type === 'success' ? 'fa-circle-check' : type === 'error' ? 'fa-triangle-exclamation' : 'fa-circle-info';
    alert.className = `alert alert-${type}`;
    alert.innerHTML = `<i class="fa-solid ${icon}" style="font-size:1.1rem;"></i> <span>${message}</span>`;
    
    // Animate in
    alert.style.opacity = '0';
    alert.style.transform = 'translateY(-10px)';
    alert.style.transition = 'all 0.3s ease';
    alert.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
    
    container.appendChild(alert);
    
    setTimeout(() => {
        alert.style.opacity = '1';
        alert.style.transform = 'translateY(0)';
    }, 10);

    setTimeout(() => {
        alert.style.opacity = '0';
        setTimeout(() => alert.remove(), 300);
    }, 4500);
}

async function loadModels() {
    try {
        const response = await axios.get(`${API_BASE}/models`);
        const models = response.data;

        // Populate model selector dropdown
        const selector = document.getElementById("modelSelector");
        if (selector) {
            selector.innerHTML = models.map(m =>
                `<option value="${m.id}"${m.active ? ' selected' : ''}>${m.name}${m.active ? ' (active)' : ''}</option>`
            ).join('');

            // Switch model on change
            selector.onchange = () => {
                const chosen = selector.value;
                if (chosen) switchModel(chosen);
            };
        }

        // Populate model history table
        const tableDiv = document.getElementById("modelHistoryTable");
        if (tableDiv) {
            if (!models.length) {
                tableDiv.innerHTML = `<p style="text-align:center; color:var(--text-muted); padding: 24px 0;">No models registered.</p>`;
            } else {
                let html = `<table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                    <thead>
                        <tr style="border-bottom:2px solid var(--border);">
                            <th style="text-align:left; padding:10px 12px; color:var(--text-muted); font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.6px;">Version</th>
                            <th style="text-align:left; padding:10px 12px; color:var(--text-muted); font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.6px;">Date</th>
                            <th style="text-align:left; padding:10px 12px; color:var(--text-muted); font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.6px;">Accuracy</th>
                            <th style="text-align:left; padding:10px 12px; color:var(--text-muted); font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.6px;">Status</th>
                            <th style="text-align:left; padding:10px 12px; color:var(--text-muted); font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.6px;">Action</th>
                        </tr>
                    </thead>
                    <tbody>`;
                models.forEach(m => {
                    const dateStr = m.date ? new Date(m.date).toLocaleDateString() : 'N/A';
                    const accStr = m.accuracy != null ? (m.accuracy * 100).toFixed(1) + '%' : '—';
                    const statusBadge = m.active
                        ? `<span style="background:rgba(16,185,129,.12); color:var(--success); padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600;">Active</span>`
                        : `<span style="background:var(--surface-2); color:var(--text-muted); padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; border:1px solid var(--border);">Available</span>`;
                    const actionBtn = m.active
                        ? `<button class="btn btn-secondary" style="padding:4px 12px; font-size:0.8rem;" disabled>Use</button>`
                        : `<button class="btn btn-secondary" style="padding:4px 12px; font-size:0.8rem;" onclick="switchModel('${m.id}')">Use</button>`;
                    html += `<tr style="border-bottom:1px solid var(--border);">
                        <td style="padding:10px 12px; color:var(--text-dark); font-weight:500;">${m.name}</td>
                        <td style="padding:10px 12px; color:var(--text-muted);">${dateStr}</td>
                        <td style="padding:10px 12px; color:var(--text-dark);">${accStr}</td>
                        <td style="padding:10px 12px;">${statusBadge}</td>
                        <td style="padding:10px 12px;">${actionBtn}</td>
                    </tr>`;
                });
                html += `</tbody></table>`;
                tableDiv.innerHTML = html;
            }
        }

        updateAccuracyChart(models);
    } catch (error) {
        const tableDiv = document.getElementById("modelHistoryTable");
        if (tableDiv) {
            tableDiv.innerHTML = `<p style="text-align:center; color:var(--text-muted); padding: 24px 0;">Unable to load model registry.</p>`;
        }
    }
}

async function switchModel(modelId) {
    try {
        await axios.post(`${API_BASE}/switch-model`, { model_id: modelId });
        await loadModels();
        await loadModelStatus();
        showAlert("Model switched successfully.", "success");
    } catch (error) {
        let msg = "Failed to switch model.";
        if (error.response && error.response.data && error.response.data.error) {
            msg = error.response.data.error;
        }
        showAlert(msg, "error");
    }
}

function updateAccuracyChart(models) {
    if (!charts.accuracy) return;
    const withAccuracy = models.filter(m => m.accuracy != null);
    if (!withAccuracy.length) return;
    charts.accuracy.data.labels = withAccuracy.map(m => m.name);
    charts.accuracy.data.datasets[0].data = withAccuracy.map(m => m.accuracy);
    charts.accuracy.update();
}

function initCharts() {
    const sharedOptions = { responsive: true, maintainAspectRatio: false };
    
    // Dist Chart
    const ctxClass = document.getElementById("classDistributionChart");
    if (ctxClass) {
        charts.classDistribution = new Chart(ctxClass, {
            type: "doughnut",
            data: {
                labels: ["With Syndrome", "Without Syndrome"],
                datasets: [{
                    data: [50, 50],
                    backgroundColor: ["#1e3a8a", "#f59e0b"],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: { ...sharedOptions, plugins: { legend: { position: "right" } } }
        });
    }

    // Confidence
    const ctxConf = document.getElementById("confidenceChart");
    if (ctxConf) {
        charts.confidence = new Chart(ctxConf, {
            type: "bar",
            data: {
                labels: ["0-20", "20-40", "40-60", "60-80", "80-100"],
                datasets: [{
                    label: "Predicted Subsets",
                    data: [1, 2, 5, 10, 25],
                    backgroundColor: "#3b82f6",
                    borderRadius: 4
                }]
            },
            options: { ...sharedOptions, plugins: { legend: { display: false } } }
        });
    }

    // Accuracy
    const ctxAcc = document.getElementById("accuracyChart");
    if (ctxAcc) {
        charts.accuracy = new Chart(ctxAcc, {
            type: "line",
            data: {
                labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                datasets: [{
                    label: "Validation Accuracy",
                    data: [0.85, 0.88, 0.90, 0.89, 0.92, 0.93, 0.94],
                    borderColor: "#1e3a8a",
                    backgroundColor: "rgba(30, 58, 138, 0.1)",
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointBackgroundColor: "#f59e0b",
                    pointRadius: 4
                }]
            },
            options: { ...sharedOptions, plugins: { legend: { display: false } }, scales: { y: { min: 0.7, max: 1.0 } } }
        });
    }
}
