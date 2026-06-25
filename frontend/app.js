const API_BASE = window.location.origin;
let feedInterval = null;
let feedRunning = true;
let feedCounters = { total: 0, safe: 0, review: 0, fraud: 0 };
let feedRateData = [];
let charts = {};

// Navigation
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('section-' + btn.dataset.section).classList.add('active');
    });
});

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const parent = btn.parentElement;
        parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        let el = parent.nextElementSibling;
        while (el && el.classList.contains('tab-content')) {
            el.classList.remove('active');
            el = el.nextElementSibling;
        }
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
});

async function api(path) {
    try {
        const r = await fetch(API_BASE + path);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
    } catch(e) {
        console.error('API error:', path, e);
        return null;
    }
}

async function apiPost(path, data) {
    try {
        const r = await fetch(API_BASE + path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
    } catch(e) {
        console.error('API error:', path, e);
        return null;
    }
}

Chart.defaults.color = '#8892b0';
Chart.defaults.borderColor = '#2a3050';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;

function createChart(id, config) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(document.getElementById(id), config);
    return charts[id];
}

async function init() {
    const [metrics, history, importance, tsne, dist, hourly, amountDist, smote, shapPre] = await Promise.all([
        api('/api/metrics'), api('/api/training_history'), api('/api/feature_importance'),
        api('/api/tsne'), api('/api/distribution'), api('/api/hourly'),
        api('/api/amount_distribution'), api('/api/smote'), api('/api/shap_precomputed')
    ]);

    if (!metrics) {
        document.getElementById('statusText').textContent = 'Backend offline — check server';
        document.getElementById('statusDot').style.background = 'var(--accent-red)';
        return;
    }

    document.getElementById('statusText').textContent = 'Connected — Models Active';
    document.getElementById('statusDot').style.background = 'var(--accent-green)';

    renderDashboard(metrics, hourly, amountDist);
    renderPerformance(metrics);
    renderFeatures(importance, tsne, dist);
    renderTraining(history);
    renderDataset(metrics, smote);
    startFeed();
}

// SECTION 1: DASHBOARD
function renderDashboard(m, hourly, amountDist) {
    const info = m.dataset_info;
    document.getElementById('kpiTotal').textContent = info.total_samples.toLocaleString();
    document.getElementById('kpiFraud').textContent = info.total_fraud.toLocaleString();
    document.getElementById('kpiFraudPct').textContent = (info.total_fraud / info.total_samples * 100).toFixed(3) + '% of total';
    document.getElementById('kpiAuc').textContent = m.xgboost.roc_auc;
    // document.getElementById('kpiAuc').textContent = m.dnn.roc_auc;
    document.getElementById('kpiFpr').textContent = (m.xgboost.fpr * 100).toFixed(4) + '%';
    // document.getElementById('kpiFpr').textContent = (m.dnn.fpr * 100).toFixed(2) + '%';

    if (hourly) {
        const hours = Array.from({length:24}, (_,i) => i);
        createChart('chartHourly', {
            type: 'bar',
            data: {
                labels: hours.map(h => h + ':00'),
                datasets: [{
                    label: 'Total', data: hours.map(h => hourly[h]?.total || 0),
                    backgroundColor: 'rgba(59,130,246,0.6)', borderRadius: 4
                }, {
                    label: 'Fraud', data: hours.map(h => hourly[h]?.fraud || 0),
                    backgroundColor: 'rgba(239,68,68,0.8)', borderRadius: 4
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { maxTicksLimit: 12 } } } }
        });

        const rates = hours.map(h => {
            const t = hourly[h]?.total || 1;
            const f = hourly[h]?.fraud || 0;
            return (f / t * 100);
        });
        createChart('chartDetection', {
            type: 'line',
            data: {
                labels: hours.map(h => h + ':00'),
                datasets: [{ label: 'Fraud Rate %', data: rates, borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', fill: true, tension: 0.4, pointRadius: 2 }]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { maxTicksLimit: 12 } } } }
        });
    }

    createChart('chartClass', {
        type: 'doughnut',
        data: {
            labels: ['Legitimate', 'Fraud'],
            datasets: [{ data: [m.dataset_info.total_legit, m.dataset_info.total_fraud], backgroundColor: ['#3b82f6', '#ef4444'], borderWidth: 0 }]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
    });

    if (amountDist) {
        createChart('chartAmountDist', {
            type: 'bar',
            data: {
                labels: amountDist.map(a => a.bucket),
                datasets: [{ label: 'Count', data: amountDist.map(a => a.count), backgroundColor: 'rgba(6,182,212,0.6)', borderRadius: 4 }]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { type: 'logarithmic' }, x: { ticks: { font: { size: 10 } } } } }
        });
    }

    const td = m.threshold_data;
    if (td && td.length > 0) {
        const mid = td.find(t => t.threshold === 0.50);
        if (mid) {
            const low = (1 - mid.recall) * 100;
            const high = mid.recall * (1 - mid.fpr) * 100;
            const med = 100 - low - high;
            document.getElementById('riskLow').textContent = low.toFixed(1) + '%';
            document.getElementById('riskLowBar').style.width = low + '%';
            document.getElementById('riskMed').textContent = Math.max(med, 0).toFixed(1) + '%';
            document.getElementById('riskMedBar').style.width = Math.max(med, 0) + '%';
            document.getElementById('riskHigh').textContent = high.toFixed(1) + '%';
            document.getElementById('riskHighBar').style.width = high + '%';
        }
    }
}

// SECTION 2: LIVE FEED
function startFeed() {
    if (feedInterval) clearInterval(feedInterval);
    feedInterval = setInterval(fetchLiveTransaction, 1500);
}

async function fetchLiveTransaction() {
    if (!feedRunning) return;
    const txn = await api('/api/live_transaction');
    if (!txn) return;

    feedCounters.total++;
    if (txn.status === 'SAFE') feedCounters.safe++;
    else if (txn.status === 'REVIEW') feedCounters.review++;
    else feedCounters.fraud++;

    document.getElementById('feedTotal').textContent = feedCounters.total;
    document.getElementById('feedSafe').textContent = feedCounters.safe;
    document.getElementById('feedReview').textContent = feedCounters.review;
    document.getElementById('feedFraud').textContent = feedCounters.fraud;

    const rate = feedCounters.total > 0 ? (feedCounters.fraud / feedCounters.total * 100) : 0;
    document.getElementById('feedRate').textContent = rate.toFixed(2) + '%';

    feedRateData.push(rate);
    if (feedRateData.length > 50) feedRateData.shift();
    updateFeedChart();

    const list = document.getElementById('feedList');
    const time = new Date().toLocaleTimeString();
    const cls = txn.status === 'FRAUD' ? 'fraud' : txn.status === 'REVIEW' ? 'review' : '';
    const badgeCls = txn.status === 'FRAUD' ? 'badge-fraud' : txn.status === 'REVIEW' ? 'badge-review' : 'badge-safe';
    const row = document.createElement('div');
    row.className = 'feed-item ' + cls;
    row.innerHTML = `
        <span class="mono" style="font-size:11px;">${time}</span>
        <span>${txn.transaction_type} · ${txn.location} · ${txn.device_status}</span>
        <span class="mono">$${txn.amount.toFixed(2)}</span>
        <span class="mono" style="color:${txn.fraud_probability > 0.5 ? 'var(--accent-red)' : 'var(--accent-green)'}">${(txn.fraud_probability * 100).toFixed(1)}%</span>
        <span class="badge ${badgeCls}">${txn.status}</span>
    `;
    list.prepend(row);
    if (list.children.length > 100) list.removeChild(list.lastChild);
}

function updateFeedChart() {
    createChart('chartFeedRate', {
        type: 'line',
        data: {
            labels: feedRateData.map((_, i) => i),
            datasets: [{ label: 'Fraud Rate %', data: feedRateData, borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', fill: true, tension: 0.4, pointRadius: 0 }]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false } }, plugins: { legend: { display: false } }, animation: { duration: 0 } }
    });
}

function toggleFeed() {
    feedRunning = !feedRunning;
    const btn = document.getElementById('feedToggle');
    btn.innerHTML = feedRunning ? '<i class="fas fa-pause"></i> Pause' : '<i class="fas fa-play"></i> Resume';
}

function clearFeed() {
    document.getElementById('feedList').innerHTML = '';
    feedCounters = { total: 0, safe: 0, review: 0, fraud: 0 };
    feedRateData = [];
    document.getElementById('feedTotal').textContent = '0';
    document.getElementById('feedSafe').textContent = '0';
    document.getElementById('feedReview').textContent = '0';
    document.getElementById('feedFraud').textContent = '0';
    document.getElementById('feedRate').textContent = '0.00%';
}

// SECTION 3: ANALYZER
const scenarios = {
    grocery: { amount: '50', hour: '12', location: 'domestic', type: 'in_store', merchant: 'established', device: 'verified' },
    online: { amount: '500', hour: '15', location: 'domestic', type: 'online', merchant: 'new', device: 'new_device' },
    latenight: { amount: '1200', hour: '2', location: 'foreign', type: 'atm', merchant: 'new', device: 'new_device' },
    wire: { amount: '5000', hour: '21', location: 'high_risk_country', type: 'wire_transfer', merchant: 'flagged', device: 'compromised' },
    small: { amount: '5', hour: '9', location: 'domestic', type: 'in_store', merchant: 'established', device: 'verified' },
    suspicious: { amount: '5000', hour: '0', location: 'high_risk_country', type: 'online', merchant: 'flagged', device: 'compromised' }
};

function fillScenario(name) {
    const s = scenarios[name];
    document.getElementById('fAmount').value = s.amount;
    document.getElementById('fHour').value = s.hour;
    document.getElementById('fLocation').value = s.location;
    document.getElementById('fType').value = s.type;
    document.getElementById('fMerchant').value = s.merchant;
    document.getElementById('fDevice').value = s.device;
    analyzeTransaction();
}

async function analyzeTransaction() {
    const params = {
        amount: parseFloat(document.getElementById('fAmount').value),
        hour: parseInt(document.getElementById('fHour').value),
        location: document.getElementById('fLocation').value,
        transaction_type: document.getElementById('fType').value,
        merchant_history: document.getElementById('fMerchant').value,
        device_status: document.getElementById('fDevice').value
    };

    document.getElementById('resultContent').innerHTML = '<div class="loading"><div class="spinner"></div>Analyzing with DNN + SHAP + LIME...</div>';
    document.getElementById('shapContent').innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    document.getElementById('limeContent').innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    document.getElementById('riskFactorsContent').innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    const result = await apiPost('/api/predict', params);
    if (!result) {
        document.getElementById('resultContent').innerHTML = '<div style="color:var(--accent-red)">Backend error. Is app.py running?</div>';
        return;
    }

    const probPct = (result.xgb_probability * 100).toFixed(1);
    const colorClass = result.decision === 'FRAUD' ? 'result-fraud' : result.decision === 'REVIEW' ? 'result-review' : 'result-safe';
    const decBadge = result.decision === 'FRAUD' ? 'badge-fraud' : result.decision === 'REVIEW' ? 'badge-review' : 'badge-safe';

    document.getElementById('resultContent').innerHTML = `
        <div class="badge ${decBadge}" style="font-size:14px;padding:6px 20px;">${result.decision}</div>
        <div class="result-prob ${colorClass}">${probPct}%</div>
        <div style="font-size:13px;color:var(--text-secondary);margin-bottom:16px;">Fraud Probability</div>
        <div class="result-meta">
            <div class="result-meta-item"><div class="result-meta-label">Confidence</div><div class="result-meta-value">${(result.confidence * 100).toFixed(1)}%</div></div>
            <div class="result-meta-item"><div class="result-meta-label">Threshold</div><div class="result-meta-value">${result.threshold}</div></div>
            <div class="result-meta-item"><div class="result-meta-label">DNN Score</div><div class="result-meta-value">${(result.fraud_probability * 100).toFixed(1)}%</div></div>
            <div class="result-meta-item"><div class="result-meta-label">XGBoost Score</div><div class="result-meta-value">${(result.xgb_probability * 100).toFixed(1)}%</div></div>
        </div>
        ${result.top_feature ? `<div style="padding:8px 12px;background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.3);border-radius:8px;margin-top:8px;font-size:12px;">
            <strong style="color:var(--accent-purple);">Top SHAP Feature:</strong> <span class="mono">${result.top_feature.feature}</span> — ${result.top_feature.direction} (${result.top_feature.ui_category})
        </div>` : ''}
        <div style="padding:12px;background:var(--bg-secondary);border-radius:8px;font-size:13px;color:var(--text-secondary);margin-top:8px;">
            <strong style="color:var(--text-primary);">Recommended Action:</strong> ${result.recommended_action}
        </div>
    `;

    // Risk Factors
    let rfHtml = '';
    if (result.top_ui_factors && result.top_ui_factors.length > 0) {
        const icons = { 'Location': '📍', 'Device Status': '📱', 'Transaction Type': '💳', 'Merchant History': '🏪', 'Amount': '💰', 'Hour': '🕐' };
        result.top_ui_factors.forEach(f => {
            const lvl = f.risk_level.toLowerCase();
            rfHtml += `<div class="risk-factor ${lvl}">
                <div class="risk-factor-icon">${icons[f.factor] || '⚡'}</div>
                <div class="risk-factor-info">
                    <div class="risk-factor-name">${f.factor}</div>
                    <div class="risk-factor-value">${f.value} (risk: ${(f.risk_score * 100).toFixed(0)}%)</div>
                </div>
                <span class="risk-factor-badge" style="background:${lvl==='high'?'rgba(239,68,68,0.15);color:var(--accent-red)':lvl==='medium'?'rgba(245,158,11,0.15);color:var(--accent-orange)':'rgba(16,185,129,0.15);color:var(--accent-green)'}">${f.risk_level}</span>
            </div>`;
        });
    }
    if (result.shap_summary && result.shap_summary.length > 0) {
        rfHtml += '<div style="margin-top:12px;font-size:12px;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;">SHAP Impact Summary</div>';
        result.shap_summary.forEach(s => {
            const maxImpact = result.shap_summary[0].total_impact;
            const pct = (s.total_impact / maxImpact * 100).toFixed(0);
            rfHtml += `<div class="xai-bar-container" style="margin-top:6px;">
                <div class="xai-bar-label"><span>${s.label}</span><span style="color:var(--text-muted)">${s.total_impact.toFixed(4)}</span></div>
                <div class="xai-bar-track"><div class="xai-bar-fill xai-bar-pos" style="width:${pct}%"></div></div>
            </div>`;
        });
    }
    document.getElementById('riskFactorsContent').innerHTML = rfHtml || '<div style="color:var(--text-muted);font-size:13px;">No significant risk factors detected.</div>';

    // SHAP
    if (result.shap && result.shap.contributions.length > 0) {
        const maxShap = Math.max(...result.shap.contributions.map(c => c.abs_shap));
        let shapHtml = `<div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;">Base value: ${result.shap.base_value.toFixed(4)}</div>`;
        result.shap.contributions.slice(0, 10).forEach(c => {
            const pct = (c.abs_shap / maxShap * 100).toFixed(0);
            const cls = c.shap_value > 0 ? 'xai-bar-pos' : 'xai-bar-neg';
            const dir = c.shap_value > 0 ? '↑ Fraud' : '↓ Safe';
            shapHtml += `<div class="xai-bar-container">
                <div class="xai-bar-label"><span>${c.feature}</span><span style="color:var(--text-muted)">${c.shap_value.toFixed(4)} (${dir})</span></div>
                <div class="xai-bar-track"><div class="xai-bar-fill ${cls}" style="width:${pct}%"></div></div>
            </div>`;
        });
        document.getElementById('shapContent').innerHTML = shapHtml;
    }

    // LIME
    if (result.lime && result.lime.contributions.length > 0) {
        const maxLime = Math.max(...result.lime.contributions.map(c => c.abs_weight));
        let limeHtml = '';
        result.lime.contributions.slice(0, 10).forEach(c => {
            const pct = (c.abs_weight / maxLime * 100).toFixed(0);
            const cls = c.weight > 0 ? 'xai-bar-pos' : 'xai-bar-neg';
            const dir = c.weight > 0 ? '↑ Fraud' : '↓ Safe';
            limeHtml += `<div class="xai-bar-container">
                <div class="xai-bar-label"><span style="font-size:11px;">${c.feature}</span><span style="color:var(--text-muted)">${c.weight.toFixed(4)} (${dir})</span></div>
                <div class="xai-bar-track"><div class="xai-bar-fill ${cls}" style="width:${pct}%"></div></div>
            </div>`;
        });
        document.getElementById('limeContent').innerHTML = limeHtml;
    }
}

