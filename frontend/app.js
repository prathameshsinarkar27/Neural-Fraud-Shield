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

