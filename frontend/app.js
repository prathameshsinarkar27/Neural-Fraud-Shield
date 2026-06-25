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
