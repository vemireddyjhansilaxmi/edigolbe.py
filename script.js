// script.js - Handles API calls and dynamic updates for the COVID-19 Dashboard

// Utility: safe fetch + json
async function safeFetchJson(url) {
    const res = await fetch(url);
    const text = await res.text();
    try {
        return { ok: res.ok, data: JSON.parse(text), raw: text, status: res.status };
    } catch {
        return { ok: res.ok, data: null, raw: text, status: res.status };
    }
}

async function loadDemographics() {
    const r = await safeFetchJson('/api/demographics');
    const el = document.getElementById('demographics');
    const plotContainer = document.getElementById('demographics-plot');
    if (!r.ok || !r.data) {
        if (el) el.innerHTML = `<p class="text-danger">Failed to load demographics: ${r.raw || r.status}</p>`;
        if (plotContainer) plotContainer.innerHTML = '';
        return;
    }
    const data = r.data || {};
    if (el) el.innerHTML = `
        <h6>Gender: ${Object.entries(data.gender || {}).map(([k,v]) => `${k}: ${v}`).join(', ') || 'N/A'}</h6>
        <h6>Age: ${data.age?.mean ? Number(data.age.mean).toFixed(1) + ' yrs avg' : 'N/A'}</h6>
        <h6>Top Region: ${Object.keys(data.region || {})[0] || 'N/A'}</h6>
    `;
    if (plotContainer) {
        plotContainer.innerHTML = `
            <img src="/api/plot/gender" class="img-fluid mb-2" alt="Gender plot">
            <img src="/api/plot/age" class="img-fluid mb-2" alt="Age plot">
            <img src="/api/plot/region" class="img-fluid" alt="Region plot">
        `;
    }
}

async function loadOutcomes() {
    const r = await safeFetchJson('/api/outcomes');
    const el = document.getElementById('recovery-stats');
    if (!r.ok) {
        if (el) el.innerHTML = `<p class="text-danger">Failed to load outcomes: ${r.raw || r.status}</p>`;
        return;
    }
    const d = r.data || {};
    if (el) el.innerHTML = `<p><strong>Recovery Stats:</strong> ${JSON.stringify(d.recovery || {})}</p>`;
    const recoveryImg = document.getElementById('recovery-plot');
    if (recoveryImg) recoveryImg.src = '/api/plot/recovery';
}

async function loadRegressionStats() {
    const r = await safeFetchJson('/api/regression_stats');
    const el = document.getElementById('regression-stats');
    if (!r.ok || !r.data) {
        if (el) el.innerHTML = `<p class="text-danger">Regression stats not available: ${r.data?.error || r.raw || r.status}</p>`;
        return;
    }
    const d = r.data;
    if (el) el.innerHTML = `
        <p><strong>R²:</strong> ${d.r2_score}</p>
        <p><strong>MSE:</strong> ${d.mse}</p>
        <p><strong>Coefficients:</strong> ${JSON.stringify(d.coefficients)}</p>
    `;
}

async function submitPrediction(event) {
    event.preventDefault();
    const resultEl = document.getElementById('prediction-result');
    if (resultEl) resultEl.textContent = 'Loading...';
    const age = Number(document.getElementById('age')?.value || 0);
    const contact_number = Number(document.getElementById('contact')?.value || 0);
    const infection_order = Number(document.getElementById('order')?.value || 0);
    const payload = { age, contact_number, infection_order };

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const text = await res.text();
        let data;
        try { data = JSON.parse(text); } catch { data = { raw: text }; }
        if (!res.ok) {
            const msg = data && data.error ? data.error : (data.raw || res.statusText);
            throw new Error(msg);
        }
        if (resultEl) resultEl.textContent = 'Predicted recovery days: ' + (data.predicted_recovery_days ?? JSON.stringify(data));
        console.log('Prediction response:', data);
    } catch (err) {
        console.error('Prediction error:', err);
        if (resultEl) resultEl.textContent = 'Error making prediction: ' + (err.message || 'unknown');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    loadDemographics();
    loadOutcomes();
    loadRegressionStats();

    const recoveryImg = document.getElementById('recovery-plot');
    if (recoveryImg) recoveryImg.src = '/api/plot/recovery';

    const form = document.getElementById('predict-form');
    if (form) form.addEventListener('submit', submitPrediction);
     fetch("http://127.0.0.1:5000/data")
  .then(res => res.json())
  .then(data => {
      console.log(data);
  });
});