'use strict';

const uploadSection   = document.getElementById('uploadSection');
const loadingSection  = document.getElementById('loadingSection');
const resultsSection  = document.getElementById('resultsSection');
const errorSection    = document.getElementById('errorSection');
const dropZone        = document.getElementById('dropZone');
const fileInput       = document.getElementById('fileInput');
const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
const errorRetryBtn   = document.getElementById('errorRetryBtn');

function showSection(id) {
  [uploadSection, loadingSection, resultsSection, errorSection].forEach(s => s.classList.add('hidden'));
  document.getElementById(id).classList.remove('hidden');
}

// Drag & Drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

dropZone.addEventListener('click', e => {
  if (e.target !== fileInput && !e.target.classList.contains('btn-upload')) fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

analyzeAnotherBtn.addEventListener('click', reset);
errorRetryBtn.addEventListener('click', reset);

function reset() {
  fileInput.value = '';
  showSection('uploadSection');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('Please upload an image file (JPG, PNG, WEBP, etc.)');
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showError('File size exceeds 16 MB limit.');
    return;
  }

  showSection('loadingSection');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || 'Unknown error from server.');
      return;
    }

    renderResults(data);
  } catch (err) {
    showError('Network error — make sure the server is running.');
  }
}

function renderResults(data) {
  const info = data.disease_info || {};
  const isHealthy = info.status === 'Healthy';
  const confidence = data.confidence;

  // Image
  document.getElementById('resultImage').src = `data:image/jpeg;base64,${data.image_b64}`;

  // Disease name & crop
  document.getElementById('diseaseName').textContent = info.display_name || data.predicted_class;
  document.getElementById('cropName').textContent = info.crop ? `Crop: ${info.crop}` : '';

  // Status badge
  const badge = document.getElementById('statusBadge');
  badge.textContent = isHealthy ? '✓' : '!';
  badge.className = 'status-badge ' + (isHealthy ? 'healthy' : 'diseased');

  // Confidence
  document.getElementById('confidenceVal').textContent = `${confidence}%`;
  const fill = document.getElementById('confidenceFill');
  fill.style.width = `${confidence}%`;
  fill.style.background = confidence >= 80 ? '#16a34a' : confidence >= 50 ? '#f59e0b' : '#ef4444';

  // Severity
  const sev = info.severity || 'Unknown';
  const sevEl = document.getElementById('severityValue');
  sevEl.textContent = sev;
  const sevKey = sev.toLowerCase().replace(/\s+to\s+.+/, '').trim();
  const sevClass = sevKey === 'none' ? 'severity-none'
    : sevKey === 'moderate' ? 'severity-moderate'
    : sevKey === 'high' ? 'severity-high'
    : sevKey === 'very' ? 'severity-veryhigh'
    : 'severity-moderate';
  sevEl.className = 'severity-value ' + sevClass;

  document.getElementById('description').textContent = info.description || '';

  // Symptoms
  const symptomsBlock = document.getElementById('symptomsBlock');
  const symptomsList  = document.getElementById('symptomsList');
  const symptoms = info.symptoms || [];
  if (symptoms.length > 0) {
    symptomsList.innerHTML = symptoms.map(s => `<li>${s}</li>`).join('');
    symptomsBlock.classList.remove('hidden');
  } else {
    symptomsBlock.classList.add('hidden');
  }

  document.getElementById('treatmentText').textContent = info.treatment || '—';
  document.getElementById('preventionText').textContent = info.prevention || '—';

  // Top-5 bars
  const top5List = document.getElementById('top5List');
  top5List.innerHTML = (data.top5 || []).map((item, idx) => {
    const label = item.class.replace(/___/g, ' — ').replace(/_/g, ' ');
    return `
      <div class="top5-item">
        <div class="top5-meta">
          <span class="top5-label">${label}</span>
          <span class="top5-pct">${item.probability}%</span>
        </div>
        <div class="top5-bar-bg">
          <div class="top5-bar-fill ${idx === 0 ? 'top1' : ''}" style="width:${item.probability}%"></div>
        </div>
      </div>`;
  }).join('');

  showSection('resultsSection');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showError(msg) {
  document.getElementById('errorMessage').textContent = msg;
  showSection('errorSection');
}
