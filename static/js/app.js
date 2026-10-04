(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const MAX_MB = 16;
  const ALLOWED = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'];

  const state = {
    file: null,          // File from the upload tab
    capture: null,       // data URL from the camera tab
    source: 'upload',
    lat: null,
    lng: null,
    stream: null,
    facing: 'environment',
    pickMap: null,
    pickMarker: null,
  };

  // ---------- Helpers ----------
  function toast(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    $('toasts').appendChild(el);
    setTimeout(() => el.remove(), type === 'error' ? 7000 : 4000);
  }

  function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  async function fetchJson(url, options) {
    let response;
    try {
      response = await fetch(url, options);
    } catch {
      throw new Error("Can't reach the server. Check that it's running and try again.");
    }
    let data = null;
    try { data = await response.json(); } catch { /* non-JSON body */ }
    if (!response.ok || (data && data.error)) {
      throw new Error((data && data.error) || `Request failed (${response.status}).`);
    }
    return data;
  }

  function levelOf(score) {
    if (score < 30) return { key: 'low', label: 'Low' };
    if (score < 70) return { key: 'medium', label: 'Medium' };
    return { key: 'high', label: 'High' };
  }

  function setStep(n) {
    document.querySelectorAll('.step').forEach((el) => {
      const s = Number(el.dataset.step);
      el.classList.toggle('is-active', s === n);
      el.classList.toggle('is-done', s < n);
    });
  }

  function hasPhoto() {
    return state.source === 'upload' ? !!state.file : !!state.capture;
  }

  function updateAnalyzeButton() {
    const ready = hasPhoto();
    $('analyzeBtn').disabled = !ready;
    $('analyzeHint').textContent = !ready
      ? 'Add a photo to continue.'
      : state.lat !== null ? 'Ready. The result will be added to the map.'
      : "Ready. Without a location we'll try the photo's GPS data.";
    setStep(ready ? 2 : 1);
  }

  // ---------- Theme ----------
  function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem('theme'); } catch { /* storage blocked */ }
    if (saved) document.documentElement.dataset.theme = saved;
    $('themeToggle').addEventListener('click', () => {
      const current = document.documentElement.dataset.theme
        || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('theme', next); } catch { /* storage blocked */ }
    });
  }

  // ---------- System status ----------
  async function checkHealth() {
    const pill = $('systemStatus');
    try {
      const h = await fetchJson('/health');
      if (!h.yolo_model_loaded) {
        pill.textContent = 'Model not loaded';
        pill.className = 'status-pill warn';
        pill.title = 'Put the weights file at Weights/best.pt and restart the server.';
      } else if (!h.database_accessible) {
        pill.textContent = 'Database offline';
        pill.className = 'status-pill warn';
      } else {
        pill.textContent = 'System ready';
        pill.className = 'status-pill ok';
      }
    } catch {
      pill.textContent = 'Server offline';
      pill.className = 'status-pill warn';
    }
  }

  // ---------- Tabs ----------
  function selectTab(source) {
    state.source = source;
    const upload = source === 'upload';
    $('tabUpload').classList.toggle('is-active', upload);
    $('tabCamera').classList.toggle('is-active', !upload);
    $('tabUpload').setAttribute('aria-selected', upload);
    $('tabCamera').setAttribute('aria-selected', !upload);
    $('panelUpload').hidden = !upload;
    $('panelCamera').hidden = upload;
    if (upload) stopCamera();
    updateAnalyzeButton();
  }

  // ---------- Upload ----------
  function setFile(file) {
    if (!file) return;
    if (!ALLOWED.includes(file.type)) {
      toast('Unsupported file type. Please choose a JPG, PNG, BMP or WEBP image.', 'error');
      return;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      toast(`That image is ${(file.size / 1048576).toFixed(1)} MB. Please use one under ${MAX_MB} MB.`, 'error');
      return;
    }
    state.file = file;
    const img = $('previewImg');
    if (img.src) URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(file);
    img.hidden = false;
    $('dropEmpty').hidden = true;
    $('fileMeta').hidden = false;
    $('fileName').textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
    updateAnalyzeButton();
  }

  function clearFile() {
    state.file = null;
    $('fileInput').value = '';
    $('previewImg').hidden = true;
    $('dropEmpty').hidden = false;
    $('fileMeta').hidden = true;
    updateAnalyzeButton();
  }

  function initUpload() {
    const zone = $('dropZone');
    $('fileInput').addEventListener('change', (e) => setFile(e.target.files[0]));
    ['dragenter', 'dragover'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add('is-over'); }));
    ['dragleave', 'drop'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove('is-over'); }));
    zone.addEventListener('drop', (e) => setFile(e.dataTransfer.files[0]));
    $('clearFile').addEventListener('click', (e) => { e.preventDefault(); clearFile(); });
    document.addEventListener('paste', (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
      if (item) { selectTab('upload'); setFile(item.getAsFile()); }
    });
  }

  // ---------- Camera ----------
  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      toast("This browser can't use the camera. Try uploading a photo instead.", 'error');
      return;
    }
    stopCamera();
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: state.facing, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false,
      });
      const video = $('video');
      video.srcObject = state.stream;
      await video.play();
      video.hidden = false;
      $('cameraPlaceholder').hidden = true;
      $('capturePreview').hidden = true;
      $('snapBtn').disabled = false;
      $('switchCamera').hidden = false;
      $('retakeBtn').hidden = true;
    } catch (err) {
      const msg = err.name === 'NotAllowedError'
        ? 'Camera access was blocked. Allow it in your browser settings, or upload a photo instead.'
        : "Couldn't start the camera. Is another app using it?";
      toast(msg, 'error');
    }
  }

  function stopCamera() {
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    $('snapBtn').disabled = true;
    $('switchCamera').hidden = true;
    if (!state.capture) $('cameraPlaceholder').hidden = false;
  }

  function snap() {
    const video = $('video');
    const canvas = $('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    state.capture = canvas.toDataURL('image/jpeg', 0.9);
    $('capturePreview').src = state.capture;
    $('capturePreview').hidden = false;
    video.hidden = true;
    stopCamera();
    $('cameraPlaceholder').hidden = true;
    $('retakeBtn').hidden = false;
    updateAnalyzeButton();
  }

  function retake() {
    state.capture = null;
    $('capturePreview').hidden = true;
    updateAnalyzeButton();
    startCamera();
  }

  function initCamera() {
    $('tabUpload').addEventListener('click', () => selectTab('upload'));
    $('tabCamera').addEventListener('click', () => selectTab('camera'));
    $('startCamera').addEventListener('click', startCamera);
    $('snapBtn').addEventListener('click', snap);
    $('retakeBtn').addEventListener('click', retake);
    $('switchCamera').addEventListener('click', () => {
      state.facing = state.facing === 'environment' ? 'user' : 'environment';
      startCamera();
    });
    window.addEventListener('pagehide', stopCamera);
  }

  // ---------- Location ----------
  function setLocation(lat, lng, label) {
    state.lat = lat;
    state.lng = lng;
    const status = $('locationStatus');
    if (lat === null) {
      status.textContent = 'No location yet.';
      status.className = 'location-status';
    } else {
      status.textContent = `${label}: ${lat.toFixed(5)}, ${lng.toFixed(5)}`;
      status.className = 'location-status ok';
      $('latInput').value = lat.toFixed(6);
      $('lngInput').value = lng.toFixed(6);
      if (state.pickMap) placePin(lat, lng, false);
    }
    updateAnalyzeButton();
  }

  function locate() {
    const status = $('locationStatus');
    if (!navigator.geolocation) {
      status.textContent = "This browser can't share location. Enter it manually below.";
      status.className = 'location-status error';
      return;
    }
    const btn = $('locateBtn');
    btn.disabled = true;
    status.textContent = 'Finding your location…';
    status.className = 'location-status';
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        btn.disabled = false;
        setLocation(pos.coords.latitude, pos.coords.longitude, `Located (±${Math.round(pos.coords.accuracy)} m)`);
      },
      (err) => {
        btn.disabled = false;
        status.textContent = err.code === err.PERMISSION_DENIED
          ? 'Location access was blocked. You can enter or pick it manually below.'
          : "Couldn't get your location. Try again or enter it manually.";
        status.className = 'location-status error';
        $('manualLocation').open = true;
        initPickMap();
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 60000 }
    );
  }

  function placePin(lat, lng, pan = true) {
    if (!state.pickMap) return;
    if (state.pickMarker) state.pickMarker.setLatLng([lat, lng]);
    else state.pickMarker = L.marker([lat, lng]).addTo(state.pickMap);
    if (pan) state.pickMap.panTo([lat, lng]);
    else state.pickMap.setView([lat, lng], Math.max(state.pickMap.getZoom(), 14));
  }

  function initPickMap() {
    if (state.pickMap) return;
    if (typeof L === 'undefined') {
      $('pickMap').innerHTML = '<p class="empty">The map could not load. Enter the coordinates above instead.</p>';
      return;
    }
    state.pickMap = L.map('pickMap').setView([20.59, 78.96], 4);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19, attribution: '&copy; OpenStreetMap contributors',
    }).addTo(state.pickMap);
    state.pickMap.on('click', (e) => setLocation(e.latlng.lat, e.latlng.lng, 'Pinned'));
    if (state.lat !== null) placePin(state.lat, state.lng, false);
    setTimeout(() => state.pickMap.invalidateSize(), 50);
  }

  function readManual() {
    const lat = parseFloat($('latInput').value);
    const lng = parseFloat($('lngInput').value);
    if (Number.isFinite(lat) && Number.isFinite(lng) && Math.abs(lat) <= 90 && Math.abs(lng) <= 180) {
      setLocation(lat, lng, 'Entered');
    } else if ($('latInput').value || $('lngInput').value) {
      const status = $('locationStatus');
      status.textContent = 'Latitude must be between -90 and 90, longitude between -180 and 180.';
      status.className = 'location-status error';
      state.lat = state.lng = null;
      updateAnalyzeButton();
    }
  }

  function initLocation() {
    $('locateBtn').addEventListener('click', locate);
    $('manualLocation').addEventListener('toggle', (e) => { if (e.target.open) initPickMap(); });
    $('latInput').addEventListener('change', readManual);
    $('lngInput').addEventListener('change', readManual);
  }

  // ---------- Analyse ----------
  async function analyze() {
    const btn = $('analyzeBtn');
    btn.disabled = true;
    btn.classList.add('is-loading');
    btn.querySelector('.btn-label').textContent = 'Analysing…';

    try {
      let data;
      if (state.source === 'upload') {
        const form = new FormData();
        form.append('file', state.file);
        if (state.lat !== null) {
          form.append('latitude', state.lat);
          form.append('longitude', state.lng);
        }
        data = await fetchJson('/upload_with_location', { method: 'POST', body: form });
      } else {
        data = await fetchJson('/capture_image', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: state.capture, latitude: state.lat, longitude: state.lng }),
        });
      }
      showResult(data);
      if (data.analysis.saved) { loadStats(); reloadMap(); }
    } catch (err) {
      toast(err.message, 'error');
    } finally {
      btn.classList.remove('is-loading');
      btn.querySelector('.btn-label').textContent = 'Analyse photo';
      updateAnalyzeButton();
    }
  }

  function showResult(data) {
    const a = data.analysis;
    const url = `/result/${encodeURIComponent(data.result_file)}?t=${Date.now()}`;
    $('resultImg').src = url;
    $('resultLink').href = url;
    $('downloadBtn').href = url;
    $('downloadBtn').setAttribute('download', data.result_file);

    const score = Number(a.pollution_score) || 0;
    const level = a.level || levelOf(score);
    $('scoreValue').textContent = Math.round(score);
    const badge = $('scoreLevel');
    badge.textContent = level.label;
    badge.className = `badge ${level.key}`;
    const fill = $('gaugeFill');
    fill.style.stroke = `var(--${level.key})`;
    requestAnimationFrame(() => { fill.style.strokeDashoffset = 157.08 * (1 - Math.min(score, 100) / 100); });

    $('itemCount').textContent = a.count === 0 ? 'None detected' : a.count;
    const where = a.location_name || (a.coordinates ? `${a.coordinates.lat.toFixed(5)}, ${a.coordinates.lng.toFixed(5)}` : 'Not available');
    $('resultLocation').textContent = a.location_source === 'photo' ? `${where} (from photo GPS)` : where;

    const breakdown = $('breakdown');
    breakdown.innerHTML = Object.entries(a.breakdown || {})
      .map(([name, n]) => `<li>${escapeHtml(name)} × ${n}</li>`).join('');

    const note = $('saveNote');
    note.textContent = a.save_message;
    note.className = `note ${a.saved ? 'ok' : 'warn'}`;

    $('resultCard').hidden = false;
    setStep(3);
    $('resultCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
    $('resultCard').focus({ preventScroll: true });
    toast(a.count ? `Found ${a.count} item${a.count > 1 ? 's' : ''} of garbage.` : 'No garbage detected in this photo.', 'success');
  }

  function newScan() {
    $('resultCard').hidden = true;
    clearFile();
    state.capture = null;
    $('capturePreview').hidden = true;
    $('retakeBtn').hidden = true;
    if (state.source === 'camera') $('cameraPlaceholder').hidden = false;
    $('gaugeFill').style.strokeDashoffset = 157.08;
    $('detect').scrollIntoView({ behavior: 'smooth' });
    updateAnalyzeButton();
  }

  // ---------- Dashboard ----------
  function renderTrend(points) {
    const el = $('trendChart');
    const withData = points.filter((p) => p.score !== null);
    if (!withData.length) {
      el.innerHTML = '<p class="empty">No reports in the last 7 days yet.</p>';
      return;
    }
    const W = 520, H = 200, padL = 32, padR = 12, padT = 12, padB = 28;
    const x = (i) => padL + (i * (W - padL - padR)) / Math.max(points.length - 1, 1);
    const y = (v) => padT + (1 - v / 100) * (H - padT - padB);
    const fmt = (d) => new Date(`${d}T00:00:00`).toLocaleDateString(undefined, { weekday: 'short' });

    let path = '';
    let area = '';
    let segStart = null;
    points.forEach((p, i) => {
      if (p.score === null) { if (segStart !== null) { area += `L${x(i - 1)},${y(0)}Z`; segStart = null; } return; }
      const cmd = segStart === null ? 'M' : 'L';
      if (segStart === null) { segStart = i; area += `M${x(i)},${y(0)}L${x(i)},${y(p.score)}`; } else area += `L${x(i)},${y(p.score)}`;
      path += `${cmd}${x(i)},${y(p.score)}`;
    });
    if (segStart !== null) area += `L${x(points.length - 1)},${y(0)}Z`;

    const grid = [0, 30, 70, 100].map((v) =>
      `<line class="axis" x1="${padL}" x2="${W - padR}" y1="${y(v)}" y2="${y(v)}" stroke-dasharray="${v ? '3 4' : ''}"/>
       <text x="${padL - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join('');
    const labels = points.map((p, i) => `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${fmt(p.date)}</text>`).join('');
    const dots = points.map((p, i) => p.score === null ? '' :
      `<circle class="pt" cx="${x(i)}" cy="${y(p.score)}" r="4"><title>${fmt(p.date)}: ${p.score} (${p.count} report${p.count > 1 ? 's' : ''})</title></circle>`).join('');

    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">${grid}
      <path class="area" d="${area}" opacity=".7"/><path class="line" d="${path}"/>${dots}${labels}</svg>`;
  }

  function renderList(el, items, render, emptyText) {
    el.innerHTML = items.length ? items.map(render).join('') : `<li class="empty">${emptyText}</li>`;
  }

  async function loadStats() {
    try {
      const d = await fetchJson('/get_pollution_data');
      const s = d.recent_stats;
      $('statReports').textContent = s.total_detections;
      $('statItems').textContent = s.items_found ?? '–';
      $('statAvg').textContent = s.avg_pollution_score;
      $('statMax').textContent = s.max_pollution_score;
      renderTrend(d.trend_data || []);
      renderList($('hotspots'), d.hotspots || [], (h) =>
        `<li><span class="dot ${h.level}"></span><span class="grow" title="${escapeHtml(h.name)}">${escapeHtml(h.name)}</span>
         <strong>${h.score}</strong><span class="muted small">${h.count} report${h.count > 1 ? 's' : ''}</span></li>`,
        'No hotspots yet. Analyse a photo with a location to get started.');
      renderList($('recentList'), d.recent || [], (r) =>
        `<li><span class="dot ${r.level}"></span><span class="grow">${escapeHtml(r.location_name || 'Unknown location')}</span>
         <span class="muted small">${new Date(r.time).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}</span>
         <strong>${r.score}</strong></li>`,
        'No reports yet.');
    } catch (err) {
      ['statReports', 'statItems', 'statAvg', 'statMax'].forEach((id) => { $(id).textContent = '–'; });
      $('trendChart').innerHTML = `<p class="empty">${escapeHtml(err.message)}</p>`;
    }
  }

  // ---------- Map ----------
  function reloadMap() {
    $('mapFrame').src = `/generate_pollution_map?t=${Date.now()}`;
  }

  // ---------- Nav highlight ----------
  function initNav() {
    const links = [...document.querySelectorAll('.nav-link')];
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) links.forEach((l) => l.classList.toggle('is-active', l.getAttribute('href') === `#${e.target.id}`));
      });
    }, { rootMargin: '-40% 0px -55% 0px' });
    ['detect', 'dashboard', 'map'].forEach((id) => observer.observe($(id)));
  }

  // ---------- Init ----------
  initTheme();
  initUpload();
  initCamera();
  initLocation();
  initNav();
  $('analyzeBtn').addEventListener('click', analyze);
  $('newScanBtn').addEventListener('click', newScan);
  $('refreshStats').addEventListener('click', loadStats);
  $('reloadMap').addEventListener('click', reloadMap);
  checkHealth();
  loadStats();
  reloadMap();
})();
