// Face Tracker frontend: streams webcam frames (or an uploaded image) to the
// FastAPI backend and draws the model's bounding box on a canvas.
(() => {
  const $ = (id) => document.getElementById(id);

  const display = $("display");
  const ctx = display.getContext("2d");
  const video = $("video");
  const spark = $("spark");
  const sctx = spark.getContext("2d");
  const capture = document.createElement("canvas");
  const cctx = capture.getContext("2d");

  const els = {
    tabs: document.querySelectorAll(".tab"),
    cameraPlaceholder: $("cameraPlaceholder"),
    dropzone: $("dropzone"),
    fileInput: $("fileInput"),
    statusPill: $("statusPill"),
    cameraToolbar: $("cameraToolbar"),
    imageToolbar: $("imageToolbar"),
    startBtn: $("startBtn"),
    startBtnInline: $("startBtnInline"),
    cameraSelect: $("cameraSelect"),
    snapshotBtn: $("snapshotBtn"),
    chooseBtn: $("chooseBtn"),
    saveImageBtn: $("saveImageBtn"),
    imageName: $("imageName"),
    faceBadge: $("faceBadge"),
    confValue: $("confValue"),
    confFill: $("confFill"),
    confMark: $("confMark"),
    statFps: $("statFps"),
    statRtt: $("statRtt"),
    statInfer: $("statInfer"),
    statBox: $("statBox"),
    threshold: $("threshold"),
    thresholdOut: $("thresholdOut"),
    squareCrop: $("squareCrop"),
    mirror: $("mirror"),
    smooth: $("smooth"),
    showZone: $("showZone"),
    modelChip: $("modelChip"),
  };

  const SEND_SIZE = 240; // longest side of frames sent to the server
  const HISTORY = 120;

  const state = {
    mode: "camera",
    stream: null,
    ws: null,
    running: false,
    inflight: false,
    sentAt: 0,
    sentCrop: null,
    responseTimes: [],
    result: null, // latest {score, box (source px), inference_ms}
    shownBox: null, // smoothed box in source px
    visibility: 0, // 0..1 fade for the box
    history: [],
    image: null,
    imageCrop: null,
  };

  const threshold = () => parseFloat(els.threshold.value);

  // ---------- helpers ----------

  function cropFor(w, h) {
    if (!els.squareCrop.checked) return { x: 0, y: 0, w, h };
    const s = Math.min(w, h);
    return { x: (w - s) / 2, y: (h - s) / 2, w: s, h: s };
  }

  function toSourceBox(box, crop) {
    return [
      crop.x + box[0] * crop.w,
      crop.y + box[1] * crop.h,
      crop.x + box[2] * crop.w,
      crop.y + box[3] * crop.h,
    ];
  }

  function encodeCrop(source, crop) {
    const scale = SEND_SIZE / Math.max(crop.w, crop.h);
    capture.width = Math.round(crop.w * scale);
    capture.height = Math.round(crop.h * scale);
    cctx.drawImage(source, crop.x, crop.y, crop.w, crop.h, 0, 0, capture.width, capture.height);
    return new Promise((resolve) => capture.toBlob(resolve, "image/jpeg", 0.9));
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function setStatus(text, isError = false) {
    els.statusPill.hidden = !text;
    els.statusPill.textContent = text || "";
    els.statusPill.classList.toggle("error", isError);
  }

  function download(canvas, name) {
    canvas.toBlob((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, "image/png");
  }

  // ---------- drawing ----------

  function drawZone(crop, W, H) {
    if (!els.showZone.checked || !els.squareCrop.checked) return;
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.beginPath();
    ctx.rect(0, 0, W, H);
    ctx.rect(crop.x, crop.y, crop.w, crop.h);
    ctx.fill("evenodd");
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.setLineDash([8, 6]);
    ctx.lineWidth = Math.max(1, W / 640);
    ctx.strokeRect(crop.x, crop.y, crop.w, crop.h);
    ctx.restore();
  }

  function drawBox(b, score, alpha, W) {
    const [x1, y1, x2, y2] = b;
    const w = x2 - x1, h = y2 - y1;
    if (w <= 1 || h <= 1) return;
    const color = cssVar("--face") || "#34d399";
    const lw = Math.max(2, W / 260);
    const corner = Math.min(w, h) * 0.18;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = lw * 0.5;
    ctx.strokeRect(x1, y1, w, h);

    // Corner brackets
    ctx.lineWidth = lw * 1.6;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (const [cx, cy, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]]) {
      ctx.moveTo(cx + dx * corner, cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + dy * corner);
    }
    ctx.stroke();

    // Label
    const fontSize = Math.max(12, W / 45);
    ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
    const text = `face ${(score * 100).toFixed(0)}%`;
    const pad = fontSize * 0.4;
    const tw = ctx.measureText(text).width + pad * 2;
    const th = fontSize + pad * 1.2;
    const ly = y1 - th - lw > 0 ? y1 - th - lw : y1 + lw;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x1, ly, tw, th, 4);
    ctx.fill();
    ctx.fillStyle = "#06120c";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x1 + pad, ly + th / 2 + 1);
    ctx.restore();
  }

  function drawSpark() {
    const dpr = window.devicePixelRatio || 1;
    const w = spark.clientWidth, h = spark.clientHeight;
    if (spark.width !== w * dpr) { spark.width = w * dpr; spark.height = h * dpr; }
    sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    sctx.clearRect(0, 0, w, h);

    const ty = h - threshold() * (h - 4) - 2;
    sctx.strokeStyle = cssVar("--border");
    sctx.setLineDash([3, 3]);
    sctx.lineWidth = 1;
    sctx.beginPath(); sctx.moveTo(0, ty); sctx.lineTo(w, ty); sctx.stroke();
    sctx.setLineDash([]);

    const pts = state.history;
    if (pts.length < 2) return;
    const step = w / (HISTORY - 1);
    const x0 = w - (pts.length - 1) * step;
    sctx.beginPath();
    pts.forEach((v, i) => {
      const x = x0 + i * step, y = h - v * (h - 4) - 2;
      i ? sctx.lineTo(x, y) : sctx.moveTo(x, y);
    });
    sctx.strokeStyle = cssVar("--face");
    sctx.lineWidth = 1.75;
    sctx.lineJoin = "round";
    sctx.stroke();
  }

  function renderCamera() {
    if (!state.running) return;
    const W = video.videoWidth, H = video.videoHeight;
    if (W && H) {
      if (display.width !== W || display.height !== H) { display.width = W; display.height = H; }
      const mirror = els.mirror.checked;
      ctx.save();
      if (mirror) { ctx.translate(W, 0); ctx.scale(-1, 1); }
      ctx.drawImage(video, 0, 0, W, H);
      ctx.restore();

      // Crop is symmetric about the centre, so mirroring doesn't move it.
      drawZone(cropFor(W, H), W, H);

      const r = state.result;
      const target = r && r.score >= threshold() ? 1 : 0;
      state.visibility += (target - state.visibility) * 0.25;
      if (state.shownBox && state.visibility > 0.02) {
        let [x1, y1, x2, y2] = state.shownBox;
        if (mirror) [x1, x2] = [W - x2, W - x1];
        drawBox([x1, y1, x2, y2], r ? r.score : 0, state.visibility, W);
      }
    }
    requestAnimationFrame(renderCamera);
  }

  function renderImage() {
    const img = state.image;
    if (!img) return;
    const W = img.naturalWidth, H = img.naturalHeight;
    display.width = W; display.height = H;
    ctx.drawImage(img, 0, 0);
    if (state.imageCrop) drawZone(state.imageCrop, W, H);
    const r = state.result;
    if (r && r.score >= threshold()) drawBox(r.sourceBox, r.score, 1, W);
  }

  // ---------- results ----------

  function applyResult(r, crop, rtt) {
    const sourceBox = toSourceBox(r.box, crop);
    state.result = { ...r, sourceBox };

    if (state.mode === "camera") {
      if (r.score >= threshold()) {
        if (!state.shownBox || state.visibility < 0.05 || !els.smooth.checked) {
          state.shownBox = sourceBox;
        } else {
          const a = 0.55;
          state.shownBox = state.shownBox.map((v, i) => v + (sourceBox[i] - v) * a);
        }
      }
      state.history.push(r.score);
      if (state.history.length > HISTORY) state.history.shift();

      const now = performance.now();
      state.responseTimes.push(now);
      while (state.responseTimes[0] < now - 1000) state.responseTimes.shift();
      els.statFps.textContent = state.responseTimes.length;
    }

    els.statRtt.textContent = rtt != null ? Math.round(rtt) : "–";
    els.statInfer.textContent = r.inference_ms.toFixed(0);
    updateDetectionUI();
  }

  function updateDetectionUI() {
    const r = state.result;
    const t = threshold();
    els.thresholdOut.textContent = t.toFixed(2);
    els.confMark.style.left = `${t * 100}%`;

    if (!r) {
      els.confValue.textContent = "–";
      els.confFill.style.width = "0";
      els.faceBadge.textContent = "Idle";
      els.faceBadge.classList.remove("on");
      els.statBox.textContent = "–";
      drawSpark();
      return;
    }
    const on = r.score >= t;
    els.confValue.textContent = r.score.toFixed(2);
    els.confFill.style.width = `${(r.score * 100).toFixed(1)}%`;
    els.confFill.classList.toggle("on", on);
    els.faceBadge.textContent = on ? "Face detected" : "No face";
    els.faceBadge.classList.toggle("on", on);
    els.statBox.textContent = on ? r.box.map((v) => v.toFixed(2)).join(", ") : "–";
    drawSpark();
  }

  function resetStats() {
    state.result = null;
    state.shownBox = null;
    state.visibility = 0;
    state.history = [];
    state.responseTimes = [];
    els.statFps.textContent = els.statRtt.textContent = els.statInfer.textContent = "–";
    updateDetectionUI();
  }

  // ---------- camera ----------

  function wsUrl() {
    return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  }

  function connect() {
    const ws = new WebSocket(wsUrl());
    ws.binaryType = "arraybuffer";
    state.ws = ws;
    state.inflight = false;
    setStatus("Connecting…");

    ws.onopen = () => { setStatus(""); pump(); };
    ws.onmessage = (ev) => {
      const r = JSON.parse(ev.data);
      const rtt = performance.now() - state.sentAt;
      state.inflight = false;
      if (r.error) setStatus(r.error, true);
      else applyResult(r, state.sentCrop, rtt);
      pump();
    };
    ws.onclose = () => {
      if (state.ws !== ws || !state.running) return;
      setStatus("Connection lost — retrying…", true);
      setTimeout(() => state.running && connect(), 1000);
    };
  }

  async function pump() {
    const ws = state.ws;
    if (!state.running || state.inflight || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (!video.videoWidth) { setTimeout(pump, 50); return; }
    state.inflight = true;
    const crop = cropFor(video.videoWidth, video.videoHeight);
    const blob = await encodeCrop(video, crop);
    if (!state.running || ws.readyState !== WebSocket.OPEN) { state.inflight = false; return; }
    state.sentCrop = crop;
    state.sentAt = performance.now();
    ws.send(await blob.arrayBuffer());
  }

  async function listCameras(activeId) {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    els.cameraSelect.innerHTML = "";
    devices.forEach((d, i) => {
      const opt = new Option(d.label || `Camera ${i + 1}`, d.deviceId);
      opt.selected = d.deviceId === activeId;
      els.cameraSelect.add(opt);
    });
    els.cameraSelect.disabled = devices.length < 2;
  }

  async function startCamera(deviceId) {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("Camera access needs a secure context (use http://localhost).", true);
      return;
    }
    stopCamera();
    setStatus("Requesting camera…");
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
                        : { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
    } catch (err) {
      setStatus(`Camera unavailable: ${err.message || err.name}`, true);
      return;
    }
    video.srcObject = state.stream;
    await video.play();
    state.running = true;
    els.cameraPlaceholder.hidden = true;
    els.startBtn.textContent = "Stop camera";
    els.snapshotBtn.disabled = false;
    resetStats();
    const activeId = state.stream.getVideoTracks()[0].getSettings().deviceId;
    listCameras(activeId).catch(() => {});
    connect();
    requestAnimationFrame(renderCamera);
  }

  function stopCamera() {
    state.running = false;
    if (state.ws) { const ws = state.ws; state.ws = null; ws.close(); }
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    video.srcObject = null;
    els.startBtn.textContent = "Start camera";
    els.snapshotBtn.disabled = true;
    setStatus("");
  }

  // ---------- image ----------

  async function detectImage() {
    const img = state.image;
    if (!img) return;
    const crop = cropFor(img.naturalWidth, img.naturalHeight);
    state.imageCrop = els.squareCrop.checked ? crop : null;
    setStatus("Detecting…");
    try {
      const form = new FormData();
      form.append("file", await encodeCrop(img, crop), "frame.jpg");
      const t0 = performance.now();
      const res = await fetch("/api/detect", { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const r = await res.json();
      setStatus("");
      applyResult(r, crop, performance.now() - t0);
      els.saveImageBtn.disabled = false;
    } catch (err) {
      setStatus(`Detection failed: ${err.message}`, true);
    }
    renderImage();
  }

  function loadImage(file) {
    if (!file || !file.type.startsWith("image/")) {
      setStatus("That file isn't an image.", true);
      return;
    }
    const img = new Image();
    img.onload = () => {
      state.image = img;
      els.dropzone.hidden = true;
      els.imageName.textContent = file.name;
      resetStats();
      renderImage();
      detectImage();
    };
    img.onerror = () => setStatus("Could not read that image.", true);
    img.src = URL.createObjectURL(file);
  }

  // ---------- mode switching ----------

  function setMode(mode) {
    if (state.mode === mode) return;
    state.mode = mode;
    els.tabs.forEach((t) => {
      const active = t.dataset.mode === mode;
      t.classList.toggle("active", active);
      t.setAttribute("aria-selected", active);
    });
    const camera = mode === "camera";
    els.cameraToolbar.hidden = !camera;
    els.imageToolbar.hidden = camera;
    els.mirror.closest(".switch").hidden = !camera;
    els.smooth.closest(".switch").hidden = !camera;

    stopCamera();
    resetStats();
    ctx.clearRect(0, 0, display.width, display.height);
    if (camera) {
      els.dropzone.hidden = true;
      els.cameraPlaceholder.hidden = false;
      state.image = null;
    } else {
      els.cameraPlaceholder.hidden = true;
      els.dropzone.hidden = !!state.image;
      renderImage();
    }
  }

  // ---------- events ----------

  els.tabs.forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));

  els.startBtn.addEventListener("click", () => {
    if (!state.running) return startCamera();
    stopCamera();
    resetStats();
    ctx.clearRect(0, 0, display.width, display.height);
    els.cameraPlaceholder.hidden = false;
  });
  els.startBtnInline.addEventListener("click", () => startCamera());
  els.cameraSelect.addEventListener("change", () => startCamera(els.cameraSelect.value));
  els.snapshotBtn.addEventListener("click", () => download(display, `face-snapshot-${Date.now()}.png`));

  els.chooseBtn.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", () => { loadImage(els.fileInput.files[0]); els.fileInput.value = ""; });
  els.saveImageBtn.addEventListener("click", () => download(display, `face-result-${Date.now()}.png`));

  const stage = $("stage");
  stage.addEventListener("dragover", (e) => {
    if (state.mode !== "image") return;
    e.preventDefault();
    els.dropzone.classList.add("dragging");
  });
  stage.addEventListener("dragleave", () => els.dropzone.classList.remove("dragging"));
  stage.addEventListener("drop", (e) => {
    if (state.mode !== "image") return;
    e.preventDefault();
    els.dropzone.classList.remove("dragging");
    loadImage(e.dataTransfer.files[0]);
  });

  els.threshold.addEventListener("input", () => {
    updateDetectionUI();
    if (state.mode === "image") renderImage();
  });
  els.squareCrop.addEventListener("change", () => {
    state.shownBox = null;
    if (state.mode === "image") detectImage();
  });
  els.showZone.addEventListener("change", () => state.mode === "image" && renderImage());
  window.addEventListener("resize", drawSpark);

  fetch("/api/info")
    .then((r) => r.json())
    .then((i) => {
      els.modelChip.textContent = `${i.model} · ${i.input_size}×${i.input_size} · ${(i.parameters / 1e6).toFixed(1)}M params · ${i.device}`;
    })
    .catch(() => { els.modelChip.textContent = "Model info unavailable"; });

  updateDetectionUI();
})();
