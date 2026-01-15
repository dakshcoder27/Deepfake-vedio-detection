// script.js - improved preview handling & error reporting
const API_ENDPOINT = '/api/predict';

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const preview = document.getElementById('preview');
const noVideo = document.getElementById('noVideo');
const predictionEl = document.getElementById('prediction');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const confidenceRow = document.getElementById('confidenceRow');
const uploadProgress = document.getElementById('uploadProgress');
const progressRow = document.getElementById('progressRow');
const progressText = document.getElementById('progressText');
const message = document.getElementById('message');
const meta = document.getElementById('meta');

let selectedFile = null;
let objectUrl = null;

// Drag & drop handlers
['dragenter','dragover'].forEach(ev=>{
  dropzone.addEventListener(ev, (e)=>{
    e.preventDefault(); e.stopPropagation();
    dropzone.classList.add('dragover');
  });
});
['dragleave','drop'].forEach(ev=>{
  dropzone.addEventListener(ev, (e)=>{
    e.preventDefault(); e.stopPropagation();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (e)=>{
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) handleFile(f);
});

// File input
fileInput.addEventListener('change', (e)=>{
  const f = e.target.files && e.target.files[0];
  if (f) handleFile(f);
});

// keyboard accessibility
dropzone.addEventListener('keydown', (e)=>{
  if (e.key === 'Enter') fileInput.click();
});

function handleFile(file){
  clearMessage();
  const allowed = ['video/mp4','video/webm','video/quicktime','video/x-matroska'];
  if (!allowed.includes(file.type) && !file.name.match(/\\.(mp4|webm|mov|mkv)$/i)){
    showMessage('Unsupported file type. Please use mp4/webm/mov/mkv', true);
    return;
  }
  selectedFile = file;
  analyzeBtn.disabled = false;
  meta.textContent = `Selected: ${file.name} (${bytesToSize(file.size)})`;
  showPreview(file);
}

// Robust preview logic
function showPreview(file){
  // Revoke previous url if any
  if (objectUrl) {
    try { URL.revokeObjectURL(objectUrl); } catch(e){ /* ignore */ }
    objectUrl = null;
  }

  // Hide UI while loading
  preview.style.display = 'none';
  noVideo.style.display = 'block';
  preview.removeAttribute('src');

  // Try createObjectURL first (fast, streamable)
  try {
    objectUrl = URL.createObjectURL(file);
    preview.src = objectUrl;
    // Force reload and listen for metadata (duration, width/height)
    preview.load();

    // If metadata loads, show video
    const onLoaded = () => {
      preview.style.display = 'block';
      noVideo.style.display = 'none';
      // Show duration in meta if needed:
      // meta.textContent += ` — duration: ${Math.round(preview.duration)}s`;
      preview.removeEventListener('loadedmetadata', onLoaded);
      preview.removeEventListener('error', onError);
    };
    const onError = (ev) => {
      console.warn('Video element error', ev);
      preview.removeEventListener('loadedmetadata', onLoaded);
      preview.removeEventListener('error', onError);
      // fallback to FileReader
      fallbackToDataURL(file);
    };

    preview.addEventListener('loadedmetadata', onLoaded);
    preview.addEventListener('error', onError);
    // also set a timeout: if nothing happens in 5s, fallback
    const fallbackTimeout = setTimeout(()=>{
      if (preview.readyState < 1) {
        // still not loaded -> fallback
        preview.removeEventListener('loadedmetadata', onLoaded);
        preview.removeEventListener('error', onError);
        fallbackToDataURL(file);
      }
    }, 5000);

    // clear timeout when loaded
    preview.addEventListener('loadedmetadata', ()=>clearTimeout(fallbackTimeout));
  } catch (err) {
    console.warn('createObjectURL failed, falling back to FileReader', err);
    fallbackToDataURL(file);
  }

  // video error events - show friendly message
  preview.onerror = function(ev) {
    console.error('Preview error event', ev);
    showMessage('Cannot play this video in your browser (unsupported codec). Try a different mp4.', true);
  };
}

function fallbackToDataURL(file){
  // Read file as data URL and assign to video.src
  const reader = new FileReader();
  reader.onerror = () => {
    showMessage('Unable to read file for preview.', true);
  };
  reader.onload = () => {
    try {
      preview.src = reader.result;
      preview.load();
      preview.style.display = 'block';
      noVideo.style.display = 'none';
    } catch (e) {
      console.warn('Data URL preview failed', e);
      showMessage('Video preview failed (codec or browser limitation). You can still upload the file.', true);
    }
  };
  reader.readAsDataURL(file);
}

// Analyze button -> upload + inference
analyzeBtn.addEventListener('click', () => {
  if (!selectedFile) {
    showMessage('Please select a video first', true);
    return;
  }
  uploadAndPredict(selectedFile);
});

// Upload with progress (XMLHttpRequest)
function uploadAndPredict(file){
  resetResultUI();
  progressRow.classList.remove('hidden');
  confidenceRow.classList.add('hidden');
  uploadProgress.value = 0;
  progressText.textContent = '0%';
  analyzeBtn.disabled = true;
  showMessage('Uploading and running model...', false);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', API_ENDPOINT, true);

  xhr.upload.onprogress = function(evt){
    if (evt.lengthComputable) {
      const percent = Math.round((evt.loaded / evt.total) * 100);
      uploadProgress.value = percent;
      progressText.textContent = percent + '%';
    }
  };

  xhr.onload = function(){
    progressRow.classList.add('hidden');
    analyzeBtn.disabled = false;
    if (xhr.status >= 200 && xhr.status < 300) {
      try {
        const data = JSON.parse(xhr.responseText);
        if (data.error) {
          showMessage('Server error: ' + data.error, true);
          return;
        }
        handleResult(data);
      } catch (err) {
        showMessage('Invalid server response: ' + err.message, true);
        console.error('Response text:', xhr.responseText);
      }
    } else {
      showMessage(`Upload failed: ${xhr.status} ${xhr.statusText}`, true);
      console.error('Server returned non-200:', xhr.responseText);
    }
  };

  xhr.onerror = function(){
    progressRow.classList.add('hidden');
    analyzeBtn.disabled = false;
    showMessage('Network error during upload', true);
  };

  const fd = new FormData();
  fd.append('video', file);
  xhr.send(fd);
}

function handleResult(data){
  const pred = (data.prediction || '').toLowerCase();
  const conf = Number(data.confidence) || 0;
  if (!pred) {
    showMessage('Server response missing prediction', true);
    return;
  }

  predictionEl.textContent = pred.toUpperCase();
  predictionEl.classList.remove('fake','real');
  if (pred === 'fake') predictionEl.classList.add('fake');
  else predictionEl.classList.add('real');

  const pct = Math.round(conf * 100);
  confidenceBar.style.width = pct + '%';
  confidenceText.textContent = pct + '%';
  confidenceRow.classList.remove('hidden');

  showMessage('Inference complete', false);
}

// Utilities
function showMessage(text, isError){
  message.textContent = text;
  message.classList.toggle('error', !!isError);
  message.style.color = isError ? '#ff7b7b' : '#9fc6ff';
}
function clearMessage(){ message.textContent = ''; }

function resetResultUI(){
  predictionEl.textContent = '—';
  predictionEl.classList.remove('fake','real');
  confidenceBar.style.width = '0%';
  confidenceText.textContent = '0%';
  confidenceRow.classList.add('hidden');
}

function bytesToSize(bytes) {
  const sizes = ['Bytes','KB','MB','GB','TB'];
  if (bytes === 0) return '0 Byte';
  const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)), 10);
  return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
}
