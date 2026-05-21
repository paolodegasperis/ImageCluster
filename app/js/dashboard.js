function setText(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value;
}

function setHTML(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = value;
}

function showElement(id, display = 'block') {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.display = display;
}

function hideElement(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.display = 'none';
}

async function loadStatus() {
  try {
    const data = await fetchJson('/api/status');
    const missing = (data.dependencies || []).filter(dep => dep.required && !dep.installed);
    setText('statusText', missing.length ? `Missing dependencies: ${missing.map(d => d.name).join(', ')}` : 'All required dependencies are available.');
    const optional = data.optional_dependencies || [];
    setHTML(
      'deps',
      data.dependencies.map(dep => `<div class="dep ${dep.installed ? 'ok' : 'missing'}">${dep.installed ? 'OK' : 'Missing'} ${dep.name}</div>`).join('') +
      optional.map(dep => `<div class="dep ${dep.installed ? 'ok' : 'missing'}">${dep.installed ? 'OK' : 'Missing'} ${dep.name} <small>(optional)</small></div>`).join('')
    );
    if (missing.length && data.install_advice) {
      const a = data.install_advice;
      showElement('installAdvice');
      setText('installAdvice', [
        `Detected platform: ${a.platform}`,
        a.recommended || '',
        '',
        'Recommended launcher:',
        a.cpu_launcher || '',
        '',
        'CUDA launcher, when applicable:',
        a.cuda_launcher || '',
        '',
        'Manual CPU command:',
        a.cpu_command || '',
        '',
        'Manual CUDA command:',
        a.cuda_command || '',
        '',
        'After PyTorch:',
        a.after_pytorch || '',
        '',
        a.imagebind_note || ''
      ].join('\n'));
    } else {
      hideElement('installAdvice');
    }
  } catch (err) {
    setText('statusText', `Cannot read status: ${err.message}`);
  }
}

async function scanImageFolder() {
  setText('dashboardImageDirStatus', 'Scanning img folder...');
  try {
    const data = await fetchJson('/api/images/scan?image_dir=img');
    if (!data.ok) {
      setText('dashboardImageDirStatus', data.error || 'The img folder could not be scanned.');
      return;
    }
    const warning = data.warnings && data.warnings.length ? ` ${data.warnings.join(' ')}` : '';
    setText('dashboardImageDirStatus', `${data.count || 0} image(s) found in img.${warning}`);
  } catch (err) {
    setText('dashboardImageDirStatus', `Cannot scan img folder: ${err.message}`);
  }
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let text = await res.text();
    try { text = JSON.parse(text).detail || text; } catch (_) {}
    throw new Error(text || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

document.getElementById('dashboardScanImg')?.addEventListener('click', scanImageFolder);
loadStatus();
