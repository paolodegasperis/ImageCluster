/* Saving things out of the app.
 *
 * Every view here is a working instrument, and an instrument is only useful if
 * what it shows can leave the screen. This file is the one place that knows how
 * to hand a file to the browser, so the projection, the graph and the analyses
 * all save the same way and name their files alike.
 *
 * Loaded as a plain script before the page's own, so these are globals. It must
 * stay free of page-specific state: nothing here may touch `state`, a canvas or
 * an element id.
 */

// A filename someone can still make sense of six months later: what produced it,
// what it is about, and when. Sorted by name it also sorts by date.
function exportFilename(surface, detail, extension) {
  const stamp = new Date().toISOString().slice(0, 16).replace('T', '_').replace(':', '');
  const slug = String(detail || '')
    .replace(/\.[a-z0-9]+$/i, '')          // drop a file extension if one was passed
    .replace(/[^a-zA-Z0-9._-]+/g, '-')
    // Runs of dots collapse: a subject like "../../etc/passwd" would otherwise
    // survive as "..-..-etc-passwd", which is not dangerous once the slashes
    // are gone but is a nonsense name to hand someone.
    .replace(/\.{2,}/g, '.')
    .replace(/^[-.]+|[-.]+$/g, '')
    .slice(0, 60);
  return ['imagecluster', surface, slug, stamp].filter(Boolean).join('_') + '.' + extension;
}

// The one place a file actually reaches the user.
function saveBlob(blob, filename) {
  const link = document.createElement('a');
  link.download = filename;
  link.href = URL.createObjectURL(blob);
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Freed on the next tick: revoking synchronously can cancel the download in
  // some browsers before it has read the blob.
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
}

function saveCanvasPng(canvas, filename) {
  const link = document.createElement('a');
  link.download = filename;
  link.href = canvas.toDataURL('image/png');
  document.body.appendChild(link);
  link.click();
  link.remove();
}

/* CSV written for the spreadsheet it will actually be opened in.
 *
 * Two deliberate choices. The separator is ';' because Excel follows the
 * system list separator, and on an Italian machine a comma-separated file lands
 * entirely in column A. And the file starts with a BOM, without which Excel
 * reads UTF-8 as the local codepage and every accented character is mangled.
 */
const CSV_SEPARATOR = ';';

function csvCell(value) {
  const text = String(value ?? '');
  // A cell containing the separator, a quote or a newline has to be quoted, and
  // quotes inside it doubled. Without this one stray value shifts a whole row.
  if (text.includes(CSV_SEPARATOR) || text.includes('"') || /[\r\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function csvText(header, rows) {
  const lines = [header.map(csvCell).join(CSV_SEPARATOR)];
  for (const row of rows) lines.push(row.map(csvCell).join(CSV_SEPARATOR));
  return '﻿' + lines.join('\r\n') + '\r\n';
}

function saveCsv(header, rows, filename) {
  // Reports rather than throws: the callers are synchronous click handlers with
  // nowhere to catch, and a button that silently does nothing reads as broken.
  try {
    saveBlob(new Blob([csvText(header, rows)], { type: 'text/csv;charset=utf-8' }), filename);
    return { ok: true, rows: rows.length };
  } catch (err) {
    return { ok: false, error: err.message || String(err) };
  }
}

/* Ask the server to build a file, then hand it over.
 *
 * The PDFs are generated server-side: the images are on disk there at full
 * resolution, while the browser can only reach them through the thumbnail
 * endpoint, which stops at 512px. Errors come back as JSON rather than a file,
 * so the response type decides how to read it.
 */
async function downloadFromServer(url, payload, filename) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body && body.detail) {
        // FastAPI returns a string for our own 400s but an array of error
        // objects for a schema violation; the array stringifies to
        // "[object Object]", which tells the reader nothing.
        detail = Array.isArray(body.detail)
          ? body.detail.map(e => e && e.msg ? `${(e.loc || []).slice(1).join('.')}: ${e.msg}` : String(e)).join('; ')
          : String(body.detail);
      }
    } catch (err) {
      /* not JSON: keep the status line */
    }
    throw new Error(detail);
  }
  const blob = await response.blob();
  // The server names the file; fall back to ours if the header is absent.
  const disposition = response.headers.get('Content-Disposition') || '';
  const match = disposition.match(/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i);
  saveBlob(blob, match ? decodeURIComponent(match[1]) : filename);

  // What the server could not include. Returned rather than swallowed, so the
  // caller can say so: a sheet quietly holding fewer images than the result it
  // came from would be wrong without looking wrong.
  const count = name => Number(response.headers.get(name) || 0);
  return {
    included: count('X-Sheet-Included'),
    skipped: count('X-Sheet-Skipped'),
    undersized: count('X-Sheet-Undersized'),
  };
}

/* Node's built-in test runner loads this file to check the CSV writer, which is
 * the one piece of formatting here that is easy to get subtly wrong and whose
 * mistakes only show up inside a spreadsheet. Browsers have no `module`, so this
 * is inert in the app. */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { exportFilename, csvCell, csvText, CSV_SEPARATOR };
}
