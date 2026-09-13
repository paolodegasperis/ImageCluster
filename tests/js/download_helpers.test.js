/* The CSV writer, under test.
 *
 * This is the one piece of formatting in the frontend whose mistakes are
 * invisible until a file is opened in a spreadsheet, and by then the export has
 * already been sent to someone. The three surfaces that now produce CSV all go
 * through the same two functions, so they are worth pinning down.
 *
 * Run with:  node --test tests/js/*.test.js
 * No dependency: the runner is built into Node.
 */

const test = require('node:test');
const assert = require('node:assert/strict');

const { exportFilename, csvCell, csvText, CSV_SEPARATOR } = require('../../app/js/download_helpers.js');

const BOM = '﻿';

// --------------------------------------------------------------------------
// Escaping

test('a plain value is written as it stands', () => {
  assert.equal(csvCell('1013.jpg'), '1013.jpg');
  assert.equal(csvCell(42), '42');
});

test('a value containing the separator is quoted', () => {
  // Without this one stray value shifts every column after it in that row.
  assert.equal(csvCell('a;b'), '"a;b"');
});

test('quotes inside a value are doubled, and the value quoted', () => {
  assert.equal(csvCell('say "hi"'), '"say ""hi"""');
});

test('a value containing a newline is quoted', () => {
  assert.equal(csvCell('line1\nline2'), '"line1\nline2"');
  assert.equal(csvCell('line1\r\nline2'), '"line1\r\nline2"');
});

test('null and undefined become empty rather than the words', () => {
  // A missing similarity must read as blank, not as "null" or "undefined".
  assert.equal(csvCell(null), '');
  assert.equal(csvCell(undefined), '');
});

test('zero and false survive as themselves', () => {
  // Rank 0 is the query row; it must not be swallowed by a falsy check.
  assert.equal(csvCell(0), '0');
  assert.equal(csvCell(false), 'false');
});

// --------------------------------------------------------------------------
// The file

test('the file begins with a UTF-8 BOM', () => {
  // Excel reads a BOM-less UTF-8 file as the local codepage, which mangles
  // every accented character in a filename.
  assert.ok(csvText(['a'], []).startsWith(BOM));
});

test('accented text is preserved', () => {
  const text = csvText(['nome'], [['città']]);
  assert.ok(text.includes('città'));
});

test('rows are separated by CRLF', () => {
  const text = csvText(['a', 'b'], [['1', '2'], ['3', '4']]);
  assert.equal(text, `${BOM}a;b\r\n1;2\r\n3;4\r\n`);
});

test('the header is escaped like any other row', () => {
  assert.ok(csvText(['a;b'], []).includes('"a;b"'));
});

test('the separator is the semicolon Excel expects on an Italian machine', () => {
  assert.equal(CSV_SEPARATOR, ';');
});

test('a realistic export round-trips through a naive reader', () => {
  const header = ['rank', 'filename', 'similarity'];
  const rows = [
    [0, '1013.jpg', ''],
    [1, 'quote"d.jpg', '0.862750'],
    [2, 'semi;colon.jpg', '0.800284'],
  ];
  const lines = csvText(header, rows).replace(BOM, '').trimEnd().split('\r\n');
  assert.equal(lines.length, 4);
  // Column count is what a spreadsheet actually cares about: a bad escape shows
  // up as a row with the wrong number of fields, not as an error.
  for (const line of lines) {
    assert.equal(splitCsvLine(line).length, 3, `wrong field count in: ${line}`);
  }
  assert.deepEqual(splitCsvLine(lines[2]), ['1', 'quote"d.jpg', '0.862750']);
  assert.deepEqual(splitCsvLine(lines[3]), ['2', 'semi;colon.jpg', '0.800284']);
});

/** A minimal RFC-4180 reader, so the test checks the output rather than restating it. */
function splitCsvLine(line) {
  const fields = [];
  let current = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') { current += '"'; i++; }
      else if (ch === '"') quoted = false;
      else current += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === CSV_SEPARATOR) { fields.push(current); current = ''; }
    else current += ch;
  }
  fields.push(current);
  return fields;
}

// --------------------------------------------------------------------------
// Filenames

test('a filename says what produced it, about what, and when', () => {
  const name = exportFilename('community', 'Community 2', 'pdf');
  assert.match(name, /^imagecluster_community_Community-2_\d{4}-\d{2}-\d{2}_\d{4}\.pdf$/);
});

test('characters a filesystem dislikes are removed', () => {
  const name = exportFilename('similar', '../../etc/passwd', 'csv');
  assert.ok(!name.includes('/'));
  assert.ok(!name.includes('..'));
});

test('an existing extension on the subject is not carried into the name', () => {
  const name = exportFilename('similar', '1013.jpg', 'csv');
  assert.ok(name.includes('_1013_'));
  assert.ok(name.endsWith('.csv'));
  assert.ok(!name.includes('.jpg'));
});

test('an empty subject leaves no double separator', () => {
  assert.match(exportFilename('projection', '', 'png'), /^imagecluster_projection_\d{4}-\d{2}-\d{2}_\d{4}\.png$/);
});
