/* Which language the in-place help panels are written in.
 *
 * The graph and the lenses each carry a long explanation inside their own page,
 * and those explanations exist in both languages. Rather than fetch a second
 * copy, both are in the markup and one is hidden: the text is a few kilobytes,
 * and a panel that opens instantly is worth more than the bytes saved.
 *
 * The choice is shared across pages and remembered, so it is made once.
 *
 * English is the default and the fallback. It is also the language of the
 * interface, and the toggle says "Italiano" rather than showing a flag,
 * because what it switches is the explanation and not the buttons - a reader
 * who expected Italian controls would be disappointed by a globe icon.
 */

const HELP_LANG_KEY = 'imagecluster.helpLanguage';

function helpLanguage() {
  try {
    return localStorage.getItem(HELP_LANG_KEY) === 'it' ? 'it' : 'en';
  } catch (err) {
    return 'en';   // private mode, or storage disabled
  }
}

function applyHelpLanguage(lang) {
  const chosen = lang === 'it' ? 'it' : 'en';
  document.querySelectorAll('[data-help-lang]').forEach(block => {
    block.hidden = block.dataset.helpLang !== chosen;
  });
  document.querySelectorAll('[data-help-lang-toggle]').forEach(button => {
    // The button names the language it would switch TO, which is the only
    // labelling people read correctly without thinking about it.
    button.textContent = chosen === 'it' ? 'English' : 'Italiano';
    button.setAttribute('aria-label', chosen === 'it'
      ? 'Read this explanation in English'
      : 'Leggi questa spiegazione in italiano');
  });
  return chosen;
}

function setHelpLanguage(lang) {
  const chosen = applyHelpLanguage(lang);
  try {
    localStorage.setItem(HELP_LANG_KEY, chosen);
  } catch (err) {
    /* not remembered, but the panel still switched */
  }
}

document.addEventListener('DOMContentLoaded', () => {
  applyHelpLanguage(helpLanguage());
  document.querySelectorAll('[data-help-lang-toggle]').forEach(button => {
    button.addEventListener('click', () => {
      setHelpLanguage(helpLanguage() === 'it' ? 'en' : 'it');
    });
  });

  // The guide is a whole page rather than a panel, so it switches by following a
  // link. It writes the same preference on the way out: otherwise choosing
  // Italian there and opening a help panel afterwards would give English, and
  // the reader would have made the same choice twice.
  document.querySelectorAll('[data-lang-switch]').forEach(link => {
    link.addEventListener('click', () => {
      const href = link.getAttribute('href') || '';
      try {
        localStorage.setItem(HELP_LANG_KEY, href.endsWith('/it') ? 'it' : 'en');
      } catch (err) {
        /* not remembered; the link still navigates */
      }
    });
  });
});
