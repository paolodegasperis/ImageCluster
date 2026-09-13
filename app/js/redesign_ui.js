function bySelector(selector) {
  return document.querySelector(selector);
}

function allBySelector(selector) {
  return [...document.querySelectorAll(selector)];
}

// Delegates to the plot's own switcher so there is one list of panes, not two
// that drift apart. The fallback keeps this file usable on a page that does not
// load clip_projection.js.
function showDockPane(name) {
  if (typeof selectDockPane === 'function') {
    selectDockPane(name);
    return;
  }
  allBySelector('[data-dock-tab]').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.dockTab === name);
  });
}

allBySelector('[data-dock-tab]').forEach(tab => {
  tab.addEventListener('click', () => showDockPane(tab.dataset.dockTab));
});

// Left-rail workflow steps collapse independently (CODEX_GUIDE §4.1 / acceptance checklist).
allBySelector('.left-rail .step-head').forEach(head => {
  head.addEventListener('click', () => head.parentElement?.classList.toggle('open'));
});

bySelector('[data-dock-toggle]')?.addEventListener('click', event => {
  const dock = bySelector('#projectionDock');
  if (!dock) return;
  dock.classList.toggle('is-expanded');
  if (dock.classList.contains('is-expanded') && typeof restoreDockHeight === 'function') {
    restoreDockHeight();
  } else if (!dock.classList.contains('is-expanded')) {
    // Collapsing clears the inline basis so the 44px rule applies again.
    dock.style.flexBasis = '';
  }
  event.currentTarget.textContent = dock.classList.contains('is-expanded') ? 'Collapse' : 'Expand';
});

bySelector('[data-search-options]')?.addEventListener('click', () => {
  const panel = bySelector('#searchOptionsPanel');
  if (panel) panel.hidden = !panel.hidden;
});

bySelector('#searchQuery')?.addEventListener('focus', () => {
  bySelector('#semanticSearchBar')?.classList.add('is-focused');
});

bySelector('#searchQuery')?.addEventListener('blur', () => {
  bySelector('#semanticSearchBar')?.classList.remove('is-focused');
});

window.addEventListener('keydown', event => {
  if (event.key === '/' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName || '')) {
    event.preventDefault();
    bySelector('#searchQuery')?.focus();
  }
});

function openWorkflowDrawer(name) {
  const drawer = bySelector('#workflowDrawer');
  if (!drawer) return;
  drawer.hidden = false;
  const titles = {
    explore: 'View & thumbnails',
    filters: 'Filters',
    session: 'Session summary',
    export: 'Export & sessions',
    help: 'Keyboard & help',
  };
  const title = bySelector('#workflowDrawerTitle');
  if (title) title.textContent = titles[name] || 'Explore';
  allBySelector('[data-drawer-panel]').forEach(panel => {
    panel.hidden = panel.dataset.drawerPanel !== name;
  });
  allBySelector('[data-open-drawer]').forEach(button => {
    button.classList.toggle('active', button.dataset.openDrawer === name);
  });
}

allBySelector('[data-open-drawer]').forEach(button => {
  button.addEventListener('click', () => openWorkflowDrawer(button.dataset.openDrawer));
});

bySelector('[data-close-workflow-drawer]')?.addEventListener('click', () => {
  const drawer = bySelector('#workflowDrawer');
  if (drawer) drawer.hidden = true;
  allBySelector('[data-open-drawer]').forEach(button => button.classList.remove('active'));
});

bySelector('#drawerResetViewBtn')?.addEventListener('click', () => bySelector('#resetViewBtn')?.click());
bySelector('#drawerExportPngBtn')?.addEventListener('click', () => bySelector('#exportPngBtn')?.click());
bySelector('#drawerDownloadBtn')?.addEventListener('click', () => bySelector('#downloadBtn')?.click());
bySelector('#drawerBuildIndexBtn')?.addEventListener('click', () => bySelector('#buildIndexBtn')?.click());
