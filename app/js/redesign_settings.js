const revealButton = document.getElementById('hfTokenRevealButton');
const revealCheckbox = document.getElementById('hfTokenReveal');
const revealInput = document.getElementById('hfTokenInput');

revealButton?.addEventListener('click', () => {
  if (!revealCheckbox || !revealInput) return;
  revealCheckbox.checked = !revealCheckbox.checked;
  revealCheckbox.dispatchEvent(new Event('change'));
  revealButton.textContent = revealCheckbox.checked ? 'Hide' : 'Show';
});

document.querySelectorAll('[data-close-drawer]').forEach(item => {
  item.addEventListener('click', () => {
    const backdrop = document.querySelector('.mt-detail-backdrop');
    if (backdrop) backdrop.hidden = true;
  });
});

const drawerObserverTarget = document.getElementById('modelDrawer');
if (drawerObserverTarget) {
  const observer = new MutationObserver(() => {
    const backdrop = document.querySelector('.mt-detail-backdrop');
    if (!backdrop) return;
    backdrop.hidden = drawerObserverTarget.hidden;
    if (!drawerObserverTarget.hidden) {
      drawerObserverTarget.style.display = 'flex';
    }
  });
  observer.observe(drawerObserverTarget, { attributes: true, attributeFilter: ['hidden'] });
}
