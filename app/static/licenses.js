'use strict';

// Render reviewed local metadata as text, never as HTML from a source card.
function line(parent, title, value) {
  const p = document.createElement('p');
  const strong = document.createElement('strong');
  strong.textContent = `${title}: `;
  p.append(strong, document.createTextNode(value));
  parent.append(p);
}

function link(parent, label, href) {
  const a = document.createElement('a');
  a.textContent = label;
  a.href = href;
  parent.append(a, document.createTextNode(' · '));
}

async function showLicences() {
  try {
    const response = await fetch('/licenses/inventory');
    if (!response.ok) throw new Error('Inventory unavailable');
    const inventory = await response.json();
    document.getElementById('review-date').textContent = `Reviewed ${inventory.reviewed_on} · record ${inventory.version}. Sources are pinned where available.`;
    for (const item of inventory.entries) {
      const card = document.createElement('article');
      card.className = 'panel licence-card';
      card.id = item.id;
      const h3 = document.createElement('h3');
      h3.textContent = item.name;
      card.append(h3);
      line(card, 'Use', item.use);
      line(card, 'Declared licence', item.license);
      line(card, 'Status', item.status);
      line(card, 'Scope & publication', item.scope);
      line(card, 'Credit', item.attribution);
      line(card, 'Source revision', item.revision);
      const links = document.createElement('p');
      links.className = 'licence-links';
      link(links, 'Original source', item.source_url);
      link(links, 'Licence / terms', item.license_url);
      for (const file of item.notices) link(links, file, `/licenses/notices/${encodeURIComponent(file)}`);
      card.append(links);
      document.getElementById(item.group === 'current' ? 'current-licences' : 'candidate-licences').append(card);
    }
  } catch (error) {
    document.getElementById('review-date').textContent = 'Source record unavailable';
    document.getElementById('licence-error').hidden = false;
  }
}
showLicences();
