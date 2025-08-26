// Anno corrente in footer
document.getElementById('year').textContent = new Date().getFullYear();

// Galleria: cerca file in docs/media elencati in una "lista"
const items = [
  // Sostituisci con i tuoi file in docs/media (relative path)
  // Tipi supportati: image o video
  // { type: 'image', src: 'media/loss_curve.webp', caption: 'Curva di loss' },
  // { type: 'image', src: 'media/road_sample.jpg', caption: 'Tracciato su strada' },
  // { type: 'image', src: 'media/cones_detection.jpg', caption: 'YOLOv8 (coni)' },
  // { type: 'video', src: 'media/test_road.mp4', caption: 'Test su strada' },
  // { type: 'video', src: 'media/test_cones.mp4', caption: 'Test con coni' },
];

// UI placeholder se non c'è nulla
const gallery = document.getElementById('gallery');
if (!items.length) {
  const empty = document.createElement('div');
  empty.className = 'tile card';
  empty.innerHTML = `
    <div class="caption">Aggiungi media in <code>docs/media/</code> e definisci gli elementi in <code>assets/main.js</code>.</div>`;
  gallery.appendChild(empty);
}

// Render
items.forEach(item => {
  const tile = document.createElement('div');
  tile.className = 'tile';
  const fig = document.createElement('figure');
  if (item.type === 'video') {
    const v = document.createElement('video');
    v.src = item.src;
    v.controls = true;
    v.playsInline = true;
    fig.appendChild(v);
  } else {
    const img = document.createElement('img');
    img.src = item.src;
    img.alt = item.caption || '';
    img.loading = 'lazy';
    fig.appendChild(img);
  }
  const cap = document.createElement('div');
  cap.className = 'caption';
  cap.textContent = item.caption || '';
  tile.appendChild(fig);
  tile.appendChild(cap);
  gallery.appendChild(tile);
});
