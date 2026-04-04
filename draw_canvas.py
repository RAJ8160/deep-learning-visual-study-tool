"""
draw_pad_fix.py  —  Reliable 28×28 drawing pad for Streamlit
=============================================================

ROOT CAUSES of why the original drawing pad didn't work:
─────────────────────────────────────────────────────────
1. components.html() ALWAYS returns None. Streamlit.setComponentValue
   only works with declare_component(). So pixels never reached Python.

2. The DOM-bridge textarea approach is fragile:
   - Streamlit re-renders the DOM dynamically, breaking querySelector selectors
   - Hidden textareas sometimes don't fire React synthetic onChange events
   - st.text_area(value=...) conflicts with session_state, causing widget errors
   - No st.rerun() was called after pixels arrived, so predictions never showed

THE RELIABLE FIX — st.query_params bridge:
──────────────────────────────────────────
1. JS encodes pixels as a compact hex string (1568 chars, safe for URLs)
2. JS writes it to the parent page URL using history.replaceState()
   — this does NOT reload the page, it just updates the query string
3. User clicks the normal Streamlit "Predict" button → rerun
4. Python reads st.query_params["pixels"], decodes hex → float32 array
5. Query param is cleared to avoid stale data on next run

WHY hex instead of CSV:
   URL query strings have length limits (~2000 chars).
   784 floats as 4-decimal CSV = ~5,500 chars  ← TOO LONG
   784 floats as uint8 hex     = 784×2 = 1,568 chars ← SAFE
   Precision: 1/255 ≈ 0.004 — negligible for digit recognition.

USAGE in streamlit_app.py:
    from draw_pad_fix import render_draw_pad
    # Delete old DRAW_PAD_HTML constant + old render_draw_pad() function
    # Everything else unchanged — session_state["drawn_pixels"] key is same
"""

import numpy as np
import streamlit as st
import streamlit.components.v1 as components


# ─────────────────────────────────────────────────────────────────
# Canvas HTML — lives in an iframe via components.html()
# JS responsibilities:
#   1. Let user draw with Gaussian brush on 28×28 grid
#   2. Show live 28×28 preview
#   3. On button click: quantise pixels → hex → push to parent URL
# ─────────────────────────────────────────────────────────────────
_CANVAS_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=JetBrains+Mono:wght@400;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:#0b0f1a;font-family:'Space Grotesk',sans-serif;
  color:#e2e8f0;padding:10px;
  display:flex;flex-direction:column;align-items:center;gap:8px;
}
h3{font-size:12px;font-weight:700;color:#4f9eff;letter-spacing:.1em;
   text-transform:uppercase;text-align:center}
.sub{font-size:10px;color:#64748b;font-family:'JetBrains Mono',monospace;
     text-align:center;margin-top:-4px}
.main-row{display:flex;gap:12px;align-items:flex-start;
          justify-content:center;flex-wrap:wrap}
.cw{border-radius:10px;overflow:hidden;cursor:crosshair;
    box-shadow:0 0 0 1px #1f2d45,0 0 24px rgba(79,158,255,.1)}
canvas{display:block}
.rc{display:flex;flex-direction:column;align-items:center;gap:7px}
.pl{font-size:9px;color:#64748b;font-family:'JetBrains Mono',monospace}
#preview{border-radius:5px;image-rendering:pixelated;border:1px solid #1f2d45}
.sr{display:flex;align-items:center;gap:5px;font-size:10px;
    color:#64748b;font-family:'JetBrains Mono',monospace}
input[type=range]{width:60px;accent-color:#4f9eff}
.tip{font-size:9px;color:#64748b;font-family:'JetBrains Mono',monospace;
     text-align:center;line-height:1.5}
.br{display:flex;gap:8px;width:100%;max-width:380px;justify-content:center}
button{
  font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;
  padding:9px 18px;border-radius:10px;border:none;cursor:pointer;transition:all .18s;
}
#bp{
  background:linear-gradient(135deg,#1d4ed8,#2563eb);color:#fff;
  box-shadow:0 4px 14px rgba(37,99,235,.45);flex:2;
}
#bp:hover{box-shadow:0 6px 22px rgba(37,99,235,.7);transform:translateY(-1px)}
#bc{background:#111827;color:#64748b;border:1px solid #1f2d45;flex:1}
#bc:hover{border-color:#f87171;color:#f87171}
#st{font-size:10px;font-family:'JetBrains Mono',monospace;
    min-height:16px;text-align:center;padding:2px 0;}
</style>
</head>
<body>

<h3>✏️ Draw a Digit (0–9)</h3>
<p class="sub">Step 1: Draw below &nbsp;·&nbsp; Step 2: Click Predict</p>

<div class="main-row">
  <div class="cw" id="cw"><canvas id="cv"></canvas></div>
  <div class="rc">
    <div class="pl">28×28 model input</div>
    <canvas id="preview" width="28" height="28"
            style="width:84px;height:84px"></canvas>
    <div class="sr">
      Brush&nbsp;
      <input type="range" id="brush" min="0.5" max="3.5" step="0.1" value="1.8">
      <span id="bv">1.8</span>
    </div>
    <div class="tip">Draw large &amp; centred<br>for best accuracy</div>
  </div>
</div>

<div id="st"></div>

<div class="br">
  <button id="bc">🗑 Clear</button>
  <button id="bp">🔍 Predict All Models</button>
</div>

<script>
const G = 28, S = 11;
const cv  = document.getElementById('cv');
const ct  = cv.getContext('2d');
const pv  = document.getElementById('preview');
const pt  = pv.getContext('2d');
const stEl = document.getElementById('st');

cv.width = G*S;
cv.height = G*S;
document.getElementById('cw').style.width  = (G*S) + 'px';
document.getElementById('cw').style.height = (G*S) + 'px';

const buf = new Float32Array(G * G);   // pixel values in [0..1]

// ── Render grid + painted pixels ─────────────────────────────────
function redraw() {
  ct.fillStyle = '#0d1117';
  ct.fillRect(0, 0, cv.width, cv.height);

  // grid lines
  ct.strokeStyle = 'rgba(31,45,69,0.6)';
  ct.lineWidth = 0.5;
  for (let i = 0; i <= G; i++) {
    ct.beginPath(); ct.moveTo(i*S, 0);   ct.lineTo(i*S, G*S); ct.stroke();
    ct.beginPath(); ct.moveTo(0,   i*S); ct.lineTo(G*S, i*S); ct.stroke();
  }

  // painted cells
  for (let r = 0; r < G; r++) {
    for (let c = 0; c < G; c++) {
      const v = buf[r*G + c];
      if (v > 0.02) {
        const a = Math.min(v, 1);
        ct.fillStyle =
          'rgba(' + Math.round(15+64*a) + ','
                  + Math.round(30+128*a) + ','
                  + Math.round(50+205*a) + ',' + a + ')';
        ct.fillRect(c*S+1, r*S+1, S-1, S-1);
      }
    }
  }
}

// ── 28×28 preview thumbnail ──────────────────────────────────────
function renderPrev() {
  const id = pt.createImageData(G, G);
  for (let i = 0; i < G*G; i++) {
    const v = Math.round(buf[i] * 255);
    id.data[i*4]   = Math.round(v * 0.3);
    id.data[i*4+1] = Math.round(v * 0.6 + 30);
    id.data[i*4+2] = Math.min(v + 60, 255);
    id.data[i*4+3] = 255;
  }
  pt.putImageData(id, 0, 0);
}

// ── Gaussian brush ───────────────────────────────────────────────
function paint(cx, cy) {
  const br  = parseFloat(document.getElementById('brush').value) || 1.8;
  const sig = br * S * 0.72;
  for (let r = 0; r < G; r++) {
    for (let c = 0; c < G; c++) {
      const px = (c + 0.5) * S;
      const py = (r + 0.5) * S;
      const d  = Math.sqrt((px-cx)*(px-cx) + (py-cy)*(py-cy));
      const v  = Math.exp(-(d*d) / (2*sig*sig));
      if (v > 0.04)
        buf[r*G + c] = Math.min(buf[r*G + c] + v * 0.9, 1.0);
    }
  }
  redraw();
  renderPrev();
}

function getXY(e) {
  const rect = cv.getBoundingClientRect();
  const sx = cv.width  / rect.width;
  const sy = cv.height / rect.height;
  if (e.touches)
    return [(e.touches[0].clientX - rect.left) * sx,
            (e.touches[0].clientY - rect.top ) * sy];
  return [(e.clientX - rect.left) * sx,
          (e.clientY - rect.top ) * sy];
}

let painting = false;
cv.addEventListener('mousedown',  function(e) { painting=true;  const p=getXY(e); paint(p[0],p[1]); });
cv.addEventListener('mousemove',  function(e) { if(painting){ const p=getXY(e); paint(p[0],p[1]); } });
window.addEventListener('mouseup',  function() { painting=false; });
cv.addEventListener('touchstart', function(e) { e.preventDefault(); painting=true;
                                                 const p=getXY(e); paint(p[0],p[1]); }, {passive:false});
cv.addEventListener('touchmove',  function(e) { e.preventDefault();
                                                 if(painting){ const p=getXY(e); paint(p[0],p[1]); } },
                                               {passive:false});
window.addEventListener('touchend', function() { painting=false; });

// ── Clear button ─────────────────────────────────────────────────
document.getElementById('bc').onclick = function() {
  buf.fill(0);
  redraw();
  renderPrev();
  stEl.textContent = '';
  stEl.style.color = '#64748b';
};

document.getElementById('brush').oninput = function() {
  document.getElementById('bv').textContent = parseFloat(this.value).toFixed(1);
};

// ── Predict button — encode pixels and push to parent URL ────────
//
// TRANSPORT MECHANISM: history.replaceState()
//
//   Why not postMessage?
//     components.html() ignores postMessage return values — always None in Python.
//
//   Why not DOM textarea injection?
//     Streamlit re-renders the DOM on every rerun, breaking querySelector.
//     React synthetic events require a native setter trick that is version-fragile.
//
//   Why history.replaceState?
//     - Changes the URL query string WITHOUT reloading the page
//     - URL params survive the next Streamlit rerun (triggered by button click)
//     - Python reads them with st.query_params — clean, official Streamlit API
//     - Works on all browsers, no cross-origin issues (same window)
//
//   ENCODING: uint8 hex  (1568 chars — well under URL limits)
//     float [0..1]  →  uint8 [0..255]  →  2-char hex per pixel
//     Python decodes: int(hex[i*2:i*2+2], 16) / 255.0
//
document.getElementById('bp').onclick = function() {
  const maxVal = buf.reduce(function(a,b){ return Math.max(a,b); }, 0);
  if (maxVal < 0.05) {
    stEl.textContent = '⚠ Draw a digit first!';
    stEl.style.color = '#fb923c';
    setTimeout(function(){ stEl.textContent = ''; }, 2000);
    return;
  }

  // Encode: quantise to uint8, convert to 2-char hex string
  var hex = '';
  for (var i = 0; i < 784; i++) {
    var byte = Math.min(Math.round(buf[i] * 255), 255);
    var h = byte.toString(16);
    hex += (h.length === 1 ? '0' + h : h);
  }

  // Push to parent URL without reloading
  try {
    var url = new URL(window.parent.location.href);
    url.searchParams.set('pixels', hex);
    window.parent.history.replaceState(null, '', url.toString());
    stEl.textContent = '✅ Ready!  Now click the blue Predict button below.';
    stEl.style.color = '#34d399';
  } catch(err) {
    stEl.textContent = '⚠ Error: ' + err.message;
    stEl.style.color = '#f87171';
  }
};

redraw();
renderPrev();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────
# Decode helper
# ─────────────────────────────────────────────────────────────────
def _decode_hex_pixels(hex_str: str) -> "np.ndarray | None":
    """
    Decode 1568-char hex string → (28,28) float32 ndarray in [0,1].
    Returns None if the string is invalid.
    """
    if not hex_str or len(hex_str.strip()) != 1568:
        return None
    try:
        h = hex_str.strip()
        arr = np.array(
            [int(h[i*2 : i*2+2], 16) / 255.0 for i in range(784)],
            dtype="float32"
        ).reshape(28, 28)
        return arr
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# Public API — drop-in replacement for the broken render_draw_pad()
# ─────────────────────────────────────────────────────────────────
def render_draw_pad() -> "np.ndarray | None":
    """
    Renders the 28×28 drawing canvas and returns pixel data to the caller.

    Full flow:
      1. Canvas renders in iframe via components.html()
      2. User draws a digit
      3. User clicks canvas "🔍 Predict All Models" button
         → JS encodes 784 pixels as 1568-char hex string
         → JS calls window.parent.history.replaceState() to add
           ?pixels=<hex> to the URL (no page reload)
      4. User clicks the Streamlit "🔍 Predict All Models" button
         → triggers a normal Streamlit rerun
      5. Python reads st.query_params["pixels"]
         → decodes hex → (28,28) float32 array
         → stores in st.session_state["drawn_pixels"]
         → clears the query param
      6. Returns the array; caller runs model predictions

    Returns:
        np.ndarray of shape (28,28), dtype float32, values in [0,1]
        — or None if the user hasn't submitted a drawing yet.
    """

    # ── 1. Read pixels from URL query params (set by JS) ─────────
    # Must happen BEFORE rendering widgets so we capture the value
    # on the rerun that was triggered by the Streamlit predict button.
    hex_pixels = st.query_params.get("pixels", None)

    if hex_pixels:
        arr = _decode_hex_pixels(hex_pixels)
        if arr is not None:
            prev = st.session_state.get("drawn_pixels")
            # Only update if meaningfully new drawing
            if prev is None or not np.allclose(arr, prev, atol=0.005):
                st.session_state["drawn_pixels"] = arr
        # Clear param so stale data doesn't persist on the next load
        try:
            del st.query_params["pixels"]
        except Exception:
            pass

    # ── 2. Render canvas inside iframe ───────────────────────────
    components.html(_CANVAS_HTML, height=430, scrolling=False)

    # ── 3. Streamlit predict button ──────────────────────────────
    # This is a normal Streamlit button — clicking it triggers a
    # rerun, which lets Python read the query params set in step 1.
    st.markdown(
        "<div style='margin-top:6px'></div>",
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🔍 Predict All Models",
            key="_draw_predict_btn",
            use_container_width=True,
        ):
            # On this rerun the query_params read above already captured
            # the hex. If something arrived, we already stored it.
            # Force another rerun to make sure predictions render.
            st.rerun()

    # ── 4. Status indicator ───────────────────────────────────────
    drawn = st.session_state.get("drawn_pixels")
    if drawn is not None:
        st.markdown(
            "<div style='text-align:center;font-size:.82rem;"
            "color:#34d399;font-family:monospace;margin-top:2px;'>"
            "✅ Drawing received — predictions shown below</div>",
            unsafe_allow_html=True
        )

    return drawn