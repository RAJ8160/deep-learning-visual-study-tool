# ═══════════════════════════════════════════════════════════════════
# LANDING PAGE INTEGRATION  —  paste this BEFORE the hero section
# in streamlit_app.py (right after st.set_page_config)
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import streamlit.components.v1 as components

# ── Session state gate ────────────────────────────────────────────
if "entered_app" not in st.session_state:
    st.session_state["entered_app"] = False

# ── Show landing page until user clicks "Explore More" ───────────
if not st.session_state["entered_app"]:

    # Listen for the postMessage event from the landing page button
    # This JS snippet injects a listener into the Streamlit page itself
    components.html("""
    <script>
    window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'streamlit:enterApp') {
            // Find and click the hidden Streamlit button
            const btn = window.parent.document.querySelector('[data-testid="stButton"] button');
            if (btn) btn.click();
        }
    });
    </script>
    """, height=0)

    # Load and render landing page HTML
    try:
        with open("landing_page.html", "r", encoding="utf-8") as f:
            landing_html = f.read()
        components.html(landing_html, height=950, scrolling=True)
    except FileNotFoundError:
        st.error("❌ landing_page.html not found. Place it next to streamlit_app.py")
        st.stop()

    # Hidden button — triggered by postMessage from the landing page
    # Place it outside any column so it's always in the DOM
    if st.button("__enter_app__", key="enter_app_btn",
                 label_visibility="hidden"):
        st.session_state["entered_app"] = True
        st.rerun()

    # Stop rendering the rest of the app
    st.stop()

# ═══════════════════════════════════════════════════════════════════
# Everything below this line = your existing streamlit_app.py code
# (hero section, tabs, model loading, etc.)
# ═══════════════════════════════════════════════════════════════════