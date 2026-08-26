from __future__ import annotations

import streamlit as st


def apply_styles() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#102a2e; --muted:#64777a; --teal:#0f766e; --mint:#d9f4e9; --line:#dbe7e3; --paper:#f6faf8; --orange:#ea580c; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    h1,h2,h3 { font-family:'Space Grotesk', sans-serif; color:var(--ink); letter-spacing:0; }
    .stApp { background: radial-gradient(circle at 100% 0%, #e4f5ed 0, transparent 34%), linear-gradient(145deg, #f8fbfa 0%, var(--paper) 55%, #eef7f3 100%); }
    [data-testid="stSidebar"] { background:#102a2e; border-right:1px solid #24524f; }
    [data-testid="stSidebar"] * { color:#edf8f3; }
    .eyebrow { color:var(--teal); font-size:.75rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
    .block-container { max-width:1440px; padding-top:2.5rem; padding-bottom:4rem; }
    .hero { background:linear-gradient(120deg,#102a2e,#155e63); border-radius:16px; padding:30px 34px; color:white; box-shadow:0 16px 35px #0b37301f; }
    .hero h1 { color:white; margin:.15rem 0 .35rem; font-size:clamp(2rem,4vw,3.5rem); }
    .hero p { color:#cce9df; margin:0; }
    .hero-stat { border-left:1px solid #ffffff38; padding-left:24px; margin-top:22px; }
    .hero-value { font:700 4rem/1 'Space Grotesk'; color:#b7f2cf; }
    .card { background:rgba(255,255,255,.92); border:1px solid var(--line); border-radius:12px; padding:18px 20px; box-shadow:0 8px 24px #17433b0c; height:100%; position:relative; overflow:hidden; transition:transform .18s ease, box-shadow .18s ease; }
    .card:hover { transform:translateY(-2px); box-shadow:0 12px 28px #17433b18; }
    .card::before { background:var(--teal); content:""; height:3px; left:0; position:absolute; right:0; top:0; }
    .card-label { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; }
    .card-value { color:var(--ink); font:700 clamp(1.45rem,2.2vw,1.9rem)/1.1 'Space Grotesk'; margin-top:10px; white-space:nowrap; }
    .badge { display:inline-block; padding:5px 10px; border-radius:99px; font-size:.75rem; font-weight:700; background:#e6f6ed; color:#166534; }
    .muted { color:var(--muted); font-size:.82rem; margin-top:7px; }
    .source-note { border-left:3px solid var(--teal); padding:10px 14px; background:#edf8f3; color:#28504d; border-radius:0 8px 8px 0; }
    .stButton > button { border-radius:8px; border:1px solid #9fcfc0; font-weight:600; }
    [data-testid="stHorizontalBlock"] { gap:1rem; }
    @media (max-width: 640px) { .block-container { padding:1.25rem 1rem 3rem; } .hero { padding:24px; } .hero-value { font-size:3rem; } .card { padding:16px; } }
    </style>
    """, unsafe_allow_html=True)
