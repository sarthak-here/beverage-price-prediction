from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).parent / "model" / "xgboost_pipeline.pkl"
CLASS_NAMES = ["100-150", "150-200", "200-250", "50-100"]


st.set_page_config(
    page_title="Beverage Price Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #eef6ff;
        --muted: #8da2bb;
        --cyan: #35e6ff;
        --violet: #8b5cf6;
        --panel: rgba(10, 21, 40, .78);
        --line: rgba(125, 211, 252, .16);
    }
    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 15% 10%, rgba(53, 230, 255, .12), transparent 27rem),
            radial-gradient(circle at 87% 8%, rgba(139, 92, 246, .14), transparent 30rem),
            linear-gradient(145deg, #040812 0%, #071225 48%, #050914 100%);
        font-family: "DM Sans", sans-serif;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: .24;
        background-image:
            linear-gradient(rgba(53, 230, 255, .06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(53, 230, 255, .06) 1px, transparent 1px);
        background-size: 52px 52px;
        mask-image: linear-gradient(to bottom, black, transparent 80%);
    }
    .block-container {max-width: 1480px; padding: 2.1rem 2.2rem 4rem;}
    header[data-testid="stHeader"] {background: transparent;}
    #MainMenu, footer {visibility: hidden;}

    .eyebrow {
        color: var(--cyan); letter-spacing: .2em; text-transform: uppercase;
        font-size: .72rem; font-weight: 700; margin-bottom: .8rem;
    }
    .hero-title {
        font-family: "Space Grotesk", sans-serif;
        font-size: clamp(2.25rem, 5vw, 4.7rem); line-height: .98;
        letter-spacing: -.055em; margin: 0; max-width: 850px;
        background: linear-gradient(105deg, #fff 15%, #8cecff 60%, #bba5ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-copy {color: var(--muted); max-width: 680px; font-size: 1.03rem; margin: 1.1rem 0 2rem;}
    .live-pill {
        display: inline-flex; align-items: center; gap: .55rem; float: right;
        border: 1px solid rgba(53, 230, 255, .25); background: rgba(5, 16, 30, .65);
        border-radius: 999px; padding: .55rem .85rem; color: #b9f7ff; font-size: .78rem;
    }
    .live-dot {width: 7px; height: 7px; border-radius: 50%; background: #3ef3a5; box-shadow: 0 0 14px #3ef3a5;}
    .section-label {font-family: "Space Grotesk"; font-weight: 600; font-size: 1.15rem; margin: .25rem 0 .15rem;}
    .section-copy {color: var(--muted); font-size: .86rem; margin-bottom: 1rem;}

    div[data-testid="stForm"] {
        background: linear-gradient(145deg, rgba(12, 26, 48, .88), rgba(6, 14, 28, .8));
        border: 1px solid var(--line); border-radius: 24px; padding: 1.3rem 1.35rem .75rem;
        box-shadow: 0 24px 80px rgba(0,0,0,.25), inset 0 1px rgba(255,255,255,.03);
        backdrop-filter: blur(18px);
    }
    div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input {
        background: rgba(5, 13, 27, .88) !important; border-color: rgba(125, 211, 252, .17) !important;
        color: #edf8ff !important; border-radius: 11px !important;
    }
    label[data-testid="stWidgetLabel"] p {color: #a9bad0; font-size: .82rem; font-weight: 500;}
    div[data-testid="stFormSubmitButton"] button {
        width: 100%; min-height: 3.25rem; border: 0; border-radius: 12px;
        color: #03111c; font-weight: 800; letter-spacing: .02em;
        background: linear-gradient(90deg, #34e7ff, #80f5da 52%, #9d8cff);
        box-shadow: 0 0 30px rgba(53, 230, 255, .22); transition: .25s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {transform: translateY(-2px); box-shadow: 0 0 42px rgba(53,230,255,.38);}
    .result-card {
        position: relative; overflow: hidden; min-height: 310px;
        background: linear-gradient(145deg, rgba(12, 27, 49, .95), rgba(8, 14, 29, .92));
        border: 1px solid rgba(53, 230, 255, .2); border-radius: 24px; padding: 1.6rem;
        box-shadow: 0 24px 80px rgba(0,0,0,.26);
    }
    .result-card::after {content:""; position:absolute; width:240px; height:240px; right:-100px; top:-110px; border-radius:50%; background:#35e6ff; filter:blur(90px); opacity:.12;}
    .result-kicker {color: var(--cyan); font-size: .72rem; font-weight: 700; letter-spacing: .17em; text-transform: uppercase;}
    .price {font-family:"Space Grotesk"; font-size: 3.5rem; font-weight:700; letter-spacing:-.05em; margin:.4rem 0 0;}
    .price span {font-size: 1.05rem; color: var(--muted); letter-spacing: 0;}
    .confidence {color:#a9bad0; margin-bottom:1.35rem;}
    .bar-row {display:grid; grid-template-columns:72px 1fr 46px; gap:.6rem; align-items:center; margin:.68rem 0; font-size:.78rem; color:#a9bad0;}
    .bar {height:7px; border-radius:99px; background:rgba(255,255,255,.07); overflow:hidden;}
    .bar > div {height:100%; border-radius:99px; background:linear-gradient(90deg,#35e6ff,#8b5cf6); box-shadow:0 0 12px rgba(53,230,255,.4);}
    .empty-state {color: var(--muted); padding: 4rem .4rem; text-align:center;}
    .signal-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:.65rem;margin-top:1.2rem;}
    .signal {background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:.75rem;}
    .signal b {display:block;color:#eafbff;font-family:"Space Grotesk";font-size:1.05rem;}.signal span{color:var(--muted);font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;}
    @media (max-width: 800px) {.block-container{padding:1.2rem 1rem 3rem}.live-pill{float:none;margin-bottom:1.5rem}.hero-title{font-size:2.7rem}.signal-grid{grid-template-columns:1fr}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def engineered_values(age: int, frequency: str, awareness: str, zone: str, income: str, brand: str, reason: str):
    age_group = pd.cut(
        pd.Series([age]),
        bins=[17, 25, 35, 45, 55, 70],
        labels=["18-25", "26-35", "36-45", "46-55", "56-70"],
    ).astype(str).iloc[0]
    frequency_score = {"0-2 times": 1, "3-4 times": 2, "5-7 times": 3}[frequency]
    awareness_score = {"0 to 1": 1, "2 to 4": 2, "above 4": 3}[awareness]
    cf_ab_score = round(frequency_score / (awareness_score + frequency_score), 2)
    zone_score = {"Urban": 3, "Metro": 4, "Rural": 1, "Semi-Urban": 2}[zone]
    income_score = {"<10L": 1, "10L - 15L": 2, "16L - 25L": 3, "26L - 35L": 4, "> 35L": 5}[income]
    zas_score = zone_score * income_score
    bsi = int(brand != "Established" and reason in ["Price", "Quality"])
    return age_group, cf_ab_score, zas_score, bsi


st.markdown('<div class="live-pill"><span class="live-dot"></span>Pricing engine online · 92.26% validated accuracy</div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">CodeX · Pricing Intelligence</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Decode the price your customer will choose.</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-copy">Transform behavioral and demographic signals into an instant beverage price-range prediction. Every result is generated by the tracked production candidate.</p>', unsafe_allow_html=True)

left, right = st.columns([1.65, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Customer signal matrix</div><div class="section-copy">Complete the profile to calculate the most likely buying range.</div>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=70, value=28)
            gender = st.selectbox("Gender", ["M", "F"])
            zone = st.selectbox("Zone", ["Urban", "Metro", "Semi-Urban", "Rural"])
            occupation = st.selectbox("Occupation", ["Working Professional", "Student", "Entrepreneur", "Retired"])
            income = st.selectbox("Income level", ["<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L"])
        with c2:
            frequency = st.selectbox("Weekly consumption", ["0-2 times", "3-4 times", "5-7 times"])
            brand = st.selectbox("Current brand", ["Newcomer", "Established"])
            size = st.selectbox("Preferred size", ["Small (250 ml)", "Medium (500 ml)", "Large (1 L)"])
            awareness = st.selectbox("Other-brand awareness", ["0 to 1", "2 to 4", "above 4"])
            reason = st.selectbox("Primary brand driver", ["Price", "Quality", "Availability", "Brand Reputation"])
        with c3:
            flavor = st.selectbox("Flavor preference", ["Traditional", "Exotic"])
            channel = st.selectbox("Purchase channel", ["Online", "Retail Store"])
            packaging = st.selectbox("Packaging preference", ["Simple", "Eco-Friendly", "Premium"])
            health = st.selectbox("Health concern", ["Low (Not very concerned)", "Medium (Moderately health-conscious)", "High (Very health-conscious)"])
            situation = st.selectbox("Consumption situation", ["Active (eg. Sports, gym)", "Casual (eg. At home)", "Social (eg. Parties)"])
        submitted = st.form_submit_button("Generate price intelligence  →")

age_group, cf_ab_score, zas_score, bsi = engineered_values(age, frequency, awareness, zone, income, brand, reason)

with right:
    st.markdown('<div class="section-label">Prediction console</div><div class="section-copy">Live model output and probability distribution.</div>', unsafe_allow_html=True)
    if submitted:
        model = load_model()
        inputs = pd.DataFrame([{
            "gender": gender, "zone": zone, "occupation": occupation,
            "income_levels": income, "consume_frequency(weekly)": frequency,
            "current_brand": brand, "preferable_consumption_size": size,
            "awareness_of_other_brands": awareness,
            "reasons_for_choosing_brands": reason, "flavor_preference": flavor,
            "purchase_channel": channel, "packaging_preference": packaging,
            "health_concerns": health, "typical_consumption_situations": situation,
            "age_group": age_group, "cf_ab_score": cf_ab_score,
            "zas_score": zas_score, "bsi": bsi,
        }])
        prediction = int(model.predict(inputs)[0])
        probabilities = model.predict_proba(inputs)[0]
        predicted_range = CLASS_NAMES[prediction]
        confidence = float(probabilities[prediction]) * 100
        action = {
            "50-100": "Lead with accessibility and value packs to protect volume.",
            "100-150": "Position as an everyday upgrade with visible value cues.",
            "150-200": "Emphasize quality, experience, and differentiated packaging.",
            "200-250": "Use premium positioning, exclusivity, and high-value channels.",
        }[predicted_range]
        bars = "".join(
            f'<div class="bar-row"><span>₹{label}</span><div class="bar"><div style="width:{probability * 100:.1f}%"></div></div><b>{probability * 100:.0f}%</b></div>'
            for label, probability in sorted(zip(CLASS_NAMES, probabilities), key=lambda item: item[1], reverse=True)
        )
        st.markdown(
            f'<div class="result-card"><div class="result-kicker">Recommended price band</div><div class="price">₹{predicted_range} <span>per unit</span></div><div class="confidence">Decision confidence · {confidence:.1f}%</div>{bars}<div style="margin-top:1.15rem;padding:.9rem 1rem;border-left:2px solid #35e6ff;background:rgba(53,230,255,.045);border-radius:0 10px 10px 0;color:#c6d8e9;font-size:.84rem"><b style="color:#76eeff">Suggested action</b><br>{action}</div><div class="signal-grid"><div class="signal"><b>{age_group}</b><span>Customer segment</span></div><div class="signal"><b>{zas_score}</b><span>Buying-power score</span></div><div class="signal"><b>{"High" if bsi else "Low"}</b><span>Switching intent</span></div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="result-card"><div class="empty-state"><div style="font-size:2.3rem;color:#35e6ff;margin-bottom:.7rem">◇</div><b style="color:#eafbff">Awaiting customer signals</b><br><span>Complete the matrix and generate a prediction.</span></div></div>', unsafe_allow_html=True)

st.markdown('<p style="text-align:center;color:#61738a;font-size:.72rem;margin-top:2.5rem">CUSTOMER-LED PRICING &nbsp; / &nbsp; FOUR PRICE SEGMENTS &nbsp; / &nbsp; SOURCE DATA PROTECTED</p>', unsafe_allow_html=True)
