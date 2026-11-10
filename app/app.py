import streamlit as st
import pandas as pd, numpy as np
import joblib
import shap
from pathlib import Path

st.set_page_config(page_title="Delivery ETA Engine")

MODELS = Path(__file__).resolve().parent.parent / "models"

@st.cache_resource
def load_all():
    return {
        "eta": joblib.load(MODELS / "lgbm_eta.pkl"),
        "clf": joblib.load(MODELS / "delay_clf.pkl"),
        "q": joblib.load(MODELS / "eta_quantiles.pkl"),
        "settings": joblib.load(MODELS / "settings.pkl"),
        "dyn": joblib.load(MODELS / "dynamic_eta.pkl"),
        "dyn_cols": joblib.load(MODELS / "dynamic_columns.pkl"),
        "cols": joblib.load(MODELS / "feature_columns.pkl"),
        "words": joblib.load(MODELS / "reason_words.pkl"),
    }

m = load_all()
cols = m["cols"]
explainer = shap.TreeExplainer(m["eta"])

st.title("Last-Mile Delivery ETA Engine")

# ---------- inputs ----------
markets = [c.replace("market_id_", "") for c in cols if c.startswith("market_id_")]
protocols = [c.replace("order_protocol_", "") for c in cols if c.startswith("order_protocol_")]
cuisines = sorted(c.replace("store_primary_category_", "") for c in cols
                  if c.startswith("store_primary_category_"))

left, right = st.columns(2)

with left:
    st.subheader("Order")
    total_items = st.number_input("Total items", 1, 30, 3)
    distinct = st.number_input("Distinct items", 1, 30, 2)
    subtotal = st.number_input("Subtotal (cents)", 100, 20000, 2500, step=100)
    min_price = st.number_input("Cheapest item (cents)", 0, 10000, 600, step=50)
    max_price = st.number_input("Priciest item (cents)", 0, 10000, 1200, step=50)
    cuisine = st.selectbox("Cuisine", cuisines)
    market = st.selectbox("Market", markets,
                          format_func=lambda x: "Unknown" if float(x) == 0 else f"Market {int(float(x))}")
    protocol = st.selectbox("Order protocol", protocols,
                            format_func=lambda x: "Unknown" if float(x) == 0 else f"Protocol {int(float(x))}")

with right:
    st.subheader("Drivers and time")
    onshift = st.number_input("Drivers on shift", 0, 150, 20)
    busy = st.number_input("Busy drivers", 0, 150, 17)
    outstanding = st.number_input("Open orders", 0, 300, 25)
    drive_min = st.number_input("Estimated drive time (min)", 1.0, 40.0, 10.0, step=0.5)
    place_sec = st.number_input("Order handling time (sec)", 0, 3000, 446, step=10)
    hour = st.slider("Hour of order (UTC)", 0, 23, 2)
    weekday = st.selectbox("Day of week", [0, 1, 2, 3, 4, 5, 6],
                           format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d])

def build_row():
    row = pd.DataFrame(0.0, index=[0], columns=cols)

    row["total_items"] = total_items
    row["num_distinct_items"] = min(distinct, total_items)
    row["subtotal"] = subtotal
    row["min_item_price"] = min_price
    row["max_item_price"] = max(max_price, min_price)
    row["total_onshift_dashers"] = onshift
    row["total_busy_dashers"] = busy
    row["total_outstanding_orders"] = outstanding
    row["estimated_order_place_duration"] = place_sec
    row["estimated_store_to_consumer_driving_duration"] = drive_min * 60   # model uses seconds
    row["hour"] = hour
    row["weekday"] = weekday


    row["available_dashers"] = onshift - busy
    row["busy_ratio"] = busy / (onshift + 1)
    row["orders_per_dasher"] = outstanding / (onshift + 1)
    row["avg_item_price"] = subtotal / max(total_items, 1)
    row["is_weekend"] = int(weekday >= 5)
    row["dasher_info_missing"] = 0

    for name in (f"market_id_{market}", f"order_protocol_{protocol}",
                 f"store_primary_category_{cuisine}"):
        if name in row.columns:
            row[name] = 1
    return row

def friendly(name):
    if name in m["words"]:
        return m["words"][name]
    for prefix, label in [("market_id_", "market"), ("order_protocol_", "order protocol"),
                          ("store_primary_category_", "cuisine")]:
        if name.startswith(prefix):
            value = name.replace(prefix, "")
            if value.replace(".", "").isdigit():
                value = "unknown" if float(value) == 0 else str(int(float(value)))
            return f"{label} {value} tends to run slower"
    return name.replace("_", " ")

# ---------- predict button ----------
if st.button("Predict", type="primary"):
    row = build_row()

    # ETA range: 10th, 50th and 90th percentile models
    low, mid, high = sorted(float(m["q"][a].predict(row)[0]) for a in (0.1, 0.5, 0.9))

    # chance of delay
    chance = float(m["clf"].predict_proba(row)[:, 1][0])

    # reasons: features that pushed the time up the most (SHAP)
    sv = explainer.shap_values(row)[0]
    push = pd.Series(sv, index=cols)
    top = push[push > 0].sort_values(ascending=False).head(3)

    st.session_state["row"] = row
    st.session_state["result"] = {
        "low": low, "mid": mid, "high": high, "chance": chance,
        "reasons": [friendly(n) for n in top.index],
    }

# ---------- displaying the saved result ----------
if "result" in st.session_state:
    r = st.session_state["result"]
    limit = m["settings"]["delay_minutes"]
    alert_at = m["settings"]["alert_probability"]
    risk = "Low" if r["chance"] < alert_at else ("Medium" if r["chance"] < 0.6 else "High")

    st.divider()
    st.metric("Estimated delivery time", f"about {r['mid']:.0f} min")
    st.write(f"Usually between **{r['low']:.0f} and {r['high']:.0f} min** (about 4 in 5 deliveries land inside).")

    a, b = st.columns(2)
    a.metric("Chance of delay", f"{r['chance']:.0%}")
    b.metric("Delay risk", risk)
    st.caption(f"Delay means taking longer than {limit:.0f} min. Alert level: {alert_at:.0%}. "
               "Shows the last Predict click.")

    if r["reasons"]:
        st.subheader("What pushes the time up most")
        for reason in r["reasons"]:
            st.write("- " + reason)

# ---------- dynamic ETA ----------
st.divider()
st.subheader("Dynamic ETA: update as time passes")
elapsed = st.select_slider("Minutes since the order was placed", options=[0, 10, 20, 30, 40], value=0)

if "row" in st.session_state:
    drow = st.session_state["row"].copy()
    drow["elapsed_min"] = elapsed                       
    left_min = max(float(m["dyn"].predict(drow[m["dyn_cols"]])[0]), 1.0)

    st.metric("Time left", f"about {left_min:.0f} min")
    st.caption(f"Total: about {elapsed + left_min:.0f} min. The estimate only knows the order is "
               "still running; there is no GPS or live driver data.")
else:
    st.info("Click Predict first, then move the slider.")

st.caption("Trained on about 4 weeks of DoorDash orders from 2015. Prep time, traffic and weather "
           "are missing, so the typical error is about 11 minutes.")