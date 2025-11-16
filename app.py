# app.py
# Streamlit app: Daewoo scraper + bus delay prediction
# Usage: streamlit run app.py

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ----------------------
# Config & Constants
# ----------------------
BASE_URL = "https://daewooinfo.pk"
ROUTES = {
    "Lahore ↔ Islamabad": "/",
    "Karachi ↔ Lahore": "/",
    "Karachi ↔ Islamabad": "/",
    "Multan ↔ Rawalpindi": "/",
    "Multan ↔ Lahore": "/",
    "Peshawar ↔ Islamabad": "/",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DaewooScraper/1.0; +https://example.com)"}

# ----------------------
# Streamlit Page Config
# ----------------------
st.set_page_config(page_title="Daewoo Scraper + Delay Predictor", page_icon="🚌", layout="wide")
st.title("🚌 Daewoo Scraper & Bus Delay Predictor")

st.markdown("""
This app scrapes tables from `daewooinfo.pk` for different routes and allows you to predict bus delay times 
based on user input features such as route, day, departure hour, traffic, and weather.
""")

# ----------------------
# Sidebar Controls
# ----------------------
with st.sidebar:
    st.header("Scraper Controls")
    route_name = st.selectbox("Choose route", list(ROUTES.keys()))
    refresh = st.button("Refresh data (scrape live)")
    st.write("---")
    st.markdown("**Cache settings**")
    cache_seconds = st.number_input("Cache TTL (seconds)", min_value=60, max_value=86400, value=600, step=60)
    st.write("---")
    st.header("Predict Bus Delay")
    pred_route = st.selectbox("Route for prediction", list(ROUTES.keys()), index=0)
    day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    departure_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23, value=9)
    traffic = st.selectbox("Traffic Level", ["Low","Medium","High"])
    weather = st.selectbox("Weather", ["Clear","Rainy","Foggy","Other"])
    predict_button = st.button("Predict Delay")

# ----------------------
# HTML Fetching
# ----------------------
route_slug = ROUTES[route_name]
route_url = urljoin(BASE_URL, route_slug)

if "cache_buster" not in st.session_state:
    st.session_state["cache_buster"] = 0

@st.cache_data(ttl=3600)
def fetch_html_cached(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        return resp.status_code, resp.text
    except requests.RequestException as e:
        return None, str(e)

def fetch_html(url, use_cache=True):
    if use_cache:
        status, text = fetch_html_cached(url)
    else:
        try:
            resp = requests.get(url + "?t=" + str(time.time()), headers=HEADERS, timeout=15)
            status, text = resp.status_code, resp.text
        except requests.RequestException as e:
            status, text = None, str(e)
    return status, text

# ----------------------
# Table Parsing
# ----------------------
def parse_tables(html):
    """Return tables as list of (name, df) tuples"""
    tables = []
    try:
        for i, t in enumerate(pd.read_html(html)):
            tables.append((f"table_{i+1}", t))
    except Exception:
        pass
    return tables

def parse_custom(html):
    soup = BeautifulSoup(html, "lxml")
    results = []
    for table in soup.find_all("table"):
        rows = []
        headers = []
        thead = table.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all(["th","td"])]
        for tr in table.find_all("tr"):
            rows.append([td.get_text(strip=True) for td in tr.find_all(["td","th"])])
        if rows:
            try:
                df = pd.DataFrame(rows[1:], columns=headers) if headers and len(headers)==len(rows[0]) else pd.DataFrame(rows)
            except:
                df = pd.DataFrame(rows)
            results.append((f"parsed_table", df))
    return results

def scrape_route(url, use_cache=True):
    st.info(f"Scraping: {url} {'(live)' if not use_cache else '(cached)'}")
    start = time.time()
    status, html = fetch_html(url, use_cache)
    if not status or status >= 400:
        return [], {"url": url, "fetched_at": None, "elapsed_seconds": 0, "num_tables":0}, f"Error fetching page: {status}"
    tables = parse_tables(html) + parse_custom(html)
    elapsed = time.time() - start
    meta = {"url": url, "fetched_at": datetime.utcnow().isoformat()+"Z", "elapsed_seconds": round(elapsed,3), "num_tables":len(tables)}
    return tables, meta, None

def filter_tables(tables, route_name):
    words = [w.strip().lower() for w in route_name.replace("↔"," ").split()]
    filtered = []
    for name, df in tables:
        text = " ".join(df.astype(str).values.flatten()).lower()
        if any(word in text for word in words):
            filtered.append((name, df))
    return filtered

@st.cache_data
def get_route_data(route_url, cache_seconds, cache_buster):
    return scrape_route(route_url, use_cache=True)

# ----------------------
# Scrape & Display Tables
# ----------------------
col1, col2 = st.columns([3,1])

with col1:
    st.subheader(f"Route: {route_name}")
    if refresh:
        st.session_state["cache_buster"] +=1
        tables, meta, error = scrape_route(route_url, use_cache=False)
    else:
        tables, meta, error = get_route_data(route_url, cache_seconds, st.session_state["cache_buster"])

    st.markdown("**Metadata**")
    st.write(meta)
    if error: st.error(error)

    if tables:
        filtered = filter_tables(tables, route_name)
        display_tables = filtered if filtered else tables
        table_names = [name for name, df in display_tables]
        idx = st.selectbox("Select Table", range(len(table_names)), format_func=lambda i: table_names[i])
        name, df = display_tables[idx]
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"{route_name.replace(' ','_')}_{name}.csv")
    else:
        st.warning("No tables found.")

with col2:
    st.markdown("**Quick Table Preview**")
    if tables:
        for name, df in tables[:6]:
            st.write(f"- {name}: {df.shape[0]} × {df.shape[1]}")
    else:
        st.write("(none)")

# ----------------------
# Bus Delay Prediction
# ----------------------
st.write("---")
st.subheader("Predict Bus Delay (minutes)")

# For demo, create or load a simple RandomForest model trained on CSV
@st.cache_resource
def load_model():
    # For demo, we create a fake dataset
    df = pd.DataFrame({
        "Route": ["Lahore ↔ Islamabad","Karachi ↔ Lahore","Multan ↔ Lahore"]*10,
        "Day": ["Monday","Tuesday","Wednesday"]*10,
        "DepartureHour": [8,9,10]*10,
        "Traffic": ["Low","Medium","High"]*10,
        "Weather": ["Clear","Rainy","Foggy"]*10,
        "Delay": [5,10,15]*10
    })
    # Encode categorical features
    le_route = LabelEncoder().fit(df["Route"])
    le_day = LabelEncoder().fit(df["Day"])
    le_traffic = LabelEncoder().fit(df["Traffic"])
    le_weather = LabelEncoder().fit(df["Weather"])
    X = pd.DataFrame({
        "Route": le_route.transform(df["Route"]),
        "Day": le_day.transform(df["Day"]),
        "DepartureHour": df["DepartureHour"],
        "Traffic": le_traffic.transform(df["Traffic"]),
        "Weather": le_weather.transform(df["Weather"])
    })
    y = df["Delay"]
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X,y)
    return model, le_route, le_day, le_traffic, le_weather

model, le_route, le_day, le_traffic, le_weather = load_model()

if predict_button:
    X_test = pd.DataFrame({
        "Route": [le_route.transform([pred_route])[0]],
        "Day": [le_day.transform([day_of_week])[0]],
        "DepartureHour": [departure_hour],
        "Traffic": [le_traffic.transform([traffic])[0]],
        "Weather": [le_weather.transform([weather])[0]]
    })
    pred_delay = model.predict(X_test)[0]
    st.success(f"Predicted Bus Delay: {round(pred_delay,1)} minutes")

st.caption("© Daewoo Scraper & Delay Predictor — Demo version")
