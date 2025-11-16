# app.py
# Streamlit app to scrape Daewoo route pages and display schedules, fares and other tables.
# Usage: streamlit run app.py

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import time

BASE_URL = "https://daewooinfo.pk"

# ROUTES: all point to homepage because this site places tables on the homepage.
ROUTES = {
    "Lahore ↔ Islamabad": "/",
    "Karachi ↔ Lahore": "/",
    "Karachi ↔ Islamabad": "/",
    "Multan ↔ Rawalpindi": "/",
    "Multan ↔ Lahore": "/",
    "Peshawar ↔ Islamabad": "/",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DaewooScraper/1.0; +https://example.com)"
}

# Streamlit page config
st.set_page_config(page_title="Daewoo Scraper", page_icon="🚌", layout="wide")

st.title("🚌 Daewoo — Route scraper & viewer")
st.markdown(
    "This app scrapes tables from `daewooinfo.pk` and shows schedules/fares. "
    "Because the site places multiple route tables on the homepage, the app fetches the homepage and "
    "filters the extracted tables by the selected route."
)

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    route_name = st.selectbox("Choose route", list(ROUTES.keys()))
    refresh = st.button("Refresh data (scrape live)")
    st.write("---")
    st.markdown("**Cache settings**")
    cache_seconds = st.number_input("Cache TTL (seconds)", min_value=60, max_value=86_400, value=600, step=60)
    st.write("---")
    st.markdown("App built from a user notebook. Use responsibly and do not overload the source site.")

route_slug = ROUTES[route_name]
route_url = urljoin(BASE_URL, route_slug)

# Use a session-state cache buster to force live scrape on button press
if "cache_buster" not in st.session_state:
    st.session_state["cache_buster"] = 0

# Cached wrapper for fetching raw HTML (cached by URL and TTL)
@st.cache_data(ttl=3600)
def fetch_html_cached(url):
    """Fetch page HTML and return (status_code, text). Cached by Streamlit."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        # Return status code so caller can handle 404 gracefully
        return resp.status_code, resp.text
    except requests.RequestException as e:
        # Return None/empty so caller can show error
        return None, str(e)

def fetch_html(url, use_cache=True):
    """
    Wrapper that uses the cached fetch by default.
    If use_cache is False, append a cache-busting query param and call requests directly.
    """
    if use_cache:
        status, text = fetch_html_cached(url)
        return status, text
    else:
        # direct live fetch (no cache)
        try:
            resp = requests.get(url + ("?t=" + str(time.time())), headers=HEADERS, timeout=15)
            return resp.status_code, resp.text
        except requests.RequestException as e:
            return None, str(e)

def parse_with_pandas_tables(html):
    """Try pandas.read_html to get any HTML tables present on the page."""
    try:
        tables = pd.read_html(html)
    except Exception:
        tables = []

    parsed = []
    for i, t in enumerate(tables):
        df = t.copy()
        parsed.append((f"table_{i+1}", df))
    return parsed

def parse_custom(html):
    """
    Attempt to parse schedules/fare blocks with BeautifulSoup for richer info when present.
    Returns list of (name, dataframe).
    """
    soup = BeautifulSoup(html, "lxml")
    results = []

    # 1) Extract <table> elements
    tables = soup.find_all("table")
    for idx, table in enumerate(tables):
        rows = []
        thead = table.find("thead")
        headers = []
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]
        for tr in table.find_all("tr"):
            cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            try:
                # If headers count matches row cell count, use headers
                if headers and len(headers) == len(rows[0]):
                    df = pd.DataFrame(rows[1:], columns=headers)
                else:
                    df = pd.DataFrame(rows)
            except Exception:
                df = pd.DataFrame(rows)
            results.append((f"parsed_table_{idx+1}", df))

    # 2) Extract blocks whose headings mention fares, schedule, departure, arrival etc.
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "strong", "b"]):
        txt = heading.get_text(" ", strip=True).lower()
        if any(k in txt for k in ("fare", "price", "schedule", "time", "departure", "arrival", "seat", "route")):
            sib = heading.find_next_sibling()
            collected = None
            if sib and sib.name == "table":
                collected = sib
            elif sib and sib.name in ("div", "ul", "ol"):
                collected = sib
            if collected:
                rows = []
                if collected.name == "table":
                    for tr in collected.find_all("tr"):
                        rows.append([td.get_text(strip=True) for td in tr.find_all(["td", "th"])])
                else:
                    for li in collected.find_all(["li", "p"]):
                        rows.append([li.get_text(strip=True)])
                if rows:
                    df = pd.DataFrame(rows)
                    results.append((f"block_{heading.name}_{txt[:20]}", df))

    return results

def scrape_route(url, use_cache=True):
    """
    Fetch and parse a route page. Returns a list of (name, dataframe) tuples and metadata.
    use_cache: when False, always does a live fetch (cache-buster).
    """
    st.info(f"Scraping: {url} {'(live)' if not use_cache else '(cached permitted)'}")
    start = time.time()
    status, html_or_err = fetch_html(url, use_cache=use_cache)

    if status is None:
        # network error
        meta = {"url": url, "fetched_at": None, "elapsed_seconds": 0, "num_tables": 0}
        return [], meta, f"Network error: {html_or_err}"

    if status == 404:
        meta = {"url": url, "fetched_at": datetime.utcnow().isoformat() + "Z", "elapsed_seconds": 0, "num_tables": 0}
        return [], meta, f"404 Not Found: {url}"

    if status and status >= 400:
        meta = {"url": url, "fetched_at": datetime.utcnow().isoformat() + "Z", "elapsed_seconds": 0, "num_tables": 0}
        return [], meta, f"HTTP error {status}"

    html = html_or_err
    parsed_tables = parse_with_pandas_tables(html)
    custom = parse_custom(html)

    # Combine and de-duplicate by keys (keep order)
    all_tables = parsed_tables + custom
    elapsed = time.time() - start
    meta = {
        "url": url,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": round(elapsed, 3),
        "num_tables": len(all_tables),
    }
    return all_tables, meta, None

def filter_tables_by_route(tables, route_name):
    """
    Return only tables that contain any of the route city names (case-insensitive).
    For robustness we search both table text and the table's name.
    """
    words = [w.strip().lower() for w in route_name.replace("↔", " ").replace("–", " ").split() if w.strip()]

    filtered = []
    for name, df in tables:
        # Check name for route words
        name_text = name.lower() if isinstance(name, str) else ""
        combined_text = " ".join(df.astype(str).values.flatten()).lower()

        # If any route word appears in the name or the combined table text, keep it
        if any(word in name_text or word in combined_text for word in words):
            filtered.append((name, df))

    return filtered

# Cache wrapper for scrape that depends on route and cache_seconds value
@st.cache_data
def get_route_data_cached(route_url, cache_ttl_seconds, cache_buster):
    # cache_buster included in key so we can force-refresh by bumping it
    tables, meta, error = scrape_route(route_url, use_cache=True)
    return tables, meta, error

# Main UI area
st.subheader(f"Route: {route_name}")
col1, col2 = st.columns([3, 1])

with col1:
    st.write(f"Source: {route_url}")

    # Decide whether to use cache or force live fetch
    if refresh:
        # bump cache_buster and perform live scrape
        st.session_state["cache_buster"] += 1
        try:
            tables, meta, error = scrape_route(route_url, use_cache=False)
        except Exception as e:
            st.error(f"Failed to scrape live: {e}")
            tables, meta, error = [], {"url": route_url, "fetched_at": None, "elapsed_seconds": 0, "num_tables": 0}, str(e)
    else:
        try:
            # call cached getter; pass cache_seconds and cache_buster so cache key includes them
            tables, meta, error = get_route_data_cached(route_url, int(cache_seconds), st.session_state["cache_buster"])
        except Exception as e:
            st.error(f"Failed to load cached data: {e}")
            tables, meta, error = [], {"url": route_url, "fetched_at": None, "elapsed_seconds": 0, "num_tables": 0}, str(e)

    st.markdown("**Metadata**")
    st.write(meta)
    if error:
        st.error(error)
    st.write("---")

    if not tables:
        st.warning("No tables found on the page. Try clicking 'Refresh data' to force a live scrape.")
    else:
        # Filter tables to only those relevant for the selected route
        filtered = filter_tables_by_route(tables, route_name)

        if not filtered:
            st.warning(
                f"No specific tables found matching route '{route_name}'. Showing all extracted tables as fallback."
            )
            display_tables = tables
        else:
            display_tables = filtered

        # Present a selectbox to choose which table to view
        table_names = [name for name, df in display_tables]
        idx = st.selectbox("Choose table block to view", list(range(len(table_names))), format_func=lambda i: table_names[i])
        name, df = display_tables[idx]
        st.markdown(f"### `{name}` — {df.shape[0]} rows x {df.shape[1]} cols")
        st.dataframe(df)

        # Provide CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name=f"{route_name.replace(' ', '_')}_{name}.csv")

with col2:
    st.markdown("**Quick preview of found blocks**")
    if tables:
        for name, df in tables[:12]:
            st.write(f"- {name}: {df.shape[0]} × {df.shape[1]}")
    else:
        st.write("(none)")

st.write("---")

st.markdown("### Notes & troubleshooting")
st.markdown(
    "- If the site changes structure, parsing may fail — inspect the page and adapt parsing.\n"
    "- Avoid repeatedly scraping; use cache and 'Refresh data' sparingly.\n"
    "- If pandas fails to find tables, the page may render tables with JavaScript — this app does not run JS."
)

st.markdown("### Developer tips")
st.code(
    """
# To run locally:
# 1) Create a virtualenv and install dependencies:
#    pip install streamlit pandas requests beautifulsoup4 lxml
# 2) Run:
#    streamlit run app.py

# To deploy to Streamlit Community Cloud:
# - Push this file to a GitHub repo
# - On share.streamlit.io, create a new app and point to this file
"""
)

st.caption("Built from a user notebook. Use responsibly. © Daewoo Scraper")
