# streamlit_app.py
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
from urllib.parse import quote_plus
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------
# Boot & Secrets
# ---------------------------
st.set_page_config(page_title="Card Scout", page_icon="🧲", layout="wide")

# Load .env locally (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

EBAY_APP_ID = os.getenv("EBAY_APP_ID")
if not EBAY_APP_ID:
    try:
        EBAY_APP_ID = st.secrets["EBAY_APP_ID"]
    except Exception:
        EBAY_APP_ID = None

# Small status badge so you know the key is seen by the app
if EBAY_APP_ID:
    st.sidebar.success("eBay key loaded")
else:
    st.sidebar.warning("No eBay key found (Demo mode recommended)")

# ---------------------------
# Sidebar: Settings
# ---------------------------
st.sidebar.title("⚙️ Settings")

if "settings" not in st.session_state:
    st.session_state.settings = {
        "fee_pct": 12.9,
        "fixed_fee": 0.30,
        "shipping": 5.50,
        "tax_pct": 0.0
    }

with st.sidebar:
    st.caption("Default costs for profit calcs")
    s = st.session_state.settings
    s["fee_pct"]   = st.number_input("Marketplace fee %", 0.0, 25.0, s["fee_pct"], 0.1)
    s["fixed_fee"] = st.number_input("Fixed fee $", 0.0, 5.0, s["fixed_fee"], 0.05)
    s["shipping"]  = st.number_input("Ship & supplies $", 0.0, 50.0, s["shipping"], 0.25)
    s["tax_pct"]   = st.number_input("Taxes on sale %", 0.0, 20.0, s["tax_pct"], 0.5)

    demo_default = EBAY_APP_ID is None
    DEMO_MODE = st.toggle(
        "Demo mode (no API)",
        value=demo_default,
        help="Use sample data if you don't have an eBay App ID or the API is flaky."
    )
    st.divider()
    st.caption("Tip: add `EBAY_APP_ID` to a local `.env` or Streamlit Secrets to enable the live API.")

# ---------------------------
# eBay API client (Finding API)
# ---------------------------
FINDING_ENDPOINT = "https://svcs.ebay.com/services/search/FindingService/v1"
CATEGORY_ID_FORCED = "261328"  # Sports Trading Cards Singles

def _session():
    s = requests.Session()
    retry = Retry(
        total=2, backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "CardScout/0.2 (+contact: you@example.com)"})
    return s

@st.cache_data(ttl=900, show_spinner=False)
def ebay_find_completed_items_last10(query: str) -> pd.DataFrame:
    """
    Return the 10 most recent SOLD/completed items for the query,
    restricted to Sports Trading Cards Singles (261328).
    Tries SOA headers first, then param-style fallback.
    """
    base_params = {
        "keywords": query,
        "paginationInput.entriesPerPage": 10,       # hard limit
        "paginationInput.pageNumber": 1,
        "itemFilter(0).name": "SoldItemsOnly",
        "itemFilter(0).value(0)": "true",
        "sortOrder": "EndTimeSoonest",
        "categoryId": CATEGORY_ID_FORCED,
    }

    s = _session()

    headers = {
        "X-EBAY-SOA-OPERATION-NAME": "findCompletedItems",
        "X-EBAY-SOA-SERVICE-VERSION": "1.13.0",
        "X-EBAY-SOA-SECURITY-APPNAME": EBAY_APP_ID,
        "X-EBAY-SOA-REQUEST-DATA-FORMAT": "JSON",
        "X-EBAY-SOA-RESPONSE-DATA-FORMAT": "JSON",
        "X-EBAY-SOA-GLOBAL-ID": "EBAY-US",
    }
    r = s.get(FINDING_ENDPOINT, headers=headers, params=base_params, timeout=20)

    # Fallback if needed
    if r.status_code >= 500:
        params_alt = {
            "OPERATION-NAME": "findCompletedItems",
            "SERVICE-VERSION": "1.13.0",
            "SECURITY-APPNAME": EBAY_APP_ID,
            "RESPONSE-DATA-FORMAT": "JSON",
            "REST-PAYLOAD": "true",
            "GLOBAL-ID": "EBAY-US",
            **base_params,
        }
        r = s.get(FINDING_ENDPOINT, params=params_alt, timeout=20)

    r.raise_for_status()
    resp = r.json().get("findCompletedItemsResponse", [{}])[0]
    if resp.get("ack", [""])[0] != "Success":
        msg = resp.get("errorMessage", [{}])[0].get("error", [{}])[0].get("message", [""])[0]
        raise requests.HTTPError(msg or "eBay API non-success ack")

    items = resp.get("searchResult", [{}])[0].get("item", [])
    rows = []
    for it in items:
        try:
            title = it.get("title", [""])[0]
            url = it.get("viewItemURL", [""])[0]
            end_time = it.get("listingInfo", [{}])[0].get("endTime", [""])[0]
            price_obj = it.get("sellingStatus", [{}])[0].get("convertedCurrentPrice", [{}])[0]
            price = float(price_obj.get("__value__", "nan"))
            curr  = price_obj.get("@currencyId", "")
            if not np.isnan(price):
                rows.append({
                    "date": pd.to_datetime(end_time, errors="coerce"),
                    "price": price,
                    "currency": curr,
                    "title": title,
                    "url": url
                })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # ensure it's the 10 most recent; display chronologically
    return df.sort_values("date", ascending=False).head(10).sort_values("date")

# ---------------------------
# Monte Carlo helpers
# ---------------------------
def winsorize(a, low=2.5, high=97.5):
    lo, hi = np.percentile(a, [low, high]);  return np.clip(a, lo, hi)

def recency_weights(n, scheme="exp"):
    if n <= 1: return np.ones(n)
    if scheme == "exp": w = 0.9 ** np.arange(n-1, -1, -1)
    elif scheme == "linear": w = np.arange(1, n+1)
    else: w = np.ones(n)
    return w / w.sum()

def monte_carlo(prices, n_iter=10000, recency="exp", noise_pct=2.0):
    p = winsorize(np.array(prices, dtype=float), 2.5, 97.5)
    w = recency_weights(len(p), recency)
    idx = np.random.choice(np.arange(len(p)), size=n_iter, replace=True, p=w)
    sim = p[idx];  noise = np.random.normal(0.0, noise_pct/100.0, size=n_iter)
    return sim * (1 + noise)

# ---------------------------
# Demo data (offline/API down)
# ---------------------------
DEMO_DF = pd.DataFrame({
    "date": pd.date_range(end=pd.Timestamp.today(), periods=15, freq="2D"),
    "price": [272, 265, 281, 269, 276, 260, 285, 292, 274, 267, 279, 286, 271, 263, 289],
    "title": ["2020 Prizm Justin Herbert Silver PSA 10"]*15,
    "url": ["https://www.ebay.com"]*15,
    "currency": ["USD"]*15
}).sort_values("date")

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(["🔎 Quick Comps", "💸 Purchases & Budget", "📦 Inventory", "📝 Listing Builder"])

# ==== TAB 1: Quick Comps (forced category, last 10) ====
with tabs[0]:
    st.subheader("🔎 eBay Sold Comps → Monte Carlo (Last 10 · Singles category)")
    if "q" not in st.session_state:
        st.session_state.q = "2020 Prizm Justin Herbert Silver PSA 10"

    with st.form("search", clear_on_submit=False):
        q = st.text_input("Search sold/completed for", value=st.session_state.q)
        submitted = st.form_submit_button("Fetch last 10 via eBay API")

    st.session_state.q = q
    comp_df = pd.DataFrame()

    if submitted:
        if not q.strip() or len(q.strip()) < 3:
            st.warning("Please enter at least 3 characters.")
        else:
            ebay_url = (
                "https://www.ebay.com/sch/i.html"
                f"?_nkw={quote_plus(q.strip())}&LH_Sold=1&LH_Complete=1&_sacat={CATEGORY_ID_FORCED}"
            )
            try:
                if DEMO_MODE or not EBAY_APP_ID:
                    comp_df = DEMO_DF.copy().sort_values("date", ascending=False).head(10).sort_values("date")
                    st.info("Demo mode: showing 10 sample sales. Add EBAY_APP_ID to hit the live API.")
                else:
                    with st.spinner("Calling eBay…"):
                        comp_df = ebay_find_completed_items_last10(q.strip())
            except Exception:
                st.info("API isn’t reachable right now; showing 10 sample sales so you can continue.")
                st.markdown(f"[Open this search on eBay (sold/completed)]({ebay_url})")
                comp_df = DEMO_DF.copy().sort_values("date", ascending=False).head(10).sort_values("date")

    if not comp_df.empty:
        st.success(f"Loaded {len(comp_df)} sold/ended results (Singles category)")
        st.dataframe(comp_df[["date", "price", "title", "url"]], use_container_width=True, hide_index=True)
        st.download_button("Download CSV (last 10)", comp_df.to_csv(index=False).encode("utf-8"),
                           file_name="ebay_comps_last10.csv", mime="text/csv")

        # Stats
        p = comp_df["price"].dropna().astype(float)
        st.markdown("**Quick stats**")
        st.write({"mean": round(p.mean(),2), "median": round(p.median(),2),
                  "min": round(p.min(),2), "max": round(p.max(),2), "n": int(p.shape[0])})

        # Monte Carlo
        st.divider()
        st.markdown("### 🎲 Monte Carlo net profit")
        s = st.session_state.settings
        cA, cB, cC = st.columns(3)
        with cA:
            cost_basis = st.number_input("Your buy price ($)", 0.0, 100000.0, float(p.median()), 1.0)
        with cB:
            shipping = st.number_input("Ship/supplies ($)", 0.0, 50.0, float(s["shipping"]), 0.25)
        with cC:
            tax_pct = st.number_input("Taxes on sale (%)", 0.0, 20.0, float(s["tax_pct"]), 0.5)

        sims = monte_carlo(p.values, n_iter=10000, recency="exp", noise_pct=2.0)
        fees  = sims * (s["fee_pct"]/100.0) + s["fixed_fee"]
        taxes = sims * (tax_pct/100.0)
        net   = sims - (fees + taxes + shipping + cost_basis)

        prob_profit = float((net > 0).mean())
        p5, p50, p95 = np.percentile(net, [5, 50, 95])
        st.write(f"**Chance of profit:** {prob_profit:.1%}")
        st.write(f"**Net pr**
