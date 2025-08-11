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
    st.session_state.settings["fee_pct"] = st.number_input("Marketplace fee %", 0.0, 25.0,
                                                           st.session_state.settings["fee_pct"], 0.1)
    st.session_state.settings["fixed_fee"] = st.number_input("Fixed fee $", 0.0, 5.0,
                                                             st.session_state.settings["fixed_fee"], 0.05)
    st.session_state.settings["shipping"] = st.number_input("Ship & supplies $", 0.0, 50.0,
                                                            st.session_state.settings["shipping"], 0.25)
    st.session_state.settings["tax_pct"] = st.number_input("Taxes on sale %", 0.0, 20.0,
                                                           st.session_state.settings["tax_pct"], 0.5)

    demo_default = EBAY_APP_ID is None
    DEMO_MODE = st.toggle("Demo mode (no API)", value=demo_default,
                          help="Use sample data if you don't have an eBay App ID or the API is flaky.")
    st.divider()
    st.caption("Tip: add `EBAY_APP_ID` to a local `.env` or Streamlit Secrets to enable the API.")

# ---------------------------
# HTTP session for eBay
# ---------------------------
FINDING_ENDPOINT = "https://svcs.ebay.com/services/search/FindingService/v1"

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
def ebay_find_completed_items(query: str,
                              entries_per_page: int = 50,
                              page: int = 1,
                              category_id: str | None = None) -> pd.DataFrame:
    """Find sold/completed via eBay Finding API (SOA headers with param fallback)."""
    base_params = {
        "keywords": query,
        "paginationInput.entriesPerPage": entries_per_page,
        "paginationInput.pageNumber": page,
        "itemFilter(0).name": "SoldItemsOnly",
        "itemFilter(0).value(0)": "true",
        "sortOrder": "EndTimeSoonest",
    }
    if category_id:
        base_params["categoryId"] = category_id

    s = _session()

    # Try header-based call
    headers = {
        "X-EBAY-SOA-OPERATION-NAME": "findCompletedItems",
        "X-EBAY-SOA-SERVICE-VERSION": "1.13.0",
        "X-EBAY-SOA-SECURITY-APPNAME": EBAY_APP_ID,
        "X-EBAY-SOA-REQUEST-DATA-FORMAT": "JSON",
        "X-EBAY-SOA-RESPONSE-DATA-FORMAT": "JSON",
        "X-EBAY-SOA-GLOBAL-ID": "EBAY-US",
    }
    r = s.get(FINDING_ENDPOINT, headers=headers, params=base_params, timeout=20)
    if r.status_code >= 500:
        # Fallback to classic param style
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
    return pd.DataFrame(rows).sort_values("date")

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
# Demo data (offline / API down)
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

# ==== TAB 1: Quick Comps ====
with tabs[0]:
    st.subheader("🔎 eBay Sold Comps → Monte Carlo")
    if "q" not in st.session_state:
        st.session_state.q = "2020 Prizm Justin Herbert Silver PSA 10"

    with st.form("search", clear_on_submit=False):
        q = st.text_input("Search sold/completed for", value=st.session_state.q)
        c1, c2 = st.columns([1,1])
        with c1:
            items = st.select_slider("Items", options=[50, 100], value=50)
        with c2:
            category_id = st.text_input("Category ID (optional)", placeholder="e.g., 261328")
        submitted = st.form_submit_button("Fetch via eBay API")

    st.session_state.q = q
    comp_df = pd.DataFrame()

    if submitted:
        if not q.strip() or len(q.strip()) < 3:
            st.warning("Please enter at least 3 characters.")
        else:
            ebay_url = f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(q.strip())}&LH_Sold=1&LH_Complete=1"
            try:
                if DEMO_MODE or not EBAY_APP_ID:
                    comp_df = DEMO_DF.copy()
                    st.info("Demo mode: showing sample data. Add EBAY_APP_ID to query the live API.")
                else:
                    with st.spinner("Calling eBay…"):
                        comp_df = ebay_find_completed_items(q.strip(), int(items), 1, category_id or None)
            except Exception as e:
                st.error(f"eBay error: {e}")
                st.markdown(f"[Open this search on eBay (sold/completed)]({ebay_url})")
                comp_df = DEMO_DF.copy()
                st.info("Loaded demo data so you can keep testing the workflow.")

    if not comp_df.empty:
        st.success(f"Loaded {len(comp_df)} sold/ended results")
        st.dataframe(comp_df[["date", "price", "title", "url"]], use_container_width=True, hide_index=True)
        st.download_button("Download CSV", comp_df.to_csv(index=False).encode("utf-8"),
                           file_name="ebay_comps.csv", mime="text/csv")

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
        st.write(f"**Net profit (P5 / P50 / P95):** ${p5:,.0f} / ${p50:,.0f} / ${p95:,.0f}")
        st.bar_chart(pd.Series(net, name="Simulated Net Profit"))
        st.caption("Tweak buy price & fees to see negotiation room.")

# ==== TAB 2: Purchases & Budget ====
with tabs[1]:
    st.subheader("💸 Purchases & Budget (Show Mode)")
    if "purchases" not in st.session_state:
        st.session_state.purchases = pd.DataFrame(columns=["when", "item", "price", "method", "notes"])

    col = st.columns(5)
    with col[0]:
        when = st.date_input("Date")
    with col[1]:
        item = st.text_input("Item / card")
    with col[2]:
        price = st.number_input("Price ($)", 0.0, 100000.0, 0.0, 1.0)
    with col[3]:
        method = st.selectbox("Paid via", ["Cash", "PayPal", "Venmo", "Other"])
    with col[4]:
        notes = st.text_input("Notes", "")

    if st.button("Add purchase"):
        new = {"when": pd.to_datetime(when), "item": item.strip(), "price": float(price),
               "method": method, "notes": notes.strip()}
        st.session_state.purchases = pd.concat([st.session_state.purchases, pd.DataFrame([new])], ignore_index=True)

    if not st.session_state.purchases.empty:
        dfp = st.session_state.purchases.sort_values("when", ascending=False)
        st.dataframe(dfp, use_container_width=True, hide_index=True)
        st.metric("Total spent", f"${dfp['price'].sum():,.2f}")
        by_method = dfp.groupby("method")["price"].sum().sort_values(ascending=False)
        st.bar_chart(by_method)

# ==== TAB 3: Inventory ====
with tabs[2]:
    st.subheader("📦 Inventory Intake")
    st.caption("Upload CSV with columns like: date_acquired, player, year, set, parallel, grade, cost_basis, sku")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        inv = pd.read_csv(up)
        st.dataframe(inv, use_container_width=True)
        if "cost_basis" in inv.columns:
            st.metric("Total cost basis", f"${float(inv['cost_basis'].fillna(0).sum()):,.2f}")

# ==== TAB 4: Listing Builder ====
def title_builder(player, year, card_set, subset, card_no, serial_no, grade, keywords):
    parts = []
    if year: parts.append(str(year))
    if player: parts.append(player)
    if card_set: parts.append(card_set)
    if subset: parts.append(subset)
    if card_no: parts.append(f"#{card_no}")
    if serial_no: parts.append(serial_no)
    if grade and grade.lower() != "raw": parts.append(grade)
    if keywords: parts.append(keywords)
    return " ".join(parts)[:80]

with tabs[3]:
    st.subheader("📝 Listing Builder")
    c1, c2, c3 = st.columns(3)
    with c1:
        player = st.text_input("Player")
        year = st.text_input("Year")
        grade = st.text_input("Grade (e.g., PSA 10 / Raw)", "PSA 10")
    with c2:
        card_set = st.text_input("Set (e.g., Prizm, Topps Chrome)")
        subset = st.text_input("Insert/Parallel (e.g., Silver, Refractor)")
        card_no = st.text_input("Card #")
    with c3:
        serial_no = st.text_input("Serial (e.g., /99)")
        keywords = st.text_input("Extra keywords (Team, RC, etc.)", "RC")
        price_anchor = st.number_input("List price anchor ($)", 0.0, 100000.0, 0.0, 1.0)

    title = title_builder(player, year, card_set, subset, card_no, serial_no, grade, keywords)
    st.text_area("Suggested Title (≤80 chars)", title, height=60)

    bullets = [
        f"{year} {card_set} {subset}".strip(),
        f"{player} {('#'+card_no) if card_no else ''} {serial_no}".strip(),
        f"Condition: {grade}",
        "Securely shipped in bubble mailer & top loader",
        "Trusted seller—message for bundle deals"
    ]
    st.text_area("Description bullets", "\n• " + "\n• ".join([b for b in bullets if b.strip()]), height=140)
