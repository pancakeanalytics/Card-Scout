# ---- Quick eBay Scrape -> Table -> Monte Carlo ----
import streamlit as st
import pandas as pd
import numpy as np
import requests, time, re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
                  "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
}

def build_ebay_sold_url(query, page=1):
    q = quote_plus(query.strip())
    base = f"https://www.ebay.com/sch/i.html?_nkw={q}&LH_Sold=1&LH_Complete=1"
    if page > 1:
        base += f"&_pgn={page}"
    return base

_price_re = re.compile(r"([\d,]+(?:\.\d{1,2})?)")
_date_re  = re.compile(r"(Ended|Sold)\s*[:\-]?\s*(.+)", re.I)

def parse_price(txt: str):
    # handle "to" ranges: "$64.99 to $79.99" -> first number
    m = _price_re.findall(txt.replace("to", " "))
    return float(m[0].replace(",", "")) if m else None

def parse_date(txt: str):
    # eBay shows "Sold  Jun 10, 2025" or "Ended: Jun 10, 2025"
    m = _date_re.search(txt)
    return m.group(2).strip() if m else None

def scrape_ebay_sold(query, max_pages=2, sleep_sec=1.2):
    rows = []
    for p in range(1, max_pages+1):
        url = build_ebay_sold_url(query, p)
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        for li in soup.select("li.s-item"):
            title_el = li.select_one(".s-item__title")
            # skip "Shop on eBay" placeholders
            if not title_el or "Shop on eBay" in title_el.get_text(strip=True):
                continue

            price_el = li.select_one(".s-item__price")
            date_el  = li.select_one(".s-item__ended-date") or li.select_one(".s-item__subtitle span")
            link_el  = li.select_one("a.s-item__link")

            title = title_el.get_text(" ", strip=True)
            price = parse_price(price_el.get_text(" ", strip=True)) if price_el else None
            date_txt = parse_date(date_el.get_text(" ", strip=True)) if date_el else None
            url_item = link_el["href"] if link_el and link_el.has_attr("href") else None

            if price is not None:
                rows.append({"date_text": date_txt, "price": price, "title": title, "url": url_item})

        # be polite; stop if there’s no next page link
        next_btn = soup.select_one("a.pagination__next")
        if not next_btn:
            break
        time.sleep(sleep_sec)
    df = pd.DataFrame(rows)
    # try to coerce date if present
    if "date_text" in df and not df["date_text"].isna().all():
        df["date"] = pd.to_datetime(df["date_text"], errors="coerce")
        df = df.sort_values("date", na_position="last")
    return df

def winsorize(a, low=2.5, high=97.5):
    lo, hi = np.percentile(a, [low, high])
    return np.clip(a, lo, hi)

def recency_weights(n, scheme="exp"):
    if n <= 1: return np.ones(n)
    if scheme == "exp":
        w = 0.9 ** np.arange(n-1, -1, -1)
    elif scheme == "linear":
        w = np.arange(1, n+1)
    else:
        w = np.ones(n)
    return w / w.sum()

def monte_carlo(prices, n_iter=10000, recency="exp", noise_pct=2.0):
    p = winsorize(np.array(prices, dtype=float), 2.5, 97.5)
    w = recency_weights(len(p), recency)
    idx = np.random.choice(np.arange(len(p)), size=n_iter, replace=True, p=w)
    sim = p[idx]
    noise = np.random.normal(0.0, noise_pct/100.0, size=n_iter)
    return sim * (1 + noise)

def quick_scrape_tab():
    st.subheader("🔎 eBay Sold Comps (Web Scrape) → Forecast/Sim")
    q = st.text_input("Card search", "2020 Prizm Justin Herbert Silver PSA 10")
    col = st.columns(3)
    max_pages = col[0].slider("Pages", 1, 5, 2)
    fee_pct   = col[1].number_input("Marketplace fee %", 0.0, 25.0, 12.9, 0.1)
    fixed_fee = col[2].number_input("Fixed fee $", 0.0, 5.0, 0.30, 0.05)

    if st.button("Fetch sold comps"):
        with st.spinner("Scraping eBay…"):
            df = scrape_ebay_sold(q, max_pages=max_pages)
        if df.empty:
            st.warning("No results found (or HTML changed). Try refining your query.")
            return
        st.success(f"Found {len(df)} sold/ended results")
        st.dataframe(df[["date", "price", "title", "url"]], use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="ebay_comps.csv", mime="text/csv")

        # Quick stats
        st.markdown("**Quick stats**")
        p = df["price"].dropna().astype(float)
        st.write({
            "mean": round(p.mean(), 2),
            "median": round(p.median(), 2),
            "min": round(p.min(), 2),
            "max": round(p.max(), 2),
            "n": int(p.shape[0]),
        })

        # Monte Carlo (resale price only)
        st.divider()
        st.markdown("### 🎲 Monte Carlo on resale price")
        cost_basis = st.number_input("Your buy price ($)", min_value=0.0, value=float(p.median() if len(p) else 0))
        shipping   = st.number_input("Your shipping/supplies ($)", min_value=0.0, value=5.50, step=0.25)
        tax_pct    = st.number_input("Taxes on sale % (optional)", min_value=0.0, value=0.0, step=0.5)
        sims = monte_carlo(p.values, n_iter=10000, recency="exp", noise_pct=2.0)

        fees = sims * (fee_pct/100.0) + fixed_fee
        taxes = sims * (tax_pct/100.0)
        net = sims - (fees + taxes + shipping + cost_basis)

        prob_profit = float((net > 0).mean())
        p5, p50, p95 = np.percentile(net, [5, 50, 95])

        st.write(f"**Chance of profit:** {prob_profit:.1%}")
        st.write(f"**Net profit (P5 / P50 / P95):** ${p5:,.0f} / ${p50:,.0f} / ${p95:,.0f}")

        st.bar_chart(pd.Series(net, name="Simulated Net Profit"))
        st.caption("Tip: tweak fee %, shipping, and buy price to see negotiation room.")

# Call quick_scrape_tab() inside one of your tabs in app.py
# with tabN:
#     quick_scrape_tab()

