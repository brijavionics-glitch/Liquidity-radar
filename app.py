"""Liquidity Radar: live NSE stock scoring with news, global cues and profit booking risk.
Prices: Yahoo Finance (may be a few minutes late). News: BSE and NSE corporate announcements.
This is a screening tool, not financial advice.
"""
import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Liquidity Radar", layout="centered")
IST = "Asia/Kolkata"

# ---------------- Stock list ----------------
DEFAULT_LIST = """PGEL,Electronics
DIXON,Electronics
RELIANCE,Energy
ONGC,Energy
HDFCBANK,Bank
ICICIBANK,Bank
SBIN,Bank
AXISBANK,Bank
KOTAKBANK,Bank
BAJFINANCE,Finance
INFY,IT
TCS,IT
WIPRO,IT
HCLTECH,IT
BHARTIARTL,Telecom
ITC,FMCG
HINDUNILVR,FMCG
NESTLEIND,FMCG
LT,Infra
ADANIENT,Infra
M&M,Auto
MARUTI,Auto
ASIANPAINT,Paint
INDIGO,Aviation
SUNPHARMA,Pharma
NTPC,Power
POWERGRID,Power
COALINDIA,Metal
TATASTEEL,Metal
JSWSTEEL,Metal
TITAN,Consumer
ULTRACEMCO,Cement"""

# Company names used to match BSE and NSE news to a stock
NAMES = {
    "PGEL": "pg electroplast", "DIXON": "dixon technologies", "RELIANCE": "reliance industries",
    "ONGC": "oil and natural gas", "HDFCBANK": "hdfc bank", "ICICIBANK": "icici bank",
    "SBIN": "state bank of india", "AXISBANK": "axis bank", "KOTAKBANK": "kotak mahindra bank",
    "BAJFINANCE": "bajaj finance", "INFY": "infosys", "TCS": "tata consultancy",
    "WIPRO": "wipro", "HCLTECH": "hcl technologies", "BHARTIARTL": "bharti airtel",
    "ITC": "itc ltd", "HINDUNILVR": "hindustan unilever", "NESTLEIND": "nestle india",
    "LT": "larsen", "ADANIENT": "adani enterprises", "M&M": "mahindra & mahindra",
    "MARUTI": "maruti suzuki", "ASIANPAINT": "asian paints", "INDIGO": "interglobe aviation",
    "SUNPHARMA": "sun pharmaceutical", "NTPC": "ntpc", "POWERGRID": "power grid",
    "COALINDIA": "coal india", "TATASTEEL": "tata steel", "JSWSTEEL": "jsw steel",
    "TITAN": "titan company", "ULTRACEMCO": "ultratech cement",
}

POS = ["order", "contract", "wins", "won", "bags", "record", "profit rise", "profit up", "beat",
       "upgrade", "buyback", "buy back", "bonus", "dividend", "acquisition", "acquires", "approval",
       "approves", "approved", "launch", "expansion", "capacity", "partnership", "higher target",
       "raises target", "growth", "strong", "award", "allotment of contract", "joint venture"]
NEG = ["downgrade", "loss", "miss", "decline", "fall", "fraud", "raid", "search", "penalty", "fine ",
       "sebi order", "probe", "investigation", "resign", "resignation", "auditor", "pledge", "default",
       "fire", "shutdown", "strike", "recall", "cut", "warning", "lower target", "weak", "clarification",
       "litigation", "show cause", "insolvency"]

W = {
    "For today": dict(flow=14, vol=9, vwap=8, lvl=7, gap=5, rng=6, w52=2, trend=3, rsi=4, news=12,
                      sector=6, macro=4, breadth=5, mood=15, k=0.25),
    "Next few days": dict(flow=9, vol=7, vwap=3, lvl=9, gap=2, rng=3, w52=7, trend=10, rsi=6, news=14,
                          sector=8, macro=4, breadth=2, mood=9, k=0.12),
}


def clamp(v, a, b):
    return max(a, min(b, v))


def sgn(v):
    return 1 if v > 0 else -1 if v < 0 else 0


def pct(v):
    return f"{v:+.1f}%"


def sentiment(text):
    t = " " + str(text).lower().replace("_", " ") + " "
    s = sum(w in t for w in POS) - sum(w in t for w in NEG)
    return sgn(s)


def rsi(series, n=14):
    d = series.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    if l.iloc[-1] == 0:
        return 100.0
    return float(100 - 100 / (1 + g.iloc[-1] / l.iloc[-1]))


def now_ist():
    return pd.Timestamp.now(tz=IST)


def phase():
    n = now_ist()
    m = n.hour * 60 + n.minute
    if n.weekday() >= 5:
        return "Weekend, market closed. Showing last trading day.", 1.0
    if m < 540:
        return "Before market. Showing last trading day.", 1.0
    if m < 555:
        return "Pre-open session", 0.6
    if m < 585:
        return "Opening 30 minutes, noisy, scores toned down", 0.7
    if m < 870:
        return "Main session", 1.0
    if m < 930:
        return "Closing hour", 0.9
    return "Market closed. Showing today's final data.", 1.0


# ---------------- Data fetch ----------------
def sub(df, t):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if t not in df.columns.get_level_values(0):
            return None
        out = df[t].dropna(how="all")
    else:
        out = df.dropna(how="all")
    return out if not out.empty else None


@st.cache_data(ttl=55, show_spinner=False)
def get_intraday(tickers):
    return yf.download(" ".join(tickers), period="5d", interval="5m", group_by="ticker",
                       progress=False, auto_adjust=False, threads=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_daily(tickers):
    return yf.download(" ".join(tickers), period="1y", interval="1d", group_by="ticker",
                       progress=False, auto_adjust=False, threads=True)


@st.cache_data(ttl=300, show_spinner=False)
def get_cues():
    m = {"dow": "^DJI", "nasdaq": "^IXIC", "nikkei": "^N225", "hangseng": "^HSI",
         "crude": "BZ=F", "usdinr": "INR=X", "vix": "^INDIAVIX", "nifty": "^NSEI"}
    df = yf.download(" ".join(m.values()), period="10d", interval="1d", group_by="ticker",
                     progress=False, auto_adjust=False, threads=True)
    out = {}
    for k, t in m.items():
        d = sub(df, t)
        if d is None or len(d["Close"].dropna()) < 2:
            out[k] = None
            continue
        c = d["Close"].dropna()
        out[k] = float(c.iloc[-1]) if k == "vix" else float((c.iloc[-1] / c.iloc[-2] - 1) * 100)
    return out


UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "application/json, text/plain, */*"}


@st.cache_data(ttl=60, show_spinner=False)
def get_bse_news():
    today = now_ist()
    start = (today - pd.Timedelta(days=2)).strftime("%Y%m%d")
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
    params = dict(pageno=1, strCat=-1, strPrevDate=start, strScrip="", strSearch="P",
                  strToDate=today.strftime("%Y%m%d"), strType="C", subcategory=-1)
    try:
        r = requests.get(url, params=params, headers={**UA, "Referer": "https://www.bseindia.com/",
                                                     "Origin": "https://www.bseindia.com"}, timeout=12)
        rows = r.json().get("Table", [])
        items = []
        for x in rows:
            ts = pd.to_datetime(x.get("NEWS_DT") or x.get("DissemDT"), errors="coerce")
            if pd.isna(ts):
                continue
            ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
            head = x.get("HEADLINE") or x.get("NEWSSUB") or ""
            items.append(dict(ts=ts, name=str(x.get("SLONGNAME", "")), sym="", src="BSE",
                              h=str(head).replace("_", " ").strip()))
        return items, f"BSE ok ({len(items)})"
    except Exception as e:
        return [], f"BSE not reachable ({type(e).__name__})"


@st.cache_data(ttl=60, show_spinner=False)
def get_nse_news():
    try:
        s = requests.Session()
        s.headers.update(UA)
        s.get("https://www.nseindia.com/companies-listing/corporate-filings-announcements", timeout=10)
        r = s.get("https://www.nseindia.com/api/corporate-announcements?index=equities", timeout=12)
        rows = r.json()
        items = []
        for x in rows[:300]:
            ts = pd.to_datetime(x.get("an_dt") or x.get("sort_date"), errors="coerce", dayfirst=True)
            if pd.isna(ts):
                continue
            ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
            head = (x.get("desc") or "") + ": " + str(x.get("attchmntText") or "")[:180]
            items.append(dict(ts=ts, name=str(x.get("sm_name", "")), sym=str(x.get("symbol", "")),
                              src="NSE", h=head.strip(": ")))
        return items, f"NSE ok ({len(items)})"
    except Exception as e:
        return [], f"NSE not reachable ({type(e).__name__})"


def match_news(items, symbols):
    out = []
    for n in items:
        sym = n["sym"] if n["sym"] in symbols else ""
        if not sym:
            nm = n["name"].lower()
            for s in symbols:
                key = NAMES.get(s)
                if key and key in nm:
                    sym = s
                    break
        if sym:
            out.append({**n, "sym": sym, "sent": sentiment(n["h"])})
    return out


# ---------------- Per stock numbers ----------------
def compute(sym, sec, d5, dd):
    d5 = d5.copy()
    d5.index = d5.index.tz_convert(IST) if d5.index.tz is not None else d5.index.tz_localize(IST)
    d5 = d5[d5["Volume"].fillna(0) >= 0].dropna(subset=["Close"])
    dates = d5.index.date
    last = dates[-1]
    today = d5[dates == last]
    prevd = d5[dates < last]
    if today.empty or prevd.empty:
        return None
    pday = prevd[prevd.index.date == prevd.index.date[-1]]
    prev = float(pday["Close"].iloc[-1])
    o, h, l, c = float(today["Open"].iloc[0]), float(today["High"].max()), float(today["Low"].min()), float(today["Close"].iloc[-1])
    v = today["Volume"].fillna(0)
    tp = (today["High"] + today["Low"] + today["Close"]) / 3
    vwap = float((tp * v).sum() / v.sum()) if v.sum() > 0 else c
    val = today["Close"] * v / 1e7
    up = today["Close"] > today["Open"]
    flat = today["Close"] == today["Open"]
    buy = float(val[up].sum() + val[flat].sum() / 2)
    sell = float(val[~up & ~flat].sum() + val[flat].sum() / 2)

    ddp = dd[dd.index.date < last] if dd is not None else None
    if ddp is None or len(ddp) < 25:
        return None
    avgv = float(ddp["Volume"].tail(20).mean())
    frac = clamp(len(today) * 5 / 375, 0.05, 1)
    volr = float(v.sum() / (avgv * frac)) if avgv > 0 else 1.0
    vol_now = float(v.tail(3).mean() / v.mean()) if v.mean() > 0 else 1.0
    closes = pd.concat([ddp["Close"].dropna(), pd.Series([c])], ignore_index=True)
    return dict(
        s=sym, sec=sec, prev=prev, open=o, high=h, low=l, ltp=c, vwap=vwap, buy=buy, sell=sell,
        pdh=float(pday["High"].max()), pdl=float(pday["Low"].min()),
        w52h=float(max(ddp["High"].max(), h)),
        dma20=float(ddp["Close"].tail(20).mean()), dma50=float(ddp["Close"].tail(50).mean()),
        rsi=rsi(closes.tail(120)), vol=volr, vol_now=vol_now, day=last,
        chg=(c / prev - 1) * 100, gap=(o / prev - 1) * 100, vwap_dist=(c / vwap - 1) * 100,
        rng=(c - l) / (h - l) if h > l else 0.5, from_high=(h - c) / h * 100, from_low=(c - l) / l * 100,
    )


# ---------------- Scoring ----------------
def market_mood(cues, fii):
    def idx(v, w):
        if v is None:
            return 0
        return w if v > 0.5 else -w if v < -0.5 else w / 2 if v > 0 else -w / 2 if v < 0 else 0
    m = idx(cues.get("dow"), 2) + idx(cues.get("nasdaq"), 2) + idx(cues.get("nikkei"), 1.5)
    m += idx(cues.get("hangseng"), 1.5) + idx(cues.get("nifty"), 4)
    cr, fx, vx = cues.get("crude"), cues.get("usdinr"), cues.get("vix")
    if cr is not None:
        m += -1.5 if cr > 1 else 1.5 if cr < -1 else 0
    if fx is not None:
        m += -1 if fx > 0.2 else 1 if fx < -0.2 else 0
    if vx is not None:
        m += -2 if vx > 18 else -1 if vx > 15 else 1 if vx < 13 else 0
    m += 2 if fii > 1000 else 1 if fii > 0 else -2 if fii < -1000 else -1 if fii < 0 else 0
    return clamp(m, -15, 15)


def news_factor(sym, news):
    s, items = 0.0, []
    now = now_ist()
    for n in news:
        if n["sym"] != sym:
            continue
        age = (now - n["ts"]).total_seconds() / 60
        w = 1 if age <= 60 else 0.7 if age <= 240 else 0.4 if age <= 1440 else 0.2
        s += n["sent"] * w
        items.append(n)
    return clamp(s, -1, 1), items


def reversal(r, nf):
    d, a, parts = sgn(r["chg"]), abs(r["chg"]), []
    if not d:
        return 0, parts

    def add(name, v):
        if v:
            parts.append((name, v))
    add(f"Move size {pct(r['chg'])}", 30 if a >= 7 else 20 if a >= 5 else 10 if a >= 3 else 0)
    vd = r["vwap_dist"] * d
    add(f"Stretched from VWAP {pct(r['vwap_dist'])}", 20 if vd >= 4 else 10 if vd >= 2.5 else 0)
    rx = r["rsi"] if d > 0 else 100 - r["rsi"]
    add(f"RSI {r['rsi']:.0f}", 20 if rx >= 80 else 10 if rx >= 72 else 0)
    fe = r["from_high"] if d > 0 else r["from_low"]
    add(("Slipped from day high " if d > 0 else "Lifted from day low ") + f"{fe:.1f}%",
        20 if fe >= 2 else 10 if fe >= 1 else 0)
    add("Volume fading", 10 if a >= 3 and r["vol_now"] < 0.7 else 0)
    if nf * d > 0.3:
        add("Fresh news supports move", -10)
    elif a >= 5:
        add("No news behind move", 10)
    return clamp(sum(p[1] for p in parts), 0, 100), parts


def score(r, mode, mood, breadth, conf, sector_chg, cues, fii, news):
    w, parts, d = W[mode], [], sgn(r["chg"]) or 1

    def add(g, name, note, f, wt):
        parts.append(dict(Group=g, Rule=name, Value=note, Points=round(f * wt, 1), wt=wt))
    ratio = (r["buy"] - r["sell"]) / (r["buy"] + r["sell"]) if r["buy"] + r["sell"] > 0 else 0
    add("Money flow", "Buy vs sell liquidity (estimate)", f"{ratio * 100:.0f}% net", clamp(ratio * 4, -1, 1), w["flow"])
    vf = 1 if r["vol"] >= 2 else 0.6 if r["vol"] >= 1.5 else 0.2 if r["vol"] >= 1 else -0.3
    add("Money flow", "Volume vs normal", f"{r['vol']:.1f}x", vf * d, w["vol"])
    add("Price", "VWAP", ("above " if r["vwap_dist"] >= 0 else "below ") + pct(r["vwap_dist"]), 1 if r["vwap_dist"] >= 0 else -1, w["vwap"])
    lf = 1 if r["ltp"] > r["pdh"] else -1 if r["ltp"] < r["pdl"] else 0
    add("Price", "Breakout", "above yesterday high" if lf > 0 else "below yesterday low" if lf < 0 else "inside range", lf, w["lvl"])
    gf, gn = 0, "no gap"
    if r["gap"] > 0.8:
        gf = 1 if r["ltp"] >= r["open"] else -1
        gn = "gap up " + ("holding" if gf > 0 else "filling")
    elif r["gap"] < -0.8:
        gf = -1 if r["ltp"] <= r["open"] else 1
        gn = "gap down " + ("holding" if gf < 0 else "recovering")
    add("Price", "Gap at open", gn, gf, w["gap"])
    add("Price", "Day range", f"{r['rng'] * 100:.0f}% from low", (r["rng"] - 0.5) * 2, w["rng"])
    n52 = r["ltp"] >= r["w52h"] * 0.99
    add("Price", "52 week high", "at or near" if n52 else "not near", 1 if n52 else 0, w["w52"])
    tf = 1 if r["ltp"] > r["dma20"] > r["dma50"] else -1 if r["ltp"] < r["dma20"] < r["dma50"] else 0
    add("Price", "Trend (20 and 50 day average)", "uptrend" if tf > 0 else "downtrend" if tf < 0 else "mixed", tf, w["trend"])
    rf = 0.6 if 55 <= r["rsi"] <= 72 else -0.6 if 28 <= r["rsi"] <= 45 else 0
    add("Price", "RSI", f"{r['rsi']:.0f}", rf, w["rsi"])
    nf, items = news_factor(r["s"], news)
    add("News", "Official news", f"{len(items)} item(s)" if items else "none", nf, w["news"])
    sc = sector_chg.get(r["sec"], 0)
    add("Market", f"Sector {r['sec']}", pct(sc), clamp(sc, -1, 1), w["sector"])
    mf, mn = 0, "no link"
    fx, cr = cues.get("usdinr") or 0, cues.get("crude") or 0
    if r["sec"] == "IT":
        mf, mn = (1 if fx > 0.2 else -1 if fx < -0.2 else 0), "rupee " + ("weaker" if fx > 0 else "stronger")
    elif r["sec"] in ("Auto", "FMCG", "Paint", "Aviation"):
        mf, mn = (1 if cr < -1 else -1 if cr > 1 else 0), f"crude {pct(cr)}"
    elif r["sec"] in ("Bank", "Finance"):
        mf, mn = (1 if fii > 1000 else -1 if fii < -1000 else 0), f"FII {fii:,.0f} cr"
    add("Market", "Rupee, crude, FII", mn, mf, w["macro"])
    add("Market", "Breadth", f"{breadth * 100:.0f}% net", clamp(breadth * 2, -1, 1), w["breadth"])
    add("Market", "Global mood", f"{mood:+.1f}", mood / 15, w["mood"])
    total = sum(p["Points"] for p in parts)
    mx = sum(p["wt"] for p in parts)
    s = 50 + 50 * total / mx
    if mode == "For today":
        s = 50 + (s - 50) * conf
    risk, rparts = reversal(r, nf)
    adj = risk * w["k"] * sgn(r["chg"])
    s -= adj
    return int(round(clamp(s, 0, 100))), parts, risk, rparts, adj, items


def signal(s):
    return "Strong up" if s >= 70 else "Up" if s >= 60 else "Neutral" if s > 40 else "Down" if s > 30 else "Strong down"


# ---------------- UI ----------------
st.title("Liquidity Radar")

with st.sidebar:
    st.header("Settings")
    raw = st.text_area("Stocks (SYMBOL,Sector one per line)", DEFAULT_LIST, height=260)
    fii = st.number_input("FII net previous day (crore)", value=0.0, step=100.0)
    refresh = st.select_slider("Auto refresh every (seconds)", [30, 60, 120, 300], value=60)
    st.subheader("Add news yourself")
    with st.form("nf", clear_on_submit=True):
        ns = st.text_input("Symbol")
        nh = st.text_input("Headline")
        if st.form_submit_button("Add") and ns and nh:
            st.session_state.setdefault("manual", []).append(
                dict(ts=now_ist(), name="", sym=ns.strip().upper(), src="You", h=nh.strip()))

stocks = []
for line in raw.splitlines():
    p = [x.strip() for x in line.split(",")]
    if p and p[0]:
        stocks.append((p[0].upper(), p[1] if len(p) > 1 and p[1] else "Other"))
symbols = [s for s, _ in stocks]
sector_of = dict(stocks)
mode = st.radio("View", list(W.keys()), horizontal=True)


@st.fragment(run_every=refresh)
def live():
    ph, conf = phase()
    tick = [s + ".NS" for s in symbols]
    with st.spinner("Getting live data"):
        d5all, ddall, cues = get_intraday(tick), get_daily(tick), get_cues()
        bse, bst = get_bse_news()
        nse, nst = get_nse_news()
    news = match_news(bse + nse + st.session_state.get("manual", []), set(symbols))

    rows = []
    for s in symbols:
        try:
            d5, dd = sub(d5all, s + ".NS"), sub(ddall, s + ".NS")
            if d5 is None or dd is None:
                continue
            r = compute(s, sector_of[s], d5, dd)
            if r:
                rows.append(r)
        except Exception:
            continue
    if not rows:
        st.error("Could not get prices right now. Wait a minute, it will retry.")
        return

    sector_chg = pd.DataFrame(rows).groupby("sec")["chg"].mean().to_dict()
    adv = sum(r["chg"] > 0 for r in rows)
    dec = sum(r["chg"] < 0 for r in rows)
    breadth = (adv - dec) / len(rows)
    mood = market_mood(cues, fii)
    for r in rows:
        r["score"], r["parts"], r["risk"], r["rparts"], r["adj"], r["news"] = score(
            r, mode, mood, breadth, conf, sector_chg, cues, fii, news)

    st.caption(f"{ph} · Updated {now_ist():%I:%M:%S %p} · Prices from Yahoo Finance, can be a few minutes late · {bst} · {nst}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Global mood", "Positive" if mood > 5 else "Negative" if mood < -5 else "Mixed", f"{mood:+.1f}")
    nifty = cues.get("nifty")
    c2.metric("Nifty", "-" if nifty is None else pct(nifty))
    c3.metric("Up vs down", f"{adv} / {dec}")

    df = pd.DataFrame([{
        "Stock": r["s"], "Price": round(r["ltp"], 1), "Change %": round(r["chg"], 2), "Score": r["score"],
        "Signal": signal(r["score"]),
        "Warning": (("Profit booking " if r["chg"] > 0 else "Bounce ") + str(r["risk"])) if r["risk"] >= 50 else "",
    } for r in rows])
    cfg = {"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
           "Change %": st.column_config.NumberColumn(format="%+.2f")}

    st.subheader("Top 10 likely to go up")
    st.dataframe(df.sort_values("Score", ascending=False).head(10), hide_index=True, column_config=cfg, use_container_width=True)
    st.subheader("Top 10 likely to go down")
    st.dataframe(df.sort_values("Score").head(10), hide_index=True, column_config=cfg, use_container_width=True)

    def watch(cond):
        w = [r for r in rows if cond(r) and r["risk"] >= 40]
        w.sort(key=lambda r: -r["risk"])
        return pd.DataFrame([{"Stock": r["s"], "Change %": round(r["chg"], 2), "Risk": r["risk"],
                              "Why": ", ".join(p[0] for p in r["rparts"] if p[1] > 0)} for r in w[:5]])
    st.subheader("Profit booking watch")
    st.caption("Up a lot today, signs that selling may start")
    pb = watch(lambda r: r["chg"] > 0)
    if pb.empty:
        st.write("None right now.")
    else:
        st.dataframe(pb, hide_index=True, use_container_width=True)
    st.subheader("Bounce watch")
    st.caption("Down a lot today, signs that buyers may step in")
    bn = watch(lambda r: r["chg"] < 0)
    if bn.empty:
        st.write("None right now.")
    else:
        st.dataframe(bn, hide_index=True, use_container_width=True)

    st.subheader("Why this score")
    pick = st.selectbox("Pick a stock", sorted(r["s"] for r in rows), key="pick")
    r = next(x for x in rows if x["s"] == pick)
    st.write(f"**{r['s']}** score {r['score']} ({signal(r['score'])}), price {r['ltp']:.1f}, change {pct(r['chg'])}")
    st.dataframe(pd.DataFrame(r["parts"]).drop(columns="wt"), hide_index=True, use_container_width=True)
    if r["rparts"]:
        st.write(("Profit booking risk " if r["chg"] > 0 else "Bounce chance ") + f"{r['risk']}, score changed by {-r['adj']:+.1f}")
        st.dataframe(pd.DataFrame(r["rparts"], columns=["Sign", "Points"]), hide_index=True, use_container_width=True)
    for n in r["news"]:
        st.write(f"{n['ts']:%d %b %I:%M %p} · {n['src']} · {'good' if n['sent'] > 0 else 'bad' if n['sent'] < 0 else 'neutral'} · {n['h']}")

    st.subheader("Latest news for your stocks")
    if news:
        nd = pd.DataFrame([{"Time": f"{n['ts']:%d %b %I:%M %p}", "Stock": n["sym"], "Source": n["src"],
                            "Type": "good" if n["sent"] > 0 else "bad" if n["sent"] < 0 else "neutral",
                            "Headline": n["h"], "_t": n["ts"]} for n in news])
        st.dataframe(nd.sort_values("_t", ascending=False).drop(columns="_t").head(40), hide_index=True, use_container_width=True)
    else:
        st.write("No news for your stocks in the last 2 days.")

    st.subheader("Buying and selling liquidity (crore, estimate)")
    lq = pd.DataFrame([{"Stock": r["s"], "Buy": round(r["buy"], 1), "Sell": round(r["sell"], 1)} for r in rows])
    a, b = st.columns(2)
    a.write("Top 10 buying")
    a.dataframe(lq.sort_values("Buy", ascending=False).head(10)[["Stock", "Buy"]], hide_index=True, use_container_width=True)
    b.write("Top 10 selling")
    b.dataframe(lq.sort_values("Sell", ascending=False).head(10)[["Stock", "Sell"]], hide_index=True, use_container_width=True)


live()

with st.expander("How the score works"):
    st.markdown("""
1. Buy vs sell liquidity: money traded in rising 5 minute candles vs falling ones.
2. Volume compared with the normal amount for this time of day.
3. Price above or below VWAP (Volume Weighted Average Price).
4. Breakout above yesterday's high or below yesterday's low.
5. Gap at open, holding or filling.
6. Where the price sits in today's range.
7. Near 52 week high.
8. Trend from 20 day and 50 day averages.
9. RSI (Relative Strength Index) zone.
10. Official news from BSE and NSE announcements. Newer news counts more.
11. Sector strength today.
12. Rupee, crude and FII (Foreign Institutional Investors) link to the sector.
13. Market breadth: stocks up vs down.
14. Global mood: US, Japan, Hong Kong, Nifty, crude, rupee, India VIX (Volatility Index), FII.
15. Time of day: first 30 minutes are toned down.
16. Profit booking risk: big move, stretched from VWAP, high RSI, slipping from the high, fading volume, no news. Fresh news lowers the risk.
""")
st.caption("Screening tool only. Not financial advice.")
