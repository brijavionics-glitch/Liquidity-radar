"""Liquidity Radar v2: top 5 buy ideas for Today, 1 week and 1 month on NSE.
Scans Nifty 100/200/500 with technicals, 25+ candle patterns, liquidity, news and global cues.
Gives buy price, two targets and a stop loss for every pick.
Prices: Yahoo Finance (can be a few minutes late). News: NSE corporate announcements.
Screening tool only. Not financial advice.
"""
import io

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Liquidity Radar", layout="wide", initial_sidebar_state="collapsed")
IST = "Asia/Kolkata"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "*/*"}

# ======================= UNIVERSE =======================
FALLBACK = """RELIANCE,Energy;HDFCBANK,Bank;ICICIBANK,Bank;INFY,IT;TCS,IT;SBIN,Bank;BHARTIARTL,Telecom;ITC,FMCG;LT,Infra;
AXISBANK,Bank;KOTAKBANK,Bank;HINDUNILVR,FMCG;BAJFINANCE,Finance;BAJAJFINSV,Finance;MARUTI,Auto;M&M,Auto;SUNPHARMA,Pharma;
HCLTECH,IT;WIPRO,IT;TECHM,IT;LTIM,IT;ASIANPAINT,Paint;TITAN,Consumer;ULTRACEMCO,Cement;NESTLEIND,FMCG;NTPC,Power;
POWERGRID,Power;ONGC,Energy;COALINDIA,Metal;TATASTEEL,Metal;JSWSTEEL,Metal;HINDALCO,Metal;ADANIENT,Infra;ADANIPORTS,Infra;
GRASIM,Cement;CIPLA,Pharma;DRREDDY,Pharma;DIVISLAB,Pharma;APOLLOHOSP,Health;EICHERMOT,Auto;HEROMOTOCO,Auto;BAJAJ-AUTO,Auto;
TVSMOTOR,Auto;BRITANNIA,FMCG;TATACONSUM,FMCG;INDUSINDBK,Bank;SBILIFE,Insurance;HDFCLIFE,Insurance;SHRIRAMFIN,Finance;
BEL,Defence;HAL,Defence;TRENT,Retail;DMART,Retail;ETERNAL,Consumer;JIOFIN,Finance;PIDILITIND,Chemicals;DABUR,FMCG;
GODREJCP,FMCG;SIEMENS,Capital goods;ABB,Capital goods;DLF,Realty;GODREJPROP,Realty;BANKBARODA,Bank;PNB,Bank;CANBK,Bank;
UNIONBANK,Bank;IOC,Energy;BPCL,Energy;GAIL,Energy;TATAPOWER,Power;ADANIGREEN,Power;VEDL,Metal;AMBUJACEM,Cement;
SHREECEM,Cement;HAVELLS,Consumer;POLYCAB,Capital goods;DIXON,Electronics;PGEL,Electronics;INDIGO,Aviation;IRCTC,Travel;
IRFC,Finance;PFC,Finance;RECLTD,Finance;LICI,Insurance;ICICIPRULI,Insurance;ICICIGI,Insurance;CHOLAFIN,Finance;
MUTHOOTFIN,Finance;NAUKRI,IT;PERSISTENT,IT;COFORGE,IT;MPHASIS,IT;TORNTPHARM,Pharma;LUPIN,Pharma;AUROPHARMA,Pharma;
ZYDUSLIFE,Pharma;MARICO,FMCG;COLPAL,FMCG;UPL,Chemicals;SRF,Chemicals;BHEL,Capital goods;CGPOWER,Capital goods;
CUMMINSIND,Capital goods;MOTHERSON,Auto;BOSCHLTD,Auto;TIINDIA,Auto;FEDERALBNK,Bank;IDFCFIRSTB,Bank;AUBANK,Bank;
VOLTAS,Consumer;NMDC,Metal;SAIL,Metal;JINDALSTEL,Metal;INDHOTEL,Hotels;LODHA,Realty;OBEROIRLTY,Realty;MAXHEALTH,Health"""


@st.cache_data(ttl=86400, show_spinner=False)
def get_universe(name):
    f = {"Nifty 100": "ind_nifty100list.csv", "Nifty 200": "ind_nifty200list.csv", "Nifty 500": "ind_nifty500list.csv"}[name]
    for base in ("https://nsearchives.nseindia.com/content/indices/", "https://archives.nseindia.com/content/indices/"):
        try:
            r = requests.get(base + f, headers=UA, timeout=15)
            if r.ok and "Symbol" in r.text:
                df = pd.read_csv(io.StringIO(r.text))
                return [(s.strip(), str(i).strip()) for s, i in zip(df["Symbol"], df["Industry"])], f"{name} list from NSE"
        except Exception:
            pass
    rows = [x.strip().split(",") for x in FALLBACK.replace("\n", "").split(";") if x.strip()]
    return [(a, b) for a, b in rows], "built-in list of 115 big stocks"


# ======================= DATA =======================
def sub(df, t):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if t not in df.columns.get_level_values(0):
            return None
        out = df[t]
    else:
        out = df
    out = out.dropna(subset=["Close"]) if "Close" in out else None
    return out if out is not None and not out.empty else None


@st.cache_data(ttl=600, show_spinner=False)
def get_daily(tickers):
    parts = []
    for i in range(0, len(tickers), 100):
        try:
            parts.append(yf.download(" ".join(tickers[i:i + 100]), period="1y", interval="1d", group_by="ticker",
                                     progress=False, auto_adjust=False, threads=True))
        except Exception:
            pass
    return parts


@st.cache_data(ttl=540, show_spinner=False)
def get_intraday(tickers):
    try:
        return yf.download(" ".join(tickers), period="5d", interval="5m", group_by="ticker",
                           progress=False, auto_adjust=False, threads=True)
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_cues():
    m = {"US Dow": "^DJI", "US Nasdaq": "^IXIC", "Japan": "^N225", "Hong Kong": "^HSI",
         "Crude": "BZ=F", "Dollar-Rupee": "INR=X", "India VIX": "^INDIAVIX", "Nifty": "^NSEI"}
    out, nifty = {}, None
    try:
        df = yf.download(" ".join(m.values()), period="1y", interval="1d", group_by="ticker", progress=False, auto_adjust=False)
    except Exception:
        return out, nifty
    for k, t in m.items():
        d = sub(df, t)
        if d is None or len(d) < 2:
            continue
        c = d["Close"]
        out[k] = float(c.iloc[-1]) if k == "India VIX" else float((c.iloc[-1] / c.iloc[-2] - 1) * 100)
        if k == "Nifty":
            nifty = c
    return out, nifty


@st.cache_resource(ttl=1800, show_spinner=False)
def nse_session():
    ses = requests.Session()
    ses.headers.update(UA)
    try:
        ses.get("https://www.nseindia.com/companies-listing/corporate-filings-announcements", timeout=10)
    except Exception:
        pass
    return ses


def nse_json(url):
    r = nse_session().get(url, timeout=12)
    if r.status_code in (401, 403):
        nse_session.clear()
        r = nse_session().get(url, timeout=12)
    return r.json()


SERIOUS = ["show cause", "penalty", "fraud", "raid", "search and seizure", "default", "insolvency", "forensic",
           "resignation of auditor", "auditor resign", "suspension", "adjudication", "sebi order", "income tax search"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_news(sym):
    """Last 30 days of NSE filings for one share. Returns (date, headline) of the latest serious one, or None."""
    try:
        rows = nse_json(f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={sym}")
        cut = now_ist() - pd.Timedelta(days=30)
        for x in rows[:60]:
            ts = pd.to_datetime(x.get("an_dt") or x.get("sort_date"), errors="coerce", dayfirst=True)
            if pd.isna(ts):
                continue
            ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
            if ts < cut:
                break
            h = ((x.get("desc") or "") + " " + str(x.get("attchmntText") or "")).lower()
            if any(w in h for w in SERIOUS):
                return (ts, (x.get("desc") or "")[:80])
    except Exception:
        pass
    return None


@st.cache_data(ttl=60, show_spinner=False)
def get_news():
    try:
        rows = nse_json("https://www.nseindia.com/api/corporate-announcements?index=equities")
        items = []
        for x in rows[:500]:
            ts = pd.to_datetime(x.get("an_dt") or x.get("sort_date"), errors="coerce", dayfirst=True)
            if pd.isna(ts):
                continue
            ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
            h = ((x.get("desc") or "") + ": " + str(x.get("attchmntText") or "")[:200]).strip(": ")
            items.append(dict(ts=ts, sym=str(x.get("symbol", "")), h=h, sent=sentiment(h)))
        return items, True
    except Exception:
        return [], False


@st.cache_data(ttl=3600, show_spinner=False)
def get_events():
    """Upcoming board meetings and results from the NSE event calendar: symbol -> (date, purpose)."""
    try:
        rows = nse_json("https://www.nseindia.com/api/event-calendar?index=equities")
        today = now_ist().date()
        ev = {}
        for x in rows:
            d = pd.to_datetime(x.get("date"), errors="coerce", dayfirst=True)
            if pd.isna(d):
                continue
            d = d.date()
            if d < today or (d - today).days > 45:
                continue
            sym = str(x.get("symbol", ""))
            p = str(x.get("purpose") or x.get("bm_purpose") or x.get("bm_desc") or "Board meeting")[:80]
            if sym not in ev or d < ev[sym][0]:
                ev[sym] = (d, p)
        return ev
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def get_all_symbols():
    """Every stock listed on NSE with its company name, for the search box."""
    for base in ("https://nsearchives.nseindia.com/content/equities/", "https://archives.nseindia.com/content/equities/"):
        try:
            r = requests.get(base + "EQUITY_L.csv", headers=UA, timeout=15)
            if r.ok and "SYMBOL" in r.text:
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip() for c in df.columns]
                df = df[df["SERIES"].astype(str).str.strip().isin(["EQ", "BE", "BZ"])]
                return [(str(a).strip(), str(b).strip()) for a, b in zip(df["SYMBOL"], df["NAME OF COMPANY"])]
        except Exception:
            pass
    return [(a, a) for a, _ in get_universe("Nifty 500")[0]]


@st.cache_data(ttl=21600, show_spinner=False)
def get_expected(sym):
    """Next results date and analyst estimates from Yahoo Finance, when available."""
    out = []
    try:
        cal = yf.Ticker(sym + ".NS").calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                ed = ed[0] if isinstance(ed, (list, tuple)) else ed
                out.append(f"Next results (Yahoo): {pd.Timestamp(ed):%d %b %Y}")
            if cal.get("Earnings Average") is not None:
                out.append(f"Expected EPS (Earnings Per Share): ₹{float(cal['Earnings Average']):,.2f}")
            if cal.get("Revenue Average") is not None:
                out.append(f"Expected revenue: ₹{float(cal['Revenue Average']) / 1e7:,.0f} crore")
    except Exception:
        pass
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def get_fii_dii():
    """Latest FII and DII net buying in crore from NSE."""
    try:
        rows = nse_json("https://www.nseindia.com/api/fiidiiTradeReact")
        fii = dii = None
        day = ""
        for x in rows:
            cat = str(x.get("category", "")).upper()
            net = float(str(x.get("netValue", "0")).replace(",", ""))
            day = x.get("date", day)
            if "FII" in cat or "FPI" in cat:
                fii = net
            elif "DII" in cat:
                dii = net
        return (fii, dii, day) if fii is not None else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_delivery():
    """Delivery percentage for every stock from the latest NSE bhavcopy."""
    day = now_ist().normalize().tz_localize(None)
    for i in range(7):
        d = day - pd.Timedelta(days=i)
        if d.weekday() >= 5:
            continue
        url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv"
        try:
            r = requests.get(url, headers=UA, timeout=15)
            if r.ok and "DELIV_PER" in r.text:
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip() for c in df.columns]
                df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
                val = pd.to_numeric(df["DELIV_PER"].astype(str).str.strip(), errors="coerce")
                return {str(a).strip(): float(b) for a, b in zip(df["SYMBOL"], val) if pd.notna(b)}, f"{d:%d %b}"
        except Exception:
            continue
    return {}, ""


@st.cache_data(ttl=3600, show_spinner=False)
def get_ban():
    """Stocks in the F&O (Futures and Options) ban list today."""
    try:
        r = requests.get("https://nsearchives.nseindia.com/content/fo/fo_secban.csv", headers=UA, timeout=12)
        return {ln.split(",")[-1].strip() for ln in r.text.splitlines()[1:] if "," in ln}
    except Exception:
        return set()


@st.cache_data(ttl=1800, show_spinner=False)
def get_deals():
    """Bulk and block deals: symbol -> (net buy minus sell count, client names)."""
    try:
        j = nse_json("https://www.nseindia.com/api/snapshot-capital-market-largedeal")
        out = {}
        for key in ("BULK_DEALS_DATA", "BLOCK_DEALS_DATA"):
            for x in j.get(key, []) or []:
                sym = str(x.get("symbol", ""))
                side = 1 if str(x.get("buySell", "")).upper().startswith("B") else -1
                n, who = out.get(sym, (0, []))
                out[sym] = (n + side, who + [f"{'Buy' if side > 0 else 'Sell'}: {str(x.get('clientName', ''))[:30]}"])
        return out
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def get_fund(sym):
    """P/E, ROE, debt and profit growth from Yahoo Finance."""
    try:
        t = yf.Ticker(sym + ".NS")
        i = t.info
        out = dict(pe=i.get("trailingPE"), roe=i.get("returnOnEquity"), de=i.get("debtToEquity"),
                   eg=i.get("earningsGrowth"), eps=i.get("trailingEps"), qoq=None, yoy=None,
                   tgt=i.get("targetMeanPrice"), nan=i.get("numberOfAnalystOpinions"))
        try:
            q = t.quarterly_income_stmt
            if q is not None and "Net Income" in q.index:
                ni = q.loc["Net Income"].dropna()
                if len(ni) >= 2 and ni.iloc[1] != 0:
                    out["qoq"] = float((ni.iloc[0] - ni.iloc[1]) / abs(ni.iloc[1]))
                if len(ni) >= 5 and ni.iloc[4] != 0:
                    out["yoy"] = float((ni.iloc[0] - ni.iloc[4]) / abs(ni.iloc[4]))
        except Exception:
            pass
        return out
    except Exception:
        return {}


# ======================= HELPERS =======================
POS = ["order", "contract", "wins", "won", "bags", "record", "profit rise", "profit up", "beat", "upgrade", "buyback",
       "buy back", "bonus", "dividend", "acquisition", "acquires", "approval", "approves", "approved", "launch",
       "expansion", "capacity", "partnership", "growth", "strong", "award", "joint venture", "letter of intent"]
NEG = ["downgrade", "loss", "miss", "decline", "fall", "fraud", "raid", "search", "penalty", "sebi order", "probe",
       "investigation", "resign", "auditor", "pledge", "default", "fire", "shutdown", "strike", "recall", "cut",
       "warning", "weak", "litigation", "show cause", "insolvency", "fine "]


def sentiment(t):
    t = " " + str(t).lower().replace("_", " ") + " "
    s = sum(w in t for w in POS) - sum(w in t for w in NEG)
    return 1 if s > 0 else -1 if s < 0 else 0


def clamp(v, a=-1.0, b=1.0):
    return max(a, min(b, v))


def tick(x):
    return round(round(x * 20) / 20, 2)


def now_ist():
    return pd.Timestamp.now(tz=IST)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi_s(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + g / l.replace(0, np.nan))).fillna(100)


def atr_s(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx_s(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    a = atr_s(h, l, c, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0), pdi, mdi


def supertrend(h, l, c, n=10, m=3):
    a = atr_s(h, l, c, n).values
    hl2 = ((h + l) / 2).values
    ub, lb, cc = hl2 + m * a, hl2 - m * a, c.values
    fu, fl, d = ub.copy(), lb.copy(), np.ones(len(cc))
    for i in range(1, len(cc)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or cc[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or cc[i - 1] < fl[i - 1]) else fl[i - 1]
        if d[i - 1] == 1 and cc[i] < fl[i]:
            d[i] = -1
        elif d[i - 1] == -1 and cc[i] > fu[i]:
            d[i] = 1
        else:
            d[i] = d[i - 1]
    return d


# ======================= CANDLE PATTERNS =======================
def candles(df):
    """Reads the last 1 to 5 candles. Returns list of (name, +1 bullish / -1 bearish, strength 1-3)."""
    if df is None or len(df) < 12:
        return []
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-9)
    ups = h - np.maximum(o, c)
    los = np.minimum(o, c) - l
    avgb = max(body[-11:-1].mean(), 1e-9)
    down = c[-2] < c[-7]
    up = c[-2] > c[-7]
    bull = lambda x: c[x] > o[x]
    bear = lambda x: c[x] < o[x]
    out = []
    add = lambda n, d, s: out.append((n, d, s))
    i, j, k = -1, -2, -3
    # one candle
    if los[i] >= 2 * body[i] and ups[i] <= 0.15 * rng[i] and body[i] > 0.05 * rng[i]:
        if down:
            add("Hammer", 1, 2)
        elif up:
            add("Hanging man", -1, 2)
    if ups[i] >= 2 * body[i] and los[i] <= 0.15 * rng[i] and body[i] > 0.05 * rng[i]:
        if down:
            add("Inverted hammer", 1, 1)
        elif up:
            add("Shooting star", -1, 3)
    if body[i] <= 0.1 * rng[i]:
        if los[i] >= 0.6 * rng[i] and down:
            add("Dragonfly doji", 1, 1)
        elif ups[i] >= 0.6 * rng[i] and up:
            add("Gravestone doji", -1, 2)
        elif up:
            add("Doji after rise", -1, 1)
    if bull(i) and body[i] >= 0.85 * rng[i] and body[i] > 1.3 * avgb:
        add("Bullish marubozu", 1, 2)
    if bear(i) and body[i] >= 0.85 * rng[i] and body[i] > 1.3 * avgb:
        add("Bearish marubozu", -1, 3)
    if bull(i) and body[i] > 2 * avgb and c[i] >= h[i] - 0.2 * rng[i]:
        add("Big green candle", 1, 1)
    if bear(i) and body[i] > 2 * avgb:
        add("Big red candle", -1, 2)
    if bull(i) and ups[i] > 1.5 * body[i] and up:
        add("Long upper wick (selling)", -1, 1)
    # two candles
    if bear(j) and bull(i) and o[i] <= c[j] and c[i] >= o[j] and body[i] > body[j]:
        add("Bullish engulfing", 1, 3)
    if bull(j) and bear(i) and o[i] >= c[j] and c[i] <= o[j] and body[i] > body[j]:
        add("Bearish engulfing", -1, 3)
    if bear(j) and bull(i) and body[j] > avgb and o[i] < c[j] and (o[j] + c[j]) / 2 < c[i] < o[j]:
        add("Piercing line", 1, 2)
    if bull(j) and bear(i) and body[j] > avgb and o[i] > c[j] and o[j] < c[i] < (o[j] + c[j]) / 2:
        add("Dark cloud cover", -1, 3)
    if bear(j) and body[j] > avgb and max(o[i], c[i]) < o[j] and min(o[i], c[i]) > c[j] and down:
        add("Bullish harami", 1, 1)
    if bull(j) and body[j] > avgb and max(o[i], c[i]) < c[j] and min(o[i], c[i]) > o[j] and up:
        add("Bearish harami", -1, 2)
    if abs(l[i] - l[j]) <= 0.002 * c[i] and bear(j) and bull(i) and down:
        add("Tweezer bottom", 1, 2)
    if abs(h[i] - h[j]) <= 0.002 * c[i] and bull(j) and bear(i) and up:
        add("Tweezer top", -1, 2)
    if bear(j) and bull(i) and o[i] > o[j] and body[i] > avgb and body[j] > avgb:
        add("Bullish kicker", 1, 3)
    if bull(j) and bear(i) and o[i] < o[j] and body[i] > avgb and body[j] > avgb:
        add("Bearish kicker", -1, 3)
    if l[i] > h[j]:
        add("Gap up kept open", 1, 2)
    if h[i] < l[j]:
        add("Gap down kept open", -1, 3)
    # three candles
    if h[j] < h[k] and l[j] > l[k] and c[i] > h[k]:
        add("Inside bar breakout", 1, 2)
    if h[j] < h[k] and l[j] > l[k] and c[i] < l[k]:
        add("Inside bar breakdown", -1, 2)
    if bear(k) and body[k] > avgb and body[j] < 0.5 * avgb and bull(i) and c[i] > (o[k] + c[k]) / 2:
        add("Morning star", 1, 3)
    if bull(k) and body[k] > avgb and body[j] < 0.5 * avgb and bear(i) and c[i] < (o[k] + c[k]) / 2:
        add("Evening star", -1, 3)
    if all(bull(x) for x in (k, j, i)) and c[k] < c[j] < c[i] and o[k] < o[j] < c[k] and o[j] < o[i] < c[j] \
            and min(body[k], body[j], body[i]) > 0.6 * avgb and max(ups[k], ups[j], ups[i]) < 0.4 * avgb:
        add("Three white soldiers", 1, 3)
    if all(bear(x) for x in (k, j, i)) and c[k] > c[j] > c[i] and min(body[k], body[j], body[i]) > 0.6 * avgb:
        add("Three black crows", -1, 3)
    if bear(k) and body[k] > avgb and bull(j) and max(o[j], c[j]) < o[k] and min(o[j], c[j]) > c[k] and bull(i) and c[i] > o[k]:
        add("Three inside up", 1, 2)
    if bull(k) and body[k] > avgb and bear(j) and max(o[j], c[j]) < c[k] and min(o[j], c[j]) > o[k] and bear(i) and c[i] < o[k]:
        add("Three inside down", -1, 2)
    # five candles
    a = -5
    if bull(a) and body[a] > avgb and all(l[x] > o[a] and h[x] < h[a] for x in (-4, -3, -2)) and bull(i) and c[i] > c[a]:
        add("Rising three methods", 1, 2)
    if bear(a) and body[a] > avgb and all(h[x] < o[a] and l[x] > l[a] for x in (-4, -3, -2)) and bear(i) and c[i] < c[a]:
        add("Falling three methods", -1, 2)
    return out


def candle_factor(pats):
    if not pats:
        return 0.0, "no clear pattern"
    s = sum(d * w for _, d, w in pats)
    return clamp(s / 3), ", ".join(n for n, _, _ in pats[:3])


# ======================= DAILY FEATURES =======================
def swing_highs(h):
    """Past turning-point highs (resistance) from the last year, plus the 52 week high."""
    v = h.values
    out = [float(v[i]) for i in range(5, len(v) - 5) if v[i] == v[i - 5:i + 6].max()]
    out.append(float(v.max()))
    return sorted(set(round(x, 2) for x in out))


def hit_prob(f, up, dn, n):
    """From this share's own last ~220 days: how often price rose by `up` (fraction)
    before falling by `dn` within n days. Same-day touch of both counts as a loss."""
    c, h, l = f["_c"], f["_h"], f["_l"]
    N = len(c)
    wins = total = 0
    for i in range(max(0, N - 1 - 220), N - 1):
        tu, td = c[i] * (1 + up), c[i] * (1 - dn)
        total += 1
        for j in range(i + 1, min(N, i + 1 + n)):
            if l[j] <= td:
                break
            if h[j] >= tu:
                wins += 1
                break
    return wins / total if total else 0.0


# ATR multiples for stop and targets (based on how far a share normally moves in that time)
MULT = {"today": dict(sl=.6, t1=.7, t2=1.2, n=1), "week": dict(sl=1.3, t1=1.3, t2=2.2, n=5),
        "month": dict(sl=2.2, t1=2.5, t2=4.0, n=20)}


def daily_features(sym, sec, d, nifty):
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    if len(d) < 120:
        return None
    o, h, l, c, v = d["Open"], d["High"], d["Low"], d["Close"], d["Volume"].fillna(0)
    e9, e20, e50, e200 = ema(c, 9), ema(c, 20), ema(c, 50), ema(c, 200)
    r = rsi_s(c)
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    hist = macd - sig
    adx, pdi, mdi = adx_s(h, l, c)
    st_dir = supertrend(h, l, c)
    atr = atr_s(h, l, c)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    bw = (4 * sd / mid)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    val = c * v / 1e7
    w = d.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    dd = c.pct_change()
    upv = val.tail(10)[dd.tail(10) > 0].sum()
    dnv = val.tail(10)[dd.tail(10) < 0].sum()
    first, second = d.iloc[-10:-5], d.iloc[-5:]
    nret = lambda n: float(nifty.iloc[-1] / nifty.iloc[-1 - n] - 1) * 100 if nifty is not None and len(nifty) > n else 0.0
    ret = lambda n: float(c.iloc[-1] / c.iloc[-1 - n] - 1) * 100
    cross = any(macd.iloc[-x] > sig.iloc[-x] and macd.iloc[-x - 1] <= sig.iloc[-x - 1] for x in (1, 2, 3))
    return dict(
        s=sym, sec=sec, day=d.index[-1].date(), c=float(c.iloc[-1]), o=float(o.iloc[-1]), h=float(h.iloc[-1]), l=float(l.iloc[-1]),
        prev=float(c.iloc[-2]), e9=float(e9.iloc[-1]), e20=float(e20.iloc[-1]), e50=float(e50.iloc[-1]), e200=float(e200.iloc[-1]),
        rsi=float(r.iloc[-1]), macd_up=bool(macd.iloc[-1] > sig.iloc[-1]), hist_up=bool(hist.iloc[-1] > hist.iloc[-2]), cross=cross,
        adx=float(adx.iloc[-1]), pdi=float(pdi.iloc[-1]), mdi=float(mdi.iloc[-1]),
        st=int(st_dir[-1]), st_flip=bool(st_dir[-1] == 1 and -1 in st_dir[-4:-1]),
        atr=float(atr.iloc[-1]), hi20=float(h.iloc[-21:-1].max()), hi55=float(h.iloc[-56:-1].max()), hi52=float(h.max()),
        lo10=float(l.tail(10).min()), lo20=float(l.tail(20).min()),
        squeeze=bool(bw.iloc[-2] <= bw.tail(120).quantile(0.2)), bb_up=float((mid + 2 * sd).iloc[-1]),
        obv_up=bool(obv.iloc[-1] > ema(obv, 20).iloc[-1]), obv_hi=bool(obv.iloc[-1] >= obv.tail(20).max()),
        liq=float(val.tail(20).mean()), volr=float(v.iloc[-1] / max(v.iloc[-21:-1].mean(), 1)),
        acc=float(upv / dnv) if dnv > 0 else 3.0,
        rs1=ret(21) - nret(21), rs3=ret(63) - nret(63), r1=ret(1), r5=ret(5), r21=ret(21),
        hhhl=1 if (second["High"].max() > first["High"].max() and second["Low"].min() > first["Low"].min())
        else -1 if (second["High"].max() < first["High"].max() and second["Low"].min() < first["Low"].min()) else 0,
        cd=candles(d.tail(20)), cw=candles(w.tail(20)),
        res=swing_highs(h.tail(250)), age=len(d), atrp=float(atr.iloc[-1] / c.iloc[-1] * 100),
        _c=c.tail(260).values, _h=h.tail(260).values, _l=l.tail(260).values,
    )


# ======================= INTRADAY FEATURES =======================
def intraday_features(d5):
    d5 = d5.copy()
    d5.index = d5.index.tz_convert(IST) if d5.index.tz is not None else d5.index.tz_localize(IST)
    d5 = d5.dropna(subset=["Close"])
    dates = d5.index.date
    t = d5[dates == dates[-1]]
    p = d5[dates < dates[-1]]
    if len(t) < 2 or p.empty:
        return None
    v = t["Volume"].fillna(0)
    tp = (t["High"] + t["Low"] + t["Close"]) / 3
    vwap = float((tp * v).sum() / v.sum()) if v.sum() > 0 else float(t["Close"].iloc[-1])
    val = t["Close"] * v
    upc = t["Close"] > t["Open"]
    buy, sell = float(val[upc].sum()), float(val[~upc].sum())
    pv = p[p.index.date == p.index.date[-1]]["Volume"].fillna(0)
    exp = pv.head(len(t)).sum()
    c5 = t["Close"]
    ltp, hi, lo = float(c5.iloc[-1]), float(t["High"].max()), float(t["Low"].min())
    return dict(
        day=dates[-1], ltp=ltp, open=float(t["Open"].iloc[0]), hi=hi, lo=lo, vwap=vwap, prev=float(p["Close"].iloc[-1]),
        orb_h=float(t["High"].head(3).max()), orb_l=float(t["Low"].head(3).min()), bars=len(t),
        e9=float(ema(d5["Close"], 9).iloc[-1]), e20=float(ema(d5["Close"], 20).iloc[-1]),
        ivol=float(v.sum() / exp) if exp > 0 else 1.0, vnow=float(v.tail(3).mean() / max(v.mean(), 1)),
        flow=(buy - sell) / (buy + sell) if buy + sell > 0 else 0.0,
        rsi5=float(rsi_s(d5["Close"]).iloc[-1]), c5=candles(d5.tail(20)), asof=t.index[-1],
    )


# ======================= SCORING =======================
W = {
    "today": dict(liq=8, vol=3, trend=4, e200=1, macd=3, rsi=3, adx=3, st=3, brk=3, sq=1, obv=2, rs=3, w52=1, stretch=3,
                  cd=4, cw=0, hhhl=2, acc=2, news=10, sector=4, mood=2,
                  inst=3, mkt=3, deliv=2, deal=3, fund=1, hist=5,
                  vwap=8, orb=6, e5=5, ivol=6, rng=4, gap=3, flow=6, c5=6),
    "week": dict(liq=6, vol=6, trend=7, e200=3, macd=6, rsi=5, adx=5, st=6, brk=7, sq=4, obv=5, rs=6, w52=3, stretch=4,
                 cd=7, cw=3, hhhl=4, acc=5, news=8, sector=4, mood=2,
                 inst=3, mkt=3, deliv=5, deal=4, fund=4, hist=7),
    "month": dict(liq=5, vol=4, trend=8, e200=7, macd=5, rsi=4, adx=5, st=5, brk=6, sq=3, obv=6, rs=8, w52=6, stretch=3,
                  cd=3, cw=8, hhhl=6, acc=5, news=6, sector=4, mood=1,
                  inst=2, mkt=3, deliv=6, deal=3, fund=9, hist=7),
}
NAMES = dict(hist="Past hit rate of Target 1", fund="Company health (P/E, ROE, growth)", inst="FII and DII money flow", mkt="Nifty trend (market direction)", deliv="Delivery percentage",
             deal="Bulk or block deals", liq="Liquidity (money traded daily)", vol="Volume vs normal", trend="Moving average trend", e200="Above 200 day average",
             macd="MACD momentum", rsi="RSI strength", adx="ADX trend power", st="Supertrend", brk="Breakout", sq="Squeeze breakout",
             obv="OBV buying pressure", rs="Stronger than Nifty", w52="Near 52 week high", stretch="Not over-stretched",
             cd="Daily candles", cw="Weekly candles", hhhl="Higher highs and lows", acc="Big money buying (10 days)",
             news="Official news", sector="Sector strength", mood="Global mood", vwap="VWAP", orb="Opening range breakout",
             e5="5 minute trend", ivol="Volume today vs yesterday", rng="Position in day range", gap="Gap at open",
             flow="Buy vs sell liquidity today", c5="5 minute candles")


def news_factor(sym, news):
    s, last = 0.0, None
    now = now_ist()
    for n in news:
        if n["sym"] != sym:
            continue
        age = (now - n["ts"]).total_seconds() / 3600
        w = 1 if age <= 2 else 0.7 if age <= 24 else 0.3 if age <= 72 else 0
        s += n["sent"] * w
        if last is None or n["ts"] > last["ts"]:
            last = n
    return clamp(s), last


def score(f, hz, ctx, intr=None):
    w = W[hz]
    chk = {}
    put = lambda k, val, note: chk.__setitem__(k, (clamp(val), note))
    put("liq", 1 if f["liq"] >= 200 else .7 if f["liq"] >= 50 else .4 if f["liq"] >= 20 else 0 if f["liq"] >= 10 else -1, f"₹{f['liq']:,.0f} cr a day")
    sgn = 1 if f["r1"] >= 0 else -1
    vr = f["volr"]
    put("vol", (1 if vr >= 2 else .7 if vr >= 1.5 else .3 if vr >= 1 else -.3) * sgn, f"{vr:.1f}x")
    n = (f["c"] > f["e20"]) + (f["e20"] > f["e50"]) + (f["e50"] > f["e200"])
    put("trend", {3: 1, 2: .5, 1: -.3, 0: -1}[n], f"{n} of 3 in order")
    put("e200", 1 if f["c"] > f["e200"] else -1, "above" if f["c"] > f["e200"] else "below")
    put("macd", (1 if f["cross"] else .6 + .4 * f["hist_up"]) if f["macd_up"] else -.6 - .4 * (not f["hist_up"]),
        "fresh buy cross" if f["cross"] and f["macd_up"] else "positive" if f["macd_up"] else "negative")
    r = f["rsi"]
    put("rsi", 1 if 55 <= r <= 70 else .4 if 50 <= r < 55 else .2 if 70 < r <= 78 else -.6 if r > 78 else -.3 if r >= 40 else -.8, f"{r:.0f}")
    put("adx", (1 if f["adx"] > 25 else .6 if f["adx"] > 20 else .2) if f["pdi"] > f["mdi"] else (-1 if f["adx"] > 20 else -.3), f"{f['adx']:.0f}")
    put("st", 1 if f["st"] == 1 else -1, "fresh buy signal" if f["st_flip"] else "buy" if f["st"] == 1 else "sell")
    level = f["hi55"] if hz == "month" else f["hi20"]
    put("brk", 1 if f["c"] > level else .4 if f["c"] >= level * 0.98 else 0, "broke out" if f["c"] > level else f"{(level / f['c'] - 1) * 100:.1f}% below high")
    put("sq", 1 if f["squeeze"] and f["c"] > f["bb_up"] else .3 if f["squeeze"] else 0, "breaking out" if f["squeeze"] and f["c"] > f["bb_up"] else "tight" if f["squeeze"] else "no")
    put("obv", 1 if f["obv_hi"] else .6 if f["obv_up"] else -.5, "new high" if f["obv_hi"] else "rising" if f["obv_up"] else "falling")
    put("rs", 1 if f["rs1"] > 0 and f["rs3"] > 0 else .3 if f["rs1"] > 0 or f["rs3"] > 0 else -1, f"{f['rs1']:+.1f}% in a month")
    put("w52", 1 if f["c"] >= .95 * f["hi52"] else .3 if f["c"] >= .85 * f["hi52"] else -.5 if f["c"] < .7 * f["hi52"] else 0, f"{(f['c'] / f['hi52'] - 1) * 100:.0f}% from high")
    dist = (f["c"] / f["e20"] - 1) * 100
    put("stretch", -1 if dist > 12 else -.5 if dist > 8 else .5 if dist >= 0 else -.3, f"{dist:+.1f}% from 20 day average")
    cdf, cdn = candle_factor(f["cd"])
    put("cd", cdf, cdn)
    cwf, cwn = candle_factor(f["cw"])
    put("cw", cwf, cwn)
    put("hhhl", f["hhhl"], {1: "yes", -1: "lower highs", 0: "sideways"}[f["hhhl"]])
    put("acc", 1 if f["acc"] > 1.5 else .5 if f["acc"] > 1.1 else -.7 if f["acc"] < .8 else 0, f"{f['acc']:.1f}x more on up days")
    nf, last = news_factor(f["s"], ctx["news"])
    put("news", nf, last["h"][:70] if last else "none")
    sr = ctx["sector_rank"].get(f["sec"], .5)
    put("sector", (sr - .5) * 2, f"{f['sec']} rank {sr * 100:.0f}%")
    put("mood", ctx["mood"] / 15, f"{ctx['mood']:+.0f}")
    fl = ctx.get("flows")
    if fl:
        fii, dii = fl[0], fl[1] or 0
        big = 1.3 if (f["liq"] >= 200 or f["sec"] in ("Bank", "Financial Services", "Finance", "Information Technology", "IT")) else 1
        put("inst", (clamp(fii / 3000) * .6 + clamp((fii + dii) / 3000) * .4) * big, f"FII ₹{fii:,.0f} cr, DII ₹{dii:,.0f} cr")
    mt = ctx.get("mkt_trend", 0)
    put("mkt", mt, {1: "Nifty in uptrend", -1: "Nifty in downtrend", 0: "Nifty sideways"}[mt])
    dv = ctx.get("deliv", {}).get(f["s"])
    if dv is not None:
        put("deliv", (1 if dv >= 60 else .5 if dv >= 45 else 0 if dv >= 30 else -.5) * (1 if f["r1"] >= 0 else -1), f"{dv:.0f}%")
    mm = MULT[hz]
    ph = hit_prob(f, mm["t1"] * f["atr"] / f["c"], mm["sl"] * f["atr"] / f["c"], mm["n"])
    put("hist", (ph - .45) * 4, f"{ph * 100:.0f}% of past setups reached it")
    fd = ctx.get("fund", {}).get(f["s"])
    if fd:
        vals, notes = [], []
        pe, roe, de, eg = fd.get("pe"), fd.get("roe"), fd.get("de"), fd.get("eg")
        if pe is not None:
            vals.append(-1 if pe <= 0 else 1 if pe <= 25 else .5 if pe <= 40 else 0 if pe <= 60 else -.6 if pe <= 100 else -1)
            notes.append(f"P/E {pe:.0f}")
        if roe is not None:
            vals.append(1 if roe >= .18 else .5 if roe >= .12 else 0 if roe >= .06 else -.7)
            notes.append(f"ROE {roe * 100:.0f}%")
        if eg is not None:
            vals.append(1 if eg > .15 else .4 if eg > 0 else -.7 if eg < -.1 else 0)
            notes.append(f"profit growth {eg * 100:+.0f}%")
        qq, yy = fd.get("qoq"), fd.get("yoy")
        if yy is not None:
            vals.append(1 if yy > .2 else .3 if yy > 0 else -.6 if yy > -.3 else -1)
            notes.append(f"last quarter profit {yy * 100:+.0f}% vs year ago")
        if qq is not None and qq < -.4:
            vals.append(-1)
            notes.append(f"profit {qq * 100:+.0f}% vs previous quarter")
        if fd.get("tgt") and (fd.get("nan") or 0) >= 3:
            up_ = fd["tgt"] / f["c"] - 1
            vals.append(1 if up_ > .15 else .4 if up_ > .05 else -.5 if up_ < -.05 else 0)
            notes.append(f"analysts' target {up_ * 100:+.0f}%")
        if de is not None and de > 200:
            vals.append(-.6)
            notes.append("high debt")
        if vals:
            put("fund", sum(vals) / len(vals), ", ".join(notes))
    dl = ctx.get("deals", {}).get(f["s"])
    if dl:
        put("deal", clamp(dl[0]), "; ".join(dl[1][:2]))
    flags = []
    if hz == "today" and intr:
        x = intr
        vd = (x["ltp"] / x["vwap"] - 1) * 100
        put("vwap", clamp(vd / 1.0) if vd >= 0 else -1, f"{vd:+.1f}%")
        put("orb", 1 if x["ltp"] > x["orb_h"] else -1 if x["ltp"] < x["orb_l"] else 0, "above first 15 min high" if x["ltp"] > x["orb_h"] else "below first 15 min low" if x["ltp"] < x["orb_l"] else "inside")
        put("e5", 1 if x["ltp"] > x["e9"] > x["e20"] else -1 if x["ltp"] < x["e9"] < x["e20"] else 0, "up" if x["ltp"] > x["e9"] > x["e20"] else "down" if x["ltp"] < x["e9"] < x["e20"] else "mixed")
        put("ivol", 1 if x["ivol"] >= 2 else .6 if x["ivol"] >= 1.4 else .2 if x["ivol"] >= 1 else -.4, f"{x['ivol']:.1f}x")
        rp = (x["ltp"] - x["lo"]) / (x["hi"] - x["lo"]) if x["hi"] > x["lo"] else .5
        put("rng", (rp - .5) * 2, f"{rp * 100:.0f}% from low")
        gp = (x["open"] / x["prev"] - 1) * 100
        put("gap", (1 if x["ltp"] >= x["open"] else -1) if abs(gp) > .8 else 0, f"{gp:+.1f}%")
        put("flow", clamp(x["flow"] * 4), f"{x['flow'] * 100:+.0f}% net")
        c5f, c5n = candle_factor(x["c5"])
        put("c5", c5f, c5n)
        chg = (x["ltp"] / x["prev"] - 1) * 100
        risk = (30 if chg >= 7 else 20 if chg >= 5 else 10 if chg >= 3 else 0) + (20 if vd >= 4 else 10 if vd >= 2.5 else 0) \
            + (20 if x["rsi5"] >= 80 else 10 if x["rsi5"] >= 72 else 0) + (20 if (x["hi"] - x["ltp"]) / x["hi"] * 100 >= 2 else 0) \
            + (10 if chg >= 3 and x["vnow"] < .7 else 0) - (10 if nf > .3 else 0)
        if risk >= 50:
            flags.append(f"Profit booking risk {risk}")
        if any(d < 0 and s >= 3 for _, d, s in x["c5"][-3:] if True):
            flags.append("Strong bearish 5 minute candle")
        if x["ltp"] < x["vwap"]:
            flags.append("Below VWAP")
    tot = sum(v * w[k] for k, (v, _) in chk.items() if k in w)
    mx = sum(w[k] for k in chk if k in w)
    sc = 50 + 50 * tot / mx
    if hz == "today" and ctx["conf"] < 1:
        sc = 50 + (sc - 50) * ctx["conf"]
    # red flags that remove a stock from buy list
    main = f["cw"] if hz == "month" else f["cd"]
    bad = [n for n, d, s in main if d < 0 and s >= 2]
    if bad:
        flags.append("Bearish candle: " + bad[0])
    if nf <= -.5:
        flags.append("Bad news")
    if f["liq"] < 10:
        flags.append("Low liquidity")
    if hz != "today" and f["st"] != 1:
        flags.append("Supertrend says sell")
    if hz == "month" and f["c"] < f["e200"]:
        flags.append("Below 200 day average")
    if hz != "today" and f["rsi"] > 80:
        flags.append("Overbought, RSI above 80")
    if hz == "week" and f["r5"] > 15:
        flags.append("Already up over 15% this week")
    if fd and ((fd.get("eps") is not None and fd["eps"] < 0) or (fd.get("pe") is not None and fd["pe"] <= 0)):
        flags.append("Company is making a loss")
    lp = ctx.get("live_px", {}).get(f["s"])
    now_px = lp[0] if lp else f["c"]
    day_chg = (lp[0] / lp[1] - 1) * 100 if lp else 0.0
    if day_chg > (5 if hz == "month" else 2.5):
        flags.append(f"Already up {day_chg:.1f}% today. Do not chase")
    if fd and fd.get("tgt") and (fd.get("nan") or 0) >= 3 and hz != "today":
        up_ = (fd["tgt"] / now_px - 1) * 100
        if up_ < (-5 if hz == "month" else -10):
            flags.append(f"Analysts' average target ₹{fd['tgt']:,.0f} is below the price")
    if hz == "today" and f["atrp"] > 5:
        flags.append(f"Swings too much ({f['atrp']:.1f}% a day) for a same day trade")
    if hz != "today" and f["age"] < 180:
        flags.append("Listed less than 9 months ago, too little history")
    if hz != "today" and fd and ((fd.get("yoy") is not None and fd["yoy"] < -.3) or (fd.get("qoq") is not None and fd["qoq"] < -.5)):
        flags.append("Profit fell sharply in last results")
    sn = ctx.get("serious", {}).get(f["s"])
    if sn:
        flags.append(f"Serious news on {sn[0]:%d %b}: {sn[1][:60]}")
    if ph < .3:
        flags.append(f"Target 1 was reached only {ph * 100:.0f}% of the time before")
    if hz != "month" and f["s"] in ctx.get("ban", set()):
        flags.append("In F&O ban list, price can swing hard")
    if hz != "month" and dl and dl[0] < 0:
        flags.append("Big seller in bulk or block deal")
    ev = ctx.get("events", {}).get(f["s"])
    if ev and "result" in ev[1].lower():
        days = (ev[0] - now_ist().date()).days
        if (hz == "today" and days <= 1) or (hz == "week" and days <= 6):
            flags.append(f"Results on {ev[0]:%d %b}, price can jump either way")
    return int(round(clamp(sc, 0, 100))), chk, flags


def plan(f, hz, intr, live, trigger=None):
    a, m = f["atr"], MULT[hz]
    if trigger:
        e = trigger
        how = f"Wait. Buy only if price goes above ₹{tick(e):,.2f}."
    elif hz == "today" and is_early() and intr:
        e = max(intr["hi"], intr["orb_h"]) * 1.001
        how = (f"Wait until 9:45 AM. Then buy only if price goes above ₹{tick(e):,.2f} and stays above VWAP "
               f"₹{tick(intr['vwap']):,.2f}. Exit by 3:15 PM.")
    elif hz == "today":
        if live and intr:
            e = intr["ltp"]
            how = f"Buy near ₹{tick(e):,.2f} while price stays above VWAP ₹{tick(intr['vwap']):,.2f}. Exit everything by 3:15 PM."
        else:
            e = (intr["hi"] if intr else f["h"]) * 1.001
            how = f"Next session: buy only if price crosses ₹{tick(e):,.2f} after 9:30 AM. If not, skip. Exit by 3:15 PM."
    elif hz == "week":
        e = f["c"]
        how = f"Buy near ₹{tick(e):,.2f}. Hold up to 5 trading days. Exit if a day closes below the stop."
    else:
        e = f["c"]
        how = f"Buy near ₹{tick(e):,.2f}. Hold up to 4 weeks. Exit if a week closes below the stop."
    # stop: below recent swing low, kept within a sensible band of the normal move (ATR)
    if hz == "today":
        stop = e - m["sl"] * a
        if live and intr and intr["vwap"] < e:
            stop = max(stop, intr["vwap"] * .997)
        stop = min(stop, e - .35 * a)
    elif hz == "week":
        stop = min(max(f["lo10"] * .998, e - 1.5 * a), e - .9 * a)
    else:
        stop = min(max(f["lo20"] * .995, f["e50"] * .98, e - 2.6 * a), e - 1.8 * a)
    t1, t2 = e + m["t1"] * a, e + m["t2"] * a
    # targets cannot jump over the next resistance
    above = [r for r in f["res"] if r > e * 1.004]
    r1 = above[0] if above else None
    blocked = False
    if r1 and r1 < t1:
        t1 = r1 * .995
    if r1 and r1 < t2:
        nxt = above[1] if len(above) > 1 else None
        t2 = max(t1 * 1.004, min(t2, (nxt * .995) if nxt else r1 * .998))
    if t1 - e < .8 * (e - stop):
        blocked = True
        how = f"Resistance at ₹{tick(r1):,.2f} is too close. Buy only after price closes above it."
    risk = e - stop
    p1 = hit_prob(f, (t1 - e) / e, risk / e, m["n"])
    p2 = hit_prob(f, (t2 - e) / e, risk / e, m["n"])
    return dict(e=tick(e), t1=tick(t1), t2=tick(t2), sl=tick(stop), rp=risk / e * 100, g1=(t1 / e - 1) * 100,
                g2=(t2 / e - 1) * 100, how=how, p1=p1, p2=p2, r1=r1, blocked=blocked)


def market_mood(cues):
    def idx(v, w):
        if v is None:
            return 0
        return w if v > .5 else -w if v < -.5 else w / 2 if v > 0 else -w / 2 if v < 0 else 0
    m = idx(cues.get("US Dow"), 2) + idx(cues.get("US Nasdaq"), 2) + idx(cues.get("Japan"), 1.5) + idx(cues.get("Hong Kong"), 1.5) + idx(cues.get("Nifty"), 4)
    cr, fx, vx = cues.get("Crude"), cues.get("Dollar-Rupee"), cues.get("India VIX")
    m += 0 if cr is None else -1.5 if cr > 1 else 1.5 if cr < -1 else 0
    m += 0 if fx is None else -1 if fx > .2 else 1 if fx < -.2 else 0
    m += 0 if vx is None else -2 if vx > 18 else -1 if vx > 15 else 1 if vx < 13 else 0
    return clamp(m, -15, 15)


def phase():
    n = now_ist()
    m = n.hour * 60 + n.minute
    if n.weekday() >= 5:
        return "Market closed (weekend)", 1.0, False
    if m < 555:
        return "Before market opens", 1.0, False
    if m < 585:
        return "First 30 minutes. No new buys until 9:45 AM", .7, True
    if m < 870:
        return "Market open", 1.0, True
    if m < 930:
        return "Last hour of market", .9, True
    return "Market closed for today", 1.0, False


def is_early():
    """True from 9:15 to 9:45 AM on a weekday: prices are too jumpy for new buys."""
    n = now_ist()
    m = n.hour * 60 + n.minute
    return n.weekday() < 5 and 555 <= m < 585


# ======================= UI =======================
import altair as alt
from concurrent.futures import ThreadPoolExecutor

SHORT = dict(hist="Past hit rate", fund="Company health", inst="FII-DII flow", mkt="Nifty trend", deliv="Delivery", deal="Big deal",
             liq="Liquidity", vol="Volume", trend="Moving average trend", e200="Above 200 day average", macd="MACD", rsi="RSI",
             adx="ADX", st="Supertrend", brk="Breakout", sq="Squeeze", obv="OBV", rs="Beats Nifty", w52="Near 52 week high",
             stretch="Not stretched", cd="Daily candle", cw="Weekly candle", hhhl="Higher highs", acc="Accumulation", news="News",
             sector="Sector", mood="World markets", vwap="VWAP", orb="Opening breakout", e5="5 minute trend", ivol="Volume today",
             rng="Day range", gap="Gap", flow="Buy money", c5="5 minute candle")
PERIODS = {"Today": ("today", "#ff9500", "Same day, exit by 3:15 PM"), "1 Week": ("week", "#007aff", "Hold up to 5 trading days"),
           "1 Month": ("month", "#af52de", "Hold up to 4 weeks"), "Common": ("common", "#34c759", "Strong in more than one period")}
HZ_NAME = {"today": "Today", "week": "1 Week", "month": "1 Month"}
HZ_COL = {"today": "#ff9500", "week": "#007aff", "month": "#af52de"}

st.markdown("""
<style>
:root{--bg:#f2f2f7;--card:#ffffff;--ink:#1c1c1e;--sub:#8e8e93;--soft:#f5f5f8;--g:#34c759;--r:#ff3b30;--o:#ff9500;--b:#007aff}
html,body,.stApp{background:var(--bg) !important;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,sans-serif}
#MainMenu,footer{visibility:hidden}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:720px}
.big{font:800 34px -apple-system,BlinkMacSystemFont,system-ui;letter-spacing:-.6px;margin:0;color:var(--ink)}
.date{color:var(--sub);font:600 13px system-ui;text-transform:uppercase;letter-spacing:.3px;margin:2px 0 6px}
.card{background:var(--card);border-radius:18px;padding:14px 16px;margin:10px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.ct{display:flex;align-items:center;justify-content:space-between;gap:8px;font:600 16px system-ui}
.ct .m{color:var(--sub);font:500 13px system-ui}
.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px 10px;margin-top:10px}
.tile span{display:block;color:var(--sub);font:500 12px system-ui}
.tile b{display:block;font:700 19px system-ui;color:var(--ink)}
.tile i{font-style:normal;font:600 12px system-ui}
.g{color:var(--g) !important}.r{color:var(--r) !important}.o{color:var(--o) !important}.s{color:var(--sub) !important}
.note{background:#fff4e5;color:#8a5300;border-radius:14px;padding:10px 14px;font:500 14px system-ui;margin:8px 0}
.sh{display:flex;align-items:center;gap:8px;color:#e8590c;font:700 18px system-ui;margin:18px 2px 6px}
.shs{color:var(--sub);font:500 13px system-ui;margin:-2px 2px 8px}
div[data-testid="stSelectbox"] div[data-baseweb="select"]>div{border:2.5px solid #e8590c !important;border-radius:14px !important;background:#fff !important;min-height:52px;box-shadow:0 2px 10px rgba(232,89,12,.15)}
.sec{display:flex;align-items:center;justify-content:space-between;margin:18px 2px 4px}
.sec .t{font:700 20px system-ui}.sec .m{color:var(--sub);font:500 13px system-ui}
div[class*="st-key-pk_"],div[class*="st-key-sr_"]{background:#fff;border-radius:18px;padding:14px 16px 10px;box-shadow:0 1px 2px rgba(0,0,0,.04);gap:6px}
div[class*="st-key-pk_"] .stButton button,div[class*="st-key-sr_"] .stButton button{width:100%;border:0;border-radius:12px;background:var(--soft);color:var(--b);font-weight:600;min-height:40px}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
.sym{font:700 20px system-ui;color:var(--ink)}
.nm{color:var(--sub);font:500 13px system-ui;margin-top:1px}
.px{font:700 20px system-ui;text-align:right;white-space:nowrap;color:var(--ink)}
.pc{font:600 13px system-ui;text-align:right;white-space:nowrap}
.badge{display:inline-block;font:700 12px system-ui;border-radius:8px;padding:2px 7px;margin-left:6px;vertical-align:3px;color:#fff}
.lv4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:12px}
@media(max-width:480px){.lv4{grid-template-columns:repeat(2,minmax(0,1fr))}}
.lv4 div{background:var(--soft);border-radius:12px;padding:8px 10px}
.lv4 span{display:block;color:var(--sub);font:600 11.5px system-ui}
.lv4 b{display:block;font:700 17px system-ui;color:var(--ink);white-space:nowrap}
.lv4 i{font-style:normal;font:700 12.5px system-ui}
.foot{display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px 10px;color:var(--sub);font:500 13px system-ui;margin-top:10px}
.foot b{color:var(--ink)}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.chips span{font:600 11.5px system-ui;border-radius:8px;padding:3px 8px;background:#e8f8ed;color:#1f7a3a}
.pill{display:inline-block;font:700 12px system-ui;border-radius:999px;padding:3px 10px;color:#fff;margin-left:6px}
.BUY{background:var(--g)}.WAIT{background:var(--o)}.AVOID{background:var(--r)}
.hzb{border-top:1px solid #ececf0;margin-top:12px;padding-top:10px}
.hzt{display:flex;justify-content:space-between;align-items:center;font:700 16px system-ui}
.hzt .m{color:var(--sub);font:500 12.5px system-ui}
.why{color:#3a3a3c;font:500 13px system-ui;margin-top:8px}
.none{background:#fff;border-radius:18px;padding:16px;color:var(--sub);font:500 14px system-ui;margin:8px 0}
.rg{display:flex;justify-content:space-between;font:600 13px system-ui;color:var(--sub);margin:6px 2px -2px}
.ck{display:flex;justify-content:space-between;gap:8px;font:500 13px system-ui;padding:5px 0;border-bottom:1px solid #f0f0f3;color:#3a3a3c}
.ck b{white-space:nowrap}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;background:#c7c7cc}
.dot.g{background:var(--g)}.dot.r{background:var(--r)}
div[data-testid="stExpander"]{background:#fff;border-radius:14px;border:0}
</style>
""", unsafe_allow_html=True)


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rs(x):
    return f"{x:,.2f}"


def pc(x):
    return f"{x:+.1f}%"


def markers(chk, hz):
    rows = sorted(((v * W[hz][k], k, v, note) for k, (v, note) in chk.items() if W[hz].get(k, 0) > 0), reverse=True)
    return rows, [r for r in rows if r[2] > .2]


def done_bars(d, drop_today):
    """While the market is open, today's daily candle is not finished. Read only finished candles."""
    if d is not None and drop_today and len(d) and d.index[-1].date() == now_ist().date():
        return d.iloc[:-1]
    return d


@st.cache_data(ttl=600, show_spinner=False)
def scan(uni, drop_today):
    universe, src = get_universe(uni)
    parts = get_daily(tuple(s + ".NS" for s, _ in universe))
    cues, nifty = get_cues()
    nifty = done_bars(nifty, drop_today)
    feats = []
    for s_, sec in universe:
        d = None
        for part in parts:
            d = sub(part, s_ + ".NS")
            if d is not None:
                break
        if d is None:
            continue
        try:
            f = daily_features(s_, sec, done_bars(d, drop_today), nifty)
            if f:
                feats.append(f)
        except Exception:
            continue
    return feats, cues, nifty, src


def fetch_funds(syms):
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for s_, r in zip(syms, ex.map(get_fund, syms)):
            if r:
                out[s_] = r
    return out


def ok_plan(p, hz):
    """A pick must have room to its target and a fair past chance of reaching it."""
    return not p["blocked"] and p["p1"] >= (.3 if hz == "today" else .35)


def verdicts(f, intr, ctx, sec_df, live_mkt, need):
    """BUY / WAIT / AVOID for all 3 periods, always with prices."""
    live_now = live_mkt and intr is not None and intr["day"] == now_ist().date()
    price = intr["ltp"] if intr else f["c"]
    a = f["atr"]
    pf = dict(f, c=price) if live_now else f
    res = {}
    for hz in ("today", "week", "month"):
        ctx["sector_rank"] = (sec_df["r1"] if hz == "today" else sec_df["r5"]).rank(pct=True).to_dict()
        sc, chk, flags = score(f, hz, ctx, intr if (hz == "today" and live_now) else None)
        rows, hit = markers(chk, hz)
        hard = any(w.startswith(("Bad news", "Low liquidity", "Below 200", "Company is making", "Serious news", "Profit fell")) for w in flags)
        miss = [SHORT[k] for _, k, v, _ in rows if v < -.05][:3]
        pb = plan(pf, hz, intr, live_now)
        if not flags and sc >= need and not pb["blocked"] and pb["p1"] < (.3 if hz == "today" else .35):
            flags = flags + [f"Target 1 was reached only {pb['p1'] * 100:.0f}% of the time before"]
        down = None
        if not flags and sc >= need and not pb["blocked"]:
            vd, p, why = "BUY", pb, "Strong on: " + ", ".join(SHORT[k] for _, k, _, _ in hit[:4])
        elif not flags and sc >= need and pb["blocked"]:
            vd, p, why = "WAIT", plan(pf, hz, intr, live_now, trigger=pb["r1"] * 1.003), f"Price is just under a resistance at ₹{rs(pb['r1'])}."
        elif sc >= 50 and not hard:
            base = (intr["hi"] if intr else f["h"]) if hz == "today" else max(f["hi20"], f["h"]) if hz == "week" else max(f["hi55"], f["h"])
            base = min(base, price + {"today": .6, "week": 1.0, "month": 1.5}[hz] * a)
            base = max(base, price)
            vd, p = "WAIT", plan(pf, hz, intr, live_now, trigger=base * 1.002)
            why = "Holding back because: " + (flags[0] if flags else ", ".join(miss) or "score is not high enough")
        else:
            vd, p = "AVOID", None
            why = "Weak because: " + (flags[0] if flags else ", ".join(miss) or "most markers")
            m = {"today": .75, "week": 1.5, "month": 3}[hz]
            sup = f["l"] if hz == "today" else f["lo10"] if hz == "week" else f["lo20"]
            fall = tick(min(price - m * a, sup))
            hold = tick(price - {"today": .5, "week": 1.0, "month": 1.8}[hz] * a)
            down = dict(fall=fall, fp=(fall / price - 1) * 100, hold=hold, hp=(hold / price - 1) * 100)
        if sc >= 60 and not hard:
            out = ("Likely to go up", "g")
        elif sc >= 45 and not hard:
            out = ("Likely to stay flat", "o")
        else:
            out = ("Likely to go down", "r")
        res[hz] = dict(vd=vd, sc=sc, p=p, why=why, hit=len(hit), n=len(rows), chk=chk, flags=flags, out=out, down=down, price=price)
    return res


def levels(p, price):
    gap = (p["e"] / price - 1) * 100
    buy_note = "at today's price" if abs(gap) < .15 else f"{pc(gap)} from now"
    return (f'<div class="lv4">'
            f'<div><span>Buy at</span><b>₹{rs(p["e"])}</b><i class="s">{buy_note}</i></div>'
            f'<div><span>Target 1</span><b class="g">₹{rs(p["t1"])}</b><i class="g">{pc(p["g1"])}</i></div>'
            f'<div><span>Target 2</span><b class="g">₹{rs(p["t2"])}</b><i class="g">{pc(p["g2"])}</i></div>'
            f'<div><span>Stop loss</span><b class="r">₹{rs(p["sl"])}</b><i class="r">-{p["rp"]:.1f}%</i></div></div>')


def down_levels(d):
    return (f'<div class="lv4">'
            f'<div><span>May fall to</span><b class="r">₹{rs(d["fall"])}</b><i class="r">{pc(d["fp"])}</i></div>'
            f'<div><span>If you hold, sell below</span><b>₹{rs(d["hold"])}</b><i class="s">{pc(d["hp"])}</i></div></div>')


def chart_data(f, intr_df, tf):
    if tf in ("1D", "1W") and intr_df is not None:
        d = intr_df.copy()
        d.index = d.index.tz_convert(IST) if d.index.tz is not None else d.index.tz_localize(IST)
        if tf == "1D":
            d = d[d.index.date == d.index.date[-1]]
        else:
            d = d.resample("15min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        t = d.index.tz_localize(None)
    else:
        d = f["_daily"].tail({"1M": 22, "6M": 126, "1Y": 250}.get(tf, 22))
        t = d.index.tz_localize(None) if d.index.tz is not None else d.index
    df = pd.DataFrame({"t": t, "o": d["Open"].values, "h": d["High"].values, "l": d["Low"].values, "c": d["Close"].values})
    df["i"] = range(len(df))
    df["ts"] = df["t"].dt.strftime("%d %b %I:%M %p" if tf in ("1D", "1W") else "%d %b %Y")
    return df


def chart(f, intr_df, tf, p, style, height=260):
    df = chart_data(f, intr_df, tf)
    if df.empty:
        st.caption("No chart data for this range.")
        return
    up = df["c"].iloc[-1] >= df["c"].iloc[0]
    col = "#34c759" if up else "#ff3b30"
    chg = (df["c"].iloc[-1] / df["c"].iloc[0] - 1) * 100
    st.markdown(f'<div class="rg"><span>{df["ts"].iloc[0]} to {df["ts"].iloc[-1]}</span>'
                f'<span class="{"g" if up else "r"}">{pc(chg)} in {tf}</span></div>', unsafe_allow_html=True)
    lv = pd.DataFrame([("Buy", p["e"], "#007aff"), ("Stop", p["sl"], "#ff3b30"), ("T1", p["t1"], "#34c759"), ("T2", p["t2"], "#34c759")],
                      columns=["n", "y", "col"]) if p else pd.DataFrame({"n": [], "y": [], "col": []})
    lo = min(df["l"].min() if style == "Candles" else df["c"].min(), lv["y"].min() if len(lv) else 1e18)
    hi = max(df["h"].max() if style == "Candles" else df["c"].max(), lv["y"].max() if len(lv) else -1e18)
    pad = (hi - lo) * .06 or hi * .01
    dom = [lo - pad, hi + pad]
    sc_ = alt.Scale(domain=dom, zero=False, nice=False)
    x = alt.X("i:Q", axis=None, scale=alt.Scale(domain=[0, max(len(df) - 1, 1)], nice=False))
    yax = alt.Axis(title=None, orient="right", labelColor="#8e8e93", gridColor="#f0f0f3", domain=False, ticks=False, tickCount=5, labelFontSize=11)
    y = alt.Y("c:Q", scale=sc_, axis=yax)
    base = alt.Chart(df)
    layers = []
    if style == "Candles":
        ycol = alt.condition("datum.c >= datum.o", alt.value("#34c759"), alt.value("#ff3b30"))
        layers.append(base.mark_rule(strokeWidth=1).encode(x=x, y=alt.Y("l:Q", scale=sc_, axis=yax), y2="h:Q", color=ycol))
        bw = max(2, min(9, int(300 / max(len(df), 1))))
        layers.append(base.mark_bar(size=bw).encode(x=x, y=alt.Y("o:Q", scale=sc_), y2="c:Q", color=ycol))
    else:
        grad = alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                            stops=[alt.GradientStop(color="rgba(255,255,255,0)", offset=0), alt.GradientStop(color=col, offset=1)])
        layers.append(base.mark_area(color=grad, opacity=.22, interpolate="monotone", clip=True).encode(x=x, y=y, y2=alt.datum(dom[0])))
        layers.append(base.mark_line(color=col, strokeWidth=2.4, interpolate="monotone").encode(x=x, y=y))
    if len(lv):
        layers.append(alt.Chart(lv).mark_rule(strokeDash=[5, 4], strokeWidth=1.2).encode(y="y:Q", color=alt.Color("col:N", scale=None)))
        layers.append(alt.Chart(lv).mark_text(align="left", dx=3, dy=-7, fontSize=11, fontWeight="bold").encode(
            y="y:Q", x=alt.value(2), text="n:N", color=alt.Color("col:N", scale=None)))
    hover = alt.selection_point(nearest=True, on="mouseover", fields=["i"], empty=False, clear="mouseout")
    layers.append(base.mark_rule(color="#c7c7cc", strokeWidth=1).encode(
        x=x, opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("ts:N", title="Time"), alt.Tooltip("o:Q", title="Open", format=",.2f"), alt.Tooltip("h:Q", title="High", format=",.2f"),
                 alt.Tooltip("l:Q", title="Low", format=",.2f"), alt.Tooltip("c:Q", title="Close", format=",.2f")]).add_params(hover))
    layers.append(base.mark_circle(size=60, color=col, stroke="#fff", strokeWidth=2).encode(
        x=x, y=y, opacity=alt.condition(hover, alt.value(1), alt.value(0))))
    st.altair_chart(alt.layer(*layers).properties(height=height).configure_view(strokeWidth=0).configure(background="transparent"),
                    use_container_width=True)


def chart_block(key, f, intr_df, p):
    c1, c2 = st.columns([3, 2])
    tf = c1.radio("Range", ["1D", "1W", "1M", "6M", "1Y"], index=2, horizontal=True, label_visibility="collapsed", key=f"tf_{key}")
    style = c2.radio("Type", ["Line", "Candles"], horizontal=True, label_visibility="collapsed", key=f"cs_{key}")
    chart(f, intr_df, tf, p, style)


def all_markers(res):
    for hz in ("today", "week", "month"):
        rows, _ = markers(res[hz]["chk"], hz)
        st.markdown(f"**{HZ_NAME[hz]}**")
        st.markdown("".join(
            f'<div class="ck"><span><span class="dot {"g" if v > .2 else "r" if v < -.05 else ""}"></span>{esc(NAMES[k])}: {esc(note)}</span>'
            f'<b class="{"g" if c > .05 else "r" if c < -.05 else "s"}">{c:+.1f}</b></div>' for c, k, v, note in rows)
            + "".join(f'<div class="ck"><span class="r">Red flag: {esc(x)}</span></div>' for x in res[hz]["flags"]), unsafe_allow_html=True)


# ---------- settings (shown at the bottom, read first) ----------
main = st.container()
with st.expander("Settings"):
    uni = st.selectbox("Shares to scan", ["Nifty 100", "Nifty 200", "Nifty 500"], index=2,
                       help="Nifty 500 includes mid and small companies, so more lower priced shares")
    max_price = {"Any price": 1e12, "Under ₹2,000": 2000, "Under ₹1,000": 1000, "Under ₹500": 500, "Under ₹200": 200}[
        st.selectbox("Share price", ["Any price", "Under ₹2,000", "Under ₹1,000", "Under ₹500", "Under ₹200"], index=0)]
    fii_manual = st.number_input("FII net buying in crore (only if NSE data does not load)", value=0.0, step=100.0)
    min_score = int(round(st.slider("Show only shares scoring at least (out of 10)", 5.5, 8.0, 6.2, 0.1) * 10))

st.session_state.setdefault("open", None)
all_syms = get_all_symbols()
names = dict(all_syms)
labels = [f"{a_} · {b_}" for a_, b_ in all_syms]
lab2sym = {l: a_ for l, (a_, _) in zip(labels, all_syms)}


def toggle(key):
    st.session_state.open = None if st.session_state.get("open") == key else key


@st.fragment(run_every=600)
def live():
    ph, conf, live_mkt = phase()
    st.markdown(f'<div class="date">{now_ist():%A, %d %b} · {ph}</div><h1 class="big">Markets</h1>', unsafe_allow_html=True)
    with st.spinner("Scanning shares. The first load can take 2 to 3 minutes."):
        feats, cues, nifty, src = scan(uni, live_mkt)
        news, news_ok = get_news()
        events = get_events()
        flows = get_fii_dii()
        deliv, _ = get_delivery()
        ban = get_ban()
        deals = get_deals()
    if not feats:
        st.error("Could not get prices right now. It will try again in 10 minutes.")
        return
    fmap = {f["s"]: f for f in feats}
    sec_df = pd.DataFrame([(f["sec"], f["r5"], f["r1"]) for f in feats], columns=["sec", "r5", "r1"]).groupby("sec").mean()
    mood = market_mood(cues)
    if flows is None and fii_manual:
        flows = (fii_manual, 0.0, "Settings")
    mkt_trend = 0
    if nifty is not None and len(nifty) > 60:
        e20n, e50n = ema(nifty, 20).iloc[-1], ema(nifty, 50).iloc[-1]
        mkt_trend = 1 if nifty.iloc[-1] > e20n > e50n else -1 if nifty.iloc[-1] < e20n < e50n else 0
    ctx = dict(news=news, mood=mood, conf=conf, events=events, flows=flows, mkt_trend=mkt_trend, deliv=deliv, ban=ban, deals=deals, fund={})
    heavy_sell = bool(flows and flows[0] < -3000 and flows[0] + (flows[1] or 0) < 0)
    need = min_score + (6 if mood <= -6 else 0) + (5 if heavy_sell else 0) + (4 if mkt_trend < 0 else 0)

    # ---- market card ----
    nc = cues.get("Nifty")
    vix = cues.get("India VIX")
    fii, dii = (flows[0], flows[1] or 0) if flows else (None, None)
    tile = lambda label, val, sub_, cls: f'<div class="tile"><span>{label}</span><b class="{cls}">{val}</b><i class="s">{sub_}</i></div>'
    tiles = [tile("Nifty", "-" if nc is None else pc(nc), "last close", "g" if (nc or 0) >= 0 else "r"),
             tile("Market trend", {1: "Up", -1: "Down", 0: "Flat"}[mkt_trend], "Nifty 20 and 50 day", {1: "g", -1: "r", 0: "o"}[mkt_trend]),
             tile("World markets", "Good" if mood > 5 else "Weak" if mood < -5 else "Mixed", "US, Asia, crude", "g" if mood > 5 else "r" if mood < -5 else "o"),
             tile("FII", "-" if fii is None else f"{'+' if fii >= 0 else '-'}₹{abs(fii):,.0f} cr", "foreign funds", "g" if (fii or 0) >= 0 else "r"),
             tile("DII", "-" if dii is None else f"{'+' if dii >= 0 else '-'}₹{abs(dii):,.0f} cr", "Indian funds", "g" if (dii or 0) >= 0 else "r"),
             tile("India VIX", "-" if vix is None else f"{vix:.1f}", "fear level", "r" if (vix or 0) > 16 else "g")]
    st.markdown(f'<div class="card"><div class="ct"><span style="color:#007aff">Market today</span>'
                f'<span class="m">Updated {now_ist():%I:%M %p}</span></div><div class="tiles">{"".join(tiles)}</div></div>', unsafe_allow_html=True)
    if heavy_sell:
        st.markdown(f'<div class="note">Foreign funds sold ₹{-fii:,.0f} crore and Indian funds did not cover it. Only stronger setups are shown. Use smaller quantity.</div>', unsafe_allow_html=True)
    elif mkt_trend < 0 or mood <= -6:
        st.markdown('<div class="note">The market is weak. Only stronger setups are shown.</div>', unsafe_allow_html=True)

    # ---- stage 1 and 2: rank shares ----
    pool = [f for f in feats if f["liq"] >= 20 and f["c"] <= max_price]
    ctx["sector_rank"] = sec_df["r5"].rank(pct=True).to_dict()
    pre = sorted(((score(f, "week", ctx)[0], f) for f in pool), key=lambda x: -x[0])
    intr_pool = [f for _, f in pre[:40]]
    d5 = get_intraday(tuple(f["s"] + ".NS" for f in intr_pool)) if intr_pool else None
    intr_map, d5_map = {}, {}
    for f in intr_pool:
        x = sub(d5, f["s"] + ".NS")
        if x is not None:
            d5_map[f["s"]] = x
            try:
                intr_map[f["s"]] = intraday_features(x)
            except Exception:
                pass
    live_now = live_mkt and any(v and v["day"] == now_ist().date() for v in intr_map.values())
    ctx["live_px"] = {k: (v["ltp"], v["prev"]) for k, v in intr_map.items() if v and v["day"] == now_ist().date()}

    def livef(f):
        lp = ctx["live_px"].get(f["s"])
        return dict(f, c=lp[0]) if lp else f

    def rank(hz, cands):
        ctx["sector_rank"] = (sec_df["r1"] if hz == "today" else sec_df["r5"]).rank(pct=True).to_dict()
        out = []
        for f in cands:
            intr = intr_map.get(f["s"])
            sc, chk, flags = score(f, hz, ctx, intr if (hz == "today" and live_now) else None)
            if not flags and sc >= need - 6:
                out.append((sc, f, chk))
        return sorted(out, key=lambda x: -x[0])

    first = {hz: rank(hz, intr_pool if hz == "today" else pool)[:25] for hz in ("today", "week", "month")}
    cand = sorted({f["s"] for lst in first.values() for _, f, _ in lst})
    extra = [s_ for s_ in cand if s_ not in intr_map]
    if extra:
        d5x = get_intraday(tuple(s_ + ".NS" for s_ in extra))
        for s_ in extra:
            x = sub(d5x, s_ + ".NS")
            if x is not None:
                d5_map[s_] = x
                try:
                    v = intraday_features(x)
                    intr_map[s_] = v
                    if v["day"] == now_ist().date():
                        ctx["live_px"][s_] = (v["ltp"], v["prev"])
                except Exception:
                    pass
    ctx["fund"] = fetch_funds(cand)
    with ThreadPoolExecutor(max_workers=4) as ex:
        ctx["serious"] = {s_: r for s_, r in zip(cand, ex.map(get_stock_news, cand)) if r}
    final = {}
    for hz in ("today", "week", "month"):
        final[hz] = [(sc, f, chk) for sc, f, chk in rank(hz, [f for _, f, _ in first[hz]]) if sc >= need
                     and ok_plan(plan(livef(f), hz, intr_map.get(f["s"]), live_now and hz == "today"), hz)]

    def full(sym):
        """Everything needed for a share's detail: features, intraday, verdicts."""
        f = fmap.get(sym)
        dd = get_daily((sym + ".NS",))
        dsym = sub(dd[0], sym + ".NS") if dd else None
        if f is None and dsym is not None:
            f = daily_features(sym, "Other", done_bars(dsym, live_mkt), nifty)
        if f is None:
            return None
        f = dict(f, _daily=dsym)
        if sym not in ctx["fund"]:
            ctx["fund"][sym] = get_fund(sym)
        if sym not in ctx["serious"]:
            sn = get_stock_news(sym)
            if sn:
                ctx["serious"][sym] = sn
        x = d5_map.get(sym)
        if x is None:
            x = sub(get_intraday((sym + ".NS",)), sym + ".NS")
        intr = None
        if x is not None:
            try:
                intr = intraday_features(x)
            except Exception:
                intr = None
        if intr and intr["day"] == now_ist().date():
            ctx["live_px"][sym] = (intr["ltp"], intr["prev"])
        return f, x, intr, verdicts(f, intr, ctx, sec_df, live_mkt, need)

    # ---- search ----
    st.markdown('<div class="sh"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#e8590c" stroke-width="2.6" '
                'stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></svg>Search a share</div>'
                '<div class="shs">Type a few letters of the name. Pick it from the list to see its targets and chart.</div>', unsafe_allow_html=True)
    kw = dict(index=None, placeholder="Type here, for example Tata, PG, KPIT", label_visibility="collapsed", key="q")
    try:
        picked = st.selectbox("Search a share", labels, accept_new_options=True, **kw)
    except TypeError:
        picked = st.selectbox("Search a share", labels, **kw)
    ssym = None
    if picked:
        if picked in lab2sym:
            ssym = lab2sym[picked]
        elif picked.strip().upper().replace(".NS", "") in names:
            ssym = picked.strip().upper().replace(".NS", "")
        else:
            import difflib
            pool_ = {a_: l for l, a_ in lab2sym.items()}
            typed = picked.strip().upper()
            near = difflib.get_close_matches(typed.replace(" ", ""), list(pool_), n=4, cutoff=.5)
            by_name = {b_.upper(): a_ for a_, b_ in all_syms}
            near += [by_name[n] for n in difflib.get_close_matches(typed, list(by_name), n=4, cutoff=.45) if by_name[n] not in near]
            if near:
                st.markdown(f'<div class="note">No share called "{esc(picked)}". Did you mean one of these?</div>', unsafe_allow_html=True)
                cols = st.columns(2)
                for i, a_ in enumerate(near[:6]):
                    cols[i % 2].button(pool_[a_][:40], key=f"dym{i}", use_container_width=True,
                                       on_click=lambda l=pool_[a_]: st.session_state.update(q=l))
            else:
                st.markdown(f'<div class="note">No share found for "{esc(picked)}". Try the company name.</div>', unsafe_allow_html=True)
    if ssym:
        data = full(ssym)
        if data is None:
            st.markdown(f'<div class="note">Not enough price history for {esc(ssym)}.</div>', unsafe_allow_html=True)
        else:
            f, x, intr, res = data
            price, prev = (intr["ltp"], intr["prev"]) if intr else (f["c"], f["prev"])
            chg = (price / prev - 1) * 100
            with st.container(key=f"sr_{ssym}"):
                blocks = ""
                for hz in ("today", "week", "month"):
                    r = res[hz]
                    lv = levels(r["p"], price) if r["p"] else down_levels(r["down"])
                    blocks += (f'<div class="hzb"><div class="hzt"><span style="color:{HZ_COL[hz]}">{HZ_NAME[hz]}<span class="pill {r["vd"]}">{r["vd"]}</span></span>'
                               f'<span class="m">{r["sc"] / 10:.1f}/10 · {r["hit"]} of {r["n"]} markers</span></div>'
                               f'<div class="why"><b class="{r["out"][1]}">{r["out"][0]}.</b> {esc(r["why"])}</div>{lv}'
                               + (f'<div class="foot"><span>Chance to reach Target 1: <b>{r["p"]["p1"] * 100:.0f}%</b></span>'
                                  f'<span>{PERIODS[HZ_NAME[hz]][2]}</span></div>' if r["p"] else "") + '</div>')
                st.markdown(f'<div class="top"><div><div class="sym">{esc(ssym)}</div><div class="nm">{esc(names.get(ssym, ssym))} · {esc(f["sec"])}</div></div>'
                            f'<div><div class="px">₹{rs(price)}</div><div class="pc {"g" if chg >= 0 else "r"}">{price - prev:+,.2f} ({pc(chg)}) today</div></div></div>'
                            f'{blocks}', unsafe_allow_html=True)
                best = next((res[h]["p"] for h in ("week", "today", "month") if res[h]["p"]), None)
                chart_block(f"s_{ssym}", f, x, best)
                with st.expander("See all markers"):
                    all_markers(res)

    # ---- picks ----
    try:
        per = st.segmented_control("Period", list(PERIODS), default="Today", key="per", label_visibility="collapsed")
    except AttributeError:
        per = st.radio("Period", list(PERIODS), horizontal=True, key="per", label_visibility="collapsed")
    per = per or "Today"
    hz_key, col, sub_ = PERIODS[per]
    st.markdown(f'<div class="sec"><span class="t" style="color:{col}">{"Top 5 for " + per if per != "Common" else "Common picks"}</span>'
                f'<span class="m">{sub_}</span></div>', unsafe_allow_html=True)
    if hz_key == "today":
        if is_early():
            st.markdown('<div class="note">It is the first 30 minutes of trading. Do not buy yet. Each pick shows the price it must cross after 9:45 AM.</div>', unsafe_allow_html=True)
        else:
            st.caption("Live picks, refreshed every 10 minutes." if live_now else "Market closed. These picks are for the next session.")

    if hz_key == "common":
        tops = {hz: {f["s"]: sc for sc, f, _ in final[hz][:15]} for hz in ("today", "week", "month")}
        cm = {}
        for hz, d in tops.items():
            for s_, sc in d.items():
                cm.setdefault(s_, {})[hz] = sc
        items = sorted(((len(v), sum(v.values()) / len(v), s_, v) for s_, v in cm.items() if len(v) >= 2), reverse=True)[:5]
        cards = [(int(round(avg)), fmap[s_], "week" if "week" in v else "month" if "month" in v else "today",
                  " · ".join(HZ_NAME[h] for h in ("today", "week", "month") if h in v)) for _, avg, s_, v in items]
    else:
        cards = [(sc, f, hz_key, None) for sc, f, _ in final[hz_key][:5]]
    if not cards:
        st.markdown('<div class="none">No share passes every buy check right now. Not buying is also a good decision.</div>', unsafe_allow_html=True)

    for i, (sc, f, hz, tag) in enumerate(cards, 1):
        intr = intr_map.get(f["s"])
        use_live = hz == "today" and live_now
        lp = ctx["live_px"].get(f["s"])
        price, prev = lp if lp else ((intr["ltp"], intr["prev"]) if (intr and hz == "today") else (f["c"], f["prev"]))
        chg = (price / prev - 1) * 100
        p = plan(livef(f), hz, intr, use_live)
        asof = f'Price at {intr["asof"]:%I:%M %p}' if intr else "Last close"
        key = f"{per}_{f['s']}"
        _, hit = markers(score(f, hz, ctx, intr if use_live else None)[1], hz)
        chips = "".join(f"<span>{esc(SHORT[k])}</span>" for _, k, _, _ in hit[:6])
        bcol = "#34c759" if sc >= 70 else "#ff9500"
        with st.container(key=f"pk_{key}"):
            st.markdown(
                f'<div class="top"><div><div class="sym">{i}. {esc(f["s"])}<span class="badge" style="background:{bcol}">{sc / 10:.1f}/10</span></div>'
                f'<div class="nm">{esc(names.get(f["s"], f["s"]))}{" · " + tag if tag else ""}</div></div>'
                f'<div><div class="px">₹{rs(price)}</div><div class="pc {"g" if chg >= 0 else "r"}">{pc(chg)} today</div>'
                f'<div class="nm" style="text-align:right">{asof}</div></div></div>'
                f'{levels(p, price)}'
                f'<div class="foot"><span>Chance to reach Target 1: <b>{p["p1"] * 100:.0f}%</b></span><span>{PERIODS[HZ_NAME[hz]][2] if not tag else "Levels for " + HZ_NAME[hz]}</span></div>'
                f'<div class="why">{esc(p["how"])} Sell half at Target 1, then move the stop loss to your buy price.</div>'
                f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
            is_open = st.session_state.get("open") == key
            st.button("Hide chart" if is_open else "Show chart", key=f"b_{key}", on_click=toggle, args=(key,))
            if is_open:
                data = full(f["s"])
                if data:
                    f2, x2, _, res2 = data
                    chart_block(key, f2, x2, p)
                    with st.expander("See all markers"):
                        all_markers(res2)

    st.caption(f"Share list: {src}. Prices from Yahoo Finance, can be a few minutes late.")


with main:
    live()

with st.expander("How shares are picked"):
    st.markdown("""
1. Only shares with high liquidity: at least ₹20 crore traded every day.
2. Each share gets a score out of 10 from chart trend, momentum, volume, delivery, over 30 candle patterns, strength vs Nifty, sector, news, company health (P/E, ROE, profit growth) and big deals.
3. Shares with a red flag are removed: bearish candle, bad or serious news, results too close, profit falling sharply, F&O ban, or too hot (RSI above 80).
4. Targets come from how far the share normally moves in that time, and never go past the next resistance or 52 week high.
5. Chance to reach Target 1 is how often this share actually reached that target before the stop loss in the past year.
6. Market mood (Nifty trend, FII, DII, world markets) raises or lowers the bar for all shares.
7. No new buys before 9:45 AM. Shares already up more than 2.5% today (5% for 1 month) are not chased.
8. Loss-making companies, shares listed less than 9 months ago, and shares where analysts' average target is below the price are removed.
9. Candles are read only after they finish, never half-formed ones.
""")
st.caption("Screening help only. No method can promise a share will go up. Always use the stop loss. Not financial advice.")
