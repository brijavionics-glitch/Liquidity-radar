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

# ======================= STYLE =======================
st.markdown("""
<style>
#MainMenu, footer, header [data-testid="stToolbar"]{visibility:hidden}
.block-container{padding-top:1rem;padding-bottom:3rem}
html, body{overflow-x:hidden}
.hero{background:radial-gradient(120% 140% at 0% 0%,#0f2a1c 0%,#071018 55%,#05080d 100%);border:1px solid #1b3a2a;color:#e6edf3;border-radius:14px;padding:14px 16px;margin-bottom:10px}
.hero h1{font:800 22px system-ui;margin:0;color:#e6edf3;letter-spacing:.3px}
.hero h1 span{color:#22c55e}
.hero .s{font-size:12.5px;color:#8b9bb0;margin-top:2px}
.pills{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}
.pill{font:600 11px ui-monospace,Menlo,monospace;padding:2px 8px;border-radius:6px;background:#0d1622;border:1px solid #1f2d3d;color:#a9b8c9;white-space:nowrap}
.pill.g{color:#4ade80;border-color:#1f4d33}.pill.r{color:#f87171;border-color:#4d1f1f}.pill.y{color:#fbbf24;border-color:#4d3d1f}
.warn{background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.35);color:#fbbf24;border-radius:10px;padding:8px 10px;font-size:13px;margin:6px 0 10px}
.cd{background:linear-gradient(180deg,#0b1220,#070b12);border:1px solid #1f2d3d;border-radius:14px;padding:11px 12px 10px;margin:8px 0 2px;color:#e6edf3;box-shadow:0 6px 18px rgba(0,0,0,.25)}
.r1{display:flex;align-items:center;gap:8px}
.rk{font:700 11px ui-monospace,Menlo,monospace;color:#22c55e;background:rgba(34,197,94,.12);border-radius:5px;padding:2px 6px}
.sy{font:800 16px system-ui;letter-spacing:.3px;flex:1;min-width:0;overflow-wrap:anywhere;color:#f1f5f9}
.px{font:600 12.5px ui-monospace,Menlo,monospace;text-align:right;white-space:nowrap;color:#e6edf3}
.sc{font:800 15px ui-monospace,Menlo,monospace;border-radius:7px;padding:3px 7px;min-width:38px;text-align:center}
.sc.hi{background:#22c55e;color:#03140a;box-shadow:0 0 10px rgba(34,197,94,.45)}.sc.md{background:#eab308;color:#1a1300}.sc.lo{background:#ef4444;color:#fff}
.r2{display:flex;justify-content:space-between;gap:8px;font-size:11.5px;color:#8b9bb0;margin-top:3px;flex-wrap:wrap}
.lvl{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:5px;margin-top:8px}
.lvl div{background:#0f1a2a;border:1px solid #1a2a3d;border-radius:7px;padding:4px 6px}
.lvl s{display:block;text-decoration:none;font:700 9.5px ui-monospace,Menlo,monospace;color:#7d8ba1;letter-spacing:.6px}
.lvl b{display:block;font:700 clamp(11px,3.2vw,14px) ui-monospace,Menlo,monospace;color:#e6edf3;white-space:nowrap}
.lvl em{font:600 9.5px ui-monospace,Menlo,monospace;font-style:normal;color:#7d8ba1}
.lvl .e b{color:#60a5fa}.lvl .x b{color:#f87171}.lvl .t b{color:#4ade80}
.mk{display:flex;align-items:center;gap:8px;margin-top:8px;font:600 11px ui-monospace,Menlo,monospace;color:#a9b8c9;white-space:nowrap}
.mk .bar{flex:1;height:6px;background:#1a2433;border-radius:3px;overflow:hidden}
.mk .bar i{display:block;height:100%;background:linear-gradient(90deg,#15803d,#4ade80);box-shadow:0 0 8px #22c55e}
.tx{font-size:12px;color:#c3cfdc;margin-top:7px;line-height:1.4}
.ch{display:flex;flex-wrap:wrap;gap:4px;margin-top:7px}
.ch span{font:600 10.5px ui-monospace,Menlo,monospace;padding:2px 6px;border-radius:5px;background:rgba(34,197,94,.1);color:#86efac;border:1px solid rgba(34,197,94,.25)}
.ch span.p{background:rgba(96,165,250,.1);color:#93c5fd;border-color:rgba(96,165,250,.3)}
.ch span.m{background:transparent;color:#64748b;border-color:#1f2d3d}
.ev{margin-top:7px;font-size:11.5px;color:#fbbf24;background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.3);border-radius:7px;padding:4px 8px}
.nw{margin-top:7px;font-size:11.5px;color:#a9b8c9;border-left:2px solid #3b82f6;padding-left:7px}
.hz{border-top:1px dashed #1f2d3d;margin-top:9px;padding-top:8px}
.hzt{display:flex;align-items:center;gap:8px;font:700 12px ui-monospace,Menlo,monospace;color:#a9b8c9}
.vd{font:800 11px ui-monospace,Menlo,monospace;padding:2px 8px;border-radius:5px;letter-spacing:.5px}
.vd.hi{background:#22c55e;color:#03140a}.vd.md{background:#eab308;color:#1a1300}.vd.av{background:#ef4444;color:#fff}
.why{font-size:11.5px;color:#a9b8c9;margin-top:5px}
.ck{display:flex;justify-content:space-between;gap:8px;font-size:12px;padding:3px 0;border-bottom:1px solid rgba(127,127,127,.15)}
.ck b{white-space:nowrap;font-family:ui-monospace,Menlo,monospace}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;background:#64748b}
.dot.g{background:#22c55e}.dot.r{background:#ef4444}
.up{color:#4ade80}.dn{color:#f87171}
.none{color:#8b9bb0;font-size:13px;padding:8px 0}
</style>
""", unsafe_allow_html=True)

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
                   eg=i.get("earningsGrowth"), eps=i.get("trailingEps"), qoq=None, yoy=None)
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
        res=swing_highs(h.tail(250)),
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
        rsi5=float(rsi_s(d5["Close"]).iloc[-1]), c5=candles(d5.tail(20)),
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
    if hz == "month" and fd and fd.get("eps") is not None and fd["eps"] < 0:
        flags.append("Company is making a loss")
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
        return "First 30 minutes, prices are jumpy", .7, True
    if m < 870:
        return "Market open", 1.0, True
    if m < 930:
        return "Last hour of market", .9, True
    return "Market closed for today", 1.0, False


# ======================= UI =======================
import altair as alt
from concurrent.futures import ThreadPoolExecutor

SHORT = dict(hist="Past hit rate", fund="Company health", inst="FII-DII", mkt="Nifty trend", deliv="Delivery", deal="Big deal", liq="Liquidity",
             vol="Volume", trend="MA trend", e200="Above 200DMA", macd="MACD", rsi="RSI", adx="ADX", st="Supertrend",
             brk="Breakout", sq="Squeeze", obv="OBV", rs="Beats Nifty", w52="52W high", stretch="Not stretched",
             cd="Daily candle", cw="Weekly candle", hhhl="HH-HL", acc="Accumulation", news="News", sector="Sector",
             mood="Global", vwap="VWAP", orb="ORB", e5="5m trend", ivol="Vol today", rng="Day range", gap="Gap",
             flow="Buy flow", c5="5m candle")
HZ_LABEL = {"today": "Today", "week": "1 week", "month": "1 month"}
PALETTE = ["#e11d48", "#0ea5e9", "#22c55e", "#14b8a6", "#6366f1", "#f59e0b", "#a855f7", "#ef4444", "#0891b2", "#65a30d"]

st.markdown("""
<style>
.wl-h{font:600 13px system-ui;color:#7d8ba1;margin:16px 0 2px;letter-spacing:.3px}
.wrow{display:flex;align-items:center;gap:11px;padding:10px 6px;border-bottom:1px solid #141c27;border-radius:8px}
.wrow.on{background:#111b2a}
.lg{width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;font:800 11px system-ui;color:#fff;flex:none;letter-spacing:.2px}
.mid{flex:1;min-width:0}
.l1,.l2{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.s1{font:700 17px system-ui;color:#f1f5f9;white-space:nowrap}
.s1 i{font:800 10.5px ui-monospace,Menlo,monospace;font-style:normal;background:#22c55e;color:#03140a;border-radius:4px;padding:1px 5px;margin-left:6px;vertical-align:2px}
.s1 i.m{background:#eab308;color:#1a1300}
.pr{font:600 17px ui-monospace,Menlo,monospace;color:#f1f5f9;white-space:nowrap}
.l2{font-size:13px;color:#7d8ba1;margin-top:1px}
.nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.cg{white-space:nowrap;font-family:ui-monospace,Menlo,monospace}
.l3{display:flex;flex-wrap:wrap;gap:4px 10px;font:600 11.5px ui-monospace,Menlo,monospace;color:#7d8ba1;margin-top:4px}
.l3 b{font-weight:700}.l3 em{font-style:normal;font-size:10.5px;opacity:.85}.l3 .e{color:#60a5fa}.l3 .x{color:#f87171}.l3 .t{color:#4ade80}
.rg{display:flex;justify-content:space-between;align-items:baseline;font:600 12px ui-monospace,Menlo,monospace;color:#7d8ba1;margin:2px 2px -4px}
.lgd{display:flex;flex-wrap:wrap;gap:4px 12px;font:600 11.5px ui-monospace,Menlo,monospace;color:#7d8ba1;margin:2px 2px 6px}
.lgd b{font-weight:700}
.tg{font:700 10px ui-monospace,Menlo,monospace;color:#93c5fd;border:1px solid #1e3a5f;border-radius:4px;padding:0 4px;margin-left:4px}
div[class*="st-key-r_"]{position:relative;gap:0}
div[class*="st-key-r_"] .stButton{position:absolute;inset:0;z-index:3;margin:0}
div[class*="st-key-r_"] .stButton button{width:100%;height:100%;opacity:0;border:0;padding:0;min-height:0}
.cp{text-align:center;padding:6px 0 2px}
.cp .lgb{width:60px;height:60px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font:800 15px system-ui;color:#fff}
.cp .n1{font:700 17px system-ui;color:#f1f5f9;margin-top:6px}
.cp .n2{font:600 12px system-ui;color:#7d8ba1}
.cp .p1{font:800 34px ui-monospace,Menlo,monospace;color:#f1f5f9;margin-top:4px}
.cp .p1 small{font:600 13px system-ui;color:#7d8ba1;margin-left:4px}
.cp .p2{font:700 15px ui-monospace,Menlo,monospace}
.st3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin-top:6px}
.st3 div{font:700 15px ui-monospace,Menlo,monospace;color:#e6edf3}
.st3 span{display:block;font:600 10.5px system-ui;color:#7d8ba1;letter-spacing:.5px}
.vrow{display:flex;justify-content:space-between;align-items:center;gap:8px;font:600 12.5px ui-monospace,Menlo,monospace;color:#a9b8c9;padding:6px 0;border-bottom:1px solid #141c27}
</style>
""", unsafe_allow_html=True)


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rs(x):
    return f"{x:,.2f}"


def logo(sym):
    c = PALETTE[sum(map(ord, sym)) % len(PALETTE)]
    return c, esc(sym.replace("-", "")[:3])


def markers(chk, hz):
    rows = sorted(((v * W[hz][k], k, v, note) for k, (v, note) in chk.items() if W[hz].get(k, 0) > 0), reverse=True)
    return rows, [r for r in rows if r[2] > .2]


def news_factor_safe(sym, ctx):
    return news_factor(sym, ctx["news"])


@st.cache_data(ttl=600, show_spinner=False)
def scan(uni):
    universe, src = get_universe(uni)
    parts = get_daily(tuple(s + ".NS" for s, _ in universe))
    cues, nifty = get_cues()
    feats = []
    for s, sec in universe:
        d = None
        for part in parts:
            d = sub(part, s + ".NS")
            if d is not None:
                break
        if d is None:
            continue
        try:
            f = daily_features(s, sec, d, nifty)
            if f:
                feats.append(f)
        except Exception:
            continue
    return feats, cues, nifty, src


def fetch_funds(syms):
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for s, r in zip(syms, ex.map(get_fund, syms)):
            if r:
                out[s] = r
    return out


def ok_plan(p, hz):
    """A pick must have room to its target and a fair past chance of reaching it."""
    return not p["blocked"] and p["p1"] >= (.3 if hz == "today" else .35)


def verdicts(f, intr, ctx, sec_df, live_mkt, need):
    """BUY / WAIT / AVOID with levels for all 3 periods."""
    live_now = live_mkt and intr is not None and intr["day"] == now_ist().date()
    res = {}
    for hz in ("today", "week", "month"):
        ctx["sector_rank"] = (sec_df["r1"] if hz == "today" else sec_df["r5"]).rank(pct=True).to_dict()
        sc, chk, flags = score(f, hz, ctx, intr if (hz == "today" and live_now) else None)
        rows, hit = markers(chk, hz)
        hard = any(w.startswith(("Bad news", "Low liquidity", "Below 200", "Company is making")) for w in flags)
        miss = [SHORT[k] for _, k, v, _ in rows if v < -.05][:3]
        pb = plan(f, hz, intr, live_now)
        if not flags and sc >= need and not pb["blocked"] and pb["p1"] < (.3 if hz == "today" else .35):
            flags = flags + [f"Target 1 reached only {pb['p1'] * 100:.0f}% of the time before"]
        if not flags and sc >= need and not pb["blocked"]:
            vd, p, why = "BUY", pb, "Strong: " + ", ".join(SHORT[k] for _, k, _, _ in hit[:4])
        elif not flags and sc >= need and pb["blocked"]:
            vd, p, why = "WAIT", plan(f, hz, intr, live_now, trigger=pb["r1"] * 1.003), f"Resistance at ₹{rs(pb['r1'])} blocks the target"
        elif sc >= 50 and not hard:
            base = (intr["hi"] if intr else f["h"]) if hz == "today" else max(f["hi20"], f["h"]) if hz == "week" else max(f["hi55"], f["h"])
            vd, p = "WAIT", plan(f, hz, intr, live_now, trigger=base * 1.002)
            why = "Waiting on: " + (flags[0] if flags else ", ".join(miss) or "higher score")
        else:
            vd, p, why = "AVOID", None, "Weak: " + (flags[0] if flags else ", ".join(miss) or "most markers")
        price = intr["ltp"] if intr else f["c"]
        a = f["atr"]
        if sc >= 60 and not hard:
            out = ("Likely UP", "up", "")
        elif sc >= 45 and not hard:
            out = ("SIDEWAYS", "nt", f"Range ₹{rs(price - {'today': .5, 'week': 1.2, 'month': 2.5}[hz] * a)} to ₹{rs(price + {'today': .5, 'week': 1.2, 'month': 2.5}[hz] * a)}")
        else:
            m = {"today": .75, "week": 1.5, "month": 3}[hz]
            sup = f["l"] if hz == "today" else f["lo10"] if hz == "week" else f["lo20"]
            out = ("Likely DOWN", "dn", f"May fall to ₹{rs(tick(min(price - m * a, sup)))}. If you hold it, keep a stop at ₹{rs(tick(price + .5 * a))} or exit.")
        res[hz] = dict(vd=vd, sc=sc, p=p, why=why, hit=len(hit), n=len(rows), chk=chk, flags=flags, out=out)
    return res


def lvl_line(p):
    return (f'<div class="l3"><span>Buy <b class="e">{rs(p["e"])}</b></span>'
            f'<span>Stop <b class="x">{rs(p["sl"])}</b> <em class="x">-{p["rp"]:.1f}%</em></span>'
            f'<span>T1 <b class="t">{rs(p["t1"])}</b> <em class="t">+{p["g1"]:.1f}%</em></span>'
            f'<span>T2 <b class="t">{rs(p["t2"])}</b> <em class="t">+{p["g2"]:.1f}%</em></span>'
            f'<span>T1 chance <b class="{"t" if p["p1"] >= .5 else "e" if p["p1"] >= .38 else "x"}">{p["p1"] * 100:.0f}%</b></span></div>')


def row(key, sym, name, price, chg, sc, p, sel, tags=""):
    c, ini = logo(sym)
    lv = lvl_line(p) if p else ""
    html = (f'<div class="wrow{" on" if sel else ""}"><div class="lg" style="background:{c}">{ini}</div><div class="mid">'
            f'<div class="l1"><span class="s1">{esc(sym)}<i class="{"" if sc >= 70 else "m"}">{sc / 10:.1f}</i>{tags}</span><span class="pr">{rs(price)}</span></div>'
            f'<div class="l2"><span class="nm">{esc(name)}</span><span class="cg {"up" if chg >= 0 else "dn"}">{price * chg / (100 + chg):+.2f} {chg:+.2f}%</span></div>'
            f'{lv}</div></div>')
    try:
        box = st.container(key=f"r_{key}")
    except TypeError:
        box = st.container()
    with box:
        st.markdown(html, unsafe_allow_html=True)
        st.button(f"Open {sym}", key=f"b_{key}",
                  on_click=lambda: st.session_state.update(sel=None if st.session_state.get("sel_key") == key else sym,
                                                           sel_key=None if st.session_state.get("sel_key") == key else key,
                                                           sel_hz=key.split("_")[0]))


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
    df["ts"] = df["t"].dt.strftime("%d %b %H:%M" if tf in ("1D", "1W") else "%d %b %Y")
    return df


def chart(sym, f, intr_df, tf, p, style, height):
    df = chart_data(f, intr_df, tf)
    if df.empty:
        st.caption("No chart data for this range.")
        return
    up = df["c"].iloc[-1] >= df["c"].iloc[0]
    col = "#00b386" if up else "#eb5b3c"          # Groww-like green / red
    chg = (df["c"].iloc[-1] / df["c"].iloc[0] - 1) * 100
    st.markdown(f'<div class="rg"><span>{df["ts"].iloc[0]} to {df["ts"].iloc[-1]}</span>'
                f'<span class="{"up" if up else "dn"}">{chg:+.2f}% in {tf}</span></div>', unsafe_allow_html=True)
    lv = pd.DataFrame([("Buy", p["e"], "#60a5fa"), ("Stop", p["sl"], "#f87171"), ("T1", p["t1"], "#4ade80"), ("T2", p["t2"], "#4ade80")],
                      columns=["n", "y", "col"]) if p else pd.DataFrame({"n": [], "y": [], "col": []})
    lo = min(df["l"].min() if style == "Candles" else df["c"].min(), lv["y"].min() if len(lv) else 1e18)
    hi = max(df["h"].max() if style == "Candles" else df["c"].max(), lv["y"].max() if len(lv) else -1e18)
    pad = (hi - lo) * .05 or hi * .01
    dom = [lo - pad, hi + pad]
    x = alt.X("i:Q", axis=None, scale=alt.Scale(domain=[0, max(len(df) - 1, 1)], nice=False))
    yax = alt.Axis(title=None, orient="right", labelColor="#6b7a90", gridColor="rgba(148,163,184,.08)", domain=False,
                   ticks=False, tickCount=5, labelFontSize=11)
    y = alt.Y("c:Q", scale=alt.Scale(domain=dom, zero=False, nice=False), axis=yax)
    base = alt.Chart(df)
    layers = []
    if style == "Candles":
        ycol = alt.condition("datum.c >= datum.o", alt.value("#00b386"), alt.value("#eb5b3c"))
        layers.append(base.mark_rule(strokeWidth=1).encode(x=x, y=alt.Y("l:Q", scale=alt.Scale(domain=dom, zero=False, nice=False), axis=yax),
                                                             y2="h:Q", color=ycol))
        bw = max(2, min(9, int(300 / max(len(df), 1))))
        layers.append(base.mark_bar(size=bw).encode(x=x, y=alt.Y("o:Q", scale=alt.Scale(domain=dom, zero=False, nice=False)), y2="c:Q", color=ycol))
    else:
        grad = alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                            stops=[alt.GradientStop(color="rgba(0,0,0,0)", offset=0), alt.GradientStop(color=col, offset=1)])
        layers.append(base.mark_area(color=grad, opacity=.28, interpolate="monotone", clip=True).encode(x=x, y=y, y2=alt.datum(dom[0])))
        layers.append(base.mark_line(color=col, strokeWidth=2.2, interpolate="monotone").encode(x=x, y=y))
    if len(lv):
        layers.append(alt.Chart(lv).mark_rule(strokeDash=[5, 4], strokeWidth=1.1, opacity=.9).encode(y="y:Q", color=alt.Color("col:N", scale=None)))
        layers.append(alt.Chart(lv).mark_text(align="left", dx=3, dy=-6, fontSize=10.5, fontWeight="bold").encode(
            y="y:Q", x=alt.value(2), text="n:N", color=alt.Color("col:N", scale=None)))
    hover = alt.selection_point(nearest=True, on="mouseover", fields=["i"], empty=False, clear="mouseout")
    layers.append(base.mark_rule(color="#94a3b8", strokeDash=[3, 3], strokeWidth=1).encode(
        x=x, opacity=alt.condition(hover, alt.value(.8), alt.value(0)),
        tooltip=[alt.Tooltip("ts:N", title="Time"), alt.Tooltip("o:Q", title="Open", format=",.2f"), alt.Tooltip("h:Q", title="High", format=",.2f"),
                 alt.Tooltip("l:Q", title="Low", format=",.2f"), alt.Tooltip("c:Q", title="Close", format=",.2f")]).add_params(hover))
    layers.append(base.mark_circle(size=55, color=col, stroke="#05080d", strokeWidth=1.5).encode(
        x=x, y=y, opacity=alt.condition(hover, alt.value(1), alt.value(0))))
    ch = alt.layer(*layers).properties(height=height).configure_view(strokeWidth=0).configure(background="transparent")
    st.altair_chart(ch, use_container_width=True)
    if p:
        st.markdown(f'<div class="lgd"><span style="color:#60a5fa">Buy <b>{rs(p["e"])}</b></span><span style="color:#f87171">Stop <b>{rs(p["sl"])}</b></span>'
                    f'<span style="color:#4ade80">T1 <b>{rs(p["t1"])}</b></span><span style="color:#4ade80">T2 <b>{rs(p["t2"])}</b></span></div>',
                    unsafe_allow_html=True)


def panel(sym, name, f, intr, intr_df, res, sel_hz, height=280):
    c, ini = logo(sym)
    price = intr["ltp"] if intr else f["c"]
    prev = intr["prev"] if intr else f["prev"]
    chg = (price / prev - 1) * 100
    top = st.columns([5, 1])
    with top[1]:
        st.button("Close", key=f"close_{sym}", on_click=lambda: st.session_state.update(sel=None, sel_key=None))
    st.markdown(f'<div class="cp"><div class="lgb" style="background:{c}">{ini}</div><div class="n1">{esc(name)}</div>'
                f'<div class="n2">{esc(sym)} · NSE · {esc(f["sec"])}</div><div class="p1">{rs(price)}<small>INR</small></div>'
                f'<div class="p2 {"up" if chg >= 0 else "dn"}">{price - prev:+,.2f} {chg:+.2f}% today</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    tf = c1.radio("Chart range", ["1D", "1W", "1M", "6M", "1Y"], index=2, horizontal=True, label_visibility="collapsed", key=f"tf_{sym}")
    style = c2.radio("Chart type", ["Line", "Candles"], index=0, horizontal=True, label_visibility="collapsed", key=f"cs_{sym}")
    hz = sel_hz if sel_hz in res else "week"
    chart(sym, f, intr_df, tf, res[hz]["p"], style, height)
    st.markdown(f'<div class="st3"><div><span>PREV CLOSE</span>{rs(prev)}</div><div><span>OPEN</span>{rs(intr["open"] if intr else f["o"])}</div>'
                f'<div><span>DAY HIGH / LOW</span>{rs(intr["hi"] if intr else f["h"])}<br>{rs(intr["lo"] if intr else f["l"])}</div></div>',
                unsafe_allow_html=True)
    html = ""
    for k in ("today", "week", "month"):
        r = res[k]
        cls = "hi" if r["vd"] == "BUY" else "md" if r["vd"] == "WAIT" else "av"
        lv = lvl_line(r["p"]) if r["p"] else ""
        o = r["out"]
        html += (f'<div class="vrow"><span>{HZ_LABEL[k]} <span class="vd {cls}">{r["vd"]}</span> <b class="{o[1]}">{o[0]}</b></span>'
                 f'<span>{r["sc"] / 10:.1f}/10 · {r["hit"]}/{r["n"]} markers</span></div>{lv}'
                 + (f'<div class="why {o[1]}">{esc(o[2])}</div>' if o[2] else "") + f'<div class="why">{esc(r["why"])}</div>')
    st.markdown(html, unsafe_allow_html=True)
    with st.expander("All markers"):
        for k in ("today", "week", "month"):
            rows, _ = markers(res[k]["chk"], k)
            st.markdown(f"**{HZ_LABEL[k]}**")
            st.markdown("".join(
                f'<div class="ck"><span><span class="dot {"g" if v > .2 else "r" if v < -.05 else ""}"></span>{esc(NAMES[kk])}: {esc(note)}</span>'
                f'<b class="{"up" if cc > .05 else "dn" if cc < -.05 else ""}">{cc:+.1f}</b></div>' for cc, kk, v, note in rows)
                + "".join(f'<div class="ck"><span>Red flag: {esc(x)}</span></div>' for x in res[k]["flags"]), unsafe_allow_html=True)


# ---------- header and search ----------
st.markdown('<div class="hero"><h1>Liquidity <span>Radar</span></h1><div class="s">Top 5 buys for today, 1 week and 1 month. Tap a stock for its chart. Updates every 10 minutes.</div></div>', unsafe_allow_html=True)
st.session_state.setdefault("sel", None)
st.session_state.setdefault("sel_hz", "week")

all_syms = get_all_symbols()
names = dict(all_syms)
labels = [f"{a} · {b}" for a, b in all_syms]
lab2sym = {l: a for l, (a, _) in zip(labels, all_syms)}


with st.expander("Settings"):
    uni = st.selectbox("Stocks to scan", ["Nifty 100", "Nifty 200", "Nifty 500"], index=2,
                       help="Nifty 500 includes mid and small companies, so more lower-priced shares")
    max_price = {"Any price": 1e12, "Under ₹2,000": 2000, "Under ₹1,000": 1000, "Under ₹500": 500, "Under ₹200": 200}[
        st.selectbox("Share price", ["Any price", "Under ₹2,000", "Under ₹1,000", "Under ₹500", "Under ₹200"], index=0)]
    fii_manual = st.number_input("FII net buying in crore (used only if NSE data does not load)", value=0.0, step=100.0)
    wide = st.toggle("Big screen: chart on the right side", value=False, help="Turn on for tablet or laptop")
    min_score = int(round(st.slider("Show only stocks scoring at least (out of 10)", 5.5, 8.0, 6.2, 0.1) * 10))


def on_search():
    v = st.session_state.get("q")
    if v and v in lab2sym:
        st.session_state.update(sel=lab2sym[v], sel_hz="week", sel_key="search")


def search_box():
    st.markdown('<div class="wl-h">CHECK ANY SHARE</div>', unsafe_allow_html=True)
    kw = dict(index=None, placeholder="Search any stock: type PG, KPIT, Tata...", label_visibility="collapsed", key="q", on_change=on_search)
    try:
        picked = st.selectbox("Search", labels, accept_new_options=True, **kw)
    except TypeError:
        picked = st.selectbox("Search", labels, **kw)
    if picked and picked not in lab2sym:
        typed = picked.strip().upper().replace(".NS", "")
        if typed in names:
            st.session_state.update(sel=typed, sel_key="search", sel_hz="week")
        else:
            import difflib
            pool = {a: l for l, a in lab2sym.items()}
            near = difflib.get_close_matches(typed.replace(" ", ""), list(pool), n=4, cutoff=.5)
            by_name = {b.upper(): a for a, b in all_syms}
            near += [by_name[n] for n in difflib.get_close_matches(typed, list(by_name), n=4, cutoff=.45) if by_name[n] not in near]
            if near:
                st.markdown(f'<div class="warn">No stock called "{esc(picked)}". Did you mean:</div>', unsafe_allow_html=True)
                cols = st.columns(2)
                for i, a in enumerate(near[:6]):
                    cols[i % 2].button(pool[a][:40], key=f"dym{i}", use_container_width=True,
                                       on_click=lambda l=pool[a], a=a: st.session_state.update(q=l, sel=a, sel_hz="week", sel_key="search"))
            else:
                st.markdown(f'<div class="warn">No stock found for "{esc(picked)}". Try the company name or NSE symbol.</div>', unsafe_allow_html=True)



st.markdown("<style>.block-container{max-width:%s !important}</style>" % ("1300px" if wide else "720px"), unsafe_allow_html=True)


@st.fragment(run_every=600)
def live():
    ph, conf, live_mkt = phase()
    with st.spinner("Scanning stocks. First load can take 2 to 3 minutes."):
        feats, cues, nifty, src = scan(uni)
        news, news_ok = get_news()
        events = get_events()
        flows = get_fii_dii()
        deliv, deliv_day = get_delivery()
        ban = get_ban()
        deals = get_deals()
    if not feats:
        st.error("Could not get prices right now. It will try again in 10 minutes.")
        return
    fmap = {f["s"]: f for f in feats}
    sec_df = pd.DataFrame([(f["sec"], f["r5"], f["r1"]) for f in feats], columns=["sec", "r5", "r1"]).groupby("sec").mean()
    mood = market_mood(cues)
    if flows is None and fii_manual:
        flows = (fii_manual, 0.0, "typed in Settings")
    mkt_trend = 0
    if nifty is not None and len(nifty) > 60:
        e20n, e50n = ema(nifty, 20).iloc[-1], ema(nifty, 50).iloc[-1]
        mkt_trend = 1 if nifty.iloc[-1] > e20n > e50n else -1 if nifty.iloc[-1] < e20n < e50n else 0
    ctx = dict(news=news, mood=mood, conf=conf, events=events, flows=flows, mkt_trend=mkt_trend, deliv=deliv, ban=ban, deals=deals, fund={})
    heavy_sell = bool(flows and flows[0] < -3000 and flows[0] + (flows[1] or 0) < 0)
    need = min_score + (6 if mood <= -6 else 0) + (5 if heavy_sell else 0) + (4 if mkt_trend < 0 else 0)

    # status pills
    nifty_chg = cues.get("Nifty")
    pills = [(ph, "g" if live_mkt else ""), (f"Updated {now_ist():%I:%M %p}", ""), (f"{len(feats)} scanned", "")]
    if nifty_chg is not None:
        pills.append((f"Nifty {nifty_chg:+.2f}%", "g" if nifty_chg >= 0 else "r"))
    if flows:
        pills.append((f"FII {flows[0]:+,.0f} cr", "g" if flows[0] >= 0 else "r"))
        pills.append((f"DII {flows[1]:+,.0f} cr", "g" if flows[1] >= 0 else "r"))
    pills.append(({1: "Nifty uptrend", -1: "Nifty downtrend", 0: "Nifty sideways"}[mkt_trend], {1: "g", -1: "r", 0: "y"}[mkt_trend]))
    pills.append((f"Global {'good' if mood > 5 else 'weak' if mood < -5 else 'mixed'}", "g" if mood > 5 else "r" if mood < -5 else "y"))
    st.markdown('<div class="pills" style="margin:0 0 6px">' + "".join(f'<span class="pill {c}">{esc(t)}</span>' for t, c in pills) + "</div>",
                unsafe_allow_html=True)
    if heavy_sell:
        st.markdown(f'<div class="warn">FIIs sold ₹{-flows[0]:,.0f} crore and DIIs did not cover it. The bar is raised to {need / 10:.1f}. Use smaller quantity.</div>', unsafe_allow_html=True)
    elif mkt_trend < 0 or mood <= -6:
        st.markdown(f'<div class="warn">Market is weak. Only stronger setups ({need / 10:.1f}+) are shown.</div>', unsafe_allow_html=True)
    search_box()
    left, right = st.columns([1, 1.25], gap="large") if wide else (None, None)

    # ---- stage 1: technical scores, liquidity and price filter ----
    pool = [f for f in feats if f["liq"] >= 20 and f["c"] <= max_price]
    ctx["sector_rank"] = sec_df["r5"].rank(pct=True).to_dict()
    pre = sorted(((score(f, "week", ctx)[0], f) for f in pool), key=lambda x: -x[0])
    intr_pool = [f for _, f in pre[:40]]
    d5 = get_intraday(tuple(f["s"] + ".NS" for f in intr_pool)) if intr_pool else None
    intr_map = {}
    for f in intr_pool:
        x = sub(d5, f["s"] + ".NS")
        if x is not None:
            try:
                intr_map[f["s"]] = intraday_features(x)
            except Exception:
                pass
    live_now = live_mkt and any(v and v["day"] == now_ist().date() for v in intr_map.values())

    def rank(hz, cands):
        ctx["sector_rank"] = (sec_df["r1"] if hz == "today" else sec_df["r5"]).rank(pct=True).to_dict()
        out = []
        for f in cands:
            intr = intr_map.get(f["s"])
            sc, chk, flags = score(f, hz, ctx, intr if (hz == "today" and live_now) else None)
            if not flags and sc >= need - 6:
                out.append((sc, f))
        return sorted(out, key=lambda x: -x[0])

    # ---- stage 2: add company health (P/E, ROE, growth) for the best candidates, then rank again ----
    first = {hz: rank(hz, intr_pool if hz == "today" else pool)[:25] for hz in ("today", "week", "month")}
    cand = sorted({f["s"] for lst in first.values() for _, f in lst})
    ctx["fund"] = fetch_funds(cand)
    with ThreadPoolExecutor(max_workers=4) as ex:
        ctx["serious"] = {s_: r for s_, r in zip(cand, ex.map(get_stock_news, cand)) if r}
    final = {}
    for hz in ("today", "week", "month"):
        final[hz] = [(sc, f) for sc, f in rank(hz, [f for _, f in first[hz]]) if sc >= need
                     and ok_plan(plan(f, hz, intr_map.get(f["s"]), live_now and hz == "today"), hz)]

    # ---- chart panel for a tapped or searched share ----
    def open_panel(sel, sel_hz, height):
        f = fmap.get(sel)
        if f is None:
            dd = get_daily((sel + ".NS",))
            d = sub(dd[0], sel + ".NS") if dd else None
            f = daily_features(sel, "Other", d, nifty) if d is not None else None
        if f is None:
            st.markdown(f'<div class="warn">Not enough price data for {esc(sel)}.</div>', unsafe_allow_html=True)
            return
        if sel not in ctx["fund"]:
            ctx["fund"][sel] = get_fund(sel)
        if sel not in ctx.setdefault("serious", {}):
            sn = get_stock_news(sel)
            if sn:
                ctx["serious"][sel] = sn
        d5s = sub(get_intraday((sel + ".NS",)), sel + ".NS")
        intr = None
        if d5s is not None:
            try:
                intr = intraday_features(d5s)
            except Exception:
                intr = None
        dd = get_daily((sel + ".NS",))
        f = dict(f, _daily=sub(dd[0], sel + ".NS") if dd else None)
        res = verdicts(f, intr, ctx, sec_df, live_mkt, need)
        with st.container(border=True):
            panel(sel, names.get(sel, sel), f, intr, d5s, res, sel_hz, height)

    sel, sel_key = st.session_state.get("sel"), st.session_state.get("sel_key")
    if sel and (sel_key in (None, "search") or wide):
        target = right if wide else st.container()
        with target:
            open_panel(sel, st.session_state.get("sel_hz"), 340 if wide else 280)

    # ---- watchlists ----
    lst = left if wide else st.container()
    lst.caption("Live picks, refreshed every 10 minutes." if live_now else "Market closed. Picks are for the next session.")
    with lst:
        lists = {}
        for hz in ("today", "week", "month"):
            st.markdown(f'<div class="wl-h">{HZ_LABEL[hz].upper()} · TOP 5</div>', unsafe_allow_html=True)
            top = final[hz][:5]
            lists[hz] = {f["s"]: sc for sc, f in final[hz][:15]}
            if not top:
                st.markdown('<div class="none">No stock passes every buy check right now.</div>', unsafe_allow_html=True)
                continue
            for sc, f in top:
                intr = intr_map.get(f["s"])
                use_live = hz == "today" and live_now
                price = intr["ltp"] if (hz == "today" and intr) else f["c"]
                prev = intr["prev"] if (hz == "today" and intr) else f["prev"]
                p = plan(f, hz, intr, use_live)
                key = f"{hz}_{f['s']}"
                row(key, f["s"], names.get(f["s"], f["s"]), price, (price / prev - 1) * 100, sc, p, sel_key == key)
                if not wide and sel_key == key:
                    open_panel(f["s"], hz, 280)

        # ---- common picks ----
        st.markdown('<div class="wl-h">COMMON PICKS · STRONG IN MORE THAN ONE PERIOD</div>', unsafe_allow_html=True)
        common = {}
        for hz, d in lists.items():
            for s_, sc in d.items():
                common.setdefault(s_, {})[hz] = sc
        common = sorted(((len(v), sum(v.values()) / len(v), s_, v) for s_, v in common.items() if len(v) >= 2), reverse=True)[:5]
        if not common:
            st.markdown('<div class="none">No stock is strong in two periods at once right now.</div>', unsafe_allow_html=True)
        for cnt, avg, s_, v in common:
            f = fmap[s_]
            tags = "".join(f'<span class="tg">{ {"today": "1D", "week": "1W", "month": "1M"}[h] }</span>' for h in ("today", "week", "month") if h in v)
            main_hz = "week" if "week" in v else "month" if "month" in v else "today"
            p = plan(f, main_hz, intr_map.get(s_), live_now and main_hz == "today")
            key = f"common_{s_}"
            row(key, s_, names.get(s_, s_), f["c"], (f["c"] / f["prev"] - 1) * 100, int(round(avg)), p, sel_key == key, tags)
            if not wide and sel_key == key:
                open_panel(s_, main_hz, 280)

    st.caption(f"Stock list: {src}. Prices from Yahoo Finance, can be a few minutes late. Company data from Yahoo Finance.")


live()

with st.expander("How stocks are picked"):
    st.markdown("""
**Step 1.** Only shares with high liquidity are kept: at least ₹20 crore traded every day.

**Step 2.** Every share is scored out of 10 on these markers:
1. Moving average trend, and price above the 200 day average.
2. MACD (Moving Average Convergence Divergence), RSI (Relative Strength Index), ADX (Average Directional Index) and Supertrend.
3. Breakouts, Bollinger squeeze and 52 week high.
4. Volume, OBV (On Balance Volume), delivery percentage and accumulation.
5. Over 30 candle patterns on daily, weekly and 5 minute charts.
6. Strength vs Nifty and sector strength.
7. NSE news, bulk and block deals, and upcoming results.
8. FII (Foreign Institutional Investors) and DII (Domestic Institutional Investors) money flow, Nifty trend and world markets.
9. Today only: VWAP (Volume Weighted Average Price), opening range breakout, 5 minute trend and buy vs sell money.

**Step 3.** The best 25 for each period are checked for company health: P/E (Price to Earnings), ROE (Return on Equity), profit growth and debt. This counts most for 1 month.

**Step 4.** Shares with a red flag are removed: bearish candle, bad news, results too close, F&O ban, big seller deal, loss-making company (1 month), or RSI above 80.

**Common picks** are shares in the top 15 of at least two periods.

**Prices:** stop loss uses ATR (Average True Range), the share's normal daily move. Targets are 1.5x and 2.5x the risk (2x and 3x for 1 month). Share price does not affect the score, so cheap and costly shares are treated the same.
""")
st.caption("Screening tool only. No method can promise a share will go up. Always use the stop loss. Not financial advice.")
