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

st.set_page_config(page_title="Liquidity Radar", layout="centered", initial_sidebar_state="collapsed")
IST = "Asia/Kolkata"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "*/*"}

# ======================= STYLE =======================
st.markdown("""
<style>
#MainMenu, footer, header [data-testid="stToolbar"]{visibility:hidden}
.block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:720px}
html, body, [class*="css"]{overflow-x:hidden}
.hero{background:linear-gradient(135deg,#0f172a,#1e3a8a);color:#fff;border-radius:16px;padding:16px 18px;margin-bottom:12px}
.hero h1{font-size:24px;margin:0;color:#fff}
.hero .s{font-size:13px;opacity:.85;margin-top:4px}
.pills{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.pill{font-size:12px;padding:3px 10px;border-radius:20px;background:rgba(255,255,255,.14);white-space:nowrap}
.warn{background:#fff7ed;border:1px solid #fdba74;color:#9a3412;border-radius:12px;padding:10px 12px;font-size:14px;margin:6px 0 12px}
.card{border:1px solid #e5e7eb;border-radius:16px;padding:14px;margin:10px 0 4px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
.sym{font-size:19px;font-weight:700;color:#0f172a}
.sub{font-size:13px;color:#64748b;margin-top:2px}
.score{min-width:58px;text-align:center;border-radius:12px;padding:6px 4px;font-weight:800;font-size:22px;line-height:1}
.score small{display:block;font-size:10px;font-weight:600;margin-top:3px}
.hi{background:#dcfce7;color:#166534}.md{background:#fef9c3;color:#854d0e}
.lv{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
.lv div{background:#f8fafc;border-radius:10px;padding:8px 10px}
.lv span{display:block;font-size:11px;color:#64748b}
.lv b{font-size:16px;color:#0f172a}
.lv .t b{color:#15803d}.lv .x b{color:#b91c1c}
.plan{font-size:13px;color:#334155;margin-top:10px;line-height:1.45}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.chip{font-size:12px;padding:3px 9px;border-radius:20px;background:#eef2ff;color:#3730a3}
.chip.c{background:#ecfdf5;color:#065f46}
.up{color:#15803d}.dn{color:#b91c1c}
.ck{display:flex;justify-content:space-between;gap:8px;font-size:13px;padding:4px 0;border-bottom:1px solid #f1f5f9}
.ck span:first-child{color:#334155}.ck b{white-space:nowrap}
.none{color:#64748b;font-size:14px;padding:10px 0}
.hz{border-top:1px solid #e5e7eb;margin-top:12px;padding-top:10px}
.vd{font-size:13px;font-weight:700;padding:4px 10px;border-radius:20px;white-space:nowrap}
.av{background:#fee2e2;color:#991b1b}
.why{font-size:13px;color:#475569;margin-top:6px}
@media (prefers-color-scheme: dark){
 .card{background:#0f172a;border-color:#1e293b}.sym,.lv b{color:#f1f5f9}.lv div{background:#1e293b}
 .plan,.ck span:first-child{color:#cbd5e1}.ck{border-color:#1e293b}.chip{background:#1e1b4b;color:#c7d2fe}.chip.c{background:#052e16;color:#a7f3d0}
}
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


@st.cache_data(ttl=1800, show_spinner=False)
def get_daily(tickers):
    parts = []
    for i in range(0, len(tickers), 100):
        try:
            parts.append(yf.download(" ".join(tickers[i:i + 100]), period="1y", interval="1d", group_by="ticker",
                                     progress=False, auto_adjust=False, threads=True))
        except Exception:
            pass
    return parts


@st.cache_data(ttl=55, show_spinner=False)
def get_intraday(tickers):
    try:
        return yf.download(" ".join(tickers), period="5d", interval="5m", group_by="ticker",
                           progress=False, auto_adjust=False, threads=True)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=60, show_spinner=False)
def get_news():
    try:
        s = requests.Session()
        s.headers.update(UA)
        s.get("https://www.nseindia.com/companies-listing/corporate-filings-announcements", timeout=10)
        rows = s.get("https://www.nseindia.com/api/corporate-announcements?index=equities", timeout=12).json()
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
                  cd=4, cw=0, hhhl=2, acc=2, news=10, sector=5, mood=8,
                  vwap=8, orb=6, e5=5, ivol=6, rng=4, gap=3, flow=6, c5=6),
    "week": dict(liq=6, vol=6, trend=7, e200=3, macd=6, rsi=5, adx=5, st=6, brk=7, sq=4, obv=5, rs=6, w52=3, stretch=4,
                 cd=7, cw=3, hhhl=4, acc=5, news=8, sector=5, mood=4),
    "month": dict(liq=5, vol=4, trend=8, e200=7, macd=5, rsi=4, adx=5, st=5, brk=6, sq=3, obv=6, rs=8, w52=6, stretch=3,
                  cd=3, cw=8, hhhl=6, acc=5, news=6, sector=5, mood=2),
}
NAMES = dict(liq="Liquidity (money traded daily)", vol="Volume vs normal", trend="Moving average trend", e200="Above 200 day average",
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
    return int(round(clamp(sc, 0, 100))), chk, flags


def plan(f, hz, intr, live, trigger=None):
    a = f["atr"]
    if trigger:
        e = trigger
        if hz == "today":
            stop, rr = e - .5 * a, (1.5, 2.5)
        elif hz == "week":
            stop, rr = max(f["lo10"] * .998, e - 1.5 * a), (1.5, 2.5)
            if e - stop < .7 * a:
                stop = e - 1.5 * a
        else:
            stop, rr = max(f["lo20"] * .995, f["e50"] * .98, e - 3 * a), (2, 3)
            if e - stop < 1.5 * a:
                stop = e - 2 * a
        how = f"Wait. Buy only if price goes above ₹{tick(e):,.2f}."
    elif hz == "today":
        if live and intr:
            e = intr["ltp"]
            stop = max(e - .5 * a, intr["vwap"] * .997) if intr["vwap"] < e else e - .5 * a
            stop = min(stop, e * .996)
            how = f"Buy now near ₹{tick(e):,.2f}. Buy only while price stays above VWAP ₹{tick(intr['vwap']):,.2f}. Exit everything by 3:15 PM."
        else:
            base = intr["hi"] if intr else f["h"]
            e = base * 1.001
            stop = e - .5 * a
            how = f"Next session: buy only if price crosses ₹{tick(e):,.2f} after 9:30 AM. If it does not, skip. Exit everything by 3:15 PM."
        rr = (1.5, 2.5)
    elif hz == "week":
        e = f["c"]
        stop = max(f["lo10"] * .998, e - 1.5 * a)
        if e - stop < .7 * a:
            stop = e - 1.5 * a
        rr = (1.5, 2.5)
        how = f"Buy near ₹{tick(e):,.2f}, or on a small dip down to ₹{tick(e - .3 * a):,.2f}. Hold up to 5 trading days. Exit if a day closes below the stop."
    else:
        e = f["c"]
        stop = max(f["lo20"] * .995, f["e50"] * .98, e - 3 * a)
        if e - stop < 1.5 * a:
            stop = e - 2 * a
        rr = (2, 3)
        how = f"Buy near ₹{tick(e):,.2f}. Hold up to 4 weeks. Exit if a week closes below the stop."
    risk = e - stop
    t1, t2 = e + rr[0] * risk, e + rr[1] * risk
    return dict(e=tick(e), t1=tick(t1), t2=tick(t2), sl=tick(stop), rp=risk / e * 100, g1=(t1 / e - 1) * 100, g2=(t2 / e - 1) * 100, how=how)


def market_mood(cues, fii):
    def idx(v, w):
        if v is None:
            return 0
        return w if v > .5 else -w if v < -.5 else w / 2 if v > 0 else -w / 2 if v < 0 else 0
    m = idx(cues.get("US Dow"), 2) + idx(cues.get("US Nasdaq"), 2) + idx(cues.get("Japan"), 1.5) + idx(cues.get("Hong Kong"), 1.5) + idx(cues.get("Nifty"), 4)
    cr, fx, vx = cues.get("Crude"), cues.get("Dollar-Rupee"), cues.get("India VIX")
    m += 0 if cr is None else -1.5 if cr > 1 else 1.5 if cr < -1 else 0
    m += 0 if fx is None else -1 if fx > .2 else 1 if fx < -.2 else 0
    m += 0 if vx is None else -2 if vx > 18 else -1 if vx > 15 else 1 if vx < 13 else 0
    m += 2 if fii > 1000 else 1 if fii > 0 else -2 if fii < -1000 else -1 if fii < 0 else 0
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
def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def card(rank, f, sc, chk, p, hz, intr):
    price = intr["ltp"] if hz == "today" and intr else f["c"]
    prev = intr["prev"] if hz == "today" and intr else f["prev"]
    chg = (price / prev - 1) * 100
    top = sorted(((v * W[hz][k], k) for k, (v, _) in chk.items() if k in W[hz] and W[hz][k] > 0), reverse=True)[:4]
    chips = "".join(f'<span class="chip">{esc(NAMES[k])}</span>' for val, k in top if val > 0)
    pats = [n for n, d, _ in (f["cw"] if hz == "month" else (intr["c5"] if hz == "today" and intr else f["cd"])) if d > 0]
    chips += "".join(f'<span class="chip c">{esc(n)}</span>' for n in pats[:2])
    hold = {"today": "Same day", "week": "Up to 1 week", "month": "Up to 1 month"}[hz]
    st.markdown(f"""
<div class="card">
 <div class="top">
  <div><div class="sym">{rank}. {esc(f['s'])}</div>
   <div class="sub">{esc(f['sec'])} · ₹{price:,.2f} <span class="{'up' if chg >= 0 else 'dn'}">{chg:+.2f}%</span></div></div>
  <div class="score {'hi' if sc >= 70 else 'md'}">{sc}<small>of 100</small></div>
 </div>
 <div class="lv">
  <div><span>Buy at</span><b>₹{p['e']:,.2f}</b></div>
  <div class="x"><span>Stop loss ({p['rp']:.1f}% risk)</span><b>₹{p['sl']:,.2f}</b></div>
  <div class="t"><span>Target 1 (+{p['g1']:.1f}%)</span><b>₹{p['t1']:,.2f}</b></div>
  <div class="t"><span>Target 2 (+{p['g2']:.1f}%)</span><b>₹{p['t2']:,.2f}</b></div>
 </div>
 <div class="plan">{esc(p['how'])}<br>Sell half at Target 1, then move stop loss to your buy price. Sell the rest at Target 2. Holding time: {hold}.</div>
 <div class="chips">{chips}</div>
</div>""", unsafe_allow_html=True)
    with st.expander(f"Why {f['s']}? See all checks"):
        rows = sorted(((v * W[hz][k], k, note) for k, (v, note) in chk.items() if k in W[hz] and W[hz][k] > 0), reverse=True)
        st.markdown("".join(
            f'<div class="ck"><span>{esc(NAMES[k])}: {esc(note)}</span><b class="{"up" if val > 0.05 else "dn" if val < -0.05 else ""}">{val:+.1f}</b></div>'
            for val, k, note in rows), unsafe_allow_html=True)


def clean(q):
    q = q.strip().upper().replace(" ", "")
    for x in (".NS", ".BO", "NSE:", "BSE:"):
        q = q.replace(x, "")
    return q


def check_stock(q, sec_of, sec_df, ctx, nifty, live_mkt, need):
    sym = clean(q)
    if not sym:
        return
    parts = get_daily((sym + ".NS",))
    d = sub(parts[0], sym + ".NS") if parts else None
    if d is None:
        st.markdown(f'<div class="warn">Could not find {esc(sym)}. Type the NSE symbol, like TATASTEEL or HDFCBANK.</div>', unsafe_allow_html=True)
        return
    f = daily_features(sym, sec_of.get(sym, "Other"), d, nifty)
    if not f:
        st.markdown(f'<div class="warn">{esc(sym)} does not have enough price history to check.</div>', unsafe_allow_html=True)
        return
    intr = None
    x = sub(get_intraday((sym + ".NS",)), sym + ".NS")
    if x is not None:
        try:
            intr = intraday_features(x)
        except Exception:
            intr = None
    live_now = live_mkt and intr is not None and intr["day"] == now_ist().date()
    price = intr["ltp"] if intr else f["c"]
    prev = intr["prev"] if intr else f["prev"]
    chg = (price / prev - 1) * 100
    rows = ""
    for hz, label in (("today", "Today"), ("week", "1 week"), ("month", "1 month")):
        ctx["sector_rank"] = (sec_df["r1"] if hz == "today" else sec_df["r5"]).rank(pct=True).to_dict()
        sc, chk, flags = score(f, hz, ctx, intr if (hz == "today" and live_now) else None)
        good = sorted(((v * W[hz][k], k, n) for k, (v, n) in chk.items() if k in W[hz] and W[hz][k] > 0), reverse=True)
        bad = [(v, k, n) for v, k, n in reversed(good) if v < 0][:2]
        if not flags and sc >= need:
            verdict, cls = "BUY", "hi"
            p = plan(f, hz, intr, live_now)
            why = "Good: " + ", ".join(NAMES[k] for v, k, _ in good[:3] if v > 0)
        elif sc >= 50 and not any(w.startswith(("Bad news", "Low liquidity", "Below 200")) for w in flags):
            verdict, cls = "WAIT", "md"
            base = (intr["hi"] if intr else f["h"]) if hz == "today" else max(f["hi20"], f["h"]) if hz == "week" else max(f["hi55"], f["h"])
            p = plan(f, hz, intr, live_now, trigger=base * 1.002)
            why = "Problem: " + (flags[0] if flags else ", ".join(NAMES[k] for _, k, _ in bad) or "score not high enough")
        else:
            verdict, cls = "AVOID", "av"
            p = None
            why = "Problem: " + (flags[0] if flags else ", ".join(NAMES[k] for _, k, _ in bad) or "weak overall")
        lv = (f'<div class="lv"><div><span>Buy at</span><b>₹{p["e"]:,.2f}</b></div>'
              f'<div class="x"><span>Stop loss ({p["rp"]:.1f}% risk)</span><b>₹{p["sl"]:,.2f}</b></div>'
              f'<div class="t"><span>Target 1 (+{p["g1"]:.1f}%)</span><b>₹{p["t1"]:,.2f}</b></div>'
              f'<div class="t"><span>Target 2 (+{p["g2"]:.1f}%)</span><b>₹{p["t2"]:,.2f}</b></div></div>'
              f'<div class="plan">{esc(p["how"])}</div>') if p else '<div class="plan">Do not buy for this period right now.</div>'
        rows += (f'<div class="hz"><div class="top"><b>{label}</b><span class="vd {cls}">{verdict} · {sc}/100</span></div>'
                 f'{lv}<div class="why">{esc(why)}</div></div>')
    st.markdown(f"""
<div class="card">
 <div class="sym">{esc(sym)}</div>
 <div class="sub">{esc(f['sec'])} · ₹{price:,.2f} <span class="{'up' if chg >= 0 else 'dn'}">{chg:+.2f}%</span></div>
 {rows}
</div>""", unsafe_allow_html=True)


st.markdown('<div class="hero"><h1>Liquidity Radar</h1><div class="s">Top 5 stocks to buy, with buy price, targets and stop loss</div></div>', unsafe_allow_html=True)

query = st.text_input("Check any stock", placeholder="Type NSE symbol, like TATASTEEL", label_visibility="collapsed")
with st.expander("Settings"):
    uni = st.selectbox("Stocks to scan", ["Nifty 100", "Nifty 200", "Nifty 500"], index=1,
                       help="Nifty 500 scans more stocks but the first load takes longer")
    fii = st.number_input("FII (Foreign Institutional Investors) net buying yesterday, in crore", value=0.0, step=100.0)
    min_score = st.slider("Show only stocks scoring at least", 55, 80, 62)


@st.fragment(run_every=60)
def live():
    ph, conf, live_mkt = phase()
    universe, src = get_universe(uni)
    tick_list = [s + ".NS" for s, _ in universe]
    sec_of = dict(universe)
    with st.spinner(f"Scanning {len(universe)} stocks. First load can take 1 to 2 minutes."):
        parts = get_daily(tuple(tick_list))
        cues, nifty = get_cues()
        news, news_ok = get_news()
    feats = []
    for s in sec_of:
        d = None
        for part in parts:
            d = sub(part, s + ".NS")
            if d is not None:
                break
        if d is None:
            continue
        try:
            f = daily_features(s, sec_of[s], d, nifty)
            if f:
                feats.append(f)
        except Exception:
            continue
    if not feats:
        st.error("Could not get prices right now. It will try again in a minute.")
        return
    sec_df = pd.DataFrame([(f["sec"], f["r5"], f["r1"]) for f in feats], columns=["sec", "r5", "r1"]).groupby("sec").mean()
    mood = market_mood(cues, fii)
    ctx = dict(news=news, mood=mood, conf=conf)

    # status line
    nifty_chg = cues.get("Nifty")
    pills = [ph, f"Updated {now_ist():%I:%M %p}", f"{len(feats)} stocks scanned",
             f"Global mood {'good' if mood > 5 else 'weak' if mood < -5 else 'mixed'} ({mood:+.0f})"]
    if nifty_chg is not None:
        pills.append(f"Nifty {nifty_chg:+.2f}%")
    pills.append("NSE news on" if news_ok else "NSE news off")
    st.markdown('<div class="pills" style="margin:-4px 0 10px">' + "".join(f'<span class="pill" style="background:#e2e8f0;color:#0f172a">{esc(x)}</span>' for x in pills) + "</div>", unsafe_allow_html=True)
    need = min_score + (6 if mood <= -6 else 0)
    if mood <= -6:
        st.markdown(f'<div class="warn">World markets look weak today. Only very strong setups (score {need}+) are shown. Use smaller quantity.</div>', unsafe_allow_html=True)

    if query.strip():
        st.markdown("#### Your stock")
        check_stock(query, sec_of, sec_df, ctx, nifty, live_mkt, need)
        st.markdown("#### Top 5 picks")

    tabs = st.tabs(["Today", "1 week", "1 month"])
    for hz, tab in zip(("today", "week", "month"), tabs):
        with tab:
            ctx["sector_rank"] = (sec_df["r1"] if hz == "today" else sec_df["r5"]).rank(pct=True).to_dict()
            pool = feats
            intr_map = {}
            if hz == "today":
                pre = []
                for f in feats:
                    if f["liq"] < 20:
                        continue
                    sc, _, fl = score(f, "week", ctx)
                    pre.append((sc, f))
                pre.sort(key=lambda x: -x[0])
                pool = [f for _, f in pre[:40]]
                d5 = get_intraday(tuple(f["s"] + ".NS" for f in pool))
                for f in pool:
                    x = sub(d5, f["s"] + ".NS")
                    if x is not None:
                        try:
                            intr_map[f["s"]] = intraday_features(x)
                        except Exception:
                            pass
                is_today = any(v and v["day"] == now_ist().date() for v in intr_map.values())
                live_now = live_mkt and is_today
                st.caption("Live picks for today. Changes every minute with price and news." if live_now
                           else "Market is closed now. These are picks for the next trading session, with the price to wait for.")
            else:
                live_now = False
                st.caption("For holding up to 1 week." if hz == "week" else "For holding up to 1 month.")
            res = []
            for f in pool:
                intr = intr_map.get(f["s"])
                sc, chk, flags = score(f, hz, ctx, intr if (hz == "today" and live_now) else None)
                if flags or sc < need:
                    continue
                res.append((sc, f, chk, intr))
            res.sort(key=lambda x: -x[0])
            if not res:
                st.markdown('<div class="none">No stock passes all the buy checks right now. Not buying is also a good decision. Check again later.</div>', unsafe_allow_html=True)
                continue
            if len(res) < 5:
                st.caption(f"Only {len(res)} stock(s) passed every check.")
            for i, (sc, f, chk, intr) in enumerate(res[:5], 1):
                card(i, f, sc, chk, plan(f, hz, intr, live_now), hz, intr)

    with st.expander("World markets today"):
        st.markdown("".join(f'<div class="ck"><span>{k}</span><b class="{"up" if (v > 0 and k not in ("Crude", "Dollar-Rupee", "India VIX")) else ""}">'
                            f'{v:.1f}{"" if k == "India VIX" else "%"}</b></div>' for k, v in cues.items()), unsafe_allow_html=True)
    st.caption(f"Stock list: {src}. Prices from Yahoo Finance, can be a few minutes late.")


live()

with st.expander("How stocks are picked"):
    st.markdown("""
**Every stock gets a score out of 100 from these checks:**
1. Liquidity: only stocks with big money traded daily.
2. Moving averages: price above 20, 50 and 200 day averages, in that order.
3. MACD (Moving Average Convergence Divergence) turning up.
4. RSI (Relative Strength Index) in the healthy 55 to 70 zone.
5. ADX (Average Directional Index) showing a strong uptrend.
6. Supertrend on buy.
7. Breakout above the 20 day high (55 day for 1 month).
8. Bollinger band squeeze breaking out.
9. OBV (On Balance Volume) rising, meaning buyers are stronger.
10. Stronger than Nifty over 1 and 3 months.
11. Near 52 week high.
12. Not stretched too far above the 20 day average.
13. Higher highs and higher lows.
14. More money traded on up days than down days.
15. Over 30 candle patterns on daily, weekly and 5 minute charts, like hammer, engulfing, morning star, three white soldiers, shooting star, evening star and three black crows.
16. Official NSE news, good or bad.
17. Sector strength.
18. World markets, crude, rupee, India VIX (Volatility Index) and FII flows.
19. For today only: VWAP (Volume Weighted Average Price), opening range breakout, 5 minute trend, volume today, buy vs sell money, gap, and profit booking risk.

**A stock is removed from the buy list if:**
1. Its latest candle is bearish, like shooting star, bearish engulfing or evening star.
2. It has bad news.
3. Supertrend says sell (1 week and 1 month).
4. It is below the 200 day average (1 month).
5. RSI is above 80, too hot.
6. For today, it is below VWAP or shows profit booking signs.

**Prices:** stop loss is based on ATR (Average True Range), the stock's normal daily move. Targets are 1.5x and 2.5x the risk (2x and 3x for 1 month).
""")
st.caption("This is a screening tool to help you shortlist. No method can promise a stock will go up. Always use the stop loss. Not financial advice.")
