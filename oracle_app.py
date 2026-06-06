"""
Stock Oracle  v3.0  Production
Browser-based multi-factor stock analysis & discovery platform.
Run:  streamlit run oracle_app.py
"""
import warnings; warnings.filterwarnings("ignore")
import re, requests
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import ephem
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

st.set_page_config(page_title="Stock Oracle", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

for k,v in {"theme":"light","result":None,"range":"6M","chart_type":"Line","ticker":"AAPL","picks_mode":"Top Opportunities","watchlist":["AAPL","MSFT","NVDA"]}.items():
    if k not in st.session_state: st.session_state[k]=v

DARK=st.session_state.theme=="dark"
if DARK:
    BG,SFC,CARD,BDR="#0B0F1A","#111726","#161E30","#27314A"
    TXT,MUT,SUB="#F1F5F9","#94A3B8","#334155"; CBG,CGRID="#0B0F1A","#1B2335"
    G="#22C55E"; R="#F43F5E"
else:
    BG,SFC,CARD,BDR="#F7F8FB","#FFFFFF","#FFFFFF","#E6E9F0"
    TXT,MUT,SUB="#0F172A","#64748B","#CBD5E1"; CBG,CGRID="#FFFFFF","#EEF1F7"
    G="#16A34A"; R="#DC2626"

# ── Indigo & Slate palette (primary accent + supporting colors, used everywhere) ──
BLU="#4F46E5"; TEAL="#0D9488"; GLD="#F59E0B"; PRP="#7C3AED"; CYN="#0EA5E9"; ORG="#F97316"
BLU_RGB="79,70,229"  # primary accent as r,g,b for translucent fills/shadows
def _rgb(h): h=h.lstrip('#'); return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"
GR=_rgb(G); RR=_rgb(R)  # "r,g,b" strings for chart fills, kept in sync with the theme
CBASE=dict(paper_bgcolor=CBG,plot_bgcolor=CBG,font=dict(family="Inter,sans-serif",color=MUT,size=11),
           margin=dict(l=52,r=16,t=40,b=32),hovermode="x unified",
           hoverlabel=dict(bgcolor=SFC,bordercolor=BDR,font=dict(color=TXT,size=12)))
LEG_H=dict(orientation="h",y=1.03,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=11,color=MUT))
LEG_V=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=11,color=MUT))
MOD_C={"technical":BLU,"fundamental":TEAL,"news":GLD,"psychology":PRP,"historical":ORG,"lunar":CYN}
MOD_I={"technical":"📈","fundamental":"📊","news":"📰","psychology":"🧠","historical":"📅","lunar":"🌙"}
MOD_D={"technical":.25,"fundamental":.25,"news":.20,"psychology":.15,"historical":.10,"lunar":.05}
# Plain-language names shown in the UI instead of the internal keys (avoids jargon)
MOD_NAME={"technical":"Price & Trend","fundamental":"Company Health","news":"News Mood",
          "psychology":"Market Mood","historical":"Seasonal Patterns","lunar":"Moon Cycle"}
MOD_DESC={
    "technical":   "How the price is moving — its trend, momentum, and whether it looks stretched too high or low",
    "fundamental": "How the business is doing — sales growth, profitability, debt, and where analysts see the price going",
    "news":        "Whether recent headlines about the company sound positive or negative",
    "psychology":  "The mood of the market — fear vs. greed, crowd behavior, and signs of buying or selling pressure",
    "historical":  "Calendar patterns — months and trends that have historically helped or hurt this kind of stock",
    "lunar":       "A light-hearted seasonal factor: new moons have historically lined up with slightly higher returns",
}
TL={"1W":7,"1M":30,"3M":90,"6M":180,"1Y":365,"2Y":730,"5Y":1825,"MAX":9999}
EXCL={"A","I","AI","AM","AN","BE","BY","DO","GO","HE","IF","IN","IS","IT","ME","MY","NO","OF","OK","ON","OR","SO","TO","UP","US","WE","AND","ARE","BUT","FOR","HAS","HOW","NOT","NOW","OUR","OUT","THE","WAS","WHO","WHY","WITH","YOU","ALL","BIG","GET","GOT","HAD","HIM","HIS","ITS","LET","LOW","MAY","NEW","OLD","ONE","OWN","PUT","SAY","SEE","SET","SHE","SIT","SIX","TEN","TOO","TWO","USE","WAY","WIN","WON","YES","YET","YOLO","FOMO","HODL","DD","WSB","IMO","IMHO","YTD","ATH","ATL","IPO","ETF","EPS","SEC","FDA","CEO","CFO","CTO","NYSE","NASDAQ","OTC","OTM","ITM","ATM","IV","VIX","SPY","QQQ","DJI","IWM","GLD","SLV","TLT","EUR","GBP","USD","BTC","ETH","GDP","CPI","PPI","FED","LOL","OMG","WTF","TBH","SMH","FUD","REKT","APE","APES","MOON","BULL","BEAR","CALLS","PUTS","GAIN","LOSS","PLAY","EDIT","TLDR","PSA","EOD","EOW","AH","PM","HIGH","CASH","SALE","GOOD","BAD","HUGE","HELP","NEED","WANT","MAKE","TAKE","LOOK","LONG","SHORT","HOLD","SELL","BUY","WAIT","NEXT","LAST","THIS","THAT","THEN","WHEN","JUST","LIKE","ALSO","EVEN","OVER","BACK","ONLY","SAME","SUCH","WELL","ELON","MUSK","TRUMP","BIDEN"}

# ── Stock universe (powers the searchable picker + movers) ──
TICKER_NAMES={
 "AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","AMZN":"Amazon","GOOGL":"Alphabet (A)","GOOG":"Alphabet (C)",
 "META":"Meta Platforms","TSLA":"Tesla","AVGO":"Broadcom","AMD":"Advanced Micro Devices","NFLX":"Netflix",
 "ADBE":"Adobe","CRM":"Salesforce","ORCL":"Oracle","INTC":"Intel","QCOM":"Qualcomm","CSCO":"Cisco",
 "TXN":"Texas Instruments","IBM":"IBM","NOW":"ServiceNow","INTU":"Intuit","AMAT":"Applied Materials",
 "MU":"Micron","LRCX":"Lam Research","ADI":"Analog Devices","PANW":"Palo Alto Networks","SNOW":"Snowflake",
 "PLTR":"Palantir","CRWD":"CrowdStrike","DDOG":"Datadog","NET":"Cloudflare","SHOP":"Shopify","UBER":"Uber",
 "ABNB":"Airbnb","DASH":"DoorDash","ZM":"Zoom","ROKU":"Roku","XYZ":"Block","PYPL":"PayPal","COIN":"Coinbase",
 "HOOD":"Robinhood","SOFI":"SoFi","MARA":"MARA Holdings","RIOT":"Riot Platforms","MSTR":"Strategy (MicroStrategy)",
 "ASML":"ASML","TSM":"Taiwan Semiconductor","ARM":"Arm Holdings","SMCI":"Super Micro","SPOT":"Spotify",
 "SNAP":"Snap","PINS":"Pinterest","DKNG":"DraftKings",
 "JPM":"JPMorgan Chase","BAC":"Bank of America","WFC":"Wells Fargo","C":"Citigroup","GS":"Goldman Sachs",
 "MS":"Morgan Stanley","SCHW":"Charles Schwab","BLK":"BlackRock","AXP":"American Express","V":"Visa",
 "MA":"Mastercard","BRK-B":"Berkshire Hathaway","COF":"Capital One","USB":"U.S. Bancorp","PNC":"PNC Financial",
 "UNH":"UnitedHealth","JNJ":"Johnson & Johnson","LLY":"Eli Lilly","PFE":"Pfizer","MRK":"Merck","ABBV":"AbbVie",
 "TMO":"Thermo Fisher","ABT":"Abbott","DHR":"Danaher","BMY":"Bristol Myers Squibb","AMGN":"Amgen","GILD":"Gilead",
 "CVS":"CVS Health","MDT":"Medtronic","ISRG":"Intuitive Surgical","VRTX":"Vertex","REGN":"Regeneron","MRNA":"Moderna",
 "WMT":"Walmart","COST":"Costco","HD":"Home Depot","LOW":"Lowe's","TGT":"Target","MCD":"McDonald's",
 "SBUX":"Starbucks","NKE":"Nike","KO":"Coca-Cola","PEP":"PepsiCo","PG":"Procter & Gamble","CL":"Colgate-Palmolive",
 "MDLZ":"Mondelez","PM":"Philip Morris","MO":"Altria","KHC":"Kraft Heinz","GIS":"General Mills","DIS":"Disney",
 "CMG":"Chipotle","LULU":"Lululemon","EL":"Estée Lauder","YUM":"Yum Brands",
 "XOM":"Exxon Mobil","CVX":"Chevron","COP":"ConocoPhillips","SLB":"Schlumberger","OXY":"Occidental",
 "BA":"Boeing","CAT":"Caterpillar","GE":"GE Aerospace","HON":"Honeywell","UPS":"UPS","FDX":"FedEx",
 "LMT":"Lockheed Martin","RTX":"RTX","DE":"Deere","MMM":"3M","UNP":"Union Pacific","EMR":"Emerson Electric",
 "T":"AT&T","VZ":"Verizon","TMUS":"T-Mobile","CMCSA":"Comcast","WBD":"Warner Bros. Discovery",
 "F":"Ford","GM":"General Motors","RIVN":"Rivian","LCID":"Lucid","NIO":"NIO",
 "BABA":"Alibaba","PDD":"PDD Holdings","JD":"JD.com","GME":"GameStop","AMC":"AMC Entertainment","BB":"BlackBerry",
 "NOK":"Nokia","PLUG":"Plug Power","CHPT":"ChargePoint","AI":"C3.ai",
 "SPY":"S&P 500 ETF","QQQ":"Nasdaq-100 ETF","DIA":"Dow Jones ETF","IWM":"Russell 2000 ETF","VTI":"Total Market ETF",
 "VOO":"Vanguard S&P 500","ARKK":"ARK Innovation","XLF":"Financials ETF","XLE":"Energy ETF","XLK":"Technology ETF",
 "GLD":"Gold ETF","SLV":"Silver ETF","TLT":"20+ Yr Treasury","SOXL":"Semis Bull 3x","TQQQ":"Nasdaq-100 Bull 3x",
}
# Liquid single names used to compute the day's top movers
MOVERS_UNIVERSE=["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","NFLX","CRM","ORCL","INTC","QCOM",
 "MU","PLTR","SMCI","ARM","COIN","HOOD","SOFI","JPM","V","UNH","LLY","WMT","DIS","XOM","BA","F","UBER","BABA"]
# Quality watchlist for the "Stock Picks" top bar
PICKS_UNIVERSE=["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","LLY","V","MA","COST","JPM","UNH","HD","PG",
 "ORCL","CRM","AMD","NFLX","ADBE","QCOM","TXN","PLTR","NOW","INTU","ABBV","KO","MCD","WMT","XOM"]
# Broad liquid pool scanned for "Potential Picks" / Gainers / Losers (one batched download)
SCAN_UNIVERSE=sorted(set(PICKS_UNIVERSE)|set(MOVERS_UNIVERSE)|{
 "GOOG","TSM","ASML","ARM","SMCI","MRVL","PANW","CRWD","SNOW","DDOG","NET","SHOP","UBER","ABNB","DASH","COIN",
 "HOOD","SOFI","PYPL","RBLX","DKNG","SNAP","PINS","SPOT","ROKU","TTD","MDB","ZS","WDAY","ADSK",
 "CSCO","IBM","MU","LRCX","AMAT","ADI","KLAC","NXPI","ON","TGT","LOW","SBUX","NKE","LULU","CMG",
 "PEP","KO","PG","MDLZ","MO","PM","DIS","CMCSA","T","VZ","TMUS","WBD",
 "BAC","WFC","C","GS","MS","SCHW","BLK","AXP","COF","BX","KKR","SPGI","ICE","CME",
 "JNJ","PFE","MRK","TMO","ABT","DHR","BMY","AMGN","GILD","CVS","ISRG","VRTX","REGN","MRNA","HCA","ELV",
 "CVX","COP","SLB","EOG","OXY","MPC","VLO","LNG","FANG","DVN",
 "GE","HON","UPS","FDX","LMT","RTX","DE","UNP","ETN","GD","NOC","CSX",
 "GM","RIVN","LCID","NIO","TM","SE","MELI","NU","GRAB","BIDU",
 "GME","AMC","PLUG","RIOT","MARA","MSTR","CLSK","SOUN","IONQ","RGTI"})

st.html(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box}}
html,body,[data-testid="stAppViewContainer"]{{font-family:'Inter',-apple-system,sans-serif!important;background:{BG}!important;color:{TXT}!important}}
[data-testid="stSidebar"]{{background:{SFC}!important;border-right:1px solid {BDR}!important}}
[data-testid="stSidebar"] *{{color:{TXT}!important}}
.block-container{{padding:1rem 1.5rem 2rem!important;max-width:100%!important}}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="manage-app-button"]{{display:none!important}}
header[data-testid="stHeader"]{{background:transparent!important;height:0!important}}
[data-baseweb="tab-list"]{{gap:3px;background:{CARD};border-radius:10px;padding:4px;border:1px solid {BDR};flex-wrap:wrap}}
[data-baseweb="tab"]{{border-radius:7px!important;font-size:12px!important;font-weight:600!important;padding:6px 14px!important;color:{MUT}!important;background:transparent!important;transition:all .15s!important;white-space:nowrap!important}}
[aria-selected="true"]{{background:{BLU}!important;color:#fff!important;box-shadow:0 2px 8px rgba({BLU_RGB},.4)!important}}
.stButton>button{{background:{CARD}!important;color:{TXT}!important;border:1px solid {BDR}!important;border-radius:8px!important;font-family:'Inter',sans-serif!important;font-size:13px!important;font-weight:600!important;padding:9px 16px!important;transition:border-color .12s,background .12s,box-shadow .12s!important}}
.stButton>button:hover{{border-color:{BLU}!important;background:{SFC}!important;color:{TXT}!important;box-shadow:0 1px 6px rgba(0,0,0,.18)!important}}
.stButton>button:focus{{box-shadow:none!important}}
.stButton>button[kind="primary"]{{background:{BLU}!important;color:#fff!important;border:1px solid {BLU}!important;font-weight:700!important;letter-spacing:.01em!important}}
.stButton>button[kind="primary"]:hover{{background:#4338CA!important;border-color:#4338CA!important;box-shadow:0 3px 12px rgba({BLU_RGB},.40)!important}}
.stTextInput input{{background:{CARD}!important;color:{TXT}!important;border:1px solid {BDR}!important;border-radius:8px!important;font-size:14px!important;font-weight:600!important}}
.stTextInput input:focus{{border-color:{BLU}!important;box-shadow:0 0 0 2px rgba({BLU_RGB},.2)!important}}
[data-baseweb="select"]>div{{background:{CARD}!important;border:1px solid {BDR}!important;border-radius:8px!important;min-height:46px!important;font-size:14px!important;font-weight:600!important}}
[data-baseweb="select"]>div:hover{{border-color:{BLU}!important}}
[data-baseweb="popover"] [role="option"]{{font-size:14px!important}}
.movrow{{display:flex;align-items:center;justify-content:space-between;padding:9px 12px;border-radius:8px;border:1px solid {BDR};background:{CARD};margin-bottom:0}}
.movrow:hover{{border-color:{BLU}}}
.sec-h{{font-size:11px;font-weight:800;letter-spacing:.12em;color:{MUT};text-transform:uppercase;margin:8px 0 8px}}
hr{{border-color:{BDR}!important;margin:14px 0!important}}
.card{{background:{CARD};border:1px solid {BDR};border-radius:12px;padding:16px 20px;margin-bottom:12px}}
.card-sm{{background:{CARD};border:1px solid {BDR};border-radius:10px;padding:12px 16px}}
.pill{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase}}
.sig{{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid {BDR};font-size:12px;color:{MUT}}}
.sig:last-child{{border:none}}
.bar-track{{height:4px;background:{BDR};border-radius:4px;margin:6px 0 10px}}
.bar-fill{{height:4px;border-radius:4px}}
.reddit-card{{background:{CARD};border:1px solid {BDR};border-radius:10px;padding:12px 16px;margin-bottom:8px}}
.fund-row{{display:flex;justify-content:space-between;align-items:center;padding:7px 10px;border-radius:6px;margin-bottom:2px}}
.top-bar{{background:{SFC};border-bottom:1px solid {BDR};padding:8px 16px;display:flex;align-items:center;gap:20px;font-size:13px;flex-wrap:wrap}}
.expl{{background:{f'rgba({BLU_RGB},.10)' if DARK else '#EEF2FF'};border:1px solid {f'rgba({BLU_RGB},.25)' if DARK else '#C7D2FE'};border-radius:8px;padding:12px 16px;font-size:12px;color:{'#A5B4FC' if DARK else '#4338CA'};line-height:1.6;margin-top:8px}}
.seclabel{{font-size:11px;font-weight:800;letter-spacing:.14em;color:{MUT};text-transform:uppercase;margin:22px 0 2px;display:flex;align-items:center;gap:8px}}
.seclabel::before{{content:'';width:3px;height:14px;background:{BLU};border-radius:2px;display:inline-block}}
.secsub{{font-size:12px;color:{MUT};margin:0 0 10px;line-height:1.5}}
.stDownloadButton>button{{background:{CARD}!important;color:{TXT}!important;border:1px solid {BDR}!important;border-radius:8px!important;font-weight:600!important}}
.stDownloadButton>button:hover{{border-color:{BLU}!important}}
::-webkit-scrollbar{{width:5px;height:5px}}::-webkit-scrollbar-track{{background:{BG}}}::-webkit-scrollbar-thumb{{background:{BDR};border-radius:3px}}
@media(max-width:768px){{.block-container{{padding:0.5rem!important}}[data-testid="stHorizontalBlock"]{{flex-direction:column!important}}[data-testid="stHorizontalBlock"]>div{{width:100%!important;min-width:100%!important}}}}
</style>""")

def clamp(x,lo=-1.,hi=1.):
    try:    return max(lo,min(hi,float(x)))
    except: return 0.

def verdict(s):
    if   s> .60: return "Strong Buy",  G,  "▲▲"
    elif s> .30: return "Buy",          G,  "▲"
    elif s> .10: return "Weak Buy",     "#86EFAC","△"
    elif s>-.10: return "Neutral",      MUT,"─"
    elif s>-.30: return "Weak Sell",    "#FCA5A5","▽"
    elif s>-.60: return "Sell",         R,  "▼"
    else:        return "Strong Sell",  R,  "▼▼"

def plain_verdict(s):
    if   s> .60: return "Strong signals point to a buying opportunity"
    elif s> .30: return "More signals are bullish than bearish"
    elif s> .10: return "Slightly more positive signals than negative"
    elif s>-.10: return "Mixed signals — no clear direction"
    elif s>-.30: return "Slightly more negative signals than positive"
    elif s>-.60: return "More signals are bearish than bullish"
    else:        return "Strong signals suggest caution"

def fdollar(v):
    if v is None: return "—"
    try:    return f"${float(v):,.2f}"
    except: return "—"

def fnum(v,dp=2):
    if v is None: return "—"
    try:    return f"{float(v):.{dp}f}"
    except: return "—"

def fpct(v):
    if v is None: return "—"
    try:    return f"{float(v)*100:.2f}%"
    except: return "—"

def fmoney(v):
    if v is None: return "—"
    try:
        v=float(v)
        if v>=1e12: return f"${v/1e12:.2f}T"
        if v>=1e9:  return f"${v/1e9:.2f}B"
        if v>=1e6:  return f"${v/1e6:.2f}M"
        return f"${v:,.0f}"
    except: return "—"

def sc(s): return G if s>.05 else R if s<-.05 else MUT
def slice_df(df,lbl): return df.iloc[-min(TL.get(lbl,180),len(df)):]

def news_title(item):
    """Pull a headline out of a yfinance news item (handles old & new schemas)."""
    if not isinstance(item,dict): return ""
    t=item.get("title")
    if t: return str(t)
    c=item.get("content")
    if isinstance(c,dict): return str(c.get("title","") or "")
    return ""

def news_link(item):
    if not isinstance(item,dict): return ""
    l=item.get("link")
    if l: return str(l)
    c=item.get("content")
    if isinstance(c,dict):
        for k in ("canonicalUrl","clickThroughUrl"):
            u=c.get(k)
            if isinstance(u,dict) and u.get("url"): return str(u["url"])
    return ""

def market_theme(mkt):
    """Turn the index snapshot into a one-line 'what's happening' read."""
    sp=mkt.get("S&P 500",{}).get("chg") or 0.; ndq=mkt.get("Nasdaq",{}).get("chg") or 0.
    vix=mkt.get("VIX",{}).get("val") or 18.; avg=(sp+ndq)/2
    if vix>30:    return ("Nervous market",  f"Big swings are likely today (fear gauge {vix:.0f}). Investors are playing defense.",R)
    if avg>=.4 and vix<18: return ("Strong, calm day",     f"Broad gains with low fear (gauge {vix:.0f}). Buyers are in control.",G)
    if avg>=.15:  return ("Leaning up",         f"Markets are grinding higher — S&P {sp:+.1f}%, Nasdaq {ndq:+.1f}%.",G)
    if avg<=-.4:  return ("Broad sell-off",          f"Widespread selling — S&P {sp:+.1f}%, Nasdaq {ndq:+.1f}%. Watch your levels.",R)
    if avg<=-.15: return ("Leaning down",         f"Mild weakness across the board — S&P {sp:+.1f}%, fear gauge {vix:.0f}.",R)
    return ("Quiet / no clear direction", f"Markets are flat today — S&P {sp:+.1f}%, Nasdaq {ndq:+.1f}%.",GLD)

def strong_reasons(r,thresh=.45,limit=4):
    """Surface the standout single-factor reasons to buy or sell."""
    names=MOD_NAME
    out=[]
    for k in ["technical","fundamental","lunar","news","psychology","historical"]:
        mod=r["modules"].get(k,{}); s=float(mod.get("score",0))
        if abs(s)<thresh: continue
        bn,bv,bd=None,0.,""
        for sn,sv in (mod.get("signals",{}) or {}).items():
            sd,vv=(sv[0],float(sv[1])) if isinstance(sv,tuple) else (str(sv),0.)
            if bn is None or abs(vv)>abs(bv): bn,bv,bd=sn,vv,sd
        out.append({"key":k,"label":names.get(k,k.capitalize()),"score":s,
                    "dir":"BUY" if s>0 else "SELL","reason":bn or names.get(k,k),"detail":bd})
    out.sort(key=lambda x:abs(x["score"]),reverse=True)
    return out[:limit]

# ── Analysis Engine ────────────────────────────────────────
class TechAnalyzer:
    def __init__(self,df): self.df=df.copy(); self._run()
    def _s(self,fn):
        try:    return fn()
        except: return pd.Series(np.nan,index=self.df.index)
    def _run(self):
        c,h,lo,v=self.df["Close"],self.df["High"],self.df["Low"],self.df["Volume"]
        for n in [20,50,200]: self.df[f"SMA{n}"]=self._s(lambda n=n:SMAIndicator(c,n).sma_indicator())
        self.df["EMA12"]=self._s(lambda:EMAIndicator(c,12).ema_indicator())
        try:
            m=MACD(c); self.df["MACD"]=m.macd(); self.df["MACDs"]=m.macd_signal(); self.df["MACDh"]=m.macd_diff()
        except: self.df["MACD"]=self.df["MACDs"]=self.df["MACDh"]=np.nan
        self.df["RSI"]=self._s(lambda:RSIIndicator(c,14).rsi())
        try:
            bb=BollingerBands(c); self.df["BBhi"]=bb.bollinger_hband(); self.df["BBlo"]=bb.bollinger_lband()
            self.df["BBpct"]=bb.bollinger_pband(); self.df["BBmid"]=bb.bollinger_mavg()
        except: self.df["BBhi"]=self.df["BBlo"]=self.df["BBpct"]=self.df["BBmid"]=np.nan
        self.df["OBV"]=self._s(lambda:OnBalanceVolumeIndicator(c,v).on_balance_volume())
        self.df["ATR"]=self._s(lambda:AverageTrueRange(h,lo,c).average_true_range())
        for n,l in [(21,"1m"),(63,"3m"),(126,"6m")]: self.df[f"M{l}"]=self._s(lambda n=n:c.pct_change(n))
    def score(self):
        df=self.df.dropna(subset=["Close"])
        if not len(df): return {"score":0.,"signals":{}}
        r=df.iloc[-1]; p=r["Close"]; sigs={}; sc_=[]
        s50,s200=r.get("SMA50",np.nan),r.get("SMA200",np.nan)
        if not any(np.isnan(x) for x in [s50,s200]):
            if s50>s200: sigs["Trend is up (short-term average above long-term)"]=("Positive",0.80); sc_.append(0.80)
            else:        sigs["Trend is down (short-term average below long-term)"]=("Negative",-0.80); sc_.append(-0.80)
        above=sum(1 for k in ["SMA20","SMA50","SMA200"] if k in r and not np.isnan(r[k]) and p>r[k])
        mas=clamp((above/3)*2-1); sigs[f"Price is above {above} of 3 key averages"]=("Positive" if mas>0 else "Negative",mas); sc_.append(mas)
        rsi=float(r.get("RSI",50)); rsi=50 if np.isnan(rsi) else rsi
        if rsi<30:   rs=0.9;               sigs[f"Looks oversold ({rsi:.0f}/100) — could bounce"]=("Positive",rs)
        elif rsi>70: rs=-0.9;              sigs[f"Looks overbought ({rsi:.0f}/100) — could pull back"]=("Negative",rs)
        else:        rs=clamp((50-rsi)/50*.4); sigs[f"Momentum is balanced ({rsi:.0f}/100)"]=("Neutral",rs)
        sc_.append(rs)
        macd,macds,macdh=r.get("MACD",0),r.get("MACDs",0),r.get("MACDh",0)
        if not any(np.isnan(x) for x in [macd,macds,macdh]):
            ms=clamp(abs(macdh)/(abs(macd)+1e-9))*(1 if macd>macds else -1)
            sigs["Momentum turning "+("up" if ms>0 else "down")]=("Positive" if ms>0 else "Negative",ms); sc_.append(ms)
        bbp=float(r.get("BBpct",0.5)); bbp=0.5 if np.isnan(bbp) else bbp
        if bbp<.2:   bs=0.7;               sigs["Price stretched below its usual range"]=("Positive",bs)
        elif bbp>.8: bs=-0.7;              sigs["Price stretched above its usual range"]=("Negative",bs)
        else:        bs=clamp((.5-bbp)*1.4); sigs["Price within its usual range"]=("Neutral",bs)
        sc_.append(bs)
        moms=[clamp(r.get(f"M{l}",0)*5) for l in ["1m","3m","6m"] if not np.isnan(r.get(f"M{l}",np.nan))]
        if moms: ms2=clamp(np.mean(moms)); sigs["Recent momentum (1–6 months)"]=("Positive" if ms2>0 else "Negative",ms2); sc_.append(ms2)
        return {"score":clamp(np.mean(sc_)) if sc_ else 0.,"signals":sigs,"rsi":rsi,"bbp":bbp}

class NewsSentiment:
    BULL=["beat","surge","record","upgrade","profit","strong","rally","buy","outperform","soar","growth","exceed","raise"]
    BEAR=["miss","decline","loss","downgrade","lawsuit","fraud","sell","underperform","cut","fall","crash","warning","layoff","fine"]
    def __init__(self,items): self.items=items; self.v=SentimentIntensityAnalyzer()
    def _s(self,t):
        b=self.v.polarity_scores(t)["compound"]; l=t.lower()
        return clamp(b+sum(.07 for w in self.BULL if w in l)-sum(.07 for w in self.BEAR if w in l))
    def score(self):
        sc,hl=[],[]
        for i in self.items:
            t=news_title(i)
            if not t: continue
            s=self._s(t); sc.append(s); hl.append({"title":t,"score":round(s,2),"sent":"Positive" if s>.1 else "Negative" if s<-.1 else "Neutral"})
        if not sc: return {"score":0.,"headlines":[],"bull":0,"bear":0,"total":0}
        ws=clamp(float(np.average(sc,weights=np.linspace(1,.4,len(sc)))))
        return {"score":ws,"headlines":hl[:12],"bull":sum(1 for s in sc if s>.1),"bear":sum(1 for s in sc if s<-.1),"total":len(sc)}

class LunarAnalyzer:
    P=[(0,.06,"New Moon",.60),(.06,.19,"Waxing Crescent",.30),(.19,.31,"First Quarter",.10),(.31,.44,"Waxing Gibbous",-.10),(.44,.56,"Full Moon",-.50),(.56,.69,"Waning Gibbous",-.20),(.69,.81,"Last Quarter",.00),(.81,.94,"Waning Crescent",.30),(.94,1.,"Balsamic",.50)]
    def score(self):
        try:
            now=datetime.now(); d=ephem.Date(now)
            p=float(d-ephem.previous_new_moon(d))/float(ephem.next_new_moon(d)-ephem.previous_new_moon(d))
            name,bs=next(((n,s) for lo,hi,n,s in self.P if lo<=p<hi),("Balsamic",.5))
            dn=float(ephem.next_new_moon(d)-d); df_=float(ephem.next_full_moon(d)-d)
            prox=(.25*(1-dn/4) if dn<df_ and dn<4 else -.15*(1-df_/4) if df_<dn and df_<4 else 0)
            return {"score":clamp(bs+prox),"phase":name,"illum":f"{p*100:.0f}%","d_new":round(dn,1),"d_full":round(df_,1),
                    "signals":{f"Moon phase: {name}":(f"{p*100:.0f}% lit",clamp(bs)),"Days to next new moon":(f"about {round(dn,1)} days away",clamp(.25*(1-dn/7)))}}
        except: return {"score":0.,"phase":"Unknown","illum":"?","d_new":0,"d_full":0,"signals":{}}

class PsychAnalyzer:
    def __init__(self,vix,fund,df): self.vix=vix; self.f=fund; self.df=df
    def score(self):
        v=self.vix; vs=(.90 if v>40 else .50 if v>30 else .00 if v>20 else -.30 if v>15 else -.65)
        vl=(f"High fear right now (gauge {v:.0f}) — often a buying opportunity" if v>30 else f"Very calm / complacent (gauge {v:.0f}) — stay cautious" if v<15 else f"Normal mood (fear gauge {v:.0f})")
        sr=self.f.get("short_ratio"); ss=(.70 if sr and sr>10 else .25 if sr and sr>5 else -.10)
        sl=(f"Heavily bet against ({sr:.1f} days to cover) — could spike if it rises" if sr and sr>5 else f"Lightly bet against ({sr:.1f} days)" if sr else "No short-interest data")
        io=self.f.get("insider_ownership"); is_=(.60 if io and io>.20 else .20 if io and io>.05 else -.20)
        il=(f"Insiders own {io*100:.1f}% — interests aligned with shareholders" if io else "No insider-ownership data")
        try:
            rets=self.df["Close"].pct_change().dropna(); y=self.df["Close"].iloc[-30:].values; x=np.arange(len(y))
            slope,_,rr,_,_=stats.linregress(x,y); rv=rets.rolling(20).std().iloc[-1]*np.sqrt(252)
            hs=clamp(np.sign(slope)*rr**2*max(0.,1.-rv)); hl=("Crowd is chasing the rise" if hs>.3 else "Crowd is selling in fear" if hs<-.3 else "No clear crowd direction")
        except: hs=0.; hl="Not enough data"
        try:
            c=self.df["Close"]; hi52=c.rolling(252).max().iloc[-1]; lo52=c.rolling(252).min().iloc[-1]
            pos=(c.iloc[-1]-lo52)/(hi52-lo52+1e-9); vs2=self.df["Volume"].iloc[-20:].mean()/(self.df["Volume"].iloc[-60:-20].mean()+1e-9)
            if pos<.25 and vs2>1.1: wy=.70; wl=f"Quiet buying near the lows ({pos:.0%} of 1-yr range)"
            elif pos<.25:           wy=.35; wl=f"Near its 1-year low — possible bottom ({pos:.0%} of range)"
            elif pos<.75:           wy=.15; wl=f"Trading in the middle of its 1-year range ({pos:.0%})"
            elif vs2>1.2:           wy=-.70; wl=f"Heavy selling into strength ({pos:.0%} of range)"
            else:                   wy=-.30; wl=f"Near its 1-year high — stretched ({pos:.0%} of range)"
        except: wy=0.; wl="Not enough data"
        sigs={"Fear vs. greed":(vl,vs),"Bets against the stock":(sl,ss),"Insider ownership":(il,is_),"Crowd behavior":(hl,hs),"Buying vs. selling pressure":(wl,wy)}
        return {"score":clamp(float(np.average([vs,ss,is_,hs,wy],weights=[.30,.15,.15,.20,.20]))),"signals":sigs}

class FundAnalyzer:
    def __init__(self,f): self.f=f
    def score(self):
        f=self.f; sigs={}; sc=[]
        pe,fpe=f.get("pe_ratio") or 0,f.get("forward_pe") or 0
        if pe>0 and fpe>0:
            s=clamp((pe-fpe)/pe*5+(-0.4 if fpe>35 else 0.4 if fpe<15 else 0))
            sigs["Valuation (price vs. earnings)"]=(f"Paying {fpe:.1f}× next year's earnings {'(pricey)' if fpe>35 else '(reasonable)' if fpe<15 else ''}",s); sc.append((s,.20))
        rg,eg=f.get("revenue_growth") or 0,f.get("earnings_growth") or 0; gs=clamp((rg+eg)*3)
        sigs["Sales & profit growth"]=(f"Sales {rg*100:+.1f}%, profit {eg*100:+.1f}% vs. a year ago",gs); sc.append((gs,.25))
        dte=f.get("debt_to_equity") or 0; ds=(.7 if dte<.5 else .2 if dte<1.5 else -.3 if dte<3 else -.7)
        sigs["Debt load"]=(f"{'Low debt' if dte<.5 else 'High debt' if dte>3 else 'Moderate debt'} ({dte:.2f}× equity)",ds); sc.append((ds,.15))
        tgt,curr=f.get("analyst_target"),f.get("current_price")
        if tgt and curr and curr>0:
            up=(tgt-curr)/curr; as_=clamp(up*4); sigs["Analyst price target"]=(f"Analysts see ${tgt:,.2f} ({up*100:+.1f}% from today)",as_); sc.append((as_,.25))
        roe=f.get("roe") or 0; rs=clamp(roe*5); sigs["Profitability (return on equity)"]=(f"Earns {roe*100:.1f}% on shareholder money",rs); sc.append((rs,.15))
        if not sc: return {"score":0.,"signals":{}}
        tw=sum(w for _,w in sc); return {"score":clamp(sum(s*w for s,w in sc)/tw),"signals":sigs}

class HistAnalyzer:
    M={1:.40,2:.10,3:.05,4:.20,5:-.30,6:-.10,7:.15,8:-.20,9:-.45,10:.10,11:.40,12:.50}
    D={0:-.15,1:.0,2:.10,3:.05,4:.20}
    def __init__(self,df): self.df=df; self.now=datetime.now()
    def score(self):
        ms=self.M.get(self.now.month,0); ds=self.D.get(self.now.weekday(),0); sea=clamp(ms*.7+ds*.3)
        c=self.df["Close"]
        rev=(0. if len(c)<60 else clamp(-(c.iloc[-1]-c.rolling(60).mean().iloc[-1])/(c.rolling(60).std().iloc[-1]+1e-9)/2.5))
        try:
            rets=c.pct_change().dropna(); lag1,_=stats.pearsonr(rets.iloc[:-1].values,rets.iloc[1:].values)
            acr=clamp(lag1*np.sign(float(rets.iloc[-5:].sum()))*abs(float(rets.iloc[-5:].sum()))*15)
        except: acr=0.
        try:
            x=np.arange(len(c.iloc[-200:])); sl,_,rr,_,_=stats.linregress(x,c.iloc[-200:].values); trd=clamp(sl/(c.mean()+1e-9)*600)
        except: trd=0.
        month=self.now.strftime('%B')
        sigs={f"Time of year ({month})":(f"{'Historically a strong' if ms>.1 else 'Historically a weak' if ms<-.1 else 'A neutral'} month for stocks",sea),
              "Distance from recent average":("How far price sits from its 3-month average",rev),"Momentum staying power":("Whether the recent move tends to continue",acr),
              "Long-term trend":(f"{'Rising' if trd>0 else 'Falling'} over the past ~10 months",trd)}
        return {"score":clamp(float(np.average([sea,rev,acr,trd],weights=[.20,.25,.25,.30]))),"signals":sigs}

# ── Data Fetching ──────────────────────────────────────────
@st.cache_data(ttl=60,show_spinner=False)
def fetch_market():
    indices={"S&P 500":"^GSPC","Nasdaq":"^IXIC","Dow Jones":"^DJI","Russell 2000":"^RUT","VIX":"^VIX","10Y Yield":"^TNX","Gold":"GC=F","Oil":"CL=F"}
    def _one(name,sym):
        try:
            h=yf.Ticker(sym).history(period="5d")["Close"]; cur=float(h.iloc[-1]); prev=float(h.iloc[-2]) if len(h)>1 else cur
            return name,{"val":cur,"chg":(cur-prev)/prev*100}
        except: return name,{"val":None,"chg":None}
    out={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for n,d in ex.map(lambda x:_one(*x),indices.items()): out[n]=d
    return {k:out.get(k,{"val":None,"chg":None}) for k in indices}

@st.cache_data(ttl=120,show_spinner=False)
def quick_quote(ticker):
    try:
        info=yf.Ticker(ticker).info; p=info.get("currentPrice") or info.get("regularMarketPrice") or 0
        prev=info.get("regularMarketPreviousClose") or p; chg=(p-prev)/prev*100 if prev else 0
        return {"price":p,"chg":chg,"name":info.get("shortName",ticker)}
    except: return {"price":0,"chg":0,"name":ticker}

@st.cache_data(ttl=300,show_spinner=False)
def fetch_price_df(ticker,period="2y"):
    try:
        df=yf.Ticker(ticker).history(period=period)
        if df.empty: return pd.DataFrame()
        df.index=pd.to_datetime(df.index).tz_localize(None); return df
    except: return pd.DataFrame()

@st.cache_data(ttl=3600,show_spinner=False)
def fetch_insider_data(ticker:str):
    result={"transactions":[],"source":"none","net_shares":0}
    def _classify(row):
        code=str(row.get("Transaction","") or "").strip().upper()
        if code in ("P","A"): return "Buy"
        if code in ("S","S-","D"): return "Sale"
        text=str(row.get("Text","") or "").lower()
        if any(w in text for w in ["purchase","bought","acquisition"]): return "Buy"
        if any(w in text for w in ["sale","sold","disposition"]): return "Sale"
        return "Other"
    try:
        df=yf.Ticker(ticker).insider_transactions
        if df is not None and not df.empty:
            rows=[]
            for _,row in df.iterrows():
                try:
                    shares=int(row.get("Shares",0) or 0); value=float(row.get("Value",0) or 0)
                    txn_type=_classify(row); dv=row.get("Start Date") or row.get("Date")
                    date_str=(str(dv.date()) if hasattr(dv,"date") else str(dv)[:10] if dv else "—")
                    rows.append({"date":date_str,"insider":str(row.get("Insider","") or ""),"title":str(row.get("Position","") or ""),"type":txn_type,"shares":shares,"value":value,"url":str(row.get("URL","") or "")})
                except: continue
            if rows:
                result.update({"transactions":rows,"source":"yfinance",
                               "net_shares":sum(r["shares"] for r in rows if r["type"]=="Buy")-sum(r["shares"] for r in rows if r["type"]=="Sale")})
                return result
    except: pass
    try:
        end=datetime.now(); start=end-timedelta(days=90)
        resp=requests.get(f"https://efts.sec.gov/LATEST/search-index?q={ticker}&forms=4&dateRange=custom&startdt={start.strftime('%Y-%m-%d')}&enddt={end.strftime('%Y-%m-%d')}",
                          headers={"User-Agent":"StockOracle/3.0 contact@stockoracle.app"},timeout=8)
        if resp.status_code==200:
            hits=resp.json().get("hits",{}).get("hits",[])
            rows=[{"date":h.get("_source",{}).get("file_date","—")[:10],"insider":(h.get("_source",{}).get("display_names",["—"])[0]),"title":"—","type":"Form 4","shares":0,"value":0,"url":f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=4&dateb=&owner=include&count=40"} for h in hits[:20]]
            if rows: result.update({"transactions":rows,"source":"SEC EDGAR"})
    except: pass
    return result

@st.cache_data(ttl=3600,show_spinner=False)
def fetch_earnings_data(ticker:str):
    result={"calendar":{},"history":[],"has_data":False}
    try:
        stk=yf.Ticker(ticker); cal=stk.calendar
        if cal is not None: result["calendar"]=cal if isinstance(cal,dict) else cal.to_dict()
        hist=stk.earnings_history
        if hist is not None and not hist.empty:
            rows=[]
            for _,row in hist.iterrows():
                try:
                    rep=row.get("Reported EPS") or row.get("epsActual"); est=row.get("EPS Estimate") or row.get("epsEstimate"); surp=row.get("Surprise(%)") or row.get("surprisePercent")
                    if rep is None and est is None: continue
                    rows.append({"date":str(row.get("Earnings Date",""))[:10],"actual":float(rep) if rep is not None else None,"estimate":float(est) if est is not None else None,"surprise":float(surp) if surp is not None else None})
                except: continue
            result["history"]=rows[-8:]
        result["has_data"]=True
    except: pass
    return result

@st.cache_data(ttl=600,show_spinner=False)
def fetch_social_trending():
    """Most talked-about stocks right now. Primary source is StockTwits (a retail-investor
    social network) whose 'trending' API works from cloud servers where Reddit is blocked.
    Falls back to the day's biggest movers so the tab always shows something useful."""
    # 1) StockTwits trending symbols
    try:
        resp=requests.get("https://api.stocktwits.com/api/2/trending/symbols.json",
                          headers={"User-Agent":"Mozilla/5.0 (StockOracle)"},timeout=6)
        if resp.status_code==200:
            syms=resp.json().get("symbols",[])
            items=[]
            for s in syms:
                tk=str(s.get("symbol","")).replace(".","-").upper()
                if not tk or any(c in tk for c in (" ","/","^","=")): continue
                items.append({"ticker":tk,"name":str(s.get("title","") or TICKER_NAMES.get(tk,tk))[:40],
                              "watchers":int(s.get("watchlist_count",0) or 0)})
                if len(items)>=20: break
            if items: return {"source":"StockTwits","items":items}
    except: pass
    # 2) Fallback: biggest movers today (always works via yfinance)
    try:
        tl=fetch_top_lists(); movers=(tl.get("up",[])+tl.get("down",[]))
        items=[{"ticker":m["ticker"],"name":m.get("name",m["ticker"]),"watchers":0,"chg":m.get("chg",0)} for m in movers][:16]
        if items: return {"source":"Movers","items":items}
    except: pass
    return {"source":"none","items":[]}

@st.cache_data(ttl=900,show_spinner=False)
def fetch_market_news(n=4):
    """A few brief market-moving headlines (from broad-market tickers)."""
    out,seen=[],set()
    for sym in ["SPY","^GSPC","QQQ"]:
        try:
            for it in (yf.Ticker(sym).news or []):
                t=news_title(it).strip()
                if not t or t.lower() in seen: continue
                seen.add(t.lower()); out.append({"title":t,"link":news_link(it)})
                if len(out)>=n: return out
        except: continue
    return out

@st.cache_data(ttl=300,show_spinner=False)
def fetch_movers():
    """Top 5 gainers & losers (by 1-day % change) across a liquid universe."""
    try:
        data=yf.download(MOVERS_UNIVERSE,period="2d",progress=False,group_by="ticker",threads=True)
        rows=[]
        for tk in MOVERS_UNIVERSE:
            try:
                cl=data[tk]["Close"].dropna()
                if len(cl)<2: continue
                last,prev=float(cl.iloc[-1]),float(cl.iloc[-2])
                if prev<=0: continue
                rows.append({"ticker":tk,"price":last,"chg":(last-prev)/prev*100,"name":TICKER_NAMES.get(tk,tk)})
            except: continue
        if not rows: return {"up":[],"down":[]}
        rows.sort(key=lambda x:x["chg"],reverse=True)
        return {"up":rows[:5],"down":list(reversed(rows[-5:]))}
    except: return {"up":[],"down":[]}

@st.cache_data(ttl=86400,show_spinner=False)
def load_all_symbols():
    """Full list of US-listed symbols (NASDAQ + NYSE/AMEX) from the official Nasdaq
    Trader directory, so the search box covers every stock & ETF. Falls back to the
    curated list if offline. Cached for a day."""
    names=dict(TICKER_NAMES)  # curated nice names + ETFs take precedence
    hdrs={"User-Agent":"StockOracle/3.0"}
    for url in ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"):
        try:
            lines=requests.get(url,headers=hdrs,timeout=10).text.splitlines()
            if not lines: continue
            head=lines[0].split("|"); ti=head.index("Test Issue") if "Test Issue" in head else -1
            for ln in lines[1:]:
                if ln.startswith("File Creation Time"): continue
                p=ln.split("|")
                if len(p)<2: continue
                sym,nm=p[0].strip(),p[1].strip()
                if not sym or not nm or len(sym)>6: continue
                if ti>=0 and ti<len(p) and p[ti].strip()=="Y": continue   # skip test issues
                if any(ch in sym for ch in (" ","$","/")): continue        # skip exotic tickers
                nm=re.split(r"\s+-\s+",nm)[0].strip()[:48]                 # trim "- Common Stock" etc.
                names.setdefault(sym.replace(".","-"),nm)                  # yfinance uses '-' for class shares
        except: continue
    return names

@st.cache_data(ttl=600,show_spinner=False)
def fetch_top_lists():
    """One batched download powering the top bar: curated picks, momentum-ranked
    'potential picks', and the day's gainers/losers."""
    uni=SCAN_UNIVERSE
    out={"picks":[],"potential":[],"up":[],"down":[]}
    try:
        data=yf.download(uni,period="3mo",progress=False,group_by="ticker",threads=True)
        info={}
        for tk in uni:
            try:
                cl=data[tk]["Close"].dropna()
                if len(cl)<2: continue
                last=float(cl.iloc[-1]); prev=float(cl.iloc[-2])
                if prev<=0: continue
                day=(last-prev)/prev*100
                r1=(last/float(cl.iloc[-21])-1) if len(cl)>21 else 0.       # ~1 month
                r3=(last/float(cl.iloc[-63])-1) if len(cl)>63 else 0.       # ~3 months
                ma50=float(cl.iloc[-50:].mean()) if len(cl)>=50 else float(cl.mean())
                trend=1. if last>ma50 else -1.
                score=clamp((r1+r3)*2.5*.6+trend*.4)
                info[tk]={"ticker":tk,"price":last,"chg":day,"name":TICKER_NAMES.get(tk,tk),"score":score}
            except: continue
        if not info: return out
        movers=sorted(info.values(),key=lambda x:x["chg"],reverse=True)
        out["up"]=movers[:8]; out["down"]=list(reversed(movers[-8:]))
        out["picks"]=[info[t] for t in PICKS_UNIVERSE if t in info][:12]
        out["potential"]=sorted([v for v in info.values() if v["score"]>.1],key=lambda x:x["score"],reverse=True)[:8]
    except: pass
    return out

@st.cache_data(ttl=300,show_spinner=False)
def compute_core(ticker:str,period:str)->dict:
    """Heavy work: fetch data + score all six modules. Cached by ticker+period only,
    so changing signal weights re-uses this instantly (see combine())."""
    stk=yf.Ticker(ticker)
    def _info():
        try: return stk.info
        except: return {}
    def _price(): return fetch_price_df(ticker,period)
    def _vix():
        try: return float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
        except: return 20.
    def _spy():
        try: s=yf.Ticker("SPY").history(period="3mo")["Close"]; return (s.iloc[-1]/s.iloc[0]-1)*100
        except: return 0.
    def _news():
        try: return stk.news[:25]
        except: return []
    with ThreadPoolExecutor(max_workers=5) as ex:
        fi=ex.submit(_info); fp=ex.submit(_price); fv=ex.submit(_vix); fs=ex.submit(_spy); fn=ex.submit(_news)
        info=fi.result(); df_raw=fp.result(); vix=fv.result(); spy=fs.result(); news_raw=fn.result()
    if df_raw.empty: raise ValueError(f"No price data for '{ticker}'. Is the ticker symbol correct?")
    fund={"company_name":info.get("longName",""),"sector":info.get("sector","—"),"industry":info.get("industry","—"),
          "market_cap":info.get("marketCap"),"current_price":info.get("currentPrice") or info.get("regularMarketPrice"),
          "analyst_target":info.get("targetMeanPrice"),"pe_ratio":info.get("trailingPE"),"forward_pe":info.get("forwardPE"),
          "peg_ratio":info.get("pegRatio"),"price_to_book":info.get("priceToBook"),
          "debt_to_equity":(info.get("debtToEquity")/100 if info.get("debtToEquity") is not None else None),
          "profit_margin":info.get("profitMargins"),"revenue_growth":info.get("revenueGrowth"),"earnings_growth":info.get("earningsGrowth"),
          "roe":info.get("returnOnEquity"),"current_ratio":info.get("currentRatio"),"beta":info.get("beta"),
          "short_ratio":info.get("shortRatio"),"insider_ownership":info.get("heldPercentInsiders"),
          "dividend_yield":(info.get("dividendYield")/100 if info.get("dividendYield") is not None else None),
          "eps_ttm":info.get("trailingEps"),"52wh":info.get("fiftyTwoWeekHigh"),
          "52wl":info.get("fiftyTwoWeekLow"),"avg_vol":info.get("averageVolume"),"description":info.get("longBusinessSummary","")}
    ta=TechAnalyzer(df_raw); edf=ta.df
    mods={"technical":ta.score(),"fundamental":FundAnalyzer(fund).score(),"news":NewsSentiment(news_raw).score(),
          "psychology":PsychAnalyzer(vix,fund,edf).score(),"historical":HistAnalyzer(edf).score(),"lunar":LunarAnalyzer().score()}
    price=fund["current_price"] or (float(df_raw["Close"].iloc[-1]) if len(df_raw) else 0)
    prev=info.get("regularMarketPreviousClose") or (float(df_raw["Close"].iloc[-2]) if len(df_raw)>1 else price)
    quote={"price":price or 0,"chg":((price-prev)/prev*100) if prev else 0.,"name":info.get("shortName") or info.get("longName") or ticker.upper()}
    return {"ticker":ticker.upper(),"modules":mods,"df":edf,"fund":fund,"vix":vix,"spy":spy,"quote":quote,
            "ts":datetime.now().strftime("%Y-%m-%d %H:%M")}

def combine(core:dict,weights)->dict:
    """Cheap: re-weight the cached module scores into a composite verdict (instant)."""
    w=dict(weights); tw=max(sum(w.values()),1e-9); wn={k:v/tw for k,v in w.items()}
    mods=core["modules"]; comp=clamp(sum(mods[k]["score"]*wn.get(k,0) for k in mods)); vl,vc,vi=verdict(comp)
    return {**core,"composite":comp,"verdict":vl,"vcolor":vc,"vicon":vi,"plain":plain_verdict(comp),"weights":wn}

def run_analysis(ticker:str,period:str,weights)->dict:
    return combine(compute_core(ticker,period),weights)

# ── Charts ─────────────────────────────────────────────────
def price_chart(df_full,ticker,rng,ctype,opts):
    try:
        df=slice_df(df_full,rng)
        show_rsi=bool(opts.get("rsi")) and "RSI" in df.columns
        show_vol=bool(opts.get("vol"))
        panels=["price"]+(["rsi"] if show_rsi else [])+(["vol"] if show_vol else [])
        idx={n:i+1 for i,n in enumerate(panels)}
        hmap={"price":.62 if (show_rsi or show_vol) else 1.,"rsi":.2,"vol":.2}
        hs=[hmap[n] for n in panels]; tot=sum(hs); hs=[h/tot for h in hs]
        fig=make_subplots(rows=len(panels),cols=1,shared_xaxes=True,row_heights=hs,vertical_spacing=(.04 if len(panels)>1 else 0))
        pr=idx["price"]
        if ctype=="Candle":
            fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],name="Price",showlegend=False,increasing=dict(line_color=G,fillcolor=G),decreasing=dict(line_color=R,fillcolor=R),line_width=1),row=pr,col=1)
        else:
            lc=G if df["Close"].iloc[-1]>=df["Close"].iloc[0] else R; rgb=GR if lc==G else RR
            kw=dict(x=df.index,y=df["Close"],line=dict(color=lc,width=2),name="Price",showlegend=False)
            if ctype=="Area": kw.update(fill="tozeroy",fillcolor=f"rgba({rgb},.08)")
            fig.add_trace(go.Scatter(**kw),row=pr,col=1)
        for col_,color,name,show in [("SMA20",GLD,"MA 20",opts.get("sma20")),("SMA50",ORG,"MA 50",opts.get("sma50")),("SMA200",PRP,"MA 200",opts.get("sma200"))]:
            if show and col_ in df.columns: fig.add_trace(go.Scatter(x=df.index,y=df[col_],name=name,line=dict(color=color,width=1.3),opacity=.85),row=pr,col=1)
        if opts.get("bb") and "BBhi" in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df["BBhi"],name="BB Upper",line=dict(color=BLU,width=1,dash="dot"),opacity=.5),row=pr,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=df["BBlo"],name="BB Lower",line=dict(color=BLU,width=1,dash="dot"),fill="tonexty",fillcolor="rgba(37,99,235,.05)",showlegend=False,opacity=.5),row=pr,col=1)
        ax={("yaxis" if pr==1 else f"yaxis{pr}"):dict(gridcolor=CGRID,tickformat="$,.2f",side="right")}
        if show_rsi:
            rw=idx["rsi"]; rsi=df["RSI"].dropna()
            fig.add_trace(go.Scatter(x=rsi.index,y=rsi,name="RSI",line=dict(color=PRP,width=1.5),showlegend=False),row=rw,col=1)
            for y_,cc in [(70,R),(30,G)]: fig.add_shape(type="line",x0=rsi.index[0],x1=rsi.index[-1],y0=y_,y1=y_,line=dict(color=cc,width=1,dash="dot"),row=rw,col=1)
            ax[f"yaxis{rw}"]=dict(gridcolor=CGRID,range=[0,100],showgrid=False,side="right",tickvals=[30,50,70],ticktext=["30","50","70"])
        if show_vol:
            vw=idx["vol"]; vc_=[f"rgba({GR if c>=o else RR},.55)" for c,o in zip(df["Close"],df["Open"])]
            fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vc_,name="Vol",showlegend=False),row=vw,col=1)
            ax[f"yaxis{vw}"]=dict(gridcolor=CGRID,showgrid=False,showticklabels=False)
        h=360+(150 if show_rsi else 0)+(120 if show_vol else 0)
        fig.update_layout(**CBASE,height=h,legend=LEG_H,xaxis_rangeslider_visible=False,**ax)
        for i in range(1,len(panels)+1): fig.update_xaxes(gridcolor=CGRID,showgrid=False,row=i,col=1)
        return fig
    except: return None

def macd_chart(df_full,rng):
    try:
        df=slice_df(df_full,rng)
        if not all(c in df.columns for c in ["MACD","MACDs","MACDh"]): return None
        d=df[["MACD","MACDs","MACDh"]].dropna()
        if not len(d): return None
        hc=[f"rgba({GR if v>=0 else RR},.75)" for v in d["MACDh"]]
        fig=go.Figure()
        fig.add_trace(go.Bar(x=d.index,y=d["MACDh"],marker_color=hc,showlegend=False))
        fig.add_trace(go.Scatter(x=d.index,y=d["MACD"],name="MACD",line=dict(color=BLU,width=1.8)))
        fig.add_trace(go.Scatter(x=d.index,y=d["MACDs"],name="Signal",line=dict(color=GLD,width=1.4,dash="dot")))
        fig.add_shape(type="line",x0=d.index[0],x1=d.index[-1],y0=0,y1=0,line=dict(color=SUB,width=1))
        fig.update_layout(**CBASE,height=200,legend=LEG_H,yaxis=dict(gridcolor=CGRID,side="right"),xaxis=dict(gridcolor=CGRID,showgrid=False))
        return fig
    except: return None

def gauge_chart(score,vl,vc_):
    try:
        fig=go.Figure(go.Indicator(mode="gauge+number",value=score*100,
            number=dict(suffix="",font=dict(color=vc_,size=32,family="Inter"),valueformat="+.0f"),
            title=dict(text=vl,font=dict(color=vc_,size=14,family="Inter")),
            gauge=dict(axis=dict(range=[-100,100],tickvals=[-100,-60,-30,0,30,60,100],ticktext=["Strong<br>Sell","Sell","Weak<br>Sell","Hold","Weak<br>Buy","Buy","Strong<br>Buy"],tickcolor=MUT,tickfont=dict(color=MUT,size=8)),
                       bar=dict(color=vc_,thickness=.20),bgcolor=CARD,borderwidth=0,
                       steps=[dict(range=[-100,-30],color=f"rgba({RR},.12)"),dict(range=[-30,30],color="rgba(0,0,0,0)"),dict(range=[30,100],color=f"rgba({GR},.12)")])))
        fig.update_layout(paper_bgcolor=CBG,font_color=MUT,height=270,margin=dict(l=16,r=16,t=48,b=8))
        return fig
    except: return None

def radar_chart(mods):
    try:
        cats=[MOD_I[k]+" "+MOD_NAME.get(k,k) for k in MOD_C]; vals=[mods[k]["score"] for k in MOD_C]; cats.append(cats[0]); vals.append(vals[0])
        pos=[max(0,v) for v in vals]; neg=[abs(min(0,v)) for v in vals]
        fig=go.Figure()
        fig.add_trace(go.Scatterpolar(r=pos,theta=cats,fill="toself",name="Positive",fillcolor=f"rgba({GR},.18)",line=dict(color=G,width=2)))
        fig.add_trace(go.Scatterpolar(r=neg,theta=cats,fill="toself",name="Negative",fillcolor=f"rgba({RR},.12)",line=dict(color=R,width=2)))
        fig.update_layout(polar=dict(bgcolor=CBG,radialaxis=dict(visible=True,range=[0,1],gridcolor=CGRID,tickfont=dict(color=MUT,size=9)),angularaxis=dict(gridcolor=CGRID,tickfont=dict(color=MUT,size=11))),paper_bgcolor=CBG,font_color=MUT,legend=LEG_V,height=300,margin=dict(l=48,r=48,t=24,b=24))
        return fig
    except: return None

def module_bars(mods,weights):
    try:
        keys=list(weights.keys()); vals=[mods[k]["score"] for k in keys]; colors=[MOD_C[k] for k in keys]
        fig=go.Figure(go.Bar(x=vals,y=[f"{MOD_I[k]} {MOD_NAME.get(k,k)}" for k in keys],orientation="h",marker_color=colors,opacity=.88,text=[f"{v:+.2f}" for v in vals],textposition="outside",textfont=dict(color=TXT,size=11)))
        fig.add_vline(x=0,line_color=MUT,line_width=1)
        fig.update_layout(**CBASE,height=260,xaxis=dict(range=[-1.15,1.15],gridcolor=CGRID,showgrid=True),yaxis=dict(gridcolor=CGRID,showgrid=False))
        return fig
    except: return None

def mini_price_chart(ticker):
    try:
        df=yf.Ticker(ticker).history(period="1mo")["Close"]; df.index=pd.to_datetime(df.index).tz_localize(None)
        color=G if float(df.iloc[-1])>=float(df.iloc[0]) else R; rgb=GR if color==G else RR
        fig=go.Figure(go.Scatter(x=df.index,y=df.values,mode="lines",line=dict(color=color,width=2),fill="tozeroy",fillcolor=f"rgba({rgb},.08)"))
        fig.update_layout(paper_bgcolor=CBG,plot_bgcolor=CBG,margin=dict(l=0,r=0,t=0,b=0),height=55,xaxis=dict(visible=False),yaxis=dict(visible=False),showlegend=False)
        return fig
    except: return None

def insider_chart(transactions):
    try:
        rows=[t for t in transactions if t.get("shares",0)!=0]
        if not rows: return None
        df=pd.DataFrame(rows); df["date_parsed"]=pd.to_datetime(df["date"],errors="coerce")
        df=df.dropna(subset=["date_parsed"]).sort_values("date_parsed")
        df["signed"]=df.apply(lambda r:r["shares"] if r["type"]=="Buy" else -r["shares"],axis=1)
        fig=go.Figure(go.Bar(x=df["date_parsed"],y=df["signed"],marker_color=[G if v>0 else R for v in df["signed"]],opacity=.85,text=[f"{abs(v):,}" for v in df["signed"]],textposition="outside",textfont=dict(color=TXT,size=9)))
        fig.add_hline(y=0,line_color=MUT,line_width=1)
        fig.update_layout(**CBASE,height=220,yaxis=dict(gridcolor=CGRID,side="right",tickformat=","),xaxis=dict(gridcolor=CGRID,showgrid=False),title=dict(text="Insider Buy / Sell Volume",font=dict(color=TXT,size=13)))
        return fig
    except: return None

def earnings_chart(history):
    try:
        valid=[h for h in history if h.get("actual") is not None or h.get("estimate") is not None]
        if not valid: return None
        dates=[h["date"] for h in valid]; acts=[h.get("actual") for h in valid]; ests=[h.get("estimate") for h in valid]
        bc=[G if(a is not None and e is not None and a>=e) else R if(a is not None and e is not None) else MUT for a,e in zip(acts,ests)]
        fig=go.Figure()
        fig.add_trace(go.Bar(x=dates,y=acts,name="Actual EPS",marker_color=bc,opacity=.85,text=[f"${v:.2f}" if v is not None else "N/A" for v in acts],textposition="outside",textfont=dict(color=TXT,size=10)))
        fig.add_trace(go.Scatter(x=dates,y=ests,name="EPS Estimate",line=dict(color=GLD,width=2,dash="dot"),mode="lines+markers",marker=dict(size=6,color=GLD)))
        fig.update_layout(**CBASE,height=240,legend=LEG_H,yaxis=dict(gridcolor=CGRID,side="right",tickprefix="$"),xaxis=dict(gridcolor=CGRID,showgrid=False),title=dict(text="EPS: Actual vs Estimate",font=dict(color=TXT,size=13)))
        return fig
    except: return None

# ── Signal Backtest ────────────────────────────────────────
def _tech_score_row(r):
    """Replicate TechAnalyzer.score()'s composite on a single indicator row → float.
    Used to walk the technical signal through history for the backtest."""
    sc_=[]
    s50,s200=r.get("SMA50",np.nan),r.get("SMA200",np.nan)
    if not any(np.isnan(x) for x in [s50,s200]): sc_.append(0.80 if s50>s200 else -0.80)
    p=r.get("Close",np.nan)
    above=sum(1 for k in ["SMA20","SMA50","SMA200"] if not np.isnan(r.get(k,np.nan)) and p>r[k])
    sc_.append(clamp((above/3)*2-1))
    rsi=r.get("RSI",50); rsi=50 if np.isnan(rsi) else rsi
    sc_.append(0.9 if rsi<30 else -0.9 if rsi>70 else clamp((50-rsi)/50*.4))
    macd,macds,macdh=r.get("MACD",np.nan),r.get("MACDs",np.nan),r.get("MACDh",np.nan)
    if not any(np.isnan(x) for x in [macd,macds,macdh]):
        sc_.append(clamp(abs(macdh)/(abs(macd)+1e-9))*(1 if macd>macds else -1))
    bbp=r.get("BBpct",.5); bbp=.5 if np.isnan(bbp) else bbp
    sc_.append(0.7 if bbp<.2 else -0.7 if bbp>.8 else clamp((.5-bbp)*1.4))
    moms=[clamp(r.get(f"M{l}",np.nan)*5) for l in ["1m","3m","6m"] if not np.isnan(r.get(f"M{l}",np.nan))]
    if moms: sc_.append(clamp(float(np.mean(moms))))
    return clamp(float(np.mean(sc_))) if sc_ else 0.

@st.cache_data(ttl=300,show_spinner=False)
def backtest_signal(ticker:str,period:str,thresh:float=.10):
    """Walk the technical composite through history: go long the next day when the
    signal is bullish (> thresh), else hold cash. Compare to buy-and-hold.
    Technical-only because fundamentals/news/lunar aren't available historically."""
    df=fetch_price_df(ticker,period)
    if df.empty or len(df)<60: return None
    edf=TechAnalyzer(df).df.copy()
    edf["sig"]=edf.apply(_tech_score_row,axis=1)
    edf["ret"]=edf["Close"].pct_change().fillna(0.)
    # position is decided on yesterday's signal, applied to today's return (no look-ahead)
    edf["pos"]=(edf["sig"].shift(1)>thresh).astype(float)
    warm=edf["SMA200"].notna().idxmax() if edf["SMA200"].notna().any() else edf.index[50]
    bt=edf.loc[warm:].copy()
    if len(bt)<30: return None
    bt["strat_ret"]=bt["pos"]*bt["ret"]
    bt["strat_eq"]=(1+bt["strat_ret"]).cumprod()
    bt["bh_eq"]=(1+bt["ret"]).cumprod()
    fwd=bt["ret"].shift(-1)  # next-day return, to judge whether a bullish call paid off
    bull=bt["sig"]>thresh; bear=bt["sig"]<=thresh
    hits=((bull)&(fwd>0))|((bear)&(fwd<=0)); hit_rate=float(hits[fwd.notna()].mean())
    def _cagr(eq):
        yrs=max((bt.index[-1]-bt.index[0]).days/365.25,.01); end=float(eq.iloc[-1])
        return (end**(1/yrs)-1) if end>0 else -1.
    def _mdd(eq): return float(((eq/eq.cummax())-1).min())
    days_in=float(bt["pos"].mean())
    def _safe_mean(s): m=s.mean(); return float(m) if pd.notna(m) else 0.
    return {"bt":bt[["strat_eq","bh_eq","sig","Close"]],
            "strat_total":float(bt["strat_eq"].iloc[-1]-1),"bh_total":float(bt["bh_eq"].iloc[-1]-1),
            "strat_cagr":_cagr(bt["strat_eq"]),"bh_cagr":_cagr(bt["bh_eq"]),
            "strat_mdd":_mdd(bt["strat_eq"]),"bh_mdd":_mdd(bt["bh_eq"]),
            "hit_rate":hit_rate if pd.notna(hit_rate) else 0.,"days_in":days_in,
            "avg_bull":_safe_mean(fwd[bull])*100,"avg_bear":_safe_mean(fwd[bear])*100,
            "n_days":len(bt),"thresh":thresh}

def backtest_chart(bt):
    try:
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.72,.28],vertical_spacing=.05)
        fig.add_trace(go.Scatter(x=bt.index,y=bt["strat_eq"],name="Signal strategy",line=dict(color=BLU,width=2)),row=1,col=1)
        fig.add_trace(go.Scatter(x=bt.index,y=bt["bh_eq"],name="Buy & hold",line=dict(color=MUT,width=1.6,dash="dot")),row=1,col=1)
        fig.add_trace(go.Bar(x=bt.index,y=bt["sig"],marker_color=[f"rgba({GR},.6)" if v>.1 else f"rgba({RR},.45)" for v in bt["sig"]],name="Tech signal",showlegend=False),row=2,col=1)
        fig.add_hline(y=.1,line=dict(color=MUT,width=1,dash="dot"),row=2,col=1)
        fig.update_layout(**CBASE,height=420,legend=LEG_H,yaxis=dict(gridcolor=CGRID,side="right",tickformat=".2f"),
                          yaxis2=dict(gridcolor=CGRID,side="right",range=[-1,1]),xaxis2=dict(gridcolor=CGRID,showgrid=False),xaxis=dict(gridcolor=CGRID,showgrid=False))
        return fig
    except: return None

# ── Watchlist & Compare ────────────────────────────────────
@st.cache_data(ttl=300,show_spinner=False)
def watch_snapshot(ticker:str):
    """Lightweight per-ticker read for the watchlist scoreboard: one price history
    call → price, day change, and the technical signal verdict (no info/news fetch)."""
    try:
        df=fetch_price_df(ticker,"1y")
        if df.empty or len(df)<2: return None
        last=float(df["Close"].iloc[-1]); prev=float(df["Close"].iloc[-2])
        edf=TechAnalyzer(df).df
        sig=_tech_score_row(edf.iloc[-1])
        vl,vc_,_=verdict(sig)
        return {"ticker":ticker,"price":last,"chg":(last-prev)/prev*100 if prev else 0.,
                "sig":sig,"verdict":vl,"vcolor":vc_,"name":TICKER_NAMES.get(ticker,ticker)}
    except: return None

def compare_chart(dfa,dfb,ta_,tb_,rng):
    """Two tickers rebased to 100 over the selected window."""
    try:
        a=slice_df(dfa,rng)["Close"]; b=slice_df(dfb,rng)["Close"]
        if not len(a) or not len(b): return None
        a=a/float(a.iloc[0])*100; b=b/float(b.iloc[0])*100
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=a.index,y=a.values,name=ta_,line=dict(color=BLU,width=2)))
        fig.add_trace(go.Scatter(x=b.index,y=b.values,name=tb_,line=dict(color=GLD,width=2)))
        fig.add_hline(y=100,line=dict(color=MUT,width=1,dash="dot"))
        fig.update_layout(**CBASE,height=340,legend=LEG_H,yaxis=dict(gridcolor=CGRID,side="right",ticksuffix=""),xaxis=dict(gridcolor=CGRID,showgrid=False),title=dict(text="Relative performance (rebased to 100)",font=dict(color=TXT,size=13)))
        return fig
    except: return None

def compare_bars(ma,mb,ta_,tb_):
    try:
        keys=list(MOD_C.keys()); labels=[f"{MOD_I[k]} {MOD_NAME.get(k,k)}" for k in keys]
        fig=go.Figure()
        fig.add_trace(go.Bar(y=labels,x=[ma[k]["score"] for k in keys],name=ta_,orientation="h",marker_color=BLU,opacity=.85))
        fig.add_trace(go.Bar(y=labels,x=[mb[k]["score"] for k in keys],name=tb_,orientation="h",marker_color=GLD,opacity=.85))
        fig.add_vline(x=0,line_color=MUT,line_width=1)
        fig.update_layout(**CBASE,height=300,legend=LEG_H,barmode="group",xaxis=dict(range=[-1.15,1.15],gridcolor=CGRID),yaxis=dict(gridcolor=CGRID,showgrid=False))
        return fig
    except: return None

# ── Export report ──────────────────────────────────────────
def build_report_html(r):
    """Self-contained HTML snapshot of the current analysis — downloadable."""
    q=r["quote"]; f=r["fund"]; comp=r["composite"]; vl,vc_,_=verdict(comp)
    chg=q["chg"]; cc_=G if chg>=0 else R
    mod_rows="".join(
        f'<tr><td style="padding:6px 10px">{MOD_I[k]} {MOD_NAME.get(k,k)}</td>'
        f'<td style="padding:6px 10px;color:{sc(r["modules"][k]["score"])};font-weight:700;text-align:right">{r["modules"][k]["score"]:+.2f}</td>'
        f'<td style="padding:6px 10px;text-align:right;color:#64748B">{r["weights"][k]*100:.0f}%</td></tr>'
        for k in MOD_C)
    reasons=strong_reasons(r)
    reason_rows="".join(
        f'<li style="margin:4px 0"><strong style="color:{G if rc["dir"]=="BUY" else R}">{rc["dir"]} · {rc["label"]}</strong> — {rc["reason"]} <span style="color:#64748B">({rc["detail"]})</span></li>'
        for rc in reasons) or "<li>No standout single-factor signals.</li>"
    def _row(lbl,val): return f'<tr><td style="padding:5px 10px;color:#64748B">{lbl}</td><td style="padding:5px 10px;text-align:right;font-weight:600">{val}</td></tr>'
    fund_rows=("".join([_row("Market Cap",fmoney(f.get("market_cap"))),_row("Analyst Target",fdollar(f.get("analyst_target"))),
        _row("Trailing P/E",fnum(f.get("pe_ratio"))),_row("Forward P/E",fnum(f.get("forward_pe"))),
        _row("Revenue Growth",fpct(f.get("revenue_growth"))),_row("Return on Equity",fpct(f.get("roe"))),
        _row("Debt/Equity",fnum(f.get("debt_to_equity"))),_row("52W Range",f'{fdollar(f.get("52wl"))} – {fdollar(f.get("52wh"))}')]))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Stock Oracle — {r['ticker']}</title>
<style>body{{font-family:-apple-system,Segoe UI,Inter,sans-serif;max-width:820px;margin:32px auto;padding:0 20px;color:#0F172A;background:#fff}}
h1{{font-size:28px;margin:0}} table{{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0 20px}}
.card{{border:1px solid #E7EBF0;border-radius:12px;padding:18px 22px;margin:14px 0}}
.big{{font-size:34px;font-weight:900}} .muted{{color:#64748B;font-size:13px}}</style></head><body>
<div style="display:flex;justify-content:space-between;align-items:baseline;border-bottom:2px solid #E7EBF0;padding-bottom:12px">
  <h1>◈ Stock Oracle</h1><span class="muted">{r['ts']} · Educational only</span></div>
<div class="card" style="border-left:6px solid {vc_}">
  <div class="muted">{r['ticker']} · {f.get('company_name','')[:60]} · {f.get('sector','—')}</div>
  <div style="display:flex;gap:18px;align-items:baseline;margin:6px 0">
    <span class="big">${q['price']:,.2f}</span>
    <span style="color:{cc_};font-weight:700;font-size:18px">{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%</span></div>
  <div style="font-size:22px;font-weight:900;color:{vc_}">{vl.upper()} · Score {comp:+.2f}</div>
  <div class="muted" style="margin-top:6px">{r['plain']}</div></div>
<h3>Module Scores</h3><table>{mod_rows}</table>
<h3>Strongest Signals</h3><ul style="font-size:14px;padding-left:20px">{reason_rows}</ul>
<h3>Fundamentals</h3><table>{fund_rows}</table>
<p class="muted">Generated by Stock Oracle v3.0. Not financial advice — for educational use only.</p>
</body></html>"""

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.html(f"""<div style="font-size:20px;font-weight:900;letter-spacing:-.03em;margin-bottom:2px">◈ Stock <span style="color:{BLU}">Oracle</span></div><div style="color:{MUT};font-size:11px;margin-bottom:16px">Multi-factor Market Intelligence</div>""")
    c1,c2=st.columns(2)
    with c1:
        if st.button("🌙 Dark",width='stretch',key="dk",type="primary" if DARK else "secondary"): st.session_state.theme="dark"; st.rerun()
    with c2:
        if st.button("☀️ Light",width='stretch',key="lt",type="primary" if not DARK else "secondary"): st.session_state.theme="light"; st.rerun()
    st.divider()
    st.html(f"<div style='font-size:10px;font-weight:700;color:{MUT};letter-spacing:.1em;margin-bottom:6px'>SEARCH STOCK</div>")
    # a ticker chosen from another tab (movers / trending) lands here before the picker renders
    if "pending_ticker" in st.session_state:
        st.session_state["ticker_pick"]=st.session_state.pop("pending_ticker")
    st.session_state.setdefault("ticker_pick",st.session_state.get("ticker","AAPL"))
    ALLSYM=load_all_symbols()
    _uni=sorted(set(ALLSYM)|{st.session_state["ticker_pick"]})
    ticker_in=st.selectbox("Search stock",_uni,key="ticker_pick",
        format_func=lambda s:f"{s} · {ALLSYM[s]}" if s in ALLSYM else s,
        label_visibility="collapsed",placeholder="Type a symbol or company name…")
    st.session_state.ticker=ticker_in
    st.html(f"<div style='font-size:10px;color:{MUT};margin-top:4px'>{len(ALLSYM):,} stocks &amp; ETFs · type to filter by symbol or name</div>")
    _in_wl=ticker_in in st.session_state.watchlist
    if st.button(("★ Remove from watchlist" if _in_wl else "☆ Add to watchlist"),width='stretch',key="wl_toggle"):
        if _in_wl: st.session_state.watchlist=[t for t in st.session_state.watchlist if t!=ticker_in]
        else: st.session_state.watchlist=(st.session_state.watchlist+[ticker_in])[-20:]
        st.rerun()
    period_sel=st.selectbox("Analysis Lookback",["1y","2y","3y","5y"],index=1)
    st.divider()
    st.html(f"<div style='font-size:10px;font-weight:700;color:{MUT};letter-spacing:.1em;margin-bottom:8px'>CHART STYLE</div>")
    ctype=st.radio("Style",["Line","Area","Candle"],horizontal=True,label_visibility="collapsed")
    with st.expander("Indicators & panels"):
        c1,c2=st.columns(2)
        with c1:
            sma20=st.checkbox("MA 20"); sma50=st.checkbox("MA 50"); sma200=st.checkbox("MA 200"); bb=st.checkbox("Bollinger")
        with c2:
            vol=st.checkbox("Volume"); rsi=st.checkbox("RSI panel"); macd=st.checkbox("MACD panel")
    opts={"sma20":sma20,"sma50":sma50,"sma200":sma200,"bb":bb,"vol":vol,"rsi":rsi,"macd":macd}
    st.divider()
    st.html(f"<div style='font-size:10px;font-weight:700;color:{MUT};letter-spacing:.1em;margin-bottom:2px'>SIGNAL WEIGHTS</div>")
    st.html(f"<div style='font-size:10px;color:{MUT};margin-bottom:8px'>Adjust how much each factor counts</div>")
    weights={}
    for k in MOD_C: weights[k]=st.slider(f"{MOD_I[k]} {MOD_NAME.get(k,k)}",0.,1.,MOD_D[k],.05,key=f"w_{k}")
    if st.button("Reset weights",width='stretch'):
        for k in MOD_D: st.session_state[f"w_{k}"]=MOD_D[k]
        st.rerun()
    st.divider()
    if st.button("↻ Refresh data",width='stretch',type="primary",key="refresh"):
        for _c in (compute_core,fetch_movers,fetch_top_lists,fetch_market,fetch_market_news,quick_quote): _c.clear()
        st.session_state.last_sig=None; st.rerun()
    st.html(f"<div style='color:{MUT};font-size:9px;margin-top:8px;text-align:center;line-height:1.7'>Updates automatically on any change · v3.0 · Educational only</div>")

def go_analyze(tk):
    """Jump to a ticker chosen from the movers / trending lists — auto-analysis does the rest."""
    st.session_state["pending_ticker"]=tk
    st.rerun()

# ── Auto-analysis: recompute whenever ticker / period / weights change ──
_sig=(ticker_in,period_sel,tuple(sorted(weights.items())))
if ticker_in and st.session_state.get("last_sig")!=_sig:
    with st.spinner(f"Analyzing {ticker_in}…"):
        try:
            st.session_state.result=run_analysis(ticker_in,period_sel,weights); st.session_state.last_sig=_sig
        except Exception as e:
            st.session_state.result=None; st.error(f"Could not analyze '{ticker_in}': {e}")

r=st.session_state.result

# ── Prominent search bar (always visible at the top of the page) ──
st.html(f"<div style='font-size:23px;font-weight:900;letter-spacing:-.03em;margin:2px 0 6px'>◈ Stock <span style='color:{BLU}'>Oracle</span> <span style='font-size:13px;font-weight:600;color:{MUT};letter-spacing:0'>— search any US stock or ETF below</span></div>")
def _main_search_cb():
    v=st.session_state.get("main_search")
    if v: st.session_state["pending_ticker"]=v
_msuni=sorted(set(ALLSYM)|{ticker_in})
st.session_state["main_search"]=ticker_in   # keep the box in sync with the current stock
sc1,sc2=st.columns([3,1])
with sc1:
    st.selectbox("Search any stock",_msuni,key="main_search",on_change=_main_search_cb,
        format_func=lambda s:f"{s} · {ALLSYM[s]}" if s in ALLSYM else s,
        label_visibility="collapsed",placeholder="Type a symbol or company name — e.g. AAPL or Apple")
with sc2:
    st.html(f"<div style='font-size:11px;color:{MUT};padding-top:12px;text-align:right'>{len(ALLSYM):,} stocks &amp; ETFs</div>")

if r is not None:
    q_top=r["quote"]; p_top=q_top["price"]; chg_top=q_top["chg"]
    cc_top=G if chg_top>=0 else R; ar_top="UP" if chg_top>=0 else "DN"
    vl_top,vc_top,_=verdict(r["composite"])
    st.html(f"""<div class="top-bar">
      <span style="font-weight:900;font-size:15px">{r['ticker']}</span>
      <span style="color:{MUT};font-size:12px">{q_top['name'][:35]}</span>
      <span style="font-weight:800;font-size:15px">${p_top:,.2f}</span>
      <span style="color:{cc_top};font-weight:700">{'▲' if chg_top>=0 else '▼'} {abs(chg_top):.2f}%</span>
      <span style="color:{vc_top};font-weight:700;margin-left:4px">{vl_top}</span>
      <span style="color:{MUT};font-size:11px;margin-left:auto">{r['ts']}</span>
    </div>""")

mkt=fetch_market(); show_mkt=[(n,d) for n,d in mkt.items() if d["val"] is not None]

# ── Market Pulse: what's happening + what's moving the market ──
mt_title,mt_detail,mt_col=market_theme(mkt)
mkt_news=fetch_market_news(4)
news_html=""
if mkt_news:
    chips="".join(
        (f'<a href="{h["link"]}" target="_blank" style="text-decoration:none">' if h["link"] else "")
        +f'<span style="display:inline-block;background:{CARD};border:1px solid {BDR};border-radius:8px;padding:5px 10px;margin:3px 6px 3px 0;font-size:11.5px;color:{TXT};line-height:1.4">📰 {h["title"][:90]}</span>'
        +("</a>" if h["link"] else "")
        for h in mkt_news)
    news_html=f'<div style="margin-top:10px">{chips}</div>'
st.html(f"""<div class="card" style="border-left:5px solid {mt_col};margin-bottom:10px">
  <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">
    <span style="font-size:10px;font-weight:800;letter-spacing:.12em;color:{MUT}">MARKET PULSE</span>
    <span style="font-size:19px;font-weight:900;color:{mt_col};letter-spacing:-.02em">{mt_title}</span>
  </div>
  <div style="font-size:12.5px;color:{MUT};line-height:1.55;margin-top:4px">{mt_detail}</div>
  {news_html}
</div>""")
st.html(f"<div class='seclabel'>Market snapshot</div><div class='secsub'>How the major indexes and key prices are doing today</div>")
for col_,(name,data) in zip(st.columns(len(show_mkt)),show_mkt):
    with col_:
        val=data["val"]; chg=data["chg"] or 0; cc=G if chg>=0 else R; sfx="%" if "Yield" in name else ""
        st.html(f"""<div class="card-sm" style="margin-bottom:8px"><div style="font-size:9px;color:{MUT};letter-spacing:.08em;font-weight:700">{name.upper()}</div><div style="font-size:17px;font-weight:800;color:{TXT};margin:2px 0">{val:.2f}{sfx}</div><div style="font-size:11px;color:{cc};font-weight:600">{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%</div></div>""")

# ── Ideas to explore: opportunities, quality stocks, and movers ──
tl=fetch_top_lists()
if any(tl.get(k) for k in ("potential","picks","up","down")):
    keymap={"Top Opportunities":"potential","Quality Stocks":"picks","Today's Gainers":"up","Today's Losers":"down"}
    submap={"Top Opportunities":"Stocks with the strongest momentum right now",
            "Quality Stocks":"A hand-picked list of large, established companies",
            "Today's Gainers":"Biggest price gains so far today","Today's Losers":"Biggest price drops so far today"}
    st.html(f"<div class='seclabel'>Ideas to explore</div>")
    seg=st.columns(len(keymap))
    for c,label in zip(seg,keymap):
        active=st.session_state.picks_mode==label
        if c.button(label,key=f"pm_{label}",width='stretch',type="primary" if active else "secondary"):
            st.session_state.picks_mode=label; st.rerun()
    mode=st.session_state.picks_mode if st.session_state.picks_mode in keymap else "Top Opportunities"
    st.html(f"<div class='secsub'>{submap.get(mode,'')}</div>")
    items=tl.get(keymap.get(mode,"potential"),[])[:6]; show_score=(mode=="Top Opportunities")
    if not items:
        st.html(f"<div style='color:{MUT};font-size:12px;padding:2px 0 10px'>No data right now — try ↻ Refresh data.</div>")
    else:
        for col_,m in zip(st.columns(len(items)),items):
            with col_:
                c=G if m["chg"]>=0 else R; badge=""
                if show_score:
                    vlp,vcp,_=verdict(m["score"]); badge=f'<div style="margin-top:5px"><span class="pill" style="background:{vcp}1f;color:{vcp}">{vlp}</span></div>'
                st.html(f'<div class="card-sm" style="padding:10px 12px;margin-bottom:6px">'
                        f'<div style="font-weight:800;font-size:14px;color:{TXT}">{m["ticker"]}</div>'
                        f'<div style="font-size:10px;color:{MUT};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:4px">{m["name"]}</div>'
                        f'<div style="font-size:14px;font-weight:700;color:{TXT}">${m["price"]:,.2f}</div>'
                        f'<div style="font-size:11px;color:{c};font-weight:700">{"▲" if m["chg"]>=0 else "▼"} {abs(m["chg"]):.2f}%</div>'
                        f'{badge}</div>')
                if st.button("Analyze",key=f"pick_{m['ticker']}",width='stretch'): go_analyze(m["ticker"])

st.html("<div style='height:6px'></div>")
tab_mkt,tab_anal,tab_back,tab_watch,tab_cmp,tab_insider,tab_disc,tab_fund,tab_ctx=st.tabs(["🏠 Dashboard","🔬 Deep Analysis","📈 Backtest","⭐ Watchlist","⚖️ Compare","🏛️ Insider Activity","🔥 Trending","📊 Fundamentals","🌐 Market Context"])

# ── Tab 1: Dashboard ───────────────────────────────────────
with tab_mkt:
    if r is None:
        st.html(f"""<div style="text-align:center;padding:60px 0 30px"><div style="font-size:64px;margin-bottom:8px">◈</div><h1 style="font-size:2.4rem;font-weight:900;letter-spacing:-.04em;color:{TXT};margin:0 0 8px">Stock Oracle</h1><p style="color:{MUT};font-size:1rem;margin:0 0 32px">Pick a stock from the search box — analysis runs <strong style="color:{BLU}">automatically</strong></p></div>""")
        for col_,(icon,h,desc) in zip(st.columns(3),[("📈","Technical Signals","Moving averages, RSI, MACD, Bollinger Bands — reads the price chart automatically"),("🧠","Psychology & Sentiment","Fear gauge, crowd behavior, short squeeze risk, news tone, insider ownership"),("📊","Fundamentals & Earnings","Revenue growth, analyst targets, earnings beats/misses, balance sheet health")]):
            with col_: st.html(f"""<div class="card" style="border-left:4px solid {BLU}"><div style="font-size:32px;margin-bottom:8px">{icon}</div><div style="font-weight:700;font-size:15px;margin-bottom:6px">{h}</div><div style="color:{MUT};font-size:12px;line-height:1.6">{desc}</div></div>""")
    else:
        q=r["quote"]; p=q["price"]; chg=q["chg"]; cc_=G if chg>=0 else R
        comp=r["composite"]; vl,vc_,vi=verdict(comp)
        desc_s=(r["fund"].get("description","")[:200]+"…") if len(r["fund"].get("description",""))>200 else r["fund"].get("description","")
        h1,h2,h3=st.columns([2.5,1.2,1.2])
        with h1:
            st.html(f"""<div class="card" style="border-left:5px solid {vc_}"><div style="font-size:11px;color:{MUT};letter-spacing:.08em;margin-bottom:6px">{r['ticker']} · {r['fund'].get('company_name','')[:50]} · {r['fund'].get('sector','—')} · {r['ts']}</div><div style="display:flex;align-items:baseline;gap:14px;margin-bottom:8px;flex-wrap:wrap"><span style="font-size:36px;font-weight:900;color:{TXT};letter-spacing:-.03em">${p:,.2f}</span><span style="font-size:18px;font-weight:700;color:{cc_}">{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%</span></div><div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px"><span style="font-size:24px;font-weight:900;color:{vc_}">{vl.upper()}</span><span class="pill" style="background:{vc_}22;color:{vc_}">Score {comp:+.2f}</span><span style="color:{MUT};font-size:12px">SPY 3m: <span style="color:{G if r['spy']>=0 else R};font-weight:600">{r['spy']:+.1f}%</span></span></div><div style="font-size:12px;color:{MUT};line-height:1.6;border-top:1px solid {BDR};padding-top:10px"><span style="color:{vc_};font-weight:600">{r['plain']}</span>{f'<br><span style="font-size:11px">{desc_s}</span>' if desc_s else ''}</div></div>""")
        with h2:
            fig_g=gauge_chart(comp,vl,vc_)
            if fig_g: st.plotly_chart(fig_g,width='stretch',config={"displayModeBar":False})
        with h3:
            fig_mb=module_bars(r["modules"],r["weights"])
            if fig_mb: st.plotly_chart(fig_mb,width='stretch',config={"displayModeBar":False})
        dl1,dl2=st.columns([1,4])
        with dl1:
            st.download_button("⬇ Export report",data=build_report_html(r),
                file_name=f"StockOracle_{r['ticker']}_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",width='stretch',key="dl_report")
        reasons=strong_reasons(r)
        if reasons:
            st.html(f"<div class='seclabel'>Why this verdict</div><div class='secsub'>The signals pushing hardest toward buy or sell right now</div>")
            for col_,rc in zip(st.columns(len(reasons)),reasons):
                dc=G if rc["dir"]=="BUY" else R; ic=MOD_I.get(rc["key"],"•")
                with col_:
                    st.html(f"""<div class="card-sm" style="border-left:4px solid {dc};height:100%">
                      <div style="font-size:10px;letter-spacing:.08em;color:{MUT};font-weight:700">{ic} {rc['label'].upper()}</div>
                      <div style="font-size:15px;font-weight:900;color:{dc};margin:3px 0">Strong reason to {rc['dir']}</div>
                      <div style="font-size:11.5px;color:{TXT};line-height:1.45">{rc['reason']}</div>
                      <div style="font-size:11px;color:{MUT};margin-top:3px">{rc['detail']}</div>
                      <div style="font-size:11px;color:{dc};font-weight:800;margin-top:5px">Signal {rc['score']:+.2f}</div>
                    </div>""")
        st.html(f"<div class='seclabel'>Price chart</div><div class='secsub'>Pick a time range below · add indicators from the sidebar</div>")
        tl_cols=st.columns(len(TL))
        for i,(lbl,_) in enumerate(TL.items()):
            with tl_cols[i]:
                active=st.session_state.range==lbl
                if st.button(lbl,key=f"tl_{lbl}",width='stretch',type="primary" if active else "secondary"): st.session_state.range=lbl; st.rerun()
        fig_price=price_chart(r["df"],r["ticker"],st.session_state.range,ctype,opts)
        if fig_price: st.plotly_chart(fig_price,width='stretch',config={"displayModeBar":True,"modeBarButtonsToRemove":["lasso2d","select2d"],"displaylogo":False})
        else: st.warning("Chart could not be rendered.")
        if opts.get("macd"):
            fig_macd=macd_chart(r["df"],st.session_state.range)
            if fig_macd:
                st.html(f"<div style='font-size:11px;color:{MUT};margin-bottom:4px'>MACD — Momentum Signal</div>")
                st.plotly_chart(fig_macd,width='stretch',config={"displayModeBar":False})

# ── Tab 2: Deep Analysis ───────────────────────────────────
with tab_anal:
    if r is None: st.info("Search for a stock in the bar at the top of the page to begin.")
    else:
        left,right=st.columns([1.1,1.8])
        with left:
            fig_r=radar_chart(r["modules"])
            if fig_r: st.plotly_chart(fig_r,width='stretch',config={"displayModeBar":False})
            for key in MOD_C:
                mod=r["modules"][key]; ms=mod["score"]; col_=MOD_C[key]; vl2,vc2,_=verdict(ms); pct=int(abs(ms)*100)
                sigs=mod.get("signals",{}); sig_html=""
                for sn,sv in list(sigs.items())[:6]:
                    sd,sv_n=(sv[0],float(sv[1])) if isinstance(sv,tuple) else (str(sv),0.)
                    ic="▲" if sv_n>.05 else "▼" if sv_n<-.05 else "◆"; sv_c=sc(sv_n)
                    sig_html+=(f'<div class="sig"><span style="color:{col_};margin-right:6px">{ic}</span><span style="flex:1">{sn}</span><span style="color:{sv_c};font-weight:600;margin-left:8px">{sv_n:+.2f}</span></div>')
                lunar_extra="" 
                if key=="lunar": lunar_extra=(f'<div style="color:{CYN};font-size:11px;padding:8px 0">Moon: {mod.get("phase","?")} · {mod.get("illum","?")} lit · New moon in {mod.get("d_new","?")}d</div>')
                st.html(f"""<div class="card" style="border-left:4px solid {col_}"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px"><span style="font-size:11px;font-weight:700;letter-spacing:.08em;color:{col_};text-transform:uppercase">{MOD_I[key]} {MOD_NAME.get(key,key)}</span><span style="color:{MUT};font-size:10px">{r['weights'][key]*100:.0f}% weight</span></div><div style="font-size:10px;color:{MUT};margin-bottom:8px;line-height:1.5">{MOD_DESC.get(key,'')}</div><div style="display:flex;align-items:baseline;gap:10px;margin-bottom:4px"><span style="font-size:24px;font-weight:900;color:{vc2}">{ms:+.2f}</span><span class="pill" style="background:{vc2}22;color:{vc2}">{vl2}</span></div><div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{col_ if ms>=0 else R}"></div></div>{sig_html}{lunar_extra}</div>""")
        with right:
            nd=r["modules"]["news"]; ns,nc_,_=verdict(nd["score"])
            st.html(f"""<div class="card" style="border-left:4px solid {nc_};margin-bottom:16px"><div style="font-size:11px;color:{MUT};letter-spacing:.08em;margin-bottom:6px">NEWS MOOD</div><div style="display:flex;align-items:center;gap:16px"><span style="font-size:28px;font-weight:900;color:{nc_}">{nd['score']:+.2f}</span><div><div style="font-weight:700;color:{nc_}">{ns}</div><div style="font-size:11px;color:{MUT}"><span style="color:{G}">▲{nd['bull']} positive</span>&nbsp;<span style="color:{R}">▼{nd['bear']} negative</span>&nbsp;{nd['total']} headlines</div></div></div></div>""")
            for h in nd.get("headlines",[])[:10]:
                col2=G if h["sent"]=="Positive" else R if h["sent"]=="Negative" else MUT
                icon2="▲" if h["sent"]=="Positive" else "▼" if h["sent"]=="Negative" else "◆"
                st.html(f"""<div class="reddit-card"><div style="display:flex;gap:10px;align-items:flex-start"><span style="color:{col2};font-size:15px;font-weight:700;min-width:18px">{icon2}</span><span style="font-size:12px;color:{TXT};flex:1;line-height:1.5">{h['title'][:130]}</span><span style="color:{col2};font-weight:700;font-size:12px;min-width:40px;text-align:right">{h['score']:+.2f}</span></div></div>""")

# ── Tab: Backtest ──────────────────────────────────────────
with tab_back:
    if r is None: st.info("Search for a stock in the bar at the top of the page to begin.")
    else:
        st.html(f"""<div style="margin-bottom:14px"><div style="font-size:22px;font-weight:800">Signal Backtest — {r['ticker']}</div><div style="color:{MUT};font-size:13px">Would the technical signal have beaten buy-and-hold? Long the next day when the signal turns bullish, otherwise hold cash. No look-ahead.</div></div>""")
        with st.spinner("Running backtest…"): bt=backtest_signal(r["ticker"],period_sel)
        if not bt:
            st.warning("Not enough price history to backtest this ticker.")
        else:
            def _mc(lbl,sv,bv,fmt,good_high=True):
                win=(sv>bv) if good_high else (sv<bv); wc=G if win else R
                return (f'<div class="card-sm" style="flex:1;min-width:120px"><div style="font-size:10px;color:{MUT};letter-spacing:.08em">{lbl}</div>'
                        f'<div style="font-size:22px;font-weight:900;color:{wc}">{fmt(sv)}</div>'
                        f'<div style="font-size:11px;color:{MUT}">vs hold {fmt(bv)}</div></div>')
            pf=lambda v:f"{v*100:+.1f}%"
            st.html('<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px">'
                    +_mc("TOTAL RETURN",bt["strat_total"],bt["bh_total"],pf)
                    +_mc("CAGR",bt["strat_cagr"],bt["bh_cagr"],pf)
                    +_mc("MAX DRAWDOWN",bt["strat_mdd"],bt["bh_mdd"],pf,good_high=True)
                    +f'<div class="card-sm" style="flex:1;min-width:120px"><div style="font-size:10px;color:{MUT};letter-spacing:.08em">DIRECTION HIT RATE</div><div style="font-size:22px;font-weight:900;color:{TXT}">{bt["hit_rate"]*100:.0f}%</div><div style="font-size:11px;color:{MUT}">in market {bt["days_in"]*100:.0f}% of days</div></div>'
                    +'</div>')
            fig_bt=backtest_chart(bt["bt"])
            if fig_bt: st.plotly_chart(fig_bt,width='stretch',config={"displayModeBar":False})
            edge_c=G if bt["avg_bull"]>bt["avg_bear"] else R
            st.html(f"""<div class="expl"><strong>Signal edge:</strong> the day after a <strong style="color:{G}">bullish</strong> reading the stock averaged <strong style="color:{G}">{bt['avg_bull']:+.2f}%</strong>, vs <strong style="color:{R}">{bt['avg_bear']:+.2f}%</strong> after a non-bullish reading — a spread of <strong style="color:{edge_c}">{bt['avg_bull']-bt['avg_bear']:+.2f}%</strong>/day across {bt['n_days']} trading days.<br><br>Backtest covers the <strong>technical</strong> signal only (fundamentals, news, psychology and lunar inputs aren't reconstructable historically). Past performance is not indicative of future results — educational only.</div>""")

# ── Tab: Watchlist ─────────────────────────────────────────
with tab_watch:
    wl=st.session_state.watchlist
    st.html(f"""<div style="margin-bottom:14px"><div style="font-size:22px;font-weight:800">Watchlist</div><div style="color:{MUT};font-size:13px">{len(wl)} tracked · add or remove the current stock from the sidebar · signal is technical-only for speed</div></div>""")
    if not wl:
        st.info("Your watchlist is empty. Pick a stock in the sidebar and tap ☆ Add to watchlist.")
    else:
        with st.spinner("Refreshing watchlist…"):
            snaps=[s for s in (watch_snapshot(t) for t in wl) if s]
        if not snaps:
            st.warning("Couldn't load watchlist quotes right now — try ↻ Refresh data.")
        else:
            snaps.sort(key=lambda x:x["sig"],reverse=True)
            st.html(f'<div style="display:flex;font-size:10px;font-weight:700;color:{MUT};letter-spacing:.1em;padding:6px 12px;border-bottom:2px solid {BDR}"><span style="flex:0.7">TICKER</span><span style="flex:1.6">NAME</span><span style="flex:0.9;text-align:right">PRICE</span><span style="flex:0.9;text-align:right">CHANGE</span><span style="flex:1">SIGNAL</span><span style="flex:0.6;text-align:center">GO</span></div>')
            for i,s in enumerate(snaps):
                cc2=G if s["chg"]>=0 else R; stripe=f"background:{BDR}22;" if i%2==0 else ""
                rc=st.columns([0.7,1.6,0.9,0.9,1,0.6])
                with rc[0]: st.html(f'<div style="{stripe}padding:9px 6px;font-weight:800;font-size:14px">{s["ticker"]}</div>')
                with rc[1]: st.html(f'<div style="{stripe}padding:9px 6px;color:{MUT};font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{s["name"][:30]}</div>')
                with rc[2]: st.html(f'<div style="{stripe}padding:9px 6px;text-align:right;font-weight:600">${s["price"]:,.2f}</div>')
                with rc[3]: st.html(f'<div style="{stripe}padding:9px 6px;text-align:right;color:{cc2};font-weight:600">{"▲" if s["chg"]>=0 else "▼"} {abs(s["chg"]):.2f}%</div>')
                with rc[4]: st.html(f'<div style="{stripe}padding:9px 6px"><span class="pill" style="background:{s["vcolor"]}22;color:{s["vcolor"]}">{s["verdict"]}</span> <span style="color:{MUT};font-size:11px">{s["sig"]:+.2f}</span></div>')
                with rc[5]:
                    if st.button("Go",key=f"wl_go_{s['ticker']}",width='stretch'): go_analyze(s["ticker"])

# ── Tab: Compare ───────────────────────────────────────────
with tab_cmp:
    if r is None: st.info("Search for a stock in the bar at the top of the page to begin.")
    else:
        ta_=r["ticker"]
        st.html(f"""<div style="margin-bottom:10px"><div style="font-size:22px;font-weight:800">Compare</div><div style="color:{MUT};font-size:13px">Head-to-head: <strong>{ta_}</strong> vs a second stock — relative performance and module scores.</div></div>""")
        _opts=[t for t in sorted(set(ALLSYM)) if t!=ta_]
        _def="MSFT" if ta_!="MSFT" else "AAPL"
        tb_=st.selectbox("Compare with",_opts,index=(_opts.index(_def) if _def in _opts else 0),
                         format_func=lambda s:f"{s} · {ALLSYM[s]}" if s in ALLSYM else s,key="cmp_pick")
        if tb_:
            with st.spinner(f"Analyzing {tb_}…"):
                try: rb=run_analysis(tb_,period_sel,weights)
                except Exception as e: rb=None; st.error(f"Could not analyze '{tb_}': {e}")
            if rb:
                cA,cB=st.columns(2)
                for col_,rr_,clr in [(cA,r,BLU),(cB,rb,GLD)]:
                    with col_:
                        q=rr_["quote"]; comp=rr_["composite"]; vl,vc_,_=verdict(comp); chg=q["chg"]; cc_=G if chg>=0 else R
                        st.html(f"""<div class="card" style="border-top:4px solid {clr}"><div style="font-size:11px;color:{MUT}">{rr_['ticker']} · {rr_['fund'].get('company_name','')[:34]}</div><div style="display:flex;align-items:baseline;gap:10px;margin:4px 0"><span style="font-size:26px;font-weight:900">${q['price']:,.2f}</span><span style="color:{cc_};font-weight:700">{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%</span></div><div style="font-size:18px;font-weight:900;color:{vc_}">{vl.upper()}</div><span class="pill" style="background:{vc_}22;color:{vc_}">Score {comp:+.2f}</span></div>""")
                fig_c=compare_chart(r["df"],rb["df"],ta_,tb_,st.session_state.range)
                if fig_c: st.plotly_chart(fig_c,width='stretch',config={"displayModeBar":False})
                fig_cb=compare_bars(r["modules"],rb["modules"],ta_,tb_)
                if fig_cb:
                    st.html(f"<div style='font-size:13px;font-weight:700;margin:6px 0 4px'>Module Scores — {ta_} vs {tb_}</div>")
                    st.plotly_chart(fig_cb,width='stretch',config={"displayModeBar":False})

# ── Tab 3: Insider ─────────────────────────────────────────
with tab_insider:
    if r is None: st.info("Search for a stock in the bar at the top of the page to begin.")
    else:
        ticker_ins=r["ticker"]
        st.html(f"""<div style="margin-bottom:16px"><div style="font-size:22px;font-weight:800">Insider Activity — {ticker_ins}</div><div style="color:{MUT};font-size:13px">Form 4 SEC filings · Directors and officers · Last 90 days</div></div>""")
        with st.spinner("Loading insider data…"): ins=fetch_insider_data(ticker_ins)
        transactions=ins.get("transactions",[]); net=ins.get("net_shares",0); source=ins.get("source","none")
        if transactions:
            net_c=G if net>0 else R if net<0 else MUT
            net_lbl=("Insiders are net buyers — a good sign" if net>0 else "Insiders are net sellers — worth watching" if net<0 else "Roughly balanced insider activity")
            st.html(f"""<div class="card" style="border-left:5px solid {net_c};margin-bottom:16px"><div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap"><span style="font-size:28px;color:{net_c}">{'▲' if net>0 else '▼' if net<0 else '─'}</span><div><div style="font-size:16px;font-weight:800;color:{net_c}">{net_lbl}</div><div style="font-size:12px;color:{MUT}">Net shares: <strong style="color:{net_c}">{net:+,}</strong> · Source: {source}</div></div></div></div>""")
            fig_ins=insider_chart(transactions)
            if fig_ins: st.plotly_chart(fig_ins,width='stretch',config={"displayModeBar":False})
            st.html(f"""<div style="display:flex;font-size:10px;font-weight:700;color:{MUT};letter-spacing:.1em;padding:8px 12px;border-bottom:2px solid {BDR};border-top:1px solid {BDR}"><span style="flex:0.8">DATE</span><span style="flex:1.5">INSIDER</span><span style="flex:1">TITLE</span><span style="flex:0.8">TYPE</span><span style="flex:0.8;text-align:right">SHARES</span><span style="flex:0.8;text-align:right">VALUE</span></div>""")
            for i,txn in enumerate(transactions[:25]):
                is_buy=txn.get("type")=="Buy"; is_sell=txn.get("type")=="Sale"; type_c=G if is_buy else R if is_sell else MUT
                type_lbl="Buy" if is_buy else "Sell" if is_sell else txn.get("type","—"); stripe=f"background:{BDR}22;" if i%2==0 else ""
                url=txn.get("url",""); link=f'<a href="{url}" target="_blank" style="color:{BLU};text-decoration:none">Link</a>' if url else ""
                st.html(f"""<div style="{stripe}display:flex;padding:8px 12px;border-bottom:1px solid {BDR};font-size:12px;align-items:center"><span style="flex:0.8;color:{MUT}">{txn.get('date','—')}</span><span style="flex:1.5;font-weight:600;color:{TXT}">{txn.get('insider','—')[:22]} {link}</span><span style="flex:1;color:{MUT};font-size:11px">{txn.get('title','—')[:18]}</span><span style="flex:0.8;color:{type_c};font-weight:700">{type_lbl}</span><span style="flex:0.8;text-align:right;font-weight:600">{txn.get('shares',0):,}</span><span style="flex:0.8;text-align:right;color:{MUT}">{fmoney(txn.get('value',0))}</span></div>""")
            sec_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker_ins}&type=4&dateb=&owner=include&count=40"
            st.html(f'<div style="margin-top:12px;font-size:12px;color:{MUT}">Source: {source} · <a href="{sec_url}" target="_blank" style="color:{BLU}">View all on SEC EDGAR</a></div>')
        else:
            st.html(f"""<div class="card" style="text-align:center;padding:40px"><div style="font-size:32px;margin-bottom:8px">📭</div><div style="font-weight:700;font-size:15px;margin-bottom:6px">No Recent Insider Activity Found</div><div style="color:{MUT};font-size:12px">No Form 4 filings for {ticker_ins} in the last 90 days.</div></div>""")
            sec_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker_ins}&type=4&dateb=&owner=include&count=40"
            st.html(f'<div style="text-align:center;margin-top:8px;font-size:12px"><a href="{sec_url}" target="_blank" style="color:{BLU}">Check SEC EDGAR directly</a></div>')

# ── Tab: Trending (most talked-about stocks) ───────────────
with tab_disc:
    with st.spinner("Loading trending stocks…"): trend=fetch_social_trending()
    src=trend.get("source","none"); items=trend.get("items",[])
    sub=("Most talked-about stocks on StockTwits right now" if src=="StockTwits"
         else "StockTwits was unavailable — showing today's biggest movers instead" if src=="Movers"
         else "Trending data is temporarily unavailable")
    st.html(f"""<div class="seclabel">Trending now</div><div class="secsub">{sub}. Tap any stock to analyze it.</div>""")
    if not items:
        st.warning("Couldn't load trending stocks right now — try ↻ Refresh data in the sidebar.")
    else:
        # Top 3 as feature cards
        for col_,it in zip(st.columns(3),items[:3]):
            tk=it["ticker"]
            with col_:
                q2=quick_quote(tk); p2=q2["price"]; chg2=q2["chg"]; cc2=G if chg2>=0 else R
                extra=(f'<div style="font-size:18px;font-weight:800;color:{BLU}">{it["watchers"]:,}</div><div style="font-size:10px;color:{MUT}">people watching</div>'
                       if it.get("watchers") else f'<div style="font-size:18px;font-weight:800;color:{cc2}">{it.get("chg",0):+.2f}%</div><div style="font-size:10px;color:{MUT}">today</div>')
                st.html(f"""<div class="card" style="border-left:4px solid {BLU}"><div style="display:flex;justify-content:space-between;align-items:flex-start"><div><div style="font-size:26px;font-weight:900;color:{TXT}">{tk}</div><div style="font-size:11px;color:{MUT};margin-bottom:8px">{it['name'][:28]}</div></div><div style="text-align:right"><div style="font-size:17px;font-weight:700">${p2:,.2f}</div><div style="font-size:12px;color:{cc2};font-weight:600">{'▲' if chg2>=0 else '▼'} {abs(chg2):.2f}%</div></div></div><div style="margin:6px 0 4px">{extra}</div></div>""")
                fig_mini=mini_price_chart(tk)
                if fig_mini: st.plotly_chart(fig_mini,width='stretch',config={"displayModeBar":False})
                if st.button(f"Analyze {tk}",key=f"trend_{tk}",width='stretch'): go_analyze(tk)
        st.html(f"<div class='seclabel'>Full list</div>")
        metric_hdr="WATCHING" if src=="StockTwits" else "TODAY"
        st.html(f'<div style="display:flex;font-size:10px;font-weight:700;color:{MUT};letter-spacing:.1em;padding:6px 12px;border-bottom:2px solid {BDR}"><span style="flex:0.6">TICKER</span><span style="flex:1.6">NAME</span><span style="flex:0.9;text-align:right">PRICE</span><span style="flex:0.8;text-align:right">CHANGE</span><span style="flex:0.9;text-align:right">{metric_hdr}</span><span style="flex:0.5;text-align:center">GO</span></div>')
        for i,it in enumerate(items[3:18],4):
            tk=it["ticker"]; q2=quick_quote(tk); cc2=G if q2["chg"]>=0 else R; stripe=f"background:{BDR}22;" if i%2==0 else ""
            metric=(f'{it["watchers"]:,}' if src=="StockTwits" and it.get("watchers") else f'{it.get("chg",0):+.2f}%')
            rc=st.columns([0.6,1.6,0.9,0.8,0.9,0.5])
            with rc[0]: st.html(f'<div style="{stripe}padding:8px 4px;font-weight:800;font-size:13px">{tk}</div>')
            with rc[1]: st.html(f'<div style="{stripe}padding:8px 4px;font-size:12px;color:{MUT};overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{it["name"][:30]}</div>')
            with rc[2]: st.html(f'<div style="{stripe}padding:8px 4px;text-align:right;font-weight:600">${q2["price"]:,.2f}</div>')
            with rc[3]: st.html(f'<div style="{stripe}padding:8px 4px;text-align:right;color:{cc2};font-weight:600">{"▲" if q2["chg"]>=0 else "▼"} {abs(q2["chg"]):.2f}%</div>')
            with rc[4]: st.html(f'<div style="{stripe}padding:8px 4px;text-align:right;color:{BLU};font-weight:600">{metric}</div>')
            with rc[5]:
                if st.button("Go",key=f"tr_{tk}_{i}",width='stretch'): go_analyze(tk)

# ── Tab 5: Fundamentals ────────────────────────────────────
with tab_fund:
    if r is None: st.info("Search for a stock in the bar at the top of the page to begin.")
    else:
        f_=r["fund"]; nm=f_.get("company_name") or r["ticker"]
        st.html(f"""<div style="margin-bottom:20px"><div style="font-size:22px;font-weight:800">{nm}</div><div style="color:{MUT};font-size:13px">{f_.get('sector','?')} · {f_.get('industry','?')} · {r['ticker']} · {fmoney(f_.get('market_cap'))}</div></div>""")
        c_l,c_r=st.columns(2)
        HL={"Analyst Price Target","Trailing P/E","Forward P/E","Revenue Growth","Return on Equity","EPS (TTM)"}
        def fund_rows(col_,title,rows_data):
            with col_:
                st.html(f"<div style='font-size:13px;font-weight:700;margin-bottom:10px'>{title}</div>")
                for i,(lbl,val) in enumerate(rows_data):
                    bg=f"background:{BDR}22;" if i%2==0 else ""; hi_=GLD if lbl in HL else TXT
                    st.html(f"""<div class="fund-row" style="{bg}"><span style="color:{MUT};font-size:12px">{lbl}</span><span style="font-weight:600;color:{hi_};font-size:13px">{val}</span></div>""")
        fund_rows(c_l,"Key Metrics",[("Current Price",fdollar(f_.get("current_price"))),("Analyst Target",fdollar(f_.get("analyst_target"))),("52W High",fdollar(f_.get("52wh"))),("52W Low",fdollar(f_.get("52wl"))),("EPS (TTM)",fnum(f_.get("eps_ttm"))),("Beta",fnum(f_.get("beta"))),("Avg Volume",fmoney(f_.get("avg_vol"))),("Dividend Yield",fpct(f_.get("dividend_yield")))])
        fund_rows(c_r,"Valuation & Health",[("Trailing P/E",fnum(f_.get("pe_ratio"))),("Forward P/E",fnum(f_.get("forward_pe"))),("PEG Ratio",fnum(f_.get("peg_ratio"))),("Price/Book",fnum(f_.get("price_to_book"))),("Revenue Growth",fpct(f_.get("revenue_growth"))),("Earnings Growth",fpct(f_.get("earnings_growth"))),("Profit Margin",fpct(f_.get("profit_margin"))),("Return on Equity",fpct(f_.get("roe"))),("Debt/Equity",fnum(f_.get("debt_to_equity"))),("Current Ratio",fnum(f_.get("current_ratio"))),("Short Ratio",fnum(f_.get("short_ratio"))),("Insider Own.",fpct(f_.get("insider_ownership")))])
        st.divider()
        st.html(f"<div style='font-size:13px;font-weight:700;margin-bottom:12px'>Fundamental Factor Scores</div>")
        for sn,sv in r["modules"]["fundamental"].get("signals",{}).items():
            sd,sv_n=(sv[0],float(sv[1])) if isinstance(sv,tuple) else (str(sv),0.); c_=sc(sv_n); pct=int(abs(sv_n)*100)
            st.html(f"""<div class="card" style="border-left:3px solid {c_};padding:12px 16px"><div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px"><span style="font-weight:600;font-size:13px">{sn}</span><span style="color:{c_};font-weight:700;font-size:16px">{sv_n:+.2f}</span></div><div style="color:{MUT};font-size:11px;margin-bottom:6px">{sd}</div><div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{c_}"></div></div></div>""")
        st.divider()
        st.html(f"<div style='font-size:18px;font-weight:800;margin-bottom:12px'>Earnings Calendar & History</div>")
        with st.spinner("Loading earnings…"): earn=fetch_earnings_data(r["ticker"])
        cal=earn.get("calendar",{}); next_date=None
        if cal:
            ed=cal.get("Earnings Date") or cal.get("earningsDate")
            if ed is not None:
                try:
                    if isinstance(ed,(list,tuple)) and len(ed)>0: next_date=pd.to_datetime(ed[0])
                    elif isinstance(ed,dict): next_date=pd.to_datetime(list(ed.values())[0])
                    else: next_date=pd.to_datetime(ed)
                except: next_date=None
        ec1,ec2,ec3=st.columns(3)
        with ec1:
            if next_date is not None:
                try:
                    nd_n=next_date.to_pydatetime().replace(tzinfo=None); days_to=max(0,(nd_n-datetime.now()).days); urg_c=R if days_to<=7 else GLD if days_to<=30 else G
                    st.html(f"""<div class="card" style="border-left:4px solid {urg_c}"><div style="font-size:10px;color:{MUT};letter-spacing:.1em;margin-bottom:6px">NEXT EARNINGS</div><div style="font-size:26px;font-weight:900;color:{urg_c}">{days_to}d</div><div style="font-size:12px;color:{MUT}">{nd_n.strftime('%b %d, %Y')}</div></div>""")
                except: st.html(f'<div class="card-sm"><div style="color:{MUT};font-size:12px">Date unavailable</div></div>')
            else: st.html(f'<div class="card-sm"><div style="color:{MUT};font-size:12px">Date unavailable</div></div>')
        with ec2:
            eps_est=cal.get("Earnings Average") or cal.get("epsAverage")
            st.html(f"""<div class="card"><div style="font-size:10px;color:{MUT};letter-spacing:.1em;margin-bottom:6px">EPS ESTIMATE</div><div style="font-size:26px;font-weight:900;color:{TXT}">{fdollar(eps_est) if eps_est else '—'}</div><div style="font-size:12px;color:{MUT}">Analyst consensus</div></div>""")
        with ec3:
            rev_est=cal.get("Revenue Average") or cal.get("revenueAverage")
            st.html(f"""<div class="card"><div style="font-size:10px;color:{MUT};letter-spacing:.1em;margin-bottom:6px">REVENUE ESTIMATE</div><div style="font-size:26px;font-weight:900;color:{TXT}">{fmoney(rev_est) if rev_est else '—'}</div><div style="font-size:12px;color:{MUT}">Analyst consensus</div></div>""")
        history=earn.get("history",[])
        if history:
            fig_earn=earnings_chart(history)
            if fig_earn: st.plotly_chart(fig_earn,width='stretch',config={"displayModeBar":False})
            st.html(f"""<div style="display:flex;font-size:10px;font-weight:700;color:{MUT};letter-spacing:.1em;padding:8px 12px;border-bottom:2px solid {BDR};border-top:1px solid {BDR}"><span style="flex:1">QUARTER</span><span style="flex:1;text-align:right">ACTUAL EPS</span><span style="flex:1;text-align:right">ESTIMATE</span><span style="flex:1;text-align:right">SURPRISE</span><span style="flex:0.6;text-align:center">RESULT</span></div>""")
            for i,h in enumerate(reversed(history[-8:])):
                actual=h.get("actual"); estimate=h.get("estimate"); surprise=h.get("surprise")
                beat=actual is not None and estimate is not None and actual>=estimate
                res_c=G if beat else R if(actual is not None and estimate is not None) else MUT
                res_lbl="Beat" if beat else "Miss" if(actual is not None and estimate is not None) else "—"; stripe=f"background:{BDR}22;" if i%2==0 else ""
                st.html(f"""<div style="{stripe}display:flex;padding:8px 12px;border-bottom:1px solid {BDR};font-size:12px;align-items:center"><span style="flex:1;color:{MUT}">{h.get('date','—')}</span><span style="flex:1;text-align:right;font-weight:700;color:{res_c}">{fdollar(actual) if actual is not None else '—'}</span><span style="flex:1;text-align:right;color:{MUT}">{fdollar(estimate) if estimate is not None else '—'}</span><span style="flex:1;text-align:right;color:{res_c};font-weight:600">{f'{surprise:+.1f}%' if surprise is not None else '—'}</span><span style="flex:0.6;text-align:center;color:{res_c};font-weight:700">{res_lbl}</span></div>""")
        else: st.html(f'<div class="card" style="text-align:center;padding:24px"><div style="color:{MUT}">No earnings history for {r["ticker"]}</div></div>')

# ── Tab 6: Market Context ──────────────────────────────────
with tab_ctx:
    @st.cache_data(ttl=3600,show_spinner=False)
    def yield_data():
        ylds={}
        def _get(n,s):
            try: return n,round(float(yf.Ticker(s).history(period="5d")["Close"].iloc[-1]),3)
            except: return n,None
        with ThreadPoolExecutor(max_workers=4) as ex:
            for n,v in ex.map(lambda x:_get(*x),{"3M":"^IRX","5Y":"^FVX","10Y":"^TNX","30Y":"^TYX"}.items()): ylds[n]=v
        try:
            t10=yf.Ticker("^TNX").history(period="2y")["Close"]; t3=yf.Ticker("^IRX").history(period="2y")["Close"]
            com=t10.index.intersection(t3.index); sdf=pd.DataFrame({"10Y":t10.loc[com],"3M":t3.loc[com]}); sdf["Spread"]=sdf["10Y"]-sdf["3M"]
            sdf.index=pd.to_datetime(sdf.index).tz_localize(None); ylds["sdf"]=sdf
        except: ylds["sdf"]=None
        return ylds

    with st.spinner("Loading macro data…"): ylds=yield_data()
    c_yc,c_sp=st.columns(2)
    with c_yc:
        durs=["3M","5Y","10Y","30Y"]; vals=[ylds.get(d) for d in durs]
        fig_yc=go.Figure(go.Bar(x=durs,y=vals,marker_color=[G if(v or 0)>0 else R for v in vals],opacity=.85,text=[f"{v:.2f}%" if v else "N/A" for v in vals],textposition="outside",textfont=dict(color=TXT,size=12)))
        fig_yc.update_layout(**CBASE,height=260,yaxis=dict(gridcolor=CGRID,ticksuffix="%",side="right"),xaxis=dict(gridcolor=CGRID,showgrid=False),title=dict(text="US Treasury Yield Curve",font=dict(color=TXT,size=14)))
        st.plotly_chart(fig_yc,width='stretch',config={"displayModeBar":False})
        sp10_3m=(ylds["10Y"]-ylds["3M"]) if ylds.get("10Y") and ylds.get("3M") else None
        spc=G if(sp10_3m or 0)>=0 else R; status="Normal curve" if(sp10_3m or 0)>=0 else "INVERTED — has preceded every US recession since 1955"
        st.html(f"""<div class="expl"><strong>Reading the yield curve:</strong> Normally long-term rates are higher than short-term. When inverted it has historically warned of recession 12-18 months ahead.<br><br><strong>10Y-3M Spread: <span style="color:{spc}">{f'{sp10_3m:.2f}%' if sp10_3m else 'N/A'} — {status}</span></strong></div>""")
    with c_sp:
        sdf=ylds.get("sdf")
        if sdf is not None:
            sc_colors=[f"rgba({GR},.7)" if v>=0 else f"rgba({RR},.7)" for v in sdf["Spread"]]
            fig_sp=go.Figure()
            fig_sp.add_shape(type="rect",x0=sdf.index[0],x1=sdf.index[-1],y0=-20,y1=0,fillcolor=f"rgba({RR},.05)",line_width=0)
            fig_sp.add_shape(type="line",x0=sdf.index[0],x1=sdf.index[-1],y0=0,y1=0,line=dict(color=R,width=1.5,dash="dot"))
            fig_sp.add_trace(go.Bar(x=sdf.index,y=sdf["Spread"].values,marker_color=sc_colors,name="10Y-3M Spread"))
            fig_sp.update_layout(**CBASE,height=260,yaxis=dict(gridcolor=CGRID,ticksuffix="%",side="right"),xaxis=dict(gridcolor=CGRID,showgrid=False),title=dict(text="10Y-3M Treasury Spread (2Y)",font=dict(color=TXT,size=14)))
            st.plotly_chart(fig_sp,width='stretch',config={"displayModeBar":False})
        else: st.info("Spread data unavailable.")
    st.divider()
    ctx_c1,ctx_c2,ctx_c3=st.columns(3)
    vix_=r["vix"] if r else 20.; vix_c=R if vix_>28 else G if vix_<15 else GLD
    vix_txt=("Extreme fear — often a contrarian buy signal" if vix_>40 else "Elevated fear — proceed carefully" if vix_>28 else "Calm market — watch for complacency" if vix_>15 else "Very low volatility — markets may be overconfident")
    with ctx_c1:
        st.html(f"""<div class="card"><div style="font-size:10px;color:{MUT};letter-spacing:.1em;margin-bottom:8px">VIX — FEAR GAUGE</div><div style="font-size:40px;font-weight:900;color:{vix_c}">{vix_:.1f}</div><div style="font-size:12px;color:{MUT};margin-top:4px">{vix_txt}</div><div class="expl" style="margin-top:10px">VIX above 30 = panic. VIX below 15 = complacency. Contrarians buy fear and sell greed.</div></div>""")
    spy_=r["spy"] if r else 0.; spy_c=G if spy_>=0 else R
    with ctx_c2:
        st.html(f"""<div class="card"><div style="font-size:10px;color:{MUT};letter-spacing:.1em;margin-bottom:8px">S&P 500 — 3 MONTH</div><div style="font-size:40px;font-weight:900;color:{spy_c}">{spy_:+.1f}%</div><div style="font-size:12px;color:{MUT};margin-top:4px">{'Rising market — tailwind for stocks' if spy_>=0 else 'Falling market — headwind for stocks'}</div><div class="expl" style="margin-top:10px">About 70% of stocks move with the overall market. A rising tide lifts most boats.</div></div>""")
    ln_=r["modules"]["lunar"] if r else {}; lsc=ln_.get("score",0); lc=G if lsc>0 else R if lsc<0 else MUT
    with ctx_c3:
        st.html(f"""<div class="card"><div style="font-size:10px;color:{MUT};letter-spacing:.1em;margin-bottom:8px">LUNAR PHASE</div><div style="font-size:24px;font-weight:800;color:{lc}">{ln_.get('phase','Unknown')}</div><div style="font-size:13px;color:{MUT};margin-top:6px">{ln_.get('illum','?')} lit · New moon in {ln_.get('d_new','?')}d</div><div class="expl" style="margin-top:10px">Dichev &amp; Janes (2003): stocks near new moons average ~0.5% higher returns vs full moons.</div></div>""")
    st.divider()
    st.html(f"<div style='font-size:15px;font-weight:700;margin-bottom:12px'>Commodity Prices</div>")
    comm_c1,comm_c2=st.columns(2)
    for col_,(sym,name,icon) in zip([comm_c1,comm_c2],[("GC=F","Gold","Gold"),("CL=F","Crude Oil","Oil")]):
        with col_:
            try:
                ch=yf.Ticker(sym).history(period="5d")["Close"]; cv=float(ch.iloc[-1]); cp=float(ch.iloc[-2]) if len(ch)>1 else cv
                cc_c=G if cv>=cp else R; pct_c=(cv-cp)/cp*100 if cp else 0
                st.html(f"""<div class="card"><div style="font-size:10px;color:{MUT};letter-spacing:.1em;margin-bottom:6px">{icon.upper()}</div><div style="font-size:28px;font-weight:900;color:{TXT}">${cv:,.2f}</div><div style="font-size:14px;color:{cc_c};font-weight:700;margin-top:4px">{'▲' if cv>=cp else '▼'} {abs(pct_c):.2f}%</div></div>""")
            except: st.html(f'<div class="card-sm"><div style="color:{MUT}">{name}: unavailable</div></div>')
