import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.universe import load_master,provider_symbol
from src.market_data import history
from src.sentiment import fetch,aggregate
from src.predict import predict

st.set_page_config(page_title="StockAI India",page_icon="📈",layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.5rem;max-width:1450px}
.hero {padding:1.5rem;border-radius:20px;background:linear-gradient(120deg,#101d38,#2a4777);color:white;margin-bottom:1rem}
.hero h1 {font-size:2rem;margin:0 0 .4rem 0}.hero p {color:#dbe7ff;margin:0}
[data-testid="stMetric"] {background:#f7f9fc;padding:1rem;border:1px solid #e7edf5;border-radius:14px}
</style>
<div class="hero"><h1>StockAI <span style="color:#9fb7ff">India</span></h1>
<p>NSE &amp; BSE · Market intelligence · Next-session forecast · Company news sentiment</p></div>
""",unsafe_allow_html=True)
@st.cache_data(ttl=300)
def master_data(): return load_master()
try: master=master_data()
except Exception as e: st.error(f"Company master error: {e}"); st.stop()
st.caption(f"Searchable exchange listing records: {len(master):,}. Price data is provider-sourced and may be delayed.")
q=st.text_input("Search company",placeholder="Enter 2+ characters: rel, tcs, infosys…")
if len(q.strip())<2:
    st.info("Type at least two characters to search."); shown=master.head(10)
else:
    q=q.casefold().strip()
    shown=master[master.search_name.str.contains(q,regex=False,na=False)|master.search_ticker.str.contains(q,regex=False,na=False)].head(100)
if shown.empty: st.warning("No match in the loaded company master."); st.stop()
def label(i):
    r=shown.loc[i]; return f"{r.company_name} · {r.exchange} · {r.ticker or 'No ticker'}"
chosen=st.selectbox("Select a company",list(shown.index),format_func=label)
r=master.loc[chosen]
st.subheader(r.company_name)
a,b,c=st.columns(3); a.metric("Exchange",r.exchange or "—"); b.metric("Ticker",r.ticker or "—"); c.metric("Sector",r.sector or "Not available")
with st.expander("Company identifiers and details"):
    st.write("ISIN:",r.isin or "Not available"); st.write("Security code:",r.security_code or "Not available")
    st.write("Listing status:",r.listing_status or "Not verified")
    if "description" in r and str(r.description).strip(): st.write(r.description)
symbol=provider_symbol(r); st.caption(f"Price-provider mapping: {symbol or 'not available'}")
x,y=st.columns(2); refresh=x.button("↻ Refresh price data",use_container_width=True); refresh_news=y.button("◎ Refresh news sentiment",use_container_width=True)
if not symbol: st.warning("No provider symbol mapped. Company stays searchable; prediction is unavailable until mapping is validated."); st.stop()
with st.spinner("Loading market history…"): h=history(symbol,refresh)
if h.empty: st.error("No history returned by the provider. Check mapping or coverage."); st.stop()
datecol="Date"; h[datecol]=pd.to_datetime(h[datecol],errors="coerce"); h=h.dropna(subset=[datecol,"Close"]).sort_values(datecol)
actual=float(h.Close.iloc[-1]); market_date=h.Date.iloc[-1].date()
key="news_"+str(r.company_name)
if refresh_news or key not in st.session_state:
    with st.spinner("Fetching company news…"): st.session_state[key]=fetch(r.company_name)
news=st.session_state[key]; sent=aggregate(news)
with st.spinner("Evaluating chronological holdout and forecasting…"): p=predict(h,sent["sentiment_score"])
if p.get("status")!="OK": st.warning(p.get("message","Forecast unavailable.")); st.stop()
m1,m2,m3,m4=st.columns(4)
m1.metric("Latest actual close",f"₹{actual:,.2f}",help=f"Provider observation date: {market_date}")
m2.metric("Predicted next close",f"₹{p['predicted_next_close']:,.2f}",f"{p['predicted_change']:+,.2f}")
m3.metric("Expected return",f"{p['sentiment_adjusted_return_pct']:+.2f}%")
m4.metric("Signal",p["decision"],help="Rule-based estimate, not an order.")
st.caption(f"Forecast horizon: next observed trading session after {market_date}. Generated: {p['prediction_timestamp_utc']}")
fig=go.Figure(); fig.add_trace(go.Scatter(x=h.Date,y=h.Close,name="Actual close",mode="lines"))
fig.add_trace(go.Scatter(x=[market_date],y=[p["predicted_next_close"]],name="Predicted next close",mode="markers",marker=dict(size=13,symbol="diamond")))
fig.update_layout(height=370,margin=dict(l=10,r=10,t=25,b=10),legend=dict(orientation="h",y=1.1),xaxis_title="Trading date",yaxis_title="Price (₹)",hovermode="x unified")
st.plotly_chart(fig,use_container_width=True)
a,b,c=st.columns(3); a.metric("News sentiment",sent["sentiment_label"].title()); b.metric("Sentiment score",f"{sent['sentiment_score']:+.3f}"); c.metric("News items",sent["news_count"])
st.caption(sent["sentiment_status"])
with st.expander("Model quality and methodology"):
    st.write("Model:",p["model"]); st.write("Feature schema:",p["feature_schema_version"])
    st.write("Training rows:",p["training_rows"]); st.write("Holdout rows:",p["holdout_rows"])
    st.write(f"Chronological holdout MAE: ₹{p['mae_holdout']:.4f}")
    st.write(f"Chronological holdout RMSE: ₹{p['rmse_holdout']:.4f}")
    st.warning(p["warning"])
with st.expander("Latest company news"):
    if news.empty: st.info("No items retrieved; this does not mean sentiment is neutral.")
    else:
        for _,item in news.head(10).iterrows():
            st.markdown(f"**{item['title']}**"); st.caption(f"{item['published']} · {item['sentiment_label']} · {item['sentiment_score']:+.2f}")
            if item.get("source_url"): st.markdown(f"[Open article]({item['source_url']})")
with st.expander("Recent actual market data"): st.dataframe(h.tail(20),use_container_width=True,hide_index=True)
st.markdown("---")
st.caption("Research demonstration only. Data coverage, forecasts and sentiment are uncertain; this is not financial advice.")
