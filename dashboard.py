"""Vanguard ETF dashboard.

Run from the project folder:
    streamlit run dashboard.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "vetf_all.csv"
PLOTS = ROOT / "allplots"

# label -> (yfinance period, bar interval)
TIMEFRAMES = {
    "1 Day": ("1d", "5m"),
    "5 Days": ("5d", "15m"),
    "1 Month": ("1mo", "1h"),
    "3 Months": ("3mo", "1d"),
    "6 Months": ("6mo", "1d"),
    "YTD": ("ytd", "1d"),
    "1 Year": ("1y", "1d"),
    "5 Years": ("5y", "1wk"),
    "Max": ("max", "1mo"),
}
# The five panels of the original allplots figures.
PANEL_TIMEFRAMES = ["6 Months", "3 Months", "1 Month", "5 Days", "1 Day"]

RETURN_COLS = {
    "YTD": "return_ytd_pct",
    "1 Year": "return_1y_pct",
    "5 Years": "return_5y_pct",
    "10 Years": "return_10y_pct",
    "Since inception": "return_since_inception_pct",
}

st.set_page_config(page_title="Vanguard ETF Dashboard", page_icon="📈", layout="wide")


@st.cache_data
def load_etfs() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["inception_date"] = pd.to_datetime(df["inception_date"], format="mixed")
    df["category"] = df["category"].fillna("—")
    df["label"] = df["symbol"] + " · " + df["name"]
    return df


@st.cache_data(ttl=900, show_spinner="Fetching prices from Yahoo Finance…")
def fetch_close(tickers: tuple[str, ...], period: str, interval: str) -> pd.DataFrame:
    """Close prices, one column per ticker. Cached for 15 minutes."""
    raw = yf.download(list(tickers), period=period, interval=interval,
                      auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.dropna(how="all")


def hide_market_gaps(fig: go.Figure, interval: str) -> None:
    """Remove weekends (and overnight hours for intraday bars) from the x-axis."""
    if interval in ("1mo", "1wk"):
        return
    breaks = [dict(bounds=["sat", "mon"])]
    if interval.endswith(("m", "h")):
        breaks.append(dict(bounds=[16, 9.5], pattern="hour"))
    fig.update_xaxes(rangebreaks=breaks)


def period_change(series: pd.Series) -> float | None:
    s = series.dropna()
    if len(s) < 2:
        return None
    return (s.iloc[-1] / s.iloc[0] - 1) * 100


# ---------------------------------------------------------------- sidebar filters
etfs = load_etfs()

st.sidebar.title("Filters")
asset_classes = st.sidebar.multiselect("Asset class", sorted(etfs["asset_class"].unique()))
pool = etfs[etfs["asset_class"].isin(asset_classes)] if asset_classes else etfs
categories = st.sidebar.multiselect("Category", sorted(pool["category"].unique()))
risk = st.sidebar.slider("Risk level", 1, 5, (1, 5))
max_er = float(etfs["expense_ratio_pct"].max())
er_cap = st.sidebar.slider("Max expense ratio (%)", 0.0, max_er, max_er, step=0.01)
search = st.sidebar.text_input("Search symbol or name")

filtered = etfs.copy()
if asset_classes:
    filtered = filtered[filtered["asset_class"].isin(asset_classes)]
if categories:
    filtered = filtered[filtered["category"].isin(categories)]
filtered = filtered[filtered["risk_level"].between(*risk) & (filtered["expense_ratio_pct"] <= er_cap)]
if search:
    mask = (filtered["symbol"].str.contains(search, case=False)
            | filtered["name"].str.contains(search, case=False))
    filtered = filtered[mask]

st.sidebar.caption(f"{len(filtered)} of {len(etfs)} ETFs match")
st.sidebar.caption("Snapshot data: prices as of 2026-09-11, returns as of 2026-08-31. "
                   "Charts use live Yahoo Finance data.")

# ---------------------------------------------------------------- header
st.title("📈 Vanguard ETF Dashboard")
if filtered.empty:
    st.warning("No ETFs match the current filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("ETFs", len(filtered))
k2.metric("Median expense ratio", f"{filtered['expense_ratio_pct'].median():.2f}%")
k3.metric("Median YTD return", f"{filtered['return_ytd_pct'].median():.2f}%")
k4.metric("Median 1-yr return", f"{filtered['return_1y_pct'].median():.2f}%")

tab_overview, tab_perf, tab_compare, tab_gallery = st.tabs(
    ["Overview", "ETF performance", "Compare ETFs", "Saved plots"])

# ---------------------------------------------------------------- overview
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        horizon = st.selectbox("Return horizon", list(RETURN_COLS), key="ov_horizon")
        col = RETURN_COLS[horizon]
        top_n = st.slider("Show top / bottom N", 5, 30, 15)
        ranked = filtered.dropna(subset=[col]).sort_values(col, ascending=False)
        shown = pd.concat([ranked.head(top_n), ranked.tail(top_n)]).drop_duplicates("symbol")
        fig = px.bar(shown.sort_values(col), x=col, y="symbol", color="asset_class",
                     orientation="h", hover_name="name",
                     labels={col: f"{horizon} return (%)", "symbol": ""},
                     title=f"{horizon} return — top & bottom {top_n}")
        fig.update_layout(height=max(400, 22 * len(shown)))
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.scatter(filtered.dropna(subset=[col]), x="expense_ratio_pct", y=col,
                         color="asset_class", size="risk_level", hover_name="label",
                         labels={"expense_ratio_pct": "Expense ratio (%)", col: f"{horizon} return (%)"},
                         title=f"Cost vs {horizon} return")
        st.plotly_chart(fig, width="stretch")

        fig = px.box(filtered, x="risk_level", y=col, color="asset_class", points="all",
                     hover_name="symbol",
                     labels={"risk_level": "Risk level", col: f"{horizon} return (%)"},
                     title=f"{horizon} return by risk level")
        st.plotly_chart(fig, width="stretch")

    counts = filtered.groupby(["asset_class", "category"]).size().reset_index(name="count")
    fig = px.sunburst(counts, path=["asset_class", "category"], values="count",
                      title="ETFs by asset class and category")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        filtered.drop(columns=["label"]),
        width="stretch", hide_index=True,
        column_config={
            "inception_date": st.column_config.DateColumn("inception_date"),
            "price_usd": st.column_config.NumberColumn("price_usd", format="$%.2f"),
        },
    )

# ---------------------------------------------------------------- single ETF performance
with tab_perf:
    label = st.selectbox("ETF", filtered["label"], key="perf_etf")
    row = filtered[filtered["label"] == label].iloc[0]
    sym = row["symbol"]

    m = st.columns(5)
    m[0].metric("Price (snapshot)", f"${row['price_usd']:.2f}", f"{row['price_change_pct']:+.2f}%")
    m[1].metric("Expense ratio", f"{row['expense_ratio_pct']:.2f}%")
    m[2].metric("SEC yield", "—" if pd.isna(row["sec_yield_pct"]) else f"{row['sec_yield_pct']:.2f}%")
    m[3].metric("Risk", f"{row['risk_level']} / 5")
    m[4].metric("Inception", row["inception_date"].strftime("%Y-%m-%d"))

    view = st.radio("View", ["Five panels (like saved plots)", "Single timeframe"], horizontal=True)

    if view.startswith("Five"):
        fig = make_subplots(rows=1, cols=5, subplot_titles=PANEL_TIMEFRAMES)
        for i, tf in enumerate(PANEL_TIMEFRAMES, start=1):
            period, interval = TIMEFRAMES[tf]
            close = fetch_close((sym,), period, interval)
            if close.empty or sym not in close:
                continue
            s = close[sym].dropna()
            chg = period_change(s)
            color = "#16a34a" if (chg or 0) >= 0 else "#dc2626"
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=color),
                                     name=tf, hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>"),
                          row=1, col=i)
            if chg is not None:
                fig.layout.annotations[i - 1].text = f"{tf} ({chg:+.2f}%)"
            breaks = [dict(bounds=["sat", "mon"])]
            if interval.endswith(("m", "h")):
                breaks.append(dict(bounds=[16, 9.5], pattern="hour"))
            fig.update_xaxes(rangebreaks=breaks, row=1, col=i)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_layout(height=420, showlegend=False, title=f"{sym} — historical close price")
        st.plotly_chart(fig, width="stretch")
    else:
        tf = st.segmented_control("Timeframe", list(TIMEFRAMES), default="6 Months", key="perf_tf") or "6 Months"
        period, interval = TIMEFRAMES[tf]
        close = fetch_close((sym,), period, interval)
        if close.empty or sym not in close:
            st.info(f"No {tf} data returned for {sym}.")
        else:
            s = close[sym].dropna()
            chg = period_change(s)
            fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", fill="tozeroy",
                                       line=dict(color="#16a34a" if (chg or 0) >= 0 else "#dc2626")))
            fig.update_yaxes(range=[s.min() * 0.995, s.max() * 1.005], title="Price (USD)")
            fig.update_layout(height=500, title=f"{sym} — {tf}"
                              + (f" ({chg:+.2f}%)" if chg is not None else ""))
            hide_market_gaps(fig, interval)
            st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- compare
with tab_compare:
    default = filtered["label"].head(3).tolist()
    picks = st.multiselect("ETFs to compare (up to 10)", filtered["label"], default=default,
                           max_selections=10)
    c1, c2 = st.columns([3, 1])
    tf = c1.segmented_control("Timeframe", list(TIMEFRAMES), default="1 Year", key="cmp_tf") or "1 Year"
    normalize = c2.toggle("Show % change", value=True)

    if picks:
        syms = tuple(filtered.set_index("label").loc[picks, "symbol"])
        period, interval = TIMEFRAMES[tf]
        close = fetch_close(syms, period, interval)
        if close.empty:
            st.info("No price data returned.")
        else:
            plot_df = close.ffill()
            if normalize:
                plot_df = (plot_df / plot_df.bfill().iloc[0] - 1) * 100
            long = plot_df.reset_index().melt(id_vars=plot_df.index.name or "index",
                                              var_name="symbol", value_name="value")
            xcol = plot_df.index.name or "index"
            fig = px.line(long, x=xcol, y="value", color="symbol",
                          labels={"value": "Change (%)" if normalize else "Price (USD)", xcol: ""},
                          title=f"{tf} {'% change' if normalize else 'price'}")
            hide_market_gaps(fig, interval)
            fig.update_layout(height=520, hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

            summary = pd.DataFrame({
                "symbol": close.columns,
                f"{tf} change (%)": [period_change(close[c]) for c in close.columns],
                "last price ($)": [close[c].dropna().iloc[-1] if close[c].notna().any() else None
                                   for c in close.columns],
            }).merge(etfs[["symbol", "name", "expense_ratio_pct", "risk_level"]], on="symbol", how="left")
            st.dataframe(summary.sort_values(f"{tf} change (%)", ascending=False),
                         hide_index=True, width="stretch")

# ---------------------------------------------------------------- saved PNG gallery
with tab_gallery:
    available = {p.stem.removesuffix("_performance"): p for p in PLOTS.glob("*_performance.png")}
    in_gallery = filtered[filtered["symbol"].isin(available)]
    st.caption(f"Static figures from `allplots/` for the {len(in_gallery)} filtered ETFs that have one.")

    g1, g2 = st.columns([1, 3])
    per_page = g1.selectbox("Plots per page", [5, 10, 20, 50], index=1)
    pages = max(1, -(-len(in_gallery) // per_page))
    page = g2.number_input(f"Page (of {pages})", 1, pages, 1)

    for _, r in in_gallery.iloc[(page - 1) * per_page: page * per_page].iterrows():
        with st.container(border=True):
            st.markdown(f"**{r['symbol']}** — {r['name']}  ·  {r['asset_class']} / {r['category']}  ·  "
                        f"risk {r['risk_level']}  ·  ER {r['expense_ratio_pct']:.2f}%")
            st.image(str(available[r["symbol"]]), width="stretch")
