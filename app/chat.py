"""
Streamlit chat UI for the macro forecast agent.
    streamlit run app/chat.py                      # real data (vendor_data/)
    MACROAGENT_DEMO=1 streamlit run app/chat.py    # synthetic demo data
"""
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from macroagent.graph import ask  # noqa: E402
from macroagent.llm import default_llm  # noqa: E402
from macroagent.tools import DataProvider  # noqa: E402

st.set_page_config(page_title="Macro forecast agent", layout="wide")


@st.cache_resource
def provider():
    if os.environ.get("MACROAGENT_DEMO") == "1":
        from macroagent.demo import synthetic_provider
        return synthetic_provider()
    try:
        return DataProvider.from_vendor_data()
    except FileNotFoundError:
        from macroagent.demo import synthetic_provider
        return synthetic_provider()


p = provider()
llm = default_llm()
st.title("Macro forecast agent: Brent crude and UK inflation")
st.caption("LangGraph agent: input guardrail → router → forecasting tools → answer → numeric grounding check → "
           "output guardrail. Every number in an answer comes from a tool output.")
with st.sidebar:
    st.subheader("Data")
    if p.synthetic:
        st.warning("SYNTHETIC demo data: run the component repos and `python sync_vendor.py` for real data.")
    m, b = p.macro.iloc[-1], p.brent.iloc[-1]
    st.metric("Brent (USD/bbl)", f"{b['brent_usd']:.2f}", help=b["date"].strftime("%Y-%m"))
    st.metric("UK CPI inflation", f"{m['cpi_yoy']:.1f}%", help=m["date"].strftime("%Y-%m"))
    st.metric("Bank Rate", f"{m['bank_rate']:.2f}%")
    st.caption("Answers: " + ("Claude (grounding-checked)" if llm else "deterministic templates (no API key)"))
    for k, v in p.sources.items():
        st.caption(f"{k}: {v}")

examples = ["Forecast Brent crude for the next 6 months", "What will UK CPI inflation be in 12 months?",
            "What happens to UK inflation if oil rises 20%?", "How accurate are the models?"]
cols = st.columns(len(examples))
clicked = next((q for c, q in zip(cols, examples) if c.button(q)), None)
question = st.chat_input("Ask about Brent or UK inflation") or clicked

if question:
    with st.chat_message("user"):
        st.write(question)
    r = ask(question, p, llm)
    with st.chat_message("assistant"):
        (st.error if r["status"] == "blocked" else st.write)(r["answer"])
        st.caption(f"intent: {r['intent']} · router: {r['router']} · answer: {r['answer_source']} · "
                   f"grounded: {(r['grounding'] or {}).get('passed')} · guard risk: {r['guard']['risk_score']} · "
                   f"{r['latency_ms']} ms" + (" · SYNTHETIC DATA" if r["synthetic_data"] else ""))
        if r["intent"] in ("oil_forecast", "oil_forecast_to_cpi"):
            from macroagent.tools import oil_forecast
            path = oil_forecast(p, 12)["path"]
            hist = p.brent.tail(60)
            fig = go.Figure([go.Scatter(x=hist["date"], y=hist["brent_usd"], name="Brent"),
                             go.Scatter(x=pd.date_range(hist["date"].iloc[-1] + pd.DateOffset(months=1), periods=12, freq="MS"),
                                        y=path, name="Forecast", line=dict(dash="dash"))])
            st.plotly_chart(fig.update_layout(height=320, template="plotly_white"), use_container_width=True)
        if r["intent"] in ("cpi_forecast", "oil_cpi_scenario"):
            hist = p.macro.tail(120)
            st.plotly_chart(go.Figure([go.Scatter(x=hist["date"], y=hist["cpi_yoy"], name="CPI inflation"),
                                       go.Scatter(x=hist["date"], y=hist["bank_rate"], name="Bank Rate")])
                            .update_layout(height=320, template="plotly_white"), use_container_width=True)
        if r["facts"]:
            with st.expander("Facts used (tool outputs)"):
                st.dataframe(pd.DataFrame({"fact": list(r["facts"]), "value": [str(v) for v in r["facts"].values()]}),
                             hide_index=True, use_container_width=True)
                st.caption("Sources: " + "; ".join(s for s in r["sources"] if s))
