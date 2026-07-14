"""
SearchScope demo app.

Presents this project's real, already-verified findings interactively.
Does not run the live retrieval/reranking/LTR pipeline. See data.py's
module docstring for why. Every number displayed here is sourced from a
results/*.md file in the main project; run `streamlit run app.py` to
launch locally, or see README.md in this folder for deployment.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data

st.set_page_config(
    page_title="SearchScope | Search Quality Engineering Demo",
    page_icon="🔍",
    layout="wide",
)

ACCENT = "#4c6ef5"
GOOD = "#2f9e44"
BAD = "#e03131"
WARN = "#e8590c"

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("SearchScope")
st.caption(
    "Hybrid retrieval, learning-to-rank, and grounded QA built end to end "
    "against MS MARCO / TREC DL 2019. Every chart on this page is real, "
    "measured data, not a mockup."
)

col1, col2, col3 = st.columns(3)
col1.link_button("Code", "https://github.com/sanketmbhujbal/searchscope", width='stretch')
col2.link_button("Blog", "https://open.substack.com/pub/sanketbhujbal/p/why-i-stopped-trusting-my-own-evaluation?r=5p0kv6&utm_campaign=post-expanded-share&utm_medium=post%20viewer", width='stretch')
col3.link_button("Design Doc", "https://github.com/sanketmbhujbal/searchscope/", width='stretch')

st.divider()

tab_overview, tab_eval, tab_vocab, tab_personalization, tab_qa = st.tabs(
    ["Overview", "Evaluation Dashboard", "Vocabulary Mismatch", "Personalization", "QA Rejection Gate"]
)

# ---------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------
with tab_overview:
    st.subheader("Four things that looked wrong, or looked suspiciously right")
    st.write(
        "Most of what this project actually found came from double-checking "
        "results that were too clean or too convenient to take at face value. "
        "Each card below is a real finding, not an illustration."
    )

    card_style = """
    <style>
    .findings-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 8px;
    }
    .finding-card {
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 18px 20px;
        background: #ffffff;
        display: flex;
        flex-direction: column;
        gap: 8px;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .finding-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        border-color: #4c6ef5;
    }
    .finding-card .title { font-weight: 600; font-size: 1.02rem; color: #212529; }
    .finding-card .body { font-size: 0.92rem; color: #343a40; line-height: 1.5; flex-grow: 1; }
    .finding-card .source { font-size: 0.78rem; color: #868e96; }
    @media (max-width: 900px) {
        .findings-grid { grid-template-columns: 1fr; }
    }
    </style>
    """

    cards_html = "".join(
        f"""<div class="finding-card">
                <div class="title">{f['title']}</div>
                <div class="body">{f['body']}</div>
                <div class="source">Source: {f['source']}</div>
            </div>"""
        for f in data.KEY_FINDINGS
    )

    st.markdown(card_style + f'<div class="findings-grid">{cards_html}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("What this app is, and isn't")
    st.write(
        "This is a static presentation of real, already-computed results. "
        "It does not accept an arbitrary query and run it through a live "
        "index; the actual pipeline needs Pyserini (a JVM-backed BM25 "
        "index), FAISS, PyTorch, and XGBoost, a heavier stack than a "
        "lightweight hosted demo should carry. Every specific query, "
        "score, and passage shown in the other tabs came from a real run "
        "against the project's actual retrieval and ranking pipeline."
    )

# ---------------------------------------------------------------------
# Tab: Evaluation Dashboard
# ---------------------------------------------------------------------
with tab_eval:
    st.subheader("NDCG@10 across the pipeline")
    df_pipeline = pd.DataFrame(data.PIPELINE_METRICS)

    fig = go.Figure()
    colors = ["#9aa5b1", ACCENT, WARN, GOOD, GOOD]
    fig.add_trace(go.Bar(
        x=df_pipeline["stage"], y=df_pipeline["ndcg10"],
        marker_color=colors,
        text=[f"{v:.2f}" for v in df_pipeline["ndcg10"]],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_title="NDCG@10", yaxis_range=[0, 0.9],
        height=420, showlegend=False,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Hybrid fusion (orange) underperforms dense retrieval alone. "
        "Cross-encoder reranking recovers the loss and exceeds dense "
        "retrieval on its own. See results/day1-2_findings.md and "
        "results/day3_findings.md."
    )

    st.divider()
    st.subheader("The leak: a result too good to be real")
    df_leak = pd.DataFrame(data.LEAK_COMPARISON)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df_leak["label"], y=df_leak["ndcg10"],
        marker_color=["#9aa5b1", BAD, GOOD],
        text=[f"{v:.2f}" for v in df_leak["ndcg10"]],
        textposition="outside",
    ))
    fig2.update_layout(
        yaxis_title="NDCG@10", yaxis_range=[0, 1.1],
        height=380, showlegend=False,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig2, width='stretch')
    st.caption(
        "The first LTR run scored 0.97 NDCG@10 because a synthetic "
        "training feature was built from the label it was supposed to "
        "predict. The corrected number, 0.73, is roughly flat against the "
        "reranker, which is the expected result given ~40 training "
        "queries. See results/day5-6_findings.md."
    )

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Feature importance (SHAP)")
        df_shap = pd.DataFrame(data.SHAP_IMPORTANCE)
        df_shap["synthetic"] = df_shap["feature"].apply(
            lambda f: "Synthetic" if f in data.SYNTHETIC_FEATURES else "Real"
        )
        fig3 = px.bar(
            df_shap.sort_values("importance"), x="importance", y="feature",
            orientation="h", color="synthetic",
            color_discrete_map={"Real": ACCENT, "Synthetic": WARN},
        )
        fig3.update_layout(height=560, margin=dict(t=10, b=10), legend_title="")
        st.plotly_chart(fig3, width='stretch')
        st.caption(
            "Synthetic features (orange) show real, nonzero SHAP importance "
            "despite being random values with no true signal, except "
            "role_doc_affinity, which is a literal constant and correctly "
            "shows zero. See results/day7_findings.md."
        )

    with right:
        st.subheader("Ablation study")
        df_ablation = pd.DataFrame(data.ABLATION_RESULTS)
        fig4 = go.Figure()
        bar_colors = [BAD if d < -0.05 else ("#9aa5b1" if abs(d) < 0.02 else WARN)
                      for d in df_ablation["delta"]]
        fig4.add_trace(go.Bar(
            x=df_ablation["delta"], y=df_ablation["ablation"],
            orientation="h", marker_color=bar_colors,
            text=[f"{v:+.3f}" for v in df_ablation["delta"]],
            textposition="outside",
        ))
        fig4.update_layout(
            xaxis_title="NDCG@10 delta vs. full model",
            height=560, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig4, width='stretch')
        st.caption(
            "Removing semantic signals or reducing to BM25 alone causes a "
            "large drop (red). Every other category sits inside a small "
            "band that's plausibly LOQO cross-validation noise at 43 "
            "queries, not a reliably ranked difference between categories."
        )

# ---------------------------------------------------------------------
# Tab: Vocabulary Mismatch
# ---------------------------------------------------------------------
with tab_vocab:
    st.subheader("BM25 misses an answer that dense retrieval finds")
    st.write(
        "Mined directly from the project's real persisted retrieval "
        "indexes (`eval/mine_failure_cases.py`), not constructed for this "
        "demo. See results/day9_failure_analysis.md."
    )

    def render_vocab_case(case, label):
        st.markdown(f"#### {label}: *\"{case['query']}\"*")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("BM25 rank of the correct answer", case["relevant_doc_bm25_rank"])
            st.metric("Dense rank of the correct answer", case["relevant_doc_dense_rank"])
        with c2:
            st.metric("TREC relevance grade", f"{case['relevant_doc_grade']} / 3")

        st.markdown("**The actually relevant passage** (found by dense retrieval, missed by BM25):")
        st.info(case["relevant_doc_text"])

        st.markdown("**What BM25 returned instead at rank 1:**")
        st.error(case["bm25_top1_text"])

    render_vocab_case(data.VOCAB_MISMATCH_CASE, "Query")
    st.divider()
    with st.expander("See a second real example (runner-up case)"):
        render_vocab_case(data.VOCAB_MISMATCH_RUNNER_UP, "Query")

# ---------------------------------------------------------------------
# Tab: Personalization
# ---------------------------------------------------------------------
with tab_personalization:
    st.subheader(f'Same query, different results by role: "{data.PERSONALIZATION_QUERY}"')
    st.write(
        "Real output from `eval/run_personalization_demo.py`. HR shows "
        "genuine differentiation; Engineer, Sales, and Legal largely "
        "collapse into the same shared list, which is itself the finding. "
        "See results/day4_findings.md for why."
    )

    role = st.radio("Choose a role", list(data.PERSONALIZATION_RESULTS.keys()), horizontal=True)
    st.write("")
    for rank, hit in enumerate(data.PERSONALIZATION_RESULTS[role], start=1):
        with st.container(border=True):
            st.markdown(f"**#{rank}** · doc `{hit['doc_id']}` · score {hit['score']:.3f}")
            st.write(hit["text"])

    st.divider()
    st.caption(
        "Notice \"Group Policy\" (doc 1169578, an Active Directory / IT "
        "admin document) appears near the top for three of the four "
        "roles, and scores HR's single highest affinity of the entire "
        "demo. That's TF-IDF rewarding shared surface words like "
        "\"policy\" and \"group,\" not genuine role understanding."
    )

# ---------------------------------------------------------------------
# Tab: QA Rejection Gate
# ---------------------------------------------------------------------
with tab_qa:
    st.subheader("A 100% rejection rate that hides a real problem")

    m1, m2, m3 = st.columns(3)
    m1.metric("Rejection rate (mismatched context)", f"{data.QA_METRICS['rejection_rate_unanswerable']:.0%}")
    m2.metric(
        "False rejections on answerable queries",
        f"{data.QA_METRICS['false_rejections_on_answerable']} / {data.QA_METRICS['total_answerable']}",
    )
    m3.metric("Citation accuracy (after fix)", f"{data.QA_METRICS['citation_hygiene_corrected']:.0%}")

    st.write(
        "The unanswerable-set score is a perfect 100%, exactly the kind "
        "of result that invites you to stop looking. Reading the 7 cases "
        "where the model wrongly rejected a genuinely answerable query "
        "tells a more useful story. See results/day8_findings.md."
    )

    st.divider()
    for case in data.QA_FALSE_REJECTIONS:
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"**\"{case['query']}\"**")
                st.write(case["explanation"])
            with right:
                st.markdown(
                    f"<div style='text-align:right'>"
                    f"<span style='background-color:{case['verdict_color']}; "
                    f"color:white; padding:4px 10px; border-radius:12px; "
                    f"font-size:0.85em;'>{case['verdict']}</span></div>",
                    unsafe_allow_html=True,
                )
