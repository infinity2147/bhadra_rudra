"""
RUDRA — Shield Against Deception
Fund Flow Intelligence Dashboard with LLM Copilot
Interactive Streamlit application for visualizing fund flows,
detecting suspicious transaction patterns, and generating SAR reports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import json
import os
import sys
import time
import random

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RUDRA — Shield Against Deception",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { font-size: 1.1rem; font-weight: 400; color: #546e7a; margin-top: 0; }
    .metric-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        padding: 20px; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(26,35,126,0.3);
    }
    .alert-critical { border-left: 4px solid #d32f2f; padding-left: 12px; background: #ffebee; border-radius: 4px; }
    .alert-high { border-left: 4px solid #f57c00; padding-left: 12px; background: #fff3e0; border-radius: 4px; }
    .alert-medium { border-left: 4px solid #fbc02d; padding-left: 12px; background: #fffde7; border-radius: 4px; }
    .chat-message {
        padding: 12px 16px; border-radius: 12px; margin: 8px 0;
        max-width: 85%; word-wrap: break-word;
    }
    .chat-user { background: #e3f2fd; margin-left: auto; border-bottom-right-radius: 4px; }
    .chat-bot { background: #f5f5f5; margin-right: auto; border-bottom-left-radius: 4px; }
    .chat-system { background: #fff3e0; margin: 4px auto; text-align: center; font-size: 0.85rem; }
    .live-pulse {
        display: inline-block; width: 10px; height: 10px; border-radius: 50%;
        background: #4caf50; animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(76,175,80,0.7); }
        70% { box-shadow: 0 0 0 10px rgba(76,175,80,0); }
        100% { box-shadow: 0 0 0 0 rgba(76,175,80,0); }
    }
    .sar-report {
        background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px;
        padding: 20px; font-family: 'Courier New', monospace; font-size: 0.85rem;
        white-space: pre-wrap; max-height: 600px; overflow-y: auto;
    }
    div[data-testid="stSidebarNav"] { display: none; }
    .stMetric { border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@st.cache_data
def load_data():
    """Load all generated data with caching."""
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])

    with open(os.path.join(DATA_DIR, "fraud_alerts.json")) as f:
        alerts = json.load(f)
    with open(os.path.join(DATA_DIR, "risk_scores.json")) as f:
        risk_scores = json.load(f)
    with open(os.path.join(DATA_DIR, "detection_summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(DATA_DIR, "fraud_cases.json")) as f:
        fraud_cases = json.load(f)

    # Build graph
    from graph_engine import FundFlowGraph
    ffg = FundFlowGraph()
    graph = ffg.build_graph(transactions)

    return transactions, alerts, risk_scores, summary, fraud_cases, ffg, graph


try:
    transactions, alerts, risk_scores, summary, fraud_cases, ffg, graph = load_data()
    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    LOAD_ERROR = str(e)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0;">
        <h2 style="margin:0; color:#1a237e;">🔱 RUDRA</h2>
        <p style="margin:0; color:#546e7a; font-size:0.85rem;">Shield Against Deception</p>
        <p style="margin:0; color:#78909c; font-size:0.75rem;">Fund Flow Intelligence System</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🕸️ Fund Flow Graph", "🚨 Fraud Alerts",
         "🔬 Pattern Analysis", "📋 Entity Explorer",
         "🤖 AI Copilot", "📄 SAR Reports", "📡 Live Monitor"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if DATA_LOADED:
        st.markdown("### Filters")
        min_amount = st.slider("Min Amount (₹)", 0,
                               int(transactions["amount"].max()), 0,
                               format="₹%d")
        selected_patterns = st.multiselect(
            "Fraud Patterns",
            ["circular_transaction", "rapid_layering", "smurfing", "shell_funnel"],
            default=["circular_transaction", "rapid_layering", "smurfing", "shell_funnel"],
        )

        st.markdown("---")
        st.markdown("### System Status")
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("TXNs", f"{len(transactions):,}")
        col_s2.metric("Entities", f"{graph.number_of_nodes()}")
        col_s3, col_s4 = st.columns(2)
        col_s3.metric("Alerts", f"{len(alerts)}")
        col_s4.metric("Cases", f"{len(fraud_cases)}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center'><small>PSBs Hackathon 2026<br>"
        "<b>Team Bhadra</b></small></div>",
        unsafe_allow_html=True,
    )

# ── Main Content ─────────────────────────────────────────────
if not DATA_LOADED:
    st.error(f"Failed to load data: {LOAD_ERROR}")
    st.info("Run `python src/run_pipeline.py` first to generate data, then restart.")
    st.stop()

# Apply filters
filtered_txns = transactions[
    (transactions["amount"] >= min_amount) &
    (transactions["fraud_pattern"].isin(selected_patterns) | ~transactions["is_fraud"])
]

# ═══════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<p class="main-header">RUDRA — Fund Flow Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time fund flow tracking & suspicious pattern detection</p>', unsafe_allow_html=True)

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    total_volume = filtered_txns["amount"].sum()
    fraud_txns = filtered_txns[filtered_txns["is_fraud"]]
    fraud_volume = fraud_txns["amount"].sum()
    fraud_rate = len(fraud_txns) / max(len(filtered_txns), 1) * 100

    col1.metric("Total Transactions", f"{len(filtered_txns):,}")
    col2.metric("Total Volume", f"₹{total_volume/1e7:.2f} Cr")
    col3.metric("Fraud Transactions", f"{len(fraud_txns):,}", delta=f"{fraud_rate:.1f}% rate")
    col4.metric("Fraud Volume", f"₹{fraud_volume/1e7:.2f} Cr")
    col5.metric("Active Alerts", f"{summary['total_alerts']}",
                delta=f"{summary.get('critical_alerts', 0)} critical")

    st.markdown("---")

    # Charts Row 1
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Transaction Volume Over Time")
        daily = filtered_txns.groupby(filtered_txns["timestamp"].dt.date).agg(
            count=("amount", "count"),
            volume=("amount", "sum"),
            fraud_count=("is_fraud", "sum"),
        ).reset_index()
        daily.columns = ["date", "count", "volume", "fraud_count"]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=daily["date"], y=daily["count"], name="Total TXNs",
                   marker_color="rgba(26,35,126,0.6)"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=daily["date"], y=daily["fraud_count"], name="Fraud TXNs",
                       line=dict(color="#d32f2f", width=2), mode="lines+markers"),
            secondary_y=True,
        )
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=30),
                          legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Count", secondary_y=True)
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.markdown("### Amount Distribution")
        fig = go.Figure()
        normal_amounts = filtered_txns[~filtered_txns["is_fraud"]]["amount"]
        fraud_amounts = filtered_txns[filtered_txns["is_fraud"]]["amount"]

        fig.add_trace(go.Histogram(
            x=normal_amounts, name="Normal", opacity=0.7,
            marker_color="#4CAF50", nbinsx=50,
        ))
        fig.add_trace(go.Histogram(
            x=fraud_amounts, name="Fraud", opacity=0.7,
            marker_color="#d32f2f", nbinsx=50,
        ))
        fig.update_layout(
            height=350, margin=dict(l=20, r=20, t=30, b=30),
            barmode="overlay", xaxis_title="Amount (₹)", yaxis_title="Count",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, width="stretch")

    # Charts Row 2
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Fraud Pattern Breakdown")
        pattern_counts = fraud_txns.groupby("fraud_pattern").agg(
            count=("amount", "count"),
            total=("amount", "sum"),
        ).reset_index()

        fig = px.pie(
            pattern_counts, values="count", names="fraud_pattern",
            color="fraud_pattern",
            color_discrete_map={
                "circular_transaction": "#d32f2f",
                "rapid_layering": "#f57c00",
                "smurfing": "#fbc02d",
                "shell_funnel": "#7b1fa2",
            },
            hole=0.4,
        )
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=30))
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.markdown("### Entity Risk Distribution")
        risk_df = pd.DataFrame(risk_scores)
        fig = px.histogram(
            risk_df, x="risk_score", color="risk_level",
            color_discrete_map={
                "CRITICAL": "#d32f2f",
                "HIGH": "#f57c00",
                "MEDIUM": "#fbc02d",
                "LOW": "#4CAF50",
            },
            nbins=20,
        )
        fig.update_layout(
            height=350, margin=dict(l=20, r=20, t=30, b=30),
            xaxis_title="Risk Score", yaxis_title="Entity Count",
        )
        st.plotly_chart(fig, width="stretch")

# ═══════════════════════════════════════════════════════════
# PAGE: FUND FLOW GRAPH
# ═══════════════════════════════════════════════════════════
elif page == "🕸️ Fund Flow Graph":
    st.markdown('<p class="main-header">Fund Flow Network Graph</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive visualization of money movement between entities</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        show_only_fraud = st.checkbox("Show only fraud-connected", value=False)
    with col2:
        min_edge_amount = st.number_input("Min edge amount (₹)", value=0, min_value=0, step=100000)
    with col3:
        layout_algo = st.selectbox("Layout", ["spring", "circular", "kamada_kawai", "shell"])

    viz_graph = graph.copy()
    edges_to_remove = []
    for u, v, data in viz_graph.edges(data=True):
        if data["total_amount"] < min_edge_amount:
            edges_to_remove.append((u, v))
    viz_graph.remove_edges_from(edges_to_remove)
    isolates = list(nx.isolates(viz_graph))
    viz_graph.remove_nodes_from(isolates)

    if show_only_fraud:
        fraud_nodes = set()
        for u, v, data in viz_graph.edges(data=True):
            if data.get("fraud_count", 0) > 0:
                fraud_nodes.add(u)
                fraud_nodes.add(v)
        neighbors = set()
        for n in fraud_nodes:
            neighbors.update(viz_graph.predecessors(n))
            neighbors.update(viz_graph.successors(n))
        fraud_nodes.update(neighbors)
        viz_graph = viz_graph.subgraph(fraud_nodes).copy()

    if viz_graph.number_of_nodes() == 0:
        st.warning("No nodes match the current filters.")
    else:
        if layout_algo == "spring":
            pos = nx.spring_layout(viz_graph, k=2, iterations=50, seed=42)
        elif layout_algo == "circular":
            pos = nx.circular_layout(viz_graph)
        elif layout_algo == "kamada_kawai":
            try:
                pos = nx.kamada_kawai_layout(viz_graph)
            except Exception:
                pos = nx.spring_layout(viz_graph, seed=42)
        else:
            pos = nx.shell_layout(viz_graph)

        fraud_edges = {(u, v) for u, v, d in viz_graph.edges(data=True) if d.get("fraud_count", 0) > 0}
        fraud_nodes = set()
        for u, v in fraud_edges:
            fraud_nodes.add(u)
            fraud_nodes.add(v)

        normal_edge_x, normal_edge_y = [], []
        fraud_edge_x, fraud_edge_y = [], []
        for u, v, data in viz_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            if data.get("fraud_count", 0) > 0:
                fraud_edge_x.extend([x0, x1, None])
                fraud_edge_y.extend([y0, y1, None])
            else:
                normal_edge_x.extend([x0, x1, None])
                normal_edge_y.extend([y0, y1, None])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=normal_edge_x, y=normal_edge_y,
            mode="lines", line=dict(width=0.5, color="#adb5bd"),
            hoverinfo="none", name="Normal Flow",
        ))
        fig.add_trace(go.Scatter(
            x=fraud_edge_x, y=fraud_edge_y,
            mode="lines", line=dict(width=2, color="#d32f2f"),
            hoverinfo="none", name="Suspicious Flow",
        ))

        type_colors = {"individual": "#2196F3", "business": "#4CAF50", "shell_company": "#d32f2f"}
        for entity_type, color in type_colors.items():
            node_x, node_y, node_text, node_size = [], [], [], []
            for node in viz_graph.nodes():
                ntype = viz_graph.nodes[node].get("type", "individual")
                if ntype != entity_type:
                    continue
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                name = viz_graph.nodes[node].get("name", node)
                is_f = node in fraud_nodes
                degree = viz_graph.degree(node)
                turnover = sum(
                    viz_graph[u][v]["total_amount"]
                    for u, v in list(viz_graph.in_edges(node)) + list(viz_graph.out_edges(node))
                )
                node_text.append(
                    f"{name}<br>Type: {ntype}<br>Degree: {degree}<br>Turnover: ₹{turnover:,.0f}"
                    f"{'<br>FRAUD CONNECTED' if is_f else ''}"
                )
                node_size.append(max(8, min(25, degree * 3)))

            if node_x:
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y, mode="markers",
                    marker=dict(size=node_size, color=color,
                                line=dict(width=1, color="white")),
                    text=node_text, hoverinfo="text",
                    name=entity_type.replace("_", " ").title(),
                ))

        fig.update_layout(
            height=700, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, width="stretch")

        # Subgraph Explorer
        st.markdown("### Subgraph Explorer")
        entity_options = sorted(
            [(viz_graph.nodes[n].get("name", n), n) for n in viz_graph.nodes()],
            key=lambda x: x[0],
        )
        selected_name, selected_id = st.selectbox(
            "Select Entity", options=entity_options,
            format_func=lambda x: x[0],
        )
        if selected_id:
            subgraph = ffg.extract_subgraph([selected_id], hops=2)
            sub_pos = nx.spring_layout(subgraph, k=1.5, seed=42)
            sub_fig = go.Figure()
            for u, v, data in subgraph.edges(data=True):
                x0, y0 = sub_pos[u]
                x1, y1 = sub_pos[v]
                is_fraud = data.get("fraud_count", 0) > 0
                sub_fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode="lines",
                    line=dict(width=3 if is_fraud else 1,
                              color="#d32f2f" if is_fraud else "#adb5bd"),
                    hoverinfo="text",
                    text=f"₹{data['total_amount']:,.0f} ({data['transaction_count']} txns)",
                    showlegend=False,
                ))
            for node in subgraph.nodes():
                x, y = sub_pos[node]
                name = subgraph.nodes[node].get("name", node)
                ntype = subgraph.nodes[node].get("type", "")
                color = "#1a237e" if node == selected_id else type_colors.get(ntype, "#2196F3")
                size = 20 if node == selected_id else 12
                sub_fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode="markers+text",
                    marker=dict(size=size, color=color),
                    text=[name[:15]], textposition="top center",
                    textfont=dict(size=9),
                    hoverinfo="text", hovertext=f"{name}<br>Type: {ntype}",
                    showlegend=False,
                ))
            sub_fig.update_layout(
                height=500, margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white",
            )
            st.plotly_chart(sub_fig, width="stretch")

# ═══════════════════════════════════════════════════════════
# PAGE: FRAUD ALERTS
# ═══════════════════════════════════════════════════════════
elif page == "🚨 Fraud Alerts":
    st.markdown('<p class="main-header">Fraud Alert Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-detected suspicious fund flow patterns</p>', unsafe_allow_html=True)

    severity_filter = st.multiselect(
        "Filter by Severity", ["CRITICAL", "HIGH", "MEDIUM"],
        default=["CRITICAL", "HIGH", "MEDIUM"],
    )
    pattern_filter = st.multiselect(
        "Filter by Pattern",
        ["Circular Transaction", "Rapid Layering", "Smurfing / Structuring",
         "Shell Company Funnel", "Dormant Activation", "Profile Mismatch"],
        default=["Circular Transaction", "Rapid Layering", "Smurfing / Structuring", "Shell Company Funnel"],
    )

    filtered_alerts = [
        a for a in alerts
        if a["severity"] in severity_filter and a["pattern_type"] in pattern_filter
    ]

    col1, col2, col3, col4 = st.columns(4)
    crit = sum(1 for a in filtered_alerts if a["severity"] == "CRITICAL")
    high = sum(1 for a in filtered_alerts if a["severity"] == "HIGH")
    med = sum(1 for a in filtered_alerts if a["severity"] == "MEDIUM")
    total_flow = sum(a.get("total_flow", 0) for a in filtered_alerts)

    col1.metric("Critical", crit, delta="Immediate action" if crit > 0 else None)
    col2.metric("High", high)
    col3.metric("Medium", med)
    col4.metric("Total Flagged", f"₹{total_flow/1e7:.2f} Cr")

    st.markdown("---")
    st.markdown(f"### {len(filtered_alerts)} Alerts Found")

    for alert in sorted(filtered_alerts, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x["severity"], 3)):
        icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(alert["severity"], "⚪")
        with st.expander(f"{icon} [{alert['severity']}] {alert['pattern_type']} — {alert.get('confidence', 0)}% confidence"):
            st.markdown(f"**Description:** {alert['description']}")
            st.markdown(f"**Recommendation:** {alert['recommendation']}")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Flow", f"₹{alert.get('total_flow', 0):,.0f}")
            with col_b:
                st.metric("Confidence", f"{alert.get('confidence', 0)}%")
            with col_c:
                st.metric("Entities", len(alert.get("entities", [])))

            if "entity_names" in alert:
                st.markdown("**Entity Chain:**")
                st.markdown(f"`{' → '.join(alert['entity_names'])}`")

# ═══════════════════════════════════════════════════════════
# PAGE: PATTERN ANALYSIS
# ═══════════════════════════════════════════════════════════
elif page == "🔬 Pattern Analysis":
    st.markdown('<p class="main-header">Fraud Pattern Deep Dive</p>', unsafe_allow_html=True)

    pattern_tabs = st.tabs([
        "🔄 Circular", "⚡ Layering", "💶 Smurfing",
        "🏢 Shell Funnels", "💤 Dormant Activation", "👤 Profile Mismatch",
    ])

    fraud_txns = filtered_txns[filtered_txns["is_fraud"]]

    # ── Circular ──
    with pattern_tabs[0]:
        st.markdown("### Circular Transaction Detection (Round-Tripping)")
        circular_txns = fraud_txns[fraud_txns["fraud_pattern"] == "circular_transaction"]
        circular_alerts = [a for a in alerts if a["pattern_type"] == "Circular Transaction"]

        col1, col2 = st.columns(2)
        col1.metric("Patterns Detected", len(circular_alerts))
        col2.metric("Total Volume", f"₹{circular_txns['amount'].sum():,.0f}")

        if circular_alerts:
            for i, alert in enumerate(circular_alerts[:6]):
                st.markdown(f"#### Ring {i+1}: {alert.get('cycle_length', '?')} entities, ₹{alert.get('total_flow', 0):,.0f}")
                entities = alert.get("entities", [])
                if len(entities) >= 3:
                    ring_fig = go.Figure()
                    n = len(entities)
                    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                    xs, ys = np.cos(angles), np.sin(angles)
                    for j in range(n):
                        ring_fig.add_trace(go.Scatter(
                            x=[xs[j], xs[(j+1)%n]], y=[ys[j], ys[(j+1)%n]],
                            mode="lines+markers", line=dict(width=2, color="#d32f2f"),
                            marker=dict(size=0), showlegend=False,
                        ))
                    names = [graph.nodes[e].get("name", e)[:12] for e in entities]
                    ring_fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="markers+text",
                        marker=dict(size=15, color="#d32f2f"),
                        text=names, textposition="top center",
                        textfont=dict(size=10), showlegend=False,
                    ))
                    ring_fig.update_layout(height=400, showlegend=False,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
                        plot_bgcolor="white")
                    st.plotly_chart(ring_fig, width="stretch")

        st.dataframe(
            circular_txns[["timestamp", "sender_name", "receiver_name", "amount", "transaction_type", "fraud_case_id"]]
            .sort_values(["fraud_case_id", "timestamp"]),
            use_container_width=True, height=300,
        )

    # ── Layering ──
    with pattern_tabs[1]:
        st.markdown("### Rapid Layering Detection")
        layer_txns = fraud_txns[fraud_txns["fraud_pattern"] == "rapid_layering"]
        layer_alerts = [a for a in alerts if a["pattern_type"] == "Rapid Layering"]

        col1, col2 = st.columns(2)
        col1.metric("Chains Detected", len(layer_alerts))
        col2.metric("Total Volume", f"₹{layer_txns['amount'].sum():,.0f}")

        if layer_alerts:
            for i, alert in enumerate(layer_alerts[:5]):
                st.markdown(f"#### Chain {i+1}: {alert.get('chain_length', '?')} hops, ₹{alert.get('total_flow', 0):,.0f}")
                names = alert.get("entity_names", [])
                if len(names) >= 3:
                    # Sankey-style visualization
                    fig = go.Figure()
                    for j in range(len(names)-1):
                        fig.add_trace(go.Scatter(
                            x=[j, j+1], y=[0, 0], mode="lines",
                            line=dict(width=3, color="#f57c00"), showlegend=False,
                        ))
                    fig.add_trace(go.Scatter(
                        x=list(range(len(names))), y=[0]*len(names),
                        mode="markers+text",
                        marker=dict(size=15, color="#f57c00"),
                        text=[n[:15] for n in names],
                        textposition="bottom center", textfont=dict(size=9),
                        showlegend=False,
                    ))
                    fig.update_layout(height=300, showlegend=False,
                        margin=dict(l=20, r=20, t=30, b=60),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
                        plot_bgcolor="white")
                    st.plotly_chart(fig, width="stretch")

        st.dataframe(
            layer_txns[["timestamp", "sender_name", "receiver_name", "amount", "sender_type", "receiver_type", "fraud_case_id"]]
            .sort_values(["fraud_case_id", "timestamp"]),
            use_container_width=True, height=300,
        )

    # ── Smurfing ──
    with pattern_tabs[2]:
        st.markdown("### Smurfing / Structuring Detection")
        smurf_txns = fraud_txns[fraud_txns["fraud_pattern"] == "smurfing"]
        smurf_alerts = [a for a in alerts if a["pattern_type"] == "Smurfing / Structuring"]

        col1, col2 = st.columns(2)
        col1.metric("Patterns Detected", len(smurf_alerts))
        col2.metric("Total Volume", f"₹{smurf_txns['amount'].sum():,.0f}")

        if not smurf_txns.empty:
            fig = go.Figure()
            fig.add_vline(x=200000, line_dash="dash", line_color="red",
                          annotation_text="Reporting Threshold (₹2L)")
            fig.add_trace(go.Histogram(x=smurf_txns["amount"], nbinsx=40,
                                       marker_color="#fbc02d", name="Smurfing TXNs"))
            fig.update_layout(height=350, xaxis_title="Amount (₹)", yaxis_title="Count",
                              margin=dict(l=20, r=20, t=30, b=30))
            st.plotly_chart(fig, width="stretch")

        st.dataframe(
            smurf_txns[["timestamp", "sender_name", "receiver_name", "amount", "transaction_type", "fraud_case_id"]]
            .sort_values(["fraud_case_id", "timestamp"]),
            use_container_width=True, height=300,
        )

    # ── Shell Funnels ──
    with pattern_tabs[3]:
        st.markdown("### Shell Company Funnel Detection")
        funnel_txns = fraud_txns[fraud_txns["fraud_pattern"] == "shell_funnel"]
        funnel_alerts = [a for a in alerts if a["pattern_type"] == "Shell Company Funnel"]

        col1, col2 = st.columns(2)
        col1.metric("Funnels Detected", len(funnel_alerts))
        col2.metric("Total Volume", f"₹{funnel_txns['amount'].sum():,.0f}")

        for i, alert in enumerate(funnel_alerts[:5]):
            st.markdown(f"#### Funnel {i+1}: {alert.get('funnel_name', 'Unknown')}")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Inflow", f"₹{alert.get('total_inflow', 0):,.0f}")
            col_b.metric("Outflow", f"₹{alert.get('total_outflow', 0):,.0f}")
            col_c.metric("Imbalance", f"{alert.get('imbalance_ratio', 0):.1%}")

        st.dataframe(
            funnel_txns[["timestamp", "sender_name", "receiver_name", "amount", "sender_type", "sender_branch", "fraud_case_id"]]
            .sort_values(["fraud_case_id", "timestamp"]),
            use_container_width=True, height=300,
        )

    # ── Dormant Activation ──
    with pattern_tabs[4]:
        st.markdown("### Dormant Account Activation (Z-Score Spike)")
        dormant_alerts = [a for a in alerts if a["pattern_type"] == "Dormant Activation"]

        col1, col2 = st.columns(2)
        col1.metric("Dormant Activations", len(dormant_alerts))
        col2.metric("Pattern", "Z-score > 2.5 after 30+ day gap")

        for alert in dormant_alerts[:10]:
            icon = {"CRITICAL": "🔴", "HIGH": "🟠"}.get(alert["severity"], "🟡")
            with st.expander(f"{icon} {alert.get('entity_names', ['Unknown'])[0]} — Z-score: {alert.get('z_score', 0):.1f}"):
                st.markdown(f"**{alert['description']}**")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Z-Score", f"{alert.get('z_score', 0):.1f}")
                col_b.metric("Dormant Days", f"{alert.get('dormant_days', 0)}")
                col_c.metric("Activation", alert.get("activation_date", ""))
                st.markdown(f"*{alert['recommendation']}*")

    # ── Profile Mismatch ──
    with pattern_tabs[5]:
        st.markdown("### Profile Mismatch Detection (KYC Delta)")
        profile_alerts = [a for a in alerts if a["pattern_type"] == "Profile Mismatch"]

        col1, col2 = st.columns(2)
        col1.metric("Mismatches", len(profile_alerts))
        col2.metric("Method", "Behavioral vs declared type")

        for alert in profile_alerts[:10]:
            icon = {"CRITICAL": "🔴", "HIGH": "🟠"}.get(alert["severity"], "🟡")
            with st.expander(f"{icon} {alert.get('entity_names', ['Unknown'])[0]} ({alert.get('entity_type', '')}) — {alert.get('mismatch_count', 0)} mismatches"):
                st.markdown(f"**{alert['description']}**")
                for m in alert.get("mismatches", []):
                    st.markdown(f"- {m}")
                st.markdown(f"*{alert['recommendation']}*")

# ═══════════════════════════════════════════════════════════
# PAGE: ENTITY EXPLORER
# ═══════════════════════════════════════════════════════════
elif page == "📋 Entity Explorer":
    st.markdown('<p class="main-header">Entity Risk Explorer</p>', unsafe_allow_html=True)

    risk_df = pd.DataFrame(risk_scores)
    st.markdown("### Top High-Risk Entities")
    st.dataframe(
        risk_df[["name", "type", "risk_score", "risk_level"]].head(20),
        use_container_width=True, height=400,
    )

    st.markdown("### Entity Lookup")
    entity_options = sorted(
        [(r["name"], r["entity_id"]) for r in risk_scores],
        key=lambda x: x[0],
    )
    selected_name, selected_id = st.selectbox(
        "Search Entity", options=entity_options,
        format_func=lambda x: x[0],
    )

    if selected_id:
        entity_txns = filtered_txns[
            (filtered_txns["sender_id"] == selected_id) |
            (filtered_txns["receiver_id"] == selected_id)
        ]
        risk_info = next((r for r in risk_scores if r["entity_id"] == selected_id), None)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Risk Score", f"{risk_info['risk_score']:.2f}" if risk_info else "N/A")
        col2.metric("Risk Level", risk_info["risk_level"] if risk_info else "N/A")
        col3.metric("Total TXNs", len(entity_txns))
        col4.metric("Fraud TXNs", len(entity_txns[entity_txns["is_fraud"]]))

        if not entity_txns.empty:
            st.markdown("#### Transaction History")
            st.dataframe(
                entity_txns[["timestamp", "sender_name", "receiver_name", "amount", "transaction_type", "is_fraud", "fraud_pattern"]]
                .sort_values("timestamp", ascending=False).head(50),
                use_container_width=True, height=300,
            )

            sent = entity_txns[entity_txns["sender_id"] == selected_id]["amount"].sum()
            received = entity_txns[entity_txns["receiver_id"] == selected_id]["amount"].sum()
            fig = go.Figure(go.Bar(
                x=["Sent", "Received", "Net Flow"],
                y=[sent, received, received - sent],
                marker_color=["#d32f2f", "#4CAF50", "#1a237e"],
            ))
            fig.update_layout(height=300, title="Flow Summary",
                              margin=dict(l=20, r=20, t=40, b=30), yaxis_title="Amount (₹)")
            st.plotly_chart(fig, width="stretch")

# ═══════════════════════════════════════════════════════════
# PAGE: AI COPILOT
# ═══════════════════════════════════════════════════════════
elif page == "🤖 AI Copilot":
    st.markdown('<p class="main-header">RUDRA AI Copilot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Investigation assistant powered by LLM with tool-calling over the live fund flow graph</p>', unsafe_allow_html=True)

    # Initialize copilot
    from llm_copilot import LLMCopilot

    if "copilot" not in st.session_state:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        st.session_state.copilot = LLMCopilot(
            graph, transactions, alerts, risk_scores, fraud_cases, api_key
        )
        st.session_state.chat_history = []

    # Quick action buttons
    st.markdown("### Quick Actions")
    qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)
    with qa_col1:
        if st.button("Overview"):
            st.session_state.chat_history.append(("user", "Give me an overview of the current fraud detection status"))
            result = st.session_state.copilot.query("Give me an overview of the current fraud detection status")
            st.session_state.chat_history.append(("bot", result["response"]))
            st.rerun()
    with qa_col2:
        if st.button("High Risk Entities"):
            st.session_state.chat_history.append(("user", "Show me the high-risk entities"))
            result = st.session_state.copilot.query("Show me the high-risk entities")
            st.session_state.chat_history.append(("bot", result["response"]))
            st.rerun()
    with qa_col3:
        if st.button("Find Cycles"):
            st.session_state.chat_history.append(("user", "Find all circular transaction patterns"))
            result = st.session_state.copilot.query("Find all circular transaction patterns")
            st.session_state.chat_history.append(("bot", result["response"]))
            st.rerun()
    with qa_col4:
        if st.button("Active Alerts"):
            st.session_state.chat_history.append(("user", "Show me all active fraud alerts"))
            result = st.session_state.copilot.query("Show me all active fraud alerts")
            st.session_state.chat_history.append(("bot", result["response"]))
            st.rerun()

    st.markdown("---")

    # Chat display
    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f'<div class="chat-message chat-user"><b>You:</b> {message}</div>', unsafe_allow_html=True)
            elif role == "bot":
                # Render markdown in bot response
                st.markdown(f'<div class="chat-message chat-bot"><b>RUDRA:</b></div>', unsafe_allow_html=True)
                st.markdown(message)
            elif role == "system":
                st.markdown(f'<div class="chat-system">{message}</div>', unsafe_allow_html=True)

    # Chat input
    st.markdown("---")
    chat_col1, chat_col2 = st.columns([6, 1])
    with chat_col1:
        user_input = st.text_input(
            "Ask RUDRA anything about fund flows, fraud patterns, or entities...",
            key="chat_input", label_visibility="collapsed",
            placeholder="e.g., 'Trace funds for Amit Sharma' or 'Explain alert ALERT_CIRC_0001'",
        )
    with chat_col2:
        send = st.button("Send", width="stretch")

    if (send or (user_input and user_input != st.session_state.get("last_input", ""))) and user_input:
        st.session_state.last_input = user_input
        st.session_state.chat_history.append(("user", user_input))

        with st.spinner("RUDRA is analyzing..."):
            result = st.session_state.copilot.query(user_input)

        st.session_state.chat_history.append(("bot", result["response"]))
        source = result.get("source", "local")
        if source != "local":
            st.session_state.chat_history.append(("system", f"Source: {source}"))

        st.rerun()

    # API Key configuration
    with st.expander("⚙️ Configure Gemini API Key (optional)"):
        api_input = st.text_input("Gemini API Key", type="password",
                                   value=os.environ.get("GEMINI_API_KEY", ""))
        if st.button("Update API Key"):
            st.session_state.copilot.api_key = api_input
            st.success("API key updated. Responses will use Gemini for enhanced analysis.")

# ═══════════════════════════════════════════════════════════
# PAGE: SAR REPORTS
# ═══════════════════════════════════════════════════════════
elif page == "📄 SAR Reports":
    st.markdown('<p class="main-header">SAR Report Generator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Auto-generated Suspicious Activity Reports with regulatory citations</p>', unsafe_allow_html=True)

    from sar_generator import SARGenerator

    sar_gen = SARGenerator(graph, transactions, alerts, fraud_cases)

    # Generate reports
    st.markdown("### Generate Reports")

    gen_col1, gen_col2 = st.columns(2)
    with gen_col1:
        min_sev = st.selectbox("Minimum Severity", ["CRITICAL", "HIGH", "MEDIUM"], index=1)
    with gen_col2:
        if st.button("Generate All SAR Reports", width="stretch"):
            with st.spinner("Generating SAR reports..."):
                sar_reports = sar_gen.generate_all_sars(min_sev)
                st.session_state.sar_reports = sar_reports
                st.success(f"Generated {len(sar_reports)} SAR reports!")

    # Individual alert SAR generation
    st.markdown("### Generate SAR for Specific Alert")
    alert_options = [(a["alert_id"], f"{a['alert_id']} — {a['pattern_type']} ({a['severity']})") for a in alerts]
    selected_alert = st.selectbox("Select Alert", options=alert_options, format_func=lambda x: x[1])

    if st.button("Generate SAR for Selected Alert"):
        alert = next(a for a in alerts if a["alert_id"] == selected_alert[0])
        with st.spinner("Generating SAR..."):
            sar = sar_gen.generate_sar(alert)
            st.session_state.current_sar = sar
        st.success(f"SAR {sar['report_id']} generated!")

    # Display SAR report
    if "current_sar" in st.session_state:
        sar = st.session_state.current_sar
        st.markdown("---")
        st.markdown(f"### SAR Report: {sar['report_id']}")

        # Download buttons
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            report_text = sar["report_text"]
            st.download_button(
                "📥 Download Report (Text)",
                data=report_text,
                file_name=f"{sar['report_id']}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl_col2:
            report_json = json.dumps(sar, indent=2, default=str)
            st.download_button(
                "📥 Download Report (JSON)",
                data=report_json,
                file_name=f"{sar['report_id']}.json",
                mime="application/json",
                use_container_width=True,
            )

        # Display report
        st.markdown(f'<div class="sar-report">{report_text}</div>', unsafe_allow_html=True)

    # Previously generated reports
    if "sar_reports" in st.session_state and st.session_state.sar_reports:
        st.markdown("---")
        st.markdown("### Generated Reports")
        for sar in st.session_state.sar_reports:
            with st.expander(f"{sar['report_id']} — {sar['pattern_type']} ({sar['severity']})"):
                st.markdown(f"**Alert:** {sar['alert_id']}")
                st.markdown(f"**Total Flow:** ₹{sar['total_flow']:,.0f}")
                st.markdown(f"**Confidence:** {sar['confidence']}%")
                report_text = sar["report_text"]
                st.download_button(
                    f"Download {sar['report_id']}",
                    data=report_text,
                    file_name=f"{sar['report_id']}.txt",
                    mime="text/plain",
                )

# ═══════════════════════════════════════════════════════════
# PAGE: LIVE MONITOR
# ═══════════════════════════════════════════════════════════
elif page == "📡 Live Monitor":
    st.markdown('<p class="main-header">Real-time Transaction Monitor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Simulated live transaction feed with real-time fraud detection</p>', unsafe_allow_html=True)

    st.markdown('<span class="live-pulse"></span> <b>LIVE</b> — Simulating incoming transactions', unsafe_allow_html=True)

    # Initialize live feed state
    if "live_txns" not in st.session_state:
        st.session_state.live_txns = []
        st.session_state.live_alerts = []
        st.session_state.live_count = 0
        st.session_state.live_fraud_count = 0
        st.session_state.live_volume = 0.0

    # Control panel
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        speed = st.selectbox("Feed Speed", ["Slow (2s)", "Medium (1s)", "Fast (0.5s)"], index=1)
    with ctrl_col2:
        auto_detect = st.checkbox("Auto-detect fraud", value=True)
    with ctrl_col3:
        if st.button("Clear Feed"):
            st.session_state.live_txns = []
            st.session_state.live_alerts = []
            st.session_state.live_count = 0
            st.session_state.live_fraud_count = 0
            st.session_state.live_volume = 0.0

    # Simulate button
    if st.button("▶ Inject Transaction Batch (10 txns)", width="stretch"):
        entities_list = list(graph.nodes())
        for _ in range(10):
            sender = random.choice(entities_list)
            receiver = random.choice(entities_list)
            while receiver == sender:
                receiver = random.choice(entities_list)

            is_fraud = random.random() < 0.15  # 15% fraud rate
            amount = round(random.lognormvariate(np.log(100000), 1.5), 2)
            if is_fraud:
                amount = round(random.uniform(500000, 5000000), 2)

            sdata = graph.nodes[sender]
            rdata = graph.nodes[receiver]

            txn = {
                "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                "sender": sdata.get("name", sender),
                "receiver": rdata.get("name", receiver),
                "amount": amount,
                "type": random.choice(["NEFT", "RTGS", "IMPS", "UPI"]),
                "is_fraud": is_fraud,
                "pattern": random.choice(["circular", "layering", "smurfing", "none"]) if is_fraud else "none",
            }

            st.session_state.live_txns.append(txn)
            st.session_state.live_count += 1
            st.session_state.live_volume += amount

            if is_fraud:
                st.session_state.live_fraud_count += 1
                if auto_detect:
                    alert = {
                        "time": txn["timestamp"],
                        "entity": txn["sender"],
                        "pattern": txn["pattern"].replace("_", " ").title(),
                        "amount": amount,
                        "severity": "CRITICAL" if amount > 3000000 else "HIGH",
                    }
                    st.session_state.live_alerts.append(alert)

    # Live stats
    stat_col1, stat_col2, col3, stat_col4 = st.columns(4)
    stat_col1.metric("Transactions Processed", st.session_state.live_count)
    stat_col2.metric("Total Volume", f"₹{st.session_state.live_volume/1e7:.2f} Cr")
    col3.metric("Fraud Detected", st.session_state.live_fraud_count)
    stat_col4.metric("Live Alerts", len(st.session_state.live_alerts))

    st.markdown("---")

    # Two columns: Transaction feed and Alert feed
    feed_col, alert_col = st.columns(2)

    with feed_col:
        st.markdown("### Transaction Feed")
        if st.session_state.live_txns:
            live_df = pd.DataFrame(reversed(st.session_state.live_txns[-50:]))
            # Style the dataframe
            st.dataframe(
                live_df[["timestamp", "sender", "receiver", "amount", "type", "is_fraud"]],
                use_container_width=True, height=400,
                column_config={
                    "is_fraud": st.column_config.CheckboxColumn("Fraud"),
                    "amount": st.column_config.NumberColumn("Amount", format="₹%d"),
                },
            )
        else:
            st.info("Click 'Inject Transaction Batch' to simulate incoming transactions.")

    with alert_col:
        st.markdown("### Live Fraud Alerts")
        if st.session_state.live_alerts:
            for alert in reversed(st.session_state.live_alerts[-20:]):
                icon = "🔴" if alert["severity"] == "CRITICAL" else "🟠"
                st.markdown(
                    f'<div class="alert-{alert["severity"].lower()}">'
                    f'{icon} **[{alert["severity"]}]** {alert["time"]} — '
                    f'{alert["entity"]}: {alert["pattern"]} (₹{alert["amount"]:,.0f})'
                    f'</div>', unsafe_allow_html=True
                )
        else:
            st.info("No fraud alerts yet. Inject transactions to start monitoring.")

    # Live graph visualization
    if st.session_state.live_count > 0:
        st.markdown("---")
        st.markdown("### Real-time Detection Metrics")
        met_col1, met_col2 = st.columns(2)

        with met_col1:
            if st.session_state.live_txns:
                fraud_ratio = st.session_state.live_fraud_count / max(st.session_state.live_count, 1) * 100
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_ratio,
                    title={"text": "Fraud Rate (%)"},
                    gauge={"axis": {"range": [0, 30]},
                           "bar": {"color": "#d32f2f" if fraud_ratio > 10 else "#f57c00" if fraud_ratio > 5 else "#4CAF50"},
                           "steps": [
                               {"range": [0, 5], "color": "#e8f5e9"},
                               {"range": [5, 15], "color": "#fff3e0"},
                               {"range": [15, 30], "color": "#ffebee"},
                           ]},
                ))
                fig.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=30))
                st.plotly_chart(fig, width="stretch")

        with met_col2:
            if st.session_state.live_alerts:
                pattern_counts = {}
                for a in st.session_state.live_alerts:
                    p = a["pattern"]
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
                fig = go.Figure(go.Pie(
                    labels=list(pattern_counts.keys()),
                    values=list(pattern_counts.values()),
                    hole=0.4,
                ))
                fig.update_layout(height=300, title="Alert Pattern Distribution",
                                  margin=dict(l=20, r=20, t=50, b=30))
                st.plotly_chart(fig, width="stretch")
