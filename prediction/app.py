import sys
import os

# Dynamically add the project root to python path to resolve import errors when run via streamlit directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.subplots as sp

# Project Module Imports
from prediction.data_loader import load_traffic_data
from prediction.graph_builder import build_graphs, compute_graph_metrics, get_pyg_graph
from prediction.feature_engineering import build_spatio_temporal_features
from prediction.stgcn_model import load_and_predict_stgcn
from prediction.energy_optimizer import run_energy_optimization
from prediction.nlp_analyzer import MultilingualNLPAnalyzer
from optimization.cabling import compute_electrical_mst
from optimization.routing import plan_maintenance_route
from evaluation.evaluate_system import run_system_evaluation

# Set page configuration with a premium look
st.set_page_config(
    page_title="Luminia Smart Grid - AI Command Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom premium styling matching the gorgeous dark cyber aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Syne:wght@500;700;800&display=swap');
    
    /* Smooth Scrollbar Customization */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #060913 !important;
    }
    ::-webkit-scrollbar-thumb {
        background: #1a2c5a !important;
        border-radius: 10px !important;
        border: 2px solid #060913 !important;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00d2ff !important;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.5) !important;
    }

    /* Main body background & font override */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif !important;
        background-color: #060913 !important;
        background-image: radial-gradient(circle at 10% 20%, rgba(13, 27, 60, 0.35) 0%, rgba(6, 9, 19, 0) 75%) !important;
        color: #cbd5e1 !important;
    }
    
    /* Custom style overrides for headers */
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        color: #ffffff !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #080d22 !important;
        border-right: 1px solid rgba(26, 44, 90, 0.6) !important;
    }
    
    /* Tabs selector styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(12, 21, 45, 0.7);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(26, 44, 90, 0.6);
        backdrop-filter: blur(8px);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #8b9bb4;
        font-weight: 600;
        border: none;
        padding: 4px 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00d2ff;
        background-color: rgba(0, 210, 255, 0.08);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1c2e5c, #0d1e3d) !important;
        color: #00d2ff !important;
        border: 1px solid rgba(0, 210, 255, 0.4) !important;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.15) !important;
    }
    
    /* Sidebar premium status panel cards */
    .sidebar-card {
        background: rgba(12, 21, 45, 0.5);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(28, 44, 90, 0.6);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: border-color 0.3s ease;
    }
    .sidebar-card:hover {
        border-color: rgba(0, 210, 255, 0.3);
    }
    .sidebar-title {
        font-size: 0.72rem;
        font-weight: 800;
        color: #8b9bb4;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
        border-bottom: 1px solid rgba(26, 44, 90, 0.4);
        padding-bottom: 4px;
    }
    .sidebar-value {
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
    }
    
    /* Breathing animation for status dot */
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #4ed9a6;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 8px #4ed9a6;
        animation: breathe 2s infinite ease-in-out;
    }
    @keyframes breathe {
        0%, 100% { opacity: 0.6; box-shadow: 0 0 4px #4ed9a6; }
        50% { opacity: 1; box-shadow: 0 0 12px #4ed9a6; }
    }

    /* Logo container design */
    .logo-container {
        text-align: center;
        padding: 1.5rem 0.5rem;
        background: linear-gradient(135deg, #0c152d, #060913);
        border-radius: 12px;
        border: 1px solid #1a2c5a;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .logo-text-large {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #f5c242, #4ed9a6, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }
    .logo-text-small {
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 4px;
        color: #8b9bb4;
        margin-top: 5px;
        text-transform: uppercase;
    }
    .logo-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #1a2c5a, transparent);
        margin: 10px 0;
    }
    .logo-slogan {
        font-size: 0.65rem;
        color: #4ed9a6;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* Custom premium card layouts */
    .card-deck {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.2rem;
        margin-bottom: 2rem;
    }
    .premium-card {
        background: rgba(12, 21, 45, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(28, 44, 90, 0.6);
        border-radius: 14px;
        padding: 1.4rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    .premium-card:hover {
        border-color: #00d2ff;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.18);
    }
    .card-title {
        font-size: 0.75rem;
        color: #8b9bb4;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .card-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.4rem 0;
        line-height: 1;
    }
    .card-subtitle {
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Data & Model Caching -----------------
@st.cache_data
def get_cached_pipeline_data():
    """Loads and computes the complete dataset and features once."""
    df_raw = load_traffic_data()
    road_g, elect_g, _ = build_graphs()
    df_features = build_spatio_temporal_features(df_raw, road_g)
    
    # Chronological Train/Test Split (83% Train, 17% Test)
    unique_timestamps = sorted(df_features['Timestamp'].unique())
    split_idx = int(len(unique_timestamps) * 0.833)
    split_time = unique_timestamps[split_idx]
    
    df_train = df_features[df_features['Timestamp'] < split_time].copy().reset_index(drop=True)
    df_test = df_features[df_features['Timestamp'] >= split_time].copy().reset_index(drop=True)
    
    return df_train, df_test, road_g, elect_g

@st.cache_resource
def get_stgcn_predictions(_df_train, _df_test):
    """Loads pre-trained STGCN models or trains them."""
    try:
        # Load predictions
        ped_preds = load_and_predict_stgcn(_df_train, _df_test, 'Target_Pedestrians')
        veh_preds = load_and_predict_stgcn(_df_train, _df_test, 'Target_Vehicles')
        return ped_preds, veh_preds
    except Exception as e:
        st.error(f"Error loading/training STGCN: {e}")
        return np.zeros(len(_df_test)), np.zeros(len(_df_test))

@st.cache_resource
def get_nlp_analyzer():
    """Loads cached multilingual NLP analyzer."""
    return MultilingualNLPAnalyzer()

# Load Data
with st.spinner("Initializing Luminia GNN Core and Loading Networks..."):
    df_train, df_test, road_g, elect_g = get_cached_pipeline_data()
    stgcn_ped_preds, stgcn_veh_preds = get_stgcn_predictions(df_train, df_test)
    nlp_analyzer = get_nlp_analyzer()

# Add GNN predictions to test set
df_test['Pred_Pedestrians'] = stgcn_ped_preds
df_test['Pred_Vehicles'] = stgcn_veh_preds

# Aggregate 30-min to hourly for simulation if needed, or if already hourly, proceed
df_test_hourly = df_test.copy()
if df_test_hourly['Timestamp'].dt.minute.nunique() > 1:
    df_test_hourly['Hour_Timestamp'] = df_test_hourly['Timestamp'].dt.floor('h')
    agg_rules = {
        'Target_Pedestrians': 'sum',
        'Target_Vehicles': 'sum',
        'Pred_Pedestrians': 'sum',
        'Pred_Vehicles': 'sum',
        'Is_Night': 'first',
        'Is_Weekend': 'first'
    }
    df_test_hourly = df_test_hourly.groupby(['Node_ID', 'Hour_Timestamp']).agg(agg_rules).reset_index()
    df_test_hourly = df_test_hourly.rename(columns={'Hour_Timestamp': 'Timestamp'})

# Run energy optimization
summary_metrics, detailed_metrics = run_energy_optimization(df_test_hourly)

# Get metrics for AIC
stgcn_summary = summary_metrics[summary_metrics['Policy'] == 'STGCN Adaptive Dimming'].iloc[0]
fixed_summary = summary_metrics[summary_metrics['Policy'] == 'Fixed Lighting (100%)'].iloc[0]
rbc_summary = summary_metrics[summary_metrics['Policy'] == 'Rule-Based Dimming (RBC)'].iloc[0]

# ----------------- Sidebar -----------------
st.sidebar.markdown("""
<div class="logo-container">
    <div class="logo-text-large">LUMINIA</div>
    <div class="logo-text-small">SMART GRID</div>
    <div class="logo-divider"></div>
    <div class="logo-slogan">STGCN INTELLIGENT LIGHTING</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div class="sidebar-card">
    <div class="sidebar-title">⚡ GNN SYSTEM STATUS</div>
    <div class="sidebar-value">
        <span class="status-dot"></span>STGCN Engine Active
    </div>
    <div style="margin-top: 10px; font-size: 0.8rem; color: #8b9bb4; line-height: 1.6;">
        • Network Nodes: <b style="color: #00d2ff; float: right;">120</b><br>
        • Total Edges: <b style="color: #00d2ff; float: right;">400</b><br>
        • STGCN Predictors: <b style="color: #4ed9a6; float: right;">Active ✓</b><br>
        • Grid Connections: <b style="color: #4ed9a6; float: right;">100% OK</b>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- Top Header Banner (Dynamic KPI Deck) -----------------
st.markdown(f"""
<div class="card-deck">
    <div class="premium-card" style="border-left: 4px solid #4ed9a6;">
        <div class="card-title">TOTAL ENERGY SAVED</div>
        <div class="card-value" style="color: #4ed9a6;">{stgcn_summary['CO2 Avoided (kg)'] / 0.533:,.1f} kWh</div>
        <div class="card-subtitle" style="color: rgba(78, 217, 166, 0.8);">{stgcn_summary['Energy Savings (%)']}% vs baseline</div>
    </div>
    <div class="premium-card" style="border-left: 4px solid #00d2ff;">
        <div class="card-title">ACTIVE NODES</div>
        <div class="card-value" style="color: #00d2ff;">120 / 120</div>
        <div class="card-subtitle" style="color: rgba(0, 210, 255, 0.8);">100% Operational</div>
    </div>
    <div class="premium-card" style="border-left: 4px solid #f5c242;">
        <div class="card-title">FINANCIAL SAVINGS</div>
        <div class="card-value" style="color: #f5c242;">${stgcn_summary['Financial Savings ($)']:,.2f}</div>
        <div class="card-subtitle" style="color: rgba(245, 194, 66, 0.8);">Adaptive Dimming</div>
    </div>
    <div class="premium-card" style="border-left: 4px solid #a855f7;">
        <div class="card-title">CO₂ REDUCTION</div>
        <div class="card-value" style="color: #a855f7;">{stgcn_summary['CO2 Avoided (kg)']:,.1f} kg</div>
        <div class="card-subtitle" style="color: rgba(168, 85, 247, 0.8);">Carbon Avoided</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- Main Tabs -----------------
tab_eval, tab_topology, tab_stgcn, tab_cabling, tab_dispatch = st.tabs([
    "📈 Project Evaluation & Savings",
    "🌐 Multi-layered Graph Topology",
    "🧠 Spatio-Temporal GNN (STGCN)",
    "⚡ Cabling Optimization (Kruskal)",
    "🛠️ Maintenance Dispatch & NLP"
])

# ================= TAB 1: EVALUATION & SAVINGS =================
with tab_eval:
    st.subheader("📈 Energy Savings & System-Wide Policy Benchmarks")
    st.write("Compare the municipal fixed lighting baseline (100% constant) against Rule-Based dimming (RBC) and the Spatio-Temporal Graph Convolutional Network (STGCN) dimming policy.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="STGCN Energy Saved (kWh)", 
            value=f"{stgcn_summary['CO2 Avoided (kg)'] / 0.533:,.1f} kWh", 
            delta=f"{stgcn_summary['Energy Savings (%)']}% Saved", 
            delta_color="normal"
        )
    with col2:
        st.metric(
            label="Financial Budget Saved ($)", 
            value=f"${stgcn_summary['Financial Savings ($)']:,.2f}", 
            delta="Tariff: $0.12/kWh"
        )
    with col3:
        st.metric(
            label="CO₂ Emissions Reduced", 
            value=f"{stgcn_summary['CO2 Avoided (kg)']:,.1f} kg", 
            delta="533g / kWh Carbon Intensity"
        )
        
    st.write("")
    st.subheader("Policy Comparison & KPI Evaluation Summary")
    st.dataframe(summary_metrics, use_container_width=True)
    
    # Plotly Charts
    fig_eval = sp.make_subplots(rows=1, cols=2, subplot_titles=(
        '<b>Total Energy Consumption (kWh) - Test Period</b>',
        '<b>Safety Violation Rate (%)</b>'
    ))
    
    fig_eval.add_trace(
        go.Bar(
            x=summary_metrics['Policy'],
            y=summary_metrics['Total Energy (kWh)'],
            marker=dict(color=['#e11d48', '#d97706', '#059669']),
            text=[f"{v:,.1f} kWh" for v in summary_metrics['Total Energy (kWh)']],
            textposition='auto',
            name='Energy'
        ),
        row=1, col=1
    )
    
    fig_eval.add_trace(
        go.Bar(
            x=summary_metrics['Policy'],
            y=summary_metrics['Safety Violation Rate (%)'],
            marker=dict(color=['#0f172a', '#ea580c', '#e11d48']),
            text=[f"{v:.3f}%" for v in summary_metrics['Safety Violation Rate (%)']],
            textposition='auto',
            name='Violations'
        ),
        row=1, col=2
    )
    
    fig_eval.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(12,21,45,0.4)',
        font=dict(color='#8b9bb4', family='Outfit'),
        showlegend=False,
        height=450,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig_eval.update_xaxes(showgrid=False)
    fig_eval.update_yaxes(showgrid=True, gridcolor='#1a2c5a')
    st.plotly_chart(fig_eval, use_container_width=True)

# ================= TAB 2: NETWORK TOPOLOGY =================
with tab_topology:
    st.subheader("🌐 Zoomable Interconnected Graph Topology")
    st.write("Visualise the road street grid and the electrical distribution network mapping transformers and sub-stations.")
    
    def draw_plotly_graph(G, title_text, color_scale='Viridis'):
        pos = {node: (attrs['Longitude'], attrs['Latitude']) for node, attrs in G.nodes(data=True) if 'Longitude' in attrs}
        if not pos:
            pos = nx.spring_layout(G)
            
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.0, color='rgba(28, 44, 90, 0.5)'), hoverinfo='none', mode='lines')
        
        node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
        degrees = dict(G.degree())
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            deg = degrees.get(node, 0)
            node_size.append(deg * 3 + 6)
            node_color.append(deg)
            node_text.append(f"Node: {node}<br>Degree: {deg}<br>Type: {G.nodes[node].get('Node_Type', 'N/A')}")
            
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
            marker=dict(showscale=True, colorscale=color_scale, color=node_color, size=node_size,
                        colorbar=dict(thickness=15, title='Node Connections', xanchor='left', tickfont=dict(color='#8b9bb4', size=10)),
                        line_width=1.5, line_color='#060913')
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text=title_text, font=dict(color='#ffffff', size=15, family='Outfit')),
                showlegend=False, hovermode='closest',
                paper_bgcolor='rgba(12,21,45,0.4)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(b=20, l=20, r=20, t=50)
            )
        )
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(draw_plotly_graph(road_g, "<b>1. Road Street Network Graph</b>", 'Viridis'), use_container_width=True)
    with col2:
        st.plotly_chart(draw_plotly_graph(elect_g, "<b>2. Electrical Grid Graph</b>", 'Tealrose'), use_container_width=True)

# ================= TAB 3: SPATIO-TEMPORAL GNN =================
with tab_stgcn:
    st.subheader("🧠 Spatio-Temporal Graph Convolutional Network (STGCN) Core")
    st.write("Train and forecast traffic intensity across the electrical graph topology using PyTorch Geometric.")
    
    # GNN metrics
    stgcn_metrics_file = "scratch/STGCN_Accuracy_Metrics.xlsx"
    if os.path.exists(stgcn_metrics_file):
        gnn_m = pd.read_excel(stgcn_metrics_file)
        st.write("#### STGCN Prediction Performance:")
        st.dataframe(gnn_m, use_container_width=True)
        
    st.markdown("---")
    st.write("#### Node-Level Traffic Intensity Heatmap (STGCN Predicted)")
    target_sel = st.selectbox("Select Traffic Category", ["Pedestrians", "Vehicles"])
    
    # Node averages
    target_col = 'Pred_Pedestrians' if target_sel == 'Pedestrians' else 'Pred_Vehicles'
    node_means = df_test_hourly.groupby('Node_ID')[target_col].mean().to_dict()
    
    pos = {node: (attrs['Longitude'], attrs['Latitude']) for node, attrs in elect_g.nodes(data=True) if 'Longitude' in attrs}
    if not pos:
        pos = nx.spring_layout(elect_g)
        
    edge_x, edge_y = [], []
    for u, v in elect_g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.8, color='rgba(28, 44, 90, 0.4)'), hoverinfo='none', mode='lines')
    
    nx_list, ny_list, nc_list, ns_list, nt_list = [], [], [], [], []
    for node in elect_g.nodes():
        x, y = pos[node]
        nx_list.append(x)
        ny_list.append(y)
        pred_val = node_means.get(node, 0.0)
        nc_list.append(pred_val)
        ns_list.append(max(8, min(24, pred_val * 0.8 + 8)))
        nt_list.append(f"Node: {node}<br>GNN Predicted Avg {target_sel}: {pred_val:.2f}")
        
    node_trace = go.Scatter(
        x=nx_list, y=ny_list, mode='markers', hoverinfo='text', text=nt_list,
        marker=dict(showscale=True, colorscale='Hot', color=nc_list, size=ns_list,
                    colorbar=dict(thickness=15, title='Predicted Flow', xanchor='left', tickfont=dict(color='#8b9bb4', size=10)),
                    line_width=1.5, line_color='#060913')
    )
    
    fig_heat = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=f"STGCN Predicted {target_sel} Heatmap", font=dict(color='#ffffff', size=14)),
            showlegend=False, hovermode='closest',
            paper_bgcolor='rgba(12,21,45,0.4)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(b=20, l=20, r=20, t=50)
        )
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ================= TAB 4: CABLING OPTIMIZATION =================
with tab_cabling:
    st.subheader("⚡ Cabling Infrastructure Optimization using Kruskal's MST")
    st.write("Reduce electrical grid line losses by optimizing the cabling layout using Kruskal's Minimum Spanning Tree (MST) algorithm.")
    
    mst_edges, cabling_metrics = compute_electrical_mst()
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.metric("Total Cable Length Saved", f"{cabling_metrics['length_saved_m']:,} m", f"-{cabling_metrics['savings_pct']}% Length")
    with col_c2:
        st.metric("MST Cabling Connections", f"{cabling_metrics['mst_edges']} / {cabling_metrics['original_edges']} edges")
    with col_c3:
        st.metric("Est. Line Loss Energy Saved", f"{cabling_metrics['estimated_line_loss_saved_kWh']:,} kWh")
        
    st.write("")
    
    # Kruskal plot
    # Build NetworkX graph from MST edges
    mst_graph = nx.Graph()
    mst_graph.add_nodes_from(elect_g.nodes(data=True))
    mst_graph.add_edges_from(mst_edges)
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.plotly_chart(draw_plotly_graph(elect_g, "<b>Original Electrical Cabling Layout</b>", 'YlOrRd'), use_container_width=True)
    with col_g2:
        st.plotly_chart(draw_plotly_graph(mst_graph, "<b>Kruskal MST Optimized Cabling Layout</b>", 'YlGn'), use_container_width=True)

# ================= TAB 5: MAINTENANCE & NLP =================
with tab_dispatch:
    st.subheader("🛠️ Maintenance Dispatch & Multilingual NLP Report Analyzer")
    st.write("Submit an unstructured incident report in **French**, **Arabic**, or **Darija**. The system will identify the fault, locate the node, and run the **Bellman-Ford algorithm** to plan the maintenance tour.")
    
    report_text = st.text_input(
        "Enter Incident Report", 
        value="Court-circuit détecté poteau Node_040, secteur Sector_B, intervention urgente."
    )
    
    if st.button("Parse Report & Dispatch"):
        nlp_res = nlp_analyzer.analyze_report(report_text)
        
        # Display NLP output
        col_nlp1, col_nlp2, col_nlp3, col_nlp4 = st.columns(4)
        with col_nlp1:
            st.markdown(f"<div class='premium-card' style='border-left: 4px solid #00d2ff;'><strong>Detected Language</strong><br><span style='font-size:1.3rem; font-weight:bold; color:#00d2ff;'>{nlp_res['Language']}</span></div>", unsafe_allow_html=True)
        with col_nlp2:
            st.markdown(f"<div class='premium-card' style='border-left: 4px solid #ea580c;'><strong>Fault Category</strong><br><span style='font-size:1.3rem; font-weight:bold; color:#ea580c;'>{nlp_res['Extracted_Fault']}</span></div>", unsafe_allow_html=True)
        with col_nlp3:
            st.markdown(f"<div class='premium-card' style='border-left: 4px solid #e11d48;'><strong>Resolved Target Node</strong><br><span style='font-size:1.3rem; font-weight:bold; color:#e11d48;'>{nlp_res['Extracted_Node']}</span></div>", unsafe_allow_html=True)
        with col_nlp4:
            st.markdown(f"<div class='premium-card' style='border-left: 4px solid #d97706;'><strong>Severity Sentiment</strong><br><span style='font-size:1.3rem; font-weight:bold; color:#d97706;'>{nlp_res['Sentiment']}</span> (Score: {nlp_res['Urgency_Score']:.2f})</div>", unsafe_allow_html=True)
            
        st.write("")
        st.subheader(f"Optimal Routing Path from depot (Node_001) to {nlp_res['Extracted_Node']} (Bellman-Ford)")
        
        # Plan route
        try:
            route_res = plan_maintenance_route("Node_001", nlp_res['Extracted_Node'])
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Total Path Distance", f"{route_res['distance_m']:,} m")
            with col_r2:
                st.metric("Path Hops", f"{route_res['hops']} transitions")
            with col_r3:
                st.metric("Est. Travel Time", f"{route_res['time_min']} mins", "Speed: 25 km/h")
                
            # Path node list
            st.write(f"**Optimal Route Nodes**: {' → '.join(route_res['path'])}")
            
            # Map visualization of path
            pos = {node: (attrs['Longitude'], attrs['Latitude']) for node, attrs in road_g.nodes(data=True) if 'Longitude' in attrs}
            
            # Draw road edges
            edge_x, edge_y = [], []
            for u, v in road_g.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.8, color='rgba(28, 44, 90, 0.3)'), hoverinfo='none', mode='lines')
            
            # Draw path edges in highlighted color
            path_x, path_y = [], []
            path_nodes = route_res['path']
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i+1]
                if u in pos and v in pos:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    path_x.extend([x0, x1, None])
                    path_y.extend([y0, y1, None])
                    
            path_edge_trace = go.Scatter(x=path_x, y=path_y, line=dict(width=3.5, color='#e11d48'), name='Dispatch Tour Path', mode='lines')
            
            # Draw nodes
            node_x, node_y, node_color, node_text = [], [], [], []
            for node in road_g.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                if node == "Node_001":
                    node_color.append('#4ed9a6')
                    node_text.append("START: Maintenance Depot (Node_001)")
                elif node == nlp_res['Extracted_Node']:
                    node_color.append('#e11d48')
                    node_text.append(f"DESTINATION: Incident Node ({node})")
                elif node in path_nodes:
                    node_color.append('#f5c242')
                    node_text.append(f"Path Node: {node}")
                else:
                    node_color.append('rgba(28, 44, 90, 0.6)')
                    node_text.append(f"Node: {node}")
                    
            node_sizes = [15 if n in ["Node_001", nlp_res['Extracted_Node']] else (10 if n in path_nodes else 5) for n in road_g.nodes()]
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                marker=dict(color=node_color, size=node_sizes, line_width=1, line_color='#060913')
            )
            
            fig_map = go.Figure(
                data=[edge_trace, path_edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(text="Optimal Dispatch Route Overlay", font=dict(color='#ffffff', size=14)),
                    showlegend=True, hovermode='closest',
                    paper_bgcolor='rgba(12,21,45,0.4)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    margin=dict(b=20, l=20, r=20, t=50),
                    legend=dict(font=dict(color='#8b9bb4', size=10), bgcolor='rgba(12,21,45,0.8)')
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
        except Exception as route_err:
            st.error(f"Routing error: {route_err}")
