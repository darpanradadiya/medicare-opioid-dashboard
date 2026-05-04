"""
============================================================
  Medicare Part D Opioid Prescribing Dashboard
  ALY 6110 - Module 4 - Group 7
  Author: Darpan Radadiya 
============================================================
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Medicare Part D Opioid Prescribing Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": (
            "Medicare Part D Opioid Prescribing Dashboard\n"
            "ALY 6110 - Northeastern University\n"
            "Built by Darpan Radadiya & Jingqi Huo\n"
            "Data: CMS Medicare Part D Prescribers (2023)"
        )
    },
)

# ============================================================
# 2. CUSTOM CSS THEME
# ============================================================
CUSTOM_CSS = """
<style>
    .stApp { background-color: #f5f7fa; }
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #5d6d7e;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        border-left: 5px solid #1f4e79;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
        height: 100%;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.06rem;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1f4e79;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #95a5a6;
        margin-top: 0.2rem;
    }

    .section-finding {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1f4e79;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
    }
    .section-context {
        font-size: 1rem;
        color: #566573;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    .so-what {
        background: #fef5e7;
        border-left: 4px solid #e67e22;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-size: 1rem;
        color: #5d4037;
        margin-top: 1rem;
        line-height: 1.55;
    }
    .so-what-label {
        font-weight: 700;
        color: #b9770e;
        margin-right: 0.4rem;
    }

    .insight-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        height: 100%;
        border-top: 4px solid #1f4e79;
    }
    .insight-num {
        display: inline-block;
        background: #1f4e79;
        color: white;
        width: 32px; height: 32px;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        font-weight: 700;
        margin-right: 0.6rem;
    }
    .insight-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.7rem;
    }
    .insight-body {
        font-size: 0.95rem;
        color: #4a5568;
        line-height: 1.55;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        padding: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 0.7rem 1.4rem;
        font-weight: 600;
        font-size: 0.98rem;
        color: #5d6d7e;
        border: 1px solid #e1e8ed;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: #1f4e79 !important;
        color: white !important;
        border-color: #1f4e79 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: #ffffff;
        border-radius: 0 10px 10px 10px;
        padding: 1.8rem 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e1e8ed;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# 3. CACHED DATA LOADERS
# ============================================================
@st.cache_data(show_spinner="Loading prescriber sample…")
def load_main_data():
    return pd.read_parquet("dashboard_data.parquet")

@st.cache_data(show_spinner="Loading opioid records…")
def load_opioid_data():
    return pd.read_parquet("opioid_records.parquet")

@st.cache_data
def load_specialty_stats():
    return pd.read_csv("insight1_specialty.csv")

@st.cache_data
def load_state_clusters():
    return pd.read_csv("insight2_state_clusters.csv")

@st.cache_data
def load_correlations():
    return {
        "all":    pd.read_csv("insight3_corr_all.csv",       index_col=0),
        "opioid": pd.read_csv("insight3_corr_opioid.csv",    index_col=0),
        "non":    pd.read_csv("insight3_corr_nonopioid.csv", index_col=0),
    }

@st.cache_data
def load_kpis():
    with open("dashboard_kpis.json") as f:
        return json.load(f)

df       = load_main_data()
df_opi   = load_opioid_data()
spec     = load_specialty_stats()
states   = load_state_clusters()
corrs    = load_correlations()
kpis     = load_kpis()

# ============================================================
# 4. HEADER
# ============================================================
st.markdown(
    '<div class="main-title">💊 Medicare Part D Opioid Prescribing Analytics</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-subtitle">'
    "Identifying high-rate opioid prescribers across "
    f"<b>{kpis['total_records']/1e6:.1f} million</b> Medicare prescription records (2023) "
    "&nbsp;|&nbsp; ALY 6110 &middot; Northeastern University &middot; Group 7"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<hr style="border: none; height: 3px; '
    'background: linear-gradient(to right, #1f4e79 0%, #1f4e79 25%, transparent 100%); '
    'margin: 0 0 1.5rem 0;">',
    unsafe_allow_html=True,
)

# ============================================================
# 5. KPI BAR
# ============================================================
def kpi_card(label, value, sub="", border_color="#1f4e79"):
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color: {border_color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

k1, k2, k3, k4 = st.columns(4, gap="medium")
with k1:
    kpi_card("Total Prescription Records",
             f"{kpis['total_records']/1e6:.1f}M",
             f"{kpis['total_records']:,} prescriber-drug pairs")
with k2:
    kpi_card("Opioid Prescriptions",
             f"{kpis['total_opioid_records']/1e3:.0f}K",
             f"{kpis['total_opioid_records']:,} records flagged")
with k3:
    kpi_card("Opioid Prescribing Rate",
             f"{kpis['opioid_rate_pct']}%",
             "of all Medicare Part D records")
with k4:
    kpi_card("Unique Prescribers",
             f"{kpis['unique_prescribers']/1e6:.2f}M",
             f"across {kpis['unique_specialties']} specialties · "
             f"{kpis['unique_states']} states")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# 6. TABS
# ============================================================
tab_overview, tab_who, tab_where, tab_how, tab_drugs, tab_explore = st.tabs([
    "📊 Overview",
    "👨‍⚕️ Specialties",
    "🗺️ Geography",
    "🔗 Patterns",
    "💊 Drug Mix",
    "🔍 Explore",
])

# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with tab_overview:
    st.markdown(
        '<div class="section-finding">Research Question</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-context" style="font-size: 1.1rem;">'
        "<i>Can we predict whether a Medicare Part D prescriber-drug record is an "
        "opioid prescription, and predict its claim volume — using provider specialty, "
        "state, and prescribing behavior as features?</i>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("##### Three insights uncovered in this analysis")
    st.markdown("<br>", unsafe_allow_html=True)

    o1, o2, o3 = st.columns(3, gap="large")
    with o1:
        st.markdown(
            """
            <div class="insight-card">
              <div class="insight-title">
                <span class="insight-num">1</span>The Volume-vs-Rate Paradox
              </div>
              <div class="insight-body">
              Family Practice writes 201,212 opioid records — the highest count —
              but only 3.0% of their prescriptions are opioid.
              <b>Hand Surgery writes 2,387 opioid records but 44.65% of their prescriptions are opioid.</b>
              Volume rankings hide the high-rate surgical specialties entirely.
              <br><br>
              <i>See: Specialties tab →</i>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o2:
        st.markdown(
            """
            <div class="insight-card" style="border-top-color: #c0392b;">
              <div class="insight-title" style="color: #c0392b;">
                <span class="insight-num" style="background:#c0392b;">2</span>
                Geographic Clustering
              </div>
              <div class="insight-body">
              K-Means clustering grouped 52 states into 4 behavioral clusters.
              <b>Colorado, Utah, Oregon, Washington and the Mountain West lead</b>
              opioid prescribing rates — not West Virginia or Kentucky as the
              conventional crisis narrative suggests.
              <br><br>
              <i>See: Geography tab →</i>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o3:
        st.markdown(
            """
            <div class="insight-card" style="border-top-color: #e67e22;">
              <div class="insight-title" style="color: #e67e22;">
                <span class="insight-num" style="background:#e67e22;">3</span>
                Regulatory Coupling
              </div>
              <div class="insight-body">
              For non-opioid drugs, days-supply correlates only 0.13 with patient count.
              <b>For opioids, that correlation jumps to 0.72</b> — opioid prescribing
              is regulatorily constrained, while non-opioid prescribing is not.
              The numeric features tell a fundamentally different story.
              <br><br>
              <i>See: Patterns tab →</i>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-weight:600; color:#1f4e79; font-size:1rem; '
        'margin-bottom:0.8rem;">📈 Three findings at a glance</div>',
        unsafe_allow_html=True,
    )

    ov1, ov2, ov3 = st.columns(3, gap="large")

    # ---- Mini chart 1: Top 8 specialties by rate ----
    with ov1:
        top8 = (spec[spec["total_records"] >= 1000]
                .nlargest(8, "opioid_rate_pct")
                .sort_values("opioid_rate_pct", ascending=True))
        fig_ov1 = go.Figure()
        fig_ov1.add_trace(go.Bar(
            x=top8["opioid_rate_pct"],
            y=top8["Prscrbr_Type"].str[:18],
            orientation="h",
            marker=dict(color="#c0392b"),
            hovertemplate="<b>%{y}</b><br>Rate: %{x:.1f}%<extra></extra>",
        ))
        fig_ov1.add_vline(x=3.53, line_dash="dot",
                          line_color="#1f4e79", line_width=1.5)
        fig_ov1.update_layout(
            height=320, margin=dict(l=5, r=5, t=5, b=5),
            plot_bgcolor="#fafbfc", paper_bgcolor="#ffffff",
            font=dict(size=9, color="#2c3e50"),
            xaxis=dict(ticksuffix="%", gridcolor="#ecf0f1",
                       zeroline=False, tickfont=dict(size=10)),
            yaxis=dict(gridcolor="#ffffff", tickfont=dict(size=10)),
            showlegend=False,
        )
        st.plotly_chart(fig_ov1, use_container_width=True)
        st.caption("Top 8 specialties by opioid rate · dashed = national avg")

    # ---- Mini chart 2: State cluster donut ----
    with ov2:
        from collections import Counter
        cluster_labels_ov = {
            0: "Mainstream", 1: "High-Rate West",
            2: "Middle America", 3: "Outlier (PR)"
        }
        cluster_colors_ov = {
            0: "#5d6d7e", 1: "#c0392b", 2: "#e6b85c", 3: "#7d3c98"
        }
        vc = states["cluster"].value_counts().sort_index()
        fig_ov2 = go.Figure(data=go.Pie(
            labels=[cluster_labels_ov[i] for i in vc.index],
            values=vc.values,
            hole=0.45,
            marker=dict(colors=[cluster_colors_ov[i] for i in vc.index],
                        line=dict(color="#ffffff", width=1)),
            textinfo="label+value",
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value} states<extra></extra>",
        ))
        fig_ov2.update_layout(
            height=320, margin=dict(l=5, r=5, t=5, b=15),
            paper_bgcolor="#ffffff",
            font=dict(size=9, color="#2c3e50"),
            showlegend=False,
        )
        st.plotly_chart(fig_ov2, use_container_width=True)
        st.caption("52 states grouped into 4 prescribing clusters")

    # ---- Mini chart 3: Days Supply correlation comparison ----
    with ov3:
        categories   = ["Days Supply ↔ Patients",
                        "Claims ↔ Patients",
                        "Drug Cost ↔ Patients"]
        opioid_vals  = [0.72, 0.79, 0.30]
        non_vals     = [0.13, 0.94, 0.74]

        fig_ov3 = go.Figure()
        fig_ov3.add_trace(go.Bar(
            name="Opioid", x=opioid_vals, y=categories,
            orientation="h",
            marker=dict(color="#c0392b"),
            text=[f"{v:.2f}" for v in opioid_vals],
            textposition="outside", textfont=dict(size=9),
        ))
        fig_ov3.add_trace(go.Bar(
            name="Non-Opioid", x=non_vals, y=categories,
            orientation="h",
            marker=dict(color="#5d6d7e"),
            text=[f"{v:.2f}" for v in non_vals],
            textposition="outside", textfont=dict(size=9),
        ))
        fig_ov3.update_layout(
            barmode="group", height=320,
            margin=dict(l=5, r=40, t=5, b=5),
            plot_bgcolor="#fafbfc", paper_bgcolor="#ffffff",
            font=dict(size=9, color="#2c3e50"),
            xaxis=dict(range=[0, 1.15], gridcolor="#ecf0f1",
                       zeroline=False, tickfont=dict(size=10)),
            yaxis=dict(gridcolor="#ffffff", tickfont=dict(size=10)),
            legend=dict(orientation="h", y=-0.15,
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_ov3, use_container_width=True)
        st.caption("Pearson r: opioid vs non-opioid prescribing")

# ============================================================
# TAB 2: SPECIALTIES (§1 WHO)
# ============================================================
with tab_who:
    st.markdown(
        '<div class="section-finding">§1 &nbsp;·&nbsp; '
        "Who Prescribes Opioids? The Volume-vs-Rate Paradox</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-context">'
        "Volume rankings hide high-rate surgical specialties; rate rankings hide "
        "high-volume primary-care prescribers. Both lenses are shown side-by-side."
        "</div>",
        unsafe_allow_html=True,
    )

    L, R = st.columns([1.15, 1], gap="medium")

    with L:
        st.markdown(
            '<div style="font-weight:600; color:#1f4e79; font-size:1rem; '
            'margin-bottom:0.4rem;">Volume × Rate Quadrants</div>',
            unsafe_allow_html=True,
        )

        spec_chart = spec[spec["total_records"] >= 1000].copy()
        vol_thresh  = spec_chart["opioid_records"].quantile(0.75)
        rate_thresh = spec_chart["opioid_rate_pct"].quantile(0.75)

        def assign_quadrant(row):
            hv = row["opioid_records"]  >= vol_thresh
            hr = row["opioid_rate_pct"] >= rate_thresh
            if hv and hr:        return "Obvious (high vol + high rate)"
            elif (not hv) and hr: return "Hidden signal (low vol + high rate)"
            elif hv and (not hr): return "High-noise (high vol + low rate)"
            else:                 return "Background"

        spec_chart["quadrant"] = spec_chart.apply(assign_quadrant, axis=1)
        quadrant_colors = {
            "Obvious (high vol + high rate)":      "#c0392b",
            "Hidden signal (low vol + high rate)": "#e67e22",
            "High-noise (high vol + low rate)":    "#5d6d7e",
            "Background":                          "#bdc3c7",
        }

        fig1 = px.scatter(
            spec_chart,
            x="opioid_records", y="opioid_rate_pct",
            size="total_records", color="quadrant",
            color_discrete_map=quadrant_colors,
            category_orders={"quadrant": list(quadrant_colors.keys())},
            hover_name="Prscrbr_Type",
            hover_data={"opioid_records": ":,", "total_records": ":,",
                        "opioid_rate_pct": ":.2f", "quadrant": False},
            size_max=42, log_x=True,
            labels={"opioid_records":  "Opioid Records (log)",
                    "opioid_rate_pct": "Opioid Rate (%)"},
        )
        fig1.add_vline(x=vol_thresh, line_dash="dot", line_color="#34495e",
                       opacity=0.5)
        fig1.add_hline(y=rate_thresh, line_dash="dot", line_color="#34495e",
                       opacity=0.5)

        spotlight = [
            ("Hand Surgery",       -25, -30),
            ("Pain Management",     30, -25),
            ("Family Practice",    -45,  20),
            ("Nurse Practitioner", -65,  35),
        ]
        for s, ax_off, ay_off in spotlight:
            row = spec_chart[spec_chart["Prscrbr_Type"] == s]
            if not row.empty:
                r = row.iloc[0]
                fig1.add_annotation(
                    x=np.log10(r["opioid_records"]), y=r["opioid_rate_pct"],
                    text=f"<b>{s}</b>", showarrow=True, arrowhead=2,
                    arrowsize=0.7, arrowwidth=1, ax=ax_off, ay=ay_off,
                    font=dict(size=10, color="#1f4e79"),
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#1f4e79", borderwidth=1, borderpad=2,
                )

        fig1.update_layout(
            height=540, margin=dict(l=10, r=10, t=10, b=80),
            plot_bgcolor="#fafbfc", paper_bgcolor="#ffffff",
            font=dict(family="Inter, Segoe UI, sans-serif", size=11,
                      color="#2c3e50"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                        xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=10), title=None),
            xaxis=dict(gridcolor="#ecf0f1", zeroline=False,
                       title_font=dict(size=12)),
            yaxis=dict(gridcolor="#ecf0f1", zeroline=False, ticksuffix="%",
                       title_font=dict(size=12)),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with R:
        st.markdown(
            '<div style="font-weight:600; color:#1f4e79; font-size:1rem; '
            'margin-bottom:0.4rem;">Top 15 Specialties by Opioid Rate</div>',
            unsafe_allow_html=True,
        )

        def short(name, max_len=28):
            return name if len(name) <= max_len else name[:max_len-1] + "…"

        top15 = (spec_chart.sort_values("opioid_rate_pct", ascending=False)
                           .head(15)
                           .sort_values("opioid_rate_pct", ascending=True)
                           .copy())
        top15["short_name"] = top15["Prscrbr_Type"].map(short)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=top15["opioid_rate_pct"], y=top15["short_name"], orientation="h",
            marker=dict(
                color=top15["opioid_rate_pct"],
                colorscale=[[0.0, "#fadbd8"], [0.5, "#e6796d"], [1.0, "#922b21"]],
                line=dict(color="#ffffff", width=1), showscale=False,
            ),
            text=[f"<b>{rate:.1f}%</b>" for rate in top15["opioid_rate_pct"]],
            textposition="outside",
            textfont=dict(size=11, color="#2c3e50"),
            customdata=top15[["Prscrbr_Type", "opioid_records",
                              "total_records"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Opioid rate: %{x:.2f}%<br>"
                "Opioid records: %{customdata[1]:,}<br>"
                "Total records: %{customdata[2]:,}<extra></extra>"
            ),
        ))
        fig2.add_vline(x=3.53, line_dash="dash", line_color="#1f4e79",
                       line_width=2,
                       annotation_text="<b>3.53%</b>",
                       annotation_position="top",
                       annotation_font=dict(size=10, color="#1f4e79"))

        fig2.update_layout(
            height=540, margin=dict(l=10, r=70, t=10, b=40),
            plot_bgcolor="#fafbfc", paper_bgcolor="#ffffff",
            font=dict(family="Inter, Segoe UI, sans-serif", size=11,
                      color="#2c3e50"),
            xaxis=dict(title="Opioid rate (%)", ticksuffix="%",
                       gridcolor="#ecf0f1", zeroline=False,
                       range=[0, max(top15["opioid_rate_pct"]) * 1.30],
                       title_font=dict(size=12)),
            yaxis=dict(title=None, gridcolor="#ffffff", zeroline=False,
                       tickfont=dict(size=11)),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    qcount = spec_chart["quadrant"].value_counts()
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("🔴 Obvious",  qcount.get("Obvious (high vol + high rate)", 0))
    with m2: st.metric("🟠 Hidden",   qcount.get("Hidden signal (low vol + high rate)", 0))
    with m3: st.metric("⚫ Noisy",    qcount.get("High-noise (high vol + low rate)", 0))
    with m4: st.metric("⚪ Background", qcount.get("Background", 0))

    st.markdown(
        """
        <div class="so-what" style="margin-top:0.6rem;">
          <span class="so-what-label">KEY INSIGHT:</span>
          Family Practice writes 201K opioid records at a 3.0% rate (97% non-opioid).
          Hand Surgery writes only 2.4K opioid records but at <b>44.65% rate</b>.
          11 of the top 15 highest-rate specialties are surgical — invisible
          when ranked by volume alone. The classifier needs both signals.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
# ============================================================
# TAB 3: GEOGRAPHY (§2 WHERE)
# ============================================================
with tab_where:
    st.markdown(
        '<div class="section-finding">§2 &nbsp;·&nbsp; '
        "Where Are Opioids Prescribed?</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-context">'
        "K-Means clustered 52 states/territories into 4 behavioral groups. "
        "The map shows opioid rate by state; the bar chart shows the top 5 "
        "of each cluster."
        "</div>",
        unsafe_allow_html=True,
    )

    GL, GR = st.columns([1.2, 1], gap="medium")

    with GL:
        st.markdown(
            '<div style="font-weight:600; color:#1f4e79; font-size:1rem; '
            'margin-bottom:0.4rem;">'
            "Opioid Rate by State (% of all prescriptions)</div>",
            unsafe_allow_html=True,
        )

        map_states = states[~states["Prscrbr_State_Abrvtn"].isin(
            ["PR", "GU", "VI", "MP", "AS"]
        )].copy()

        fig3 = go.Figure(data=go.Choropleth(
            locations=map_states["Prscrbr_State_Abrvtn"],
            z=map_states["opioid_rate_pct"],
            locationmode="USA-states",
            colorscale=[
                [0.0, "#1f4e79"], [0.4, "#aed6f1"], [0.5, "#f5f5f5"],
                [0.6, "#f5b7b1"], [1.0, "#922b21"],
            ],
            zmin=2.5, zmax=5.5, zmid=3.53,
            marker_line_color="white", marker_line_width=1.0,
            colorbar=dict(
                title=dict(text="Rate %", font=dict(size=10)),
                tickfont=dict(size=9), ticksuffix="%",
                len=0.7, thickness=12, x=1.02,
            ),
            customdata=map_states[["opioid_records", "cluster",
                                   "unique_npis"]].values,
            hovertemplate=(
                "<b>%{location}</b><br>"
                "Rate: %{z:.2f}%<br>"
                "Opioid records: %{customdata[0]:,}<br>"
                "Prescribers: %{customdata[2]:,}<br>"
                "Cluster: %{customdata[1]}<extra></extra>"
            ),
        ))

        # State abbreviation labels overlaid on the choropleth
        state_centroids = {
            "AL": (32.8, -86.8),  "AK": (63.6, -152.0), "AZ": (34.3, -111.7),
            "AR": (34.9, -92.4),  "CA": (37.2, -119.7), "CO": (38.9, -105.6),
            "CT": (41.6, -72.7),  "DE": (39.0, -75.5),  "DC": (38.9, -77.0),
            "FL": (28.6, -82.5),  "GA": (32.7, -83.4),  "HI": (20.6, -157.5),
            "ID": (44.4, -114.6), "IL": (40.0, -89.2),  "IN": (39.9, -86.3),
            "IA": (42.1, -93.5),  "KS": (38.5, -98.4),  "KY": (37.5, -85.3),
            "LA": (31.1, -91.9),  "ME": (45.4, -69.0),  "MD": (39.0, -76.8),
            "MA": (42.2, -71.5),  "MI": (44.3, -85.4),  "MN": (46.3, -94.3),
            "MS": (32.7, -89.7),  "MO": (38.4, -92.3),  "MT": (47.0, -109.6),
            "NE": (41.5, -99.8),  "NV": (39.3, -116.6), "NH": (43.7, -71.6),
            "NJ": (40.2, -74.5),  "NM": (34.4, -106.1), "NY": (42.9, -75.5),
            "NC": (35.6, -79.8),  "ND": (47.5, -100.3), "OH": (40.3, -82.8),
            "OK": (35.5, -97.5),  "OR": (44.1, -120.5), "PA": (40.6, -77.2),
            "RI": (41.7, -71.5),  "SC": (33.9, -80.9),  "SD": (44.4, -100.2),
            "TN": (35.7, -86.7),  "TX": (31.1, -97.6),  "UT": (39.3, -111.7),
            "VT": (44.0, -72.7),  "VA": (37.5, -78.9),  "WA": (47.4, -120.5),
            "WV": (38.5, -80.6),  "WI": (44.3, -89.6),  "WY": (43.0, -107.6),
        }
        labels_lat, labels_lon, labels_text = [], [], []
        for st_code in map_states["Prscrbr_State_Abrvtn"]:
            if st_code in state_centroids:
                lat, lon = state_centroids[st_code]
                labels_lat.append(lat)
                labels_lon.append(lon)
                labels_text.append(st_code)

        fig3.add_trace(go.Scattergeo(
            lon=labels_lon, lat=labels_lat, text=labels_text,
            mode="text",
            textfont=dict(size=9, color="#1a1a1a", family="Inter, sans-serif"),
            showlegend=False, hoverinfo="skip",
        ))

        fig3.update_layout(
            geo=dict(scope="usa", projection=dict(type="albers usa"),
                     showlakes=True, lakecolor="#eaf2f8", bgcolor="#ffffff"),
            height=540, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#ffffff",
            font=dict(family="Inter, Segoe UI, sans-serif", size=11,
                      color="#2c3e50"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with GR:
        st.markdown(
            '<div style="font-weight:600; color:#1f4e79; font-size:1rem; '
            'margin-bottom:0.4rem;">Top 5 States in Each Cluster</div>',
            unsafe_allow_html=True,
        )

        cluster_labels = {
            0: "Mainstream",
            1: "High-Rate West",
            2: "Middle America",
            3: "Outlier (PR)",
        }
        cluster_colors = {
            0: "#5d6d7e", 1: "#c0392b", 2: "#e6b85c", 3: "#7d3c98",
        }

        top_per_cluster = (
            states.sort_values("opioid_rate_pct", ascending=False)
                  .groupby("cluster", group_keys=False)
                  .head(5)
                  .copy()
        )
        cluster_order = [3, 0, 2, 1]
        top_per_cluster["cluster_rank"] = top_per_cluster["cluster"].map(
            {c: i for i, c in enumerate(cluster_order)}
        )
        top_per_cluster = top_per_cluster.sort_values(
            ["cluster_rank", "opioid_rate_pct"], ascending=[True, True]
        )

        fig4 = go.Figure()
        for c in cluster_order:
            sub = top_per_cluster[top_per_cluster["cluster"] == c]
            if sub.empty:
                continue
            fig4.add_trace(go.Bar(
                x=sub["opioid_rate_pct"], y=sub["Prscrbr_State_Abrvtn"],
                orientation="h",
                name=f"{cluster_labels[c]}",
                marker=dict(color=cluster_colors[c],
                            line=dict(color="#ffffff", width=1)),
                text=[f"{r:.2f}%" for r in sub["opioid_rate_pct"]],
                textposition="outside",
                textfont=dict(size=10, color="#2c3e50"),
                hovertemplate=(
                    "<b>%{y}</b><br>Rate: %{x:.2f}%<br>"
                    f"Cluster: {cluster_labels[c]}<extra></extra>"
                ),
            ))

        fig4.add_vline(
            x=3.53, line_dash="dash", line_color="#1f4e79", line_width=2,
            annotation_text="<b>National avg 3.53%</b>",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#1f4e79"),
        )
        fig4.update_layout(
            height=540, margin=dict(l=10, r=50, t=10, b=70),
            plot_bgcolor="#fafbfc", paper_bgcolor="#ffffff",
            font=dict(family="Inter, Segoe UI, sans-serif", size=10,
                      color="#2c3e50"),
            xaxis=dict(title="Opioid rate (%)", ticksuffix="%",
                       gridcolor="#ecf0f1", zeroline=False,
                       range=[0, 6], title_font=dict(size=11)),
            yaxis=dict(title=None, gridcolor="#ffffff", zeroline=False,
                       tickfont=dict(size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                        xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=9), title=None),
            bargap=0.2,
        )
        st.plotly_chart(fig4, use_container_width=True)

    cs1, cs2, cs3, cs4 = st.columns(4)
    summary = (states.groupby("cluster")
                     .agg(n=("Prscrbr_State_Abrvtn", "count"),
                          avg_rate=("opioid_rate_pct", "mean")))
    with cs1:
        if 1 in summary.index:
            delta = round(summary.loc[1, 'avg_rate'] - 3.53, 2)
            st.metric("🔴 High-Rate West",
                      f"{summary.loc[1, 'avg_rate']:.2f}%",
                      f"+{delta}% above national avg",
                      delta_color="inverse")
    with cs2:
        if 2 in summary.index:
            delta = round(summary.loc[2, 'avg_rate'] - 3.53, 2)
            st.metric("🟡 Middle America",
                      f"{summary.loc[2, 'avg_rate']:.2f}%",
                      f"+{delta}% above national avg",
                      delta_color="inverse")
    with cs3:
        if 0 in summary.index:
            delta = round(summary.loc[0, 'avg_rate'] - 3.53, 2)
            st.metric("⚫ Mainstream",
                      f"{summary.loc[0, 'avg_rate']:.2f}%",
                      f"{delta}% vs national avg",
                      delta_color="inverse")
    with cs4:
        if 3 in summary.index:
            delta = round(summary.loc[3, 'avg_rate'] - 3.53, 2)
            st.metric("🟣 Outlier (PR)",
                      f"{summary.loc[3, 'avg_rate']:.2f}%",
                      f"{delta}% vs national avg",
                      delta_color="off")

    st.markdown(
        """
        <div class="so-what" style="margin-top:0.6rem;">
          <span class="so-what-label">KEY INSIGHT:</span>
          The high-rate cluster forms a clean geographic block — Pacific NW,
          Mountain West, Alaska, and upper New England — all at <b>4.5%+</b>.
          Meanwhile <b>West Virginia (2.97%) and Kentucky (3.01%) sit BELOW the
          national average</b> of 3.53%, contradicting the conventional opioid-crisis
          geography. Crisis narratives reflect death and misuse rates; Medicare
          prescribing concentrates elsewhere.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📋 Show all 52 states/territories", expanded=False):
        full_table = states[
            ["Prscrbr_State_Abrvtn", "opioid_rate_pct",
             "opioid_records", "total_records", "unique_npis", "cluster"]
        ].copy().sort_values("opioid_rate_pct", ascending=False)
        full_table["cluster_label"] = full_table["cluster"].map(cluster_labels)
        full_table = full_table.rename(columns={
            "Prscrbr_State_Abrvtn": "State",
            "opioid_rate_pct":      "Rate %",
            "opioid_records":       "Opioid Records",
            "total_records":        "Total Records",
            "unique_npis":          "Prescribers",
            "cluster_label":        "Cluster",
        }).drop(columns=["cluster"])
        full_table["Rate %"] = full_table["Rate %"].map(lambda x: f"{x:.2f}%")
        for c in ["Opioid Records", "Total Records", "Prescribers"]:
            full_table[c] = full_table[c].map(lambda x: f"{x:,}")
        st.dataframe(full_table, use_container_width=True, hide_index=True,
                     height=420)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-weight:600; color:#1f4e79; font-size:1rem; '
        'margin-bottom:0.4rem;">🔀 State vs State Comparison</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.92rem; color:#566573; margin-bottom:0.8rem;">'
        "Pick any two states to compare their opioid rate, prescriber count, "
        "and top specialties side by side."
        "</div>",
        unsafe_allow_html=True,
    )

    all_state_list = sorted(states["Prscrbr_State_Abrvtn"].tolist())
    cmp1, cmp2 = st.columns(2)
    with cmp1:
        state_a = st.selectbox("State A", all_state_list,
                               index=all_state_list.index("CO"))
    with cmp2:
        state_b = st.selectbox("State B", all_state_list,
                               index=all_state_list.index("WV"))

    def get_state_row(code):
        return states[states["Prscrbr_State_Abrvtn"] == code].iloc[0]

    def get_top_specs(code, n=5):
        sub = df[df["Prscrbr_State_Abrvtn"] == code]
        return (
            sub[sub["is_opioid"] == 1]["Prscrbr_Type"]
            .value_counts()
            .head(n)
            .reset_index()
            .rename(columns={"Prscrbr_Type": "Specialty",
                             "count": "Opioid Records"})
        )

    ra = get_state_row(state_a)
    rb = get_state_row(state_b)

    # ---- Metric comparison row ----
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(f"Opioid Rate — {state_a}", f"{ra['opioid_rate_pct']:.2f}%",
                  f"{ra['opioid_rate_pct'] - rb['opioid_rate_pct']:+.2f}% vs {state_b}")
    with m2:
        st.metric(f"Opioid Rate — {state_b}", f"{rb['opioid_rate_pct']:.2f}%",
                  f"{rb['opioid_rate_pct'] - ra['opioid_rate_pct']:+.2f}% vs {state_a}")
    with m3:
        ratio = ra['opioid_rate_pct'] / rb['opioid_rate_pct']
        st.metric("Rate Ratio",
                  f"{ratio:.2f}×",
                  f"{state_a} prescribes opioids {ratio:.2f}x more than {state_b}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Side by side bar charts ----
    bar1, bar2 = st.columns(2)
    specs_a = get_top_specs(state_a)
    specs_b = get_top_specs(state_b)

    def state_bar(specs_df, state_code, color):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=specs_df["Opioid Records"],
            y=specs_df["Specialty"],
            orientation="h",
            marker=dict(color=color,
                        line=dict(color="#ffffff", width=1)),
            text=[f"{v:,}" for v in specs_df["Opioid Records"]],
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>Records: %{x:,}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=f"Top 5 Opioid Specialties — {state_code}",
                       font=dict(size=12, color="#1f4e79")),
            height=280,
            margin=dict(l=10, r=80, t=40, b=20),
            plot_bgcolor="#fafbfc",
            paper_bgcolor="#ffffff",
            font=dict(family="Inter, Segoe UI, sans-serif",
                      size=10, color="#2c3e50"),
            xaxis=dict(gridcolor="#ecf0f1", zeroline=False, title=None),
            yaxis=dict(autorange="reversed", title=None,
                       gridcolor="#ffffff", tickfont=dict(size=10)),
            showlegend=False,
        )
        return fig

    with bar1:
        st.plotly_chart(state_bar(specs_a, state_a, "#c0392b"),
                        use_container_width=True)
    with bar2:
        st.plotly_chart(state_bar(specs_b, state_b, "#2471a3"),
                        use_container_width=True)

# ============================================================
# TAB 4: PATTERNS (§3 HOW)
# ============================================================
with tab_how:
    st.markdown(
        '<div class="section-finding">§3 &nbsp;·&nbsp; '
        "How Does Opioid Prescribing Differ Mathematically?</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-context">'
        "Correlation matrices computed on a 500K-row sample reveal that opioid "
        "prescribing follows a fundamentally different mathematical structure "
        "than non-opioid prescribing. Side-by-side heatmaps make the difference "
        "visible at a glance."
        "</div>",
        unsafe_allow_html=True,
    )

    HL, HR = st.columns(2, gap="medium")

    feature_labels = {
        "Tot_Clms":        "Claims",
        "Tot_30day_Fills": "30-Day Fills",
        "Tot_Day_Suply":   "Days Supply",
        "Tot_Drug_Cst":    "Drug Cost",
        "Tot_Benes":       "Patients",
    }

    def render_heatmap(corr_df, title, container):
        with container:
            st.markdown(
                f'<div style="font-weight:600; color:#1f4e79; font-size:1rem; '
                f'margin-bottom:0.4rem;">{title}</div>',
                unsafe_allow_html=True,
            )
            short_labels = [feature_labels.get(c, c) for c in corr_df.columns]
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=short_labels,
                y=short_labels,
                colorscale=[
                    [0.0, "#1f4e79"], [0.5, "#f5f5f5"], [1.0, "#922b21"],
                ],
                zmin=-0.1, zmax=1.0,
                text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
                texttemplate="%{text}",
                textfont=dict(size=12, color="#2c3e50"),
                hovertemplate=(
                    "<b>%{y} ↔ %{x}</b><br>"
                    "Correlation: %{z:.3f}<extra></extra>"
                ),
                colorbar=dict(
                    title=dict(text="r", font=dict(size=10)),
                    tickfont=dict(size=9),
                    len=0.85, thickness=12, x=1.02,
                ),
            ))
            fig.update_layout(
                height=440, margin=dict(l=10, r=40, t=10, b=10),
                plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                font=dict(family="Inter, Segoe UI, sans-serif", size=11,
                          color="#2c3e50"),
                xaxis=dict(side="bottom", tickangle=0),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    render_heatmap(corrs["opioid"],
                   "Opioid Records Only (n ≈ 9.5K)", HL)
    render_heatmap(corrs["non"],
                   "Non-Opioid Records (n ≈ 215K)", HR)

    # Difference summary metrics
    st.markdown(
        '<div style="font-weight:600; color:#1f4e79; font-size:1rem; '
        'margin: 0.6rem 0 0.4rem 0;">Where the patterns diverge most</div>',
        unsafe_allow_html=True,
    )

    diff_pairs = [
        ("Tot_Day_Suply",  "Tot_Benes"),
        ("Tot_Day_Suply",  "Tot_Clms"),
        ("Tot_Drug_Cst",   "Tot_Benes"),
        ("Tot_Drug_Cst",   "Tot_Day_Suply"),
    ]
    d1, d2, d3, d4 = st.columns(4)
    for col_obj, (a, b) in zip([d1, d2, d3, d4], diff_pairs):
        with col_obj:
            opi = corrs["opioid"].loc[a, b]
            non = corrs["non"].loc[a, b]
            label_a = feature_labels[a]
            label_b = feature_labels[b]
            st.metric(
                f"{label_a} ↔ {label_b}",
                f"{opi:.2f} vs {non:.2f}",
                f"Δ = {opi - non:+.2f}",
            )

    # Drug deep-dive bar chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-weight:600; color:#1f4e79; font-size:1rem; '
        'margin-bottom:0.4rem;">Top 10 Opioid Drugs by Record Count</div>',
        unsafe_allow_html=True,
    )

    top_drugs = (df_opi["Gnrc_Name"]
                 .value_counts()
                 .head(10)
                 .sort_values(ascending=True))

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        x=top_drugs.values, y=top_drugs.index, orientation="h",
        marker=dict(
            color=top_drugs.values,
            colorscale=[[0.0, "#fadbd8"], [0.5, "#e6796d"], [1.0, "#922b21"]],
            line=dict(color="#ffffff", width=1), showscale=False,
        ),
        text=[f"<b>{v:,}</b>" for v in top_drugs.values],
        textposition="outside",
        textfont=dict(size=11, color="#2c3e50"),
        hovertemplate=(
            "<b>%{y}</b><br>Records: %{x:,}<extra></extra>"
        ),
    ))
    fig5.update_layout(
        height=380, margin=dict(l=10, r=120, t=10, b=40),
        plot_bgcolor="#fafbfc", paper_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, sans-serif", size=11,
                  color="#2c3e50"),
        xaxis=dict(title="Number of records",
                   gridcolor="#ecf0f1", zeroline=False,
                   range=[0, max(top_drugs.values) * 1.18],
                   title_font=dict(size=12)),
        yaxis=dict(title=None, gridcolor="#ffffff", zeroline=False,
                   tickfont=dict(size=11)),
        showlegend=False,
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        """
        <div class="so-what" style="margin-top:0.6rem;">
          <span class="so-what-label">KEY INSIGHT:</span>
          For non-opioid prescribing, days-supply varies independently of patient
          count (r = 0.13) — doctors flexibly write 30-day, 60-day, 90-day refills
          based on clinical context. <b>For opioids, days-supply locks in step
          with patient count (r = 0.72)</b> because federal opioid prescribing
          guidelines constrain supply duration. This is the regulatory regime
          made visible in the data. Practical consequence:
          <b>Tot_Clms, Tot_30day_Fills, and Tot_Day_Suply are nearly perfectly
          correlated (≥0.97) for opioids</b> — including all three as features
          in a regression would cause severe multicollinearity. The classifier
          should keep one volume feature plus Tot_Benes, which is more independent.
        </div>
        """,
        unsafe_allow_html=True,
    )
# ============================================================
# TAB 5: DRUG MIX — Drug × Specialty Heatmap
# ============================================================
with tab_drugs:
    st.markdown(
        '<div class="section-finding">§4 &nbsp;·&nbsp; '
        "Which Specialty Prescribes Which Opioid?</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-context">'
        "Top 10 opioid drugs × Top 15 specialties by record count. "
        "Surgical specialties concentrate on short-acting drugs; "
        "primary care dominates long-term maintenance opioids."
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- Slider to control how many drugs/specialties to show ----
    dc1, dc2 = st.columns(2)
    with dc1:
        n_drugs = st.slider("Number of drugs to show", 5, 15, 10)
    with dc2:
        n_specs = st.slider("Number of specialties to show", 5, 20, 15)

    top_drugs_list = (df_opi["Gnrc_Name"]
                      .value_counts()
                      .head(n_drugs)
                      .index.tolist())
    top_specs_list = (df_opi["Prscrbr_Type"]
                      .value_counts()
                      .head(n_specs)
                      .index.tolist())

    heatmap_df = (
        df_opi[
            df_opi["Gnrc_Name"].isin(top_drugs_list) &
            df_opi["Prscrbr_Type"].isin(top_specs_list)
        ]
        .groupby(["Prscrbr_Type", "Gnrc_Name"], observed=True)
        .size()
        .reset_index(name="count")
        .pivot(index="Prscrbr_Type", columns="Gnrc_Name", values="count")
        .fillna(0)
    )

    heatmap_df = heatmap_df.loc[
        heatmap_df.sum(axis=1).sort_values(ascending=False).index
    ]
    heatmap_df = heatmap_df[
        heatmap_df.sum(axis=0).sort_values(ascending=False).index
    ]

    def shorten(name, n=22):
        return name if len(name) <= n else name[:n-1] + "…"

    short_cols = [shorten(c) for c in heatmap_df.columns]
    short_rows = [shorten(r) for r in heatmap_df.index]

    fig_hm = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=short_cols,
        y=short_rows,
        colorscale=[
            [0.0,  "#f5f5f5"],
            [0.15, "#fadbd8"],
            [0.4,  "#e6796d"],
            [0.7,  "#c0392b"],
            [1.0,  "#7b241c"],
        ],
        text=[[f"{int(v):,}" for v in row] for row in heatmap_df.values],
        texttemplate="%{text}",
        textfont=dict(size=10, color="#2c3e50"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Drug: <b>%{x}</b><br>"
            "Records: <b>%{z:,}</b><extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Records", font=dict(size=10)),
            tickfont=dict(size=9),
            len=0.85, thickness=12, x=1.01,
        ),
    ))

    fig_hm.update_layout(
        height=520,
        margin=dict(l=10, r=60, t=10, b=120),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, sans-serif",
                  size=11, color="#2c3e50"),
        xaxis=dict(tickangle=-35, tickfont=dict(size=10),
                   title=None, gridcolor="#ffffff"),
        yaxis=dict(tickfont=dict(size=10), title=None,
                   autorange="reversed", gridcolor="#ffffff"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown(
        """
        <div class="so-what">
          <span class="so-what-label">KEY INSIGHT:</span>
          Family Practice and Nurse Practitioners dominate
          <b>Hydrocodone/Acetaminophen</b> and <b>Tramadol</b> —
          high-volume maintenance opioids. Surgical specialties
          concentrate almost exclusively on <b>Oxycodone</b> combinations
          for short-term post-operative pain.
          <b>These are fundamentally different prescribing regimes
          hiding under the same "opioid" label.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
# ============================================================
# TAB 5: EXPLORE — Filterable Drill-Down
# ============================================================
with tab_explore:
    st.markdown(
        '<div class="section-finding">🔍 &nbsp;·&nbsp; '
        "Explore the Data</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-context">'
        "Filter the 1M-row sample by state, specialty, and opioid status. "
        "The aggregated view shows top prescribers within your selection. "
        "Download the filtered records as CSV using the button below the table."
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- Filter row ----
    f1, f2, f3 = st.columns([1, 1.2, 1])

    with f1:
        all_states = sorted(df["Prscrbr_State_Abrvtn"].dropna().unique().tolist())
        sel_states = st.multiselect(
            "State",
            options=all_states,
            default=[],
            placeholder="All states",
        )

    with f2:
        all_specs = sorted(df["Prscrbr_Type"].dropna().unique().tolist())
        sel_specs = st.multiselect(
            "Specialty",
            options=all_specs,
            default=[],
            placeholder="All specialties",
        )

    with f3:
        view_mode = st.radio(
            "View",
            options=["All records", "Opioid only", "Non-opioid only"],
            horizontal=False,
        )

    # ---- Apply filters ----
    filtered = df.copy()
    if sel_states:
        filtered = filtered[filtered["Prscrbr_State_Abrvtn"].isin(sel_states)]
    if sel_specs:
        filtered = filtered[filtered["Prscrbr_Type"].isin(sel_specs)]
    if view_mode == "Opioid only":
        filtered = filtered[filtered["is_opioid"] == 1]
    elif view_mode == "Non-opioid only":
        filtered = filtered[filtered["is_opioid"] == 0]

    # ---- Summary metrics for the filtered set ----
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("Records in selection", f"{len(filtered):,}")
    with e2:
        opi_n = int(filtered["is_opioid"].sum())
        st.metric("Opioid records", f"{opi_n:,}")
    with e3:
        rate = (filtered["is_opioid"].mean() * 100) if len(filtered) else 0
        st.metric("Opioid rate", f"{rate:.2f}%")
    with e4:
        npi_n = filtered["Prscrbr_NPI"].nunique() if len(filtered) else 0
        st.metric("Unique prescribers", f"{npi_n:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Top prescribers in selection (grouped by NPI) ----
    if len(filtered) == 0:
        st.warning("No records match your filters. Try widening your selection.")
    else:
        st.markdown(
            '<div style="font-weight:600; color:#1f4e79; font-size:1rem; '
            'margin-bottom:0.4rem;">Top 50 prescribers in current selection</div>',
            unsafe_allow_html=True,
        )

        agg = (filtered
               .groupby(["Prscrbr_NPI", "Prscrbr_Type",
                         "Prscrbr_State_Abrvtn"], observed=True)
               .agg(total_claims=("Tot_Clms", "sum"),
                    opioid_records=("is_opioid", "sum"),
                    record_count=("is_opioid", "size"),
                    total_drug_cost=("Tot_Drug_Cst", "sum"))
               .reset_index())
        agg["opioid_rate_pct"] = (agg["opioid_records"]
                                  / agg["record_count"] * 100).round(2)
        agg = agg.sort_values("total_claims", ascending=False).head(50)

        display = agg.rename(columns={
            "Prscrbr_NPI":          "NPI",
            "Prscrbr_Type":         "Specialty",
            "Prscrbr_State_Abrvtn": "State",
            "total_claims":         "Total Claims",
            "opioid_records":       "Opioid Records",
            "record_count":         "Drug Records",
            "opioid_rate_pct":      "Opioid Rate %",
            "total_drug_cost":      "Total Drug Cost ($)",
        })
        display["Total Claims"] = display["Total Claims"].map(lambda x: f"{int(x):,}")
        display["Opioid Records"] = display["Opioid Records"].map(lambda x: f"{int(x):,}")
        display["Drug Records"] = display["Drug Records"].map(lambda x: f"{int(x):,}")
        display["Opioid Rate %"] = display["Opioid Rate %"].map(lambda x: f"{x:.2f}%")
        display["Total Drug Cost ($)"] = display["Total Drug Cost ($)"].map(
            lambda x: f"${x:,.0f}")

        st.dataframe(display, use_container_width=True, hide_index=True,
                     height=480)

        # ---- Download button ----
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {len(filtered):,} filtered records as CSV",
            data=csv_bytes,
            file_name="medicare_partd_filtered.csv",
            mime="text/csv",
            use_container_width=False,
        )

# ============================================================
# 7. FOOTER
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "Medicare Part D Opioid Prescribing Dashboard · Built by Group 7 "
    "(Darpan Radadiya & Jingqi Huo) · Data: CMS Medicare Part D Prescribers (2023) · "
    "ALY 6110 Northeastern University"
)