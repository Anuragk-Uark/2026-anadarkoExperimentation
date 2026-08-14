"""
Streamlit viewer for the Anadarko master petrophysical database.

Reads master_log_database.parquet, lets the user pick a well (UWI) in the
sidebar, and renders a Plotly triple-combo plot (GR / ILD / RHOB) with
formation tops drawn as labeled dashed lines across all tracks.

Run with:  streamlit run well_viewer_app.py
"""

from csv import reader
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PARQUET_PATH = Path(__file__).resolve().parent / "master_log_database.parquet"
st.set_page_config(page_title="Anadarko Well Log Viewer", layout="wide")

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(PARQUET_PATH)


df = load_data()

# tops columns in the database -> display names
TOPS = {
    "CHSE_MD": "Herington",
    "KRDR_MD": "Krider",
    "WNFD_MD": "Winfield",
    "TWND_MD": "Towanda",
    "FRLY_MD": "Fort Riley",
    "FLOR_MD": "Florence",
    "WRFD_MD": "Wreford",
    "CCGV_MD": "Council Grove",
}





# ---------------- sidebar ----------------
st.sidebar.title("Well Selection")

wells = df[["UWI_API", "WELL_NAME", "LABEL"]].drop_duplicates("UWI_API").sort_values("WELL_NAME")
label_by_uwi = {
    row.UWI_API: f"{row.WELL_NAME} {row.LABEL} ({row.UWI_API})" for row in wells.itertuples()
}

uwi = st.sidebar.selectbox(
    "Well (UWI)",
    options=wells["UWI_API"].tolist(),
    format_func=lambda u: label_by_uwi.get(u, u),
)

well_df = df[df["UWI_API"] == uwi].sort_values("DEPT")
header = well_df.iloc[0]

st.sidebar.markdown(
    f"**Well Name:** {header['WELL_NAME']}  \n"
    f"**Operator:** {header['OPERATOR']}  \n"
    f"**TD:** {header['TD']:,.0f} ft  \n"
    f"**Completion Date:** {header['COMP_DATE']}  \n"
    f"**Log interval:** {well_df['DEPT'].min():,.0f}–{well_df['DEPT'].max():,.0f} ft  \n"
    
)

# tops present for this well
well_tops = {
    name: float(header[col]) for col, name in TOPS.items() if pd.notna(header.get(col))
}

# ---------------- triple combo plot ----------------
st.title(f"{header['WELL_NAME']} {header['LABEL']}")
st.caption("Anurag Kulkarni Summer 2026 Experiment - Anadarko Project Well Visualizer")
st.caption("Generated in Python using Claude Code, slightly modified to taste")
st.caption(f"UWI/API {uwi} — Visualization of Gamma Ray Log, Bulk Density, Deep Resistivity, and Photoelectric Log")

# two tracks: GR on the left, ILD + RHOB + PE overlaid on the right
fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    horizontal_spacing=0.02,
    column_widths=[0.35, 0.65],
)

# GR value-banded shading: fill from the left edge (0) to the curve,
# colored by where the GR reading falls
GR_BANDS = [
    (90, 150, "#1a1a1a"),   # dark gray nearing black
    (75, 90, "#4d4d4d"),    # dark grey
    (60, 75, "#969696"),    # grey
    (45, 60, "#a2b4ba"),
    (0, 45, "lightblue"),
]
gr = well_df["GR"].to_numpy()
for lo, hi, color in GR_BANDS:
    in_band = (gr >= lo) & (gr <= hi)
    # include neighbors (clipped to the band) so adjacent bands meet cleanly;
    # outside the band sit on the 150 baseline (zero-width fill) — NaN gaps
    # would make Plotly close the fill polygon across them
    near = in_band | np.roll(in_band, 1) | np.roll(in_band, -1)
    band_x = np.where(near, np.clip(gr, lo, hi), 150.0)
    band_x = np.where(np.isnan(band_x), 150.0, band_x)
    # invisible baseline at 150, then fill from the curve back to it
    fig.add_trace(
        go.Scatter(x=np.full(len(gr), 150.0), y=well_df["DEPT"], mode="lines",
                   line=dict(width=0), hoverinfo="skip", showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=band_x, y=well_df["DEPT"], mode="lines",
                   line=dict(width=0), fill="tonextx", fillcolor=color,
                   hoverinfo="skip", showlegend=False),
        row=1, col=1,
    )

fig.add_trace(
    go.Scatter(x=well_df["GR"], y=well_df["DEPT"], name="GR",
               line=dict(color="green", width=1)),
    row=1, col=1,
)
# crossover shading: yellow where RHOB plots LEFT of ILD in the track.
# The curves use different x-scales, so compare fractional track positions
# (0 = left edge, 1 = right edge of track 2) on a hidden 0-1 axis (x5).
frac_ild = (np.log10(well_df["ILD"]) - np.log10(0.2)) / (np.log10(2000) - np.log10(0.2))
frac_rhob = (well_df["RHOB"] - 2.175) / (3.175 - 2.175)
shade_edge = np.where(frac_rhob < frac_ild, frac_rhob, frac_ild)
fig.add_trace(
    go.Scatter(x=frac_ild, y=well_df["DEPT"], mode="lines",
               line=dict(width=0), hoverinfo="skip", showlegend=False,
               xaxis="x5", yaxis="y2"),
)
fig.add_trace(
    go.Scatter(x=shade_edge, y=well_df["DEPT"], mode="lines",
               line=dict(width=0), fill="tonextx",
               fillcolor="rgba(255, 255, 0, 0.5)",
               hoverinfo="skip", showlegend=False,
               xaxis="x5", yaxis="y2"),
)

fig.add_trace(
    go.Scatter(x=well_df["ILD"], y=well_df["DEPT"], name="ILD",
               line=dict(color="blue", width=2, dash="dash")),
    row=1, col=2,
)
# RHOB and PE share track 2 on their own hidden linear x-axes
fig.add_trace(
    go.Scatter(x=well_df["RHOB"], y=well_df["DEPT"], name="RHOB",
               line=dict(color="turquoise", width=1.5),
               xaxis="x3", yaxis="y2"),
)
fig.add_trace(
    go.Scatter(x=well_df["PE"], y=well_df["DEPT"], name="PE",
               line=dict(color="magenta", width=1),
               xaxis="x4", yaxis="y2"),
)

# track 1: GR 0-150 linear; track 2 grid: ILD 0.2-2000 logarithmic
fig.update_xaxes(range=[0, 150], dtick=50,
                 minor=dict(dtick=10, showgrid=True, gridcolor="#eeeeee"),
                 row=1, col=1)
fig.update_xaxes(type="log", range=[-0.699, 3.301], dtick=1,  # log10(0.2)..log10(2000)
                 minor=dict(dtick="D1", showgrid=True, gridcolor="#eeeeee"),
                 row=1, col=2)
for col in (1, 2):
    fig.update_xaxes(side="top", showgrid=True, gridcolor="lightgray",
                     mirror=True, showline=True, linecolor="black", row=1, col=col)

# overlay scales on track 2 (no grid/labels of their own; the log grid rules)
fig.update_layout(
    xaxis3=dict(overlaying="x2", side="top", range=[2.175, 3.175],
                showgrid=False, showticklabels=False),
    xaxis4=dict(overlaying="x2", side="top", range=[1, 6],
                showgrid=False, showticklabels=False),
    xaxis5=dict(overlaying="x2", side="top", range=[0, 1],
                showgrid=False, showticklabels=False),
)

# color-coded track headers with each curve's scale
for xref, yshift, text, color in [
    ("x domain", 28, "GR (gAPI) 0–150", "green"),
    ("x domain", 40, "0 - 45 API", "lightblue"),
    ("x domain", 52, "45 - 60 API", "#a2b4ba"),
    ("x domain", 63, "60 - 75 API", "#969696"),
    ("x domain", 74, "75 - 90 API", "#4d4d4d"),
    ("x domain", 85, "90 - 150 API", "#1a1a1a"),
    ("x2 domain", 28, "ILD (ohm·m) 0.2–2000", "blue"),
    ("x2 domain", 46, "RHOB (g/cc) 2.175–3.175", "turquoise"),
    ("x2 domain", 64, "PE (b/e) 1–6", "magenta"),
]:
    fig.add_annotation(
        xref=xref, x=0.5, yref="paper", y=1.0, yshift=yshift,
        yanchor="bottom", text=f"<b>{text}</b>", showarrow=False,
        font=dict(size=13, color=color),
    )

# depth column runs from 100 ft above the Herington top to 200 ft below the
# deepest available top (full log interval if the well has no tops)
top_depth = well_tops["Herington"] - 100 if "Herington" in well_tops else well_df["DEPT"].min()
bottom_depth = max(well_tops.values()) + 200 if well_tops else well_df["DEPT"].max()
fig.update_yaxes(
    range=[bottom_depth, top_depth],  # reversed: deep at bottom
    dtick=50, showgrid=True, gridcolor="lightgray",
    minor=dict(dtick=10, showgrid=True, gridcolor="#eeeeee"),
    mirror=True, showline=True, linecolor="black",
)
fig.update_yaxes(title_text="Depth (ft MD)", row=1, col=1)

# tops as dashed lines across every track, labeled on the left track
for name, depth in sorted(well_tops.items(), key=lambda kv: kv[1]):
    fig.add_hline(y=depth, line_dash="solid", line_color="black",
                  line_width=1.5, row=1, col="all")
    fig.add_annotation(
        x=3, y=depth, xref="x1", yref="y1",
        text=f"<b>{name}</b>", showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(size=11, color="black"),
    )

# fixed vertical scale: 20 ft of depth per inch of screen (96 px/in)
FT_PER_INCH = 20
PX_PER_INCH = 96
MARGIN_T, MARGIN_B = 110, 20
plot_px = (bottom_depth - top_depth) / FT_PER_INCH * PX_PER_INCH

fig.update_layout(
    height=int(plot_px + MARGIN_T + MARGIN_B),
    showlegend=False,
    plot_bgcolor="white",
    margin=dict(t=MARGIN_T, b=MARGIN_B),
)

st.plotly_chart(fig, use_container_width=True)

if not well_tops:
    st.info("No formation tops recorded for this well.")
