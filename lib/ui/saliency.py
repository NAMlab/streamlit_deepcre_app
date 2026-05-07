import itertools
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ── Styling helpers ───────────────────────────────────────────────────────────

def _section(label: str) -> None:
    st.markdown(
        f"""<div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.07em;color:#4F1787;margin:1.4rem 0 0.6rem 0;
                        padding-bottom:4px;border-bottom:1px solid #ede9f5;">
                {label}
            </div>""",
        unsafe_allow_html=True,
    )


def _chart_title(title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        title,
        subtitle=[subtitle, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        subtitleColor="#9ca3af",
        fontSize=13,
        subtitleFontSize=10,
    )


# ── Genomic region span helpers ───────────────────────────────────────────────

_REGION_SPANS = [
    (0,    999,  "grey",          "gUR (gene Upstream Region)"),
    (1000, 1499, "red",           "gTUR (gene 5′ UTR Region)"),
    (1519, 2019, "cornflowerblue","gTDR (gene 3′ UTR Region)"),
    (2020, 3020, "grey",          "gDR (gene Downstream Region)"),
]

def _region_layers() -> list[alt.Chart]:
    layers = []
    for x1, x2, color, tip in _REGION_SPANS:
        layers.append(
            alt.Chart(pd.DataFrame({"x1": [x1], "x2": [x2]}))
            .mark_rect(opacity=0.1)
            .encode(
                x=alt.X("x1", scale=alt.Scale(domain=[1, 3021]), title="Nucleotide Position"),
                x2="x2",
                color=alt.value(color),
                tooltip=alt.value(tip),
            )
        )
    return layers


def _annotation_layer(text_y: float) -> alt.Chart:
    df = pd.DataFrame([
        (1000, text_y, "TSS", "Transcription Start Site"),
        (2020, text_y, "TTS", "Transcription Termination Site"),
    ], columns=["Nucleotide Position", "Saliency Score", "marker", "description"])
    return (
        alt.Chart(df)
        .mark_text(size=14, dx=-10, dy=0, align="center", fontWeight="bold")
        .encode(
            x=alt.X("Nucleotide Position", scale=alt.Scale(domain=[1, 3021])),
            y=alt.Y("Saliency Score:Q", title="Saliency Score"),
            text="marker",
            tooltip="description",
        )
    )


def _opacity_param(name: str = "opacity"):
    return alt.param(
        value=1,
        bind=alt.binding_range(min=0.2, max=1, step=0.05, name=f"{name}:"),
    )


# ── CSV export ────────────────────────────────────────────────────────────────

@st.cache_data
def _pivot_csv(df: pd.DataFrame, column_var: str) -> bytes:
    return (
        df.pivot(index="Nucleotide Position", columns=column_var, values="Saliency Score")
        .to_csv(index=True)
        .encode("utf-8")
    )


# ── Chart builders ────────────────────────────────────────────────────────────

def _build_line_chart(df: pd.DataFrame, title: str, subtitle: str,
                      color_field: str, color_scale: alt.Scale,
                      opacity_param, y_domain=None) -> alt.Chart:
    y_enc = (
        alt.Y("Saliency Score:Q", title="Saliency Score", scale=alt.Scale(domain=y_domain))
        if y_domain else alt.Y("Saliency Score:Q", title="Saliency Score")
    )
    return (
        alt.Chart(df, title=_chart_title(title, subtitle))
        .mark_line(opacity=opacity_param)
        .encode(
            x=alt.X("Nucleotide Position", scale=alt.Scale(domain=[1, 3021]),
                    axis=alt.Axis(tickCount=10)),
            y=y_enc,
            color=alt.Color(f"{color_field}:N", scale=color_scale),
        )
        .add_params(opacity_param)
    )


def _build_scatter(df: pd.DataFrame, title: str, subtitle: str,
                   color_field: str, color_scale: alt.Scale) -> alt.Chart:
    return (
        alt.Chart(df, title=_chart_title(title, subtitle))
        .mark_circle(size=30, opacity=0.7)
        .encode(
            x=alt.X("Probability of high expression:Q", title="Probability of high expression", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Sum Saliency Score:Q", title="Sum Saliency Score"),
            color=alt.Color(f"{color_field}:N", scale=color_scale),
            tooltip=df.columns.tolist(),
        )
    )


def _composite(layers: list, line_chart: alt.Chart, text_y: float) -> alt.LayerChart:
    rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[3, 3], color="black").encode(y=alt.Y("y:Q", title="Saliency Score"))
    return alt.layer(*layers, line_chart, _annotation_layer(text_y), rule)


# ── Public entry point ────────────────────────────────────────────────────────

def show_saliency_tab(
    actual_scores_high, actual_scores_low,
    p_h, p_l, color_palette_low_high, g_h, g_l,
):
    # Guard: fill missing class with zeros
    if actual_scores_low.shape[0] == 0:
        actual_scores_low = np.zeros_like(actual_scores_high)
        p_l, g_l = [np.nan], [np.nan]
    if actual_scores_high.shape[0] == 0:
        actual_scores_high = np.zeros_like(actual_scores_low)
        p_h, g_h = [np.nan], [np.nan]

    expr_scale = alt.Scale(range=color_palette_low_high, domain=["High", "Low"])
    base_scale = alt.Scale(range=["green", "cornflowerblue", "darkorange", "red"], domain=["A", "C", "G", "T"])
    positions  = np.arange(1, 3021)

    # ── 1. Average saliency map ───────────────────────────────────────────────
    _section("📈 Average Saliency Map")
    avg_line_col, avg_scat_col = st.columns([0.6, 0.4], vertical_alignment="top", gap="medium")

    avg_df = pd.DataFrame({
        "Nucleotide Position": np.tile(positions, 2),
        "Saliency Score": np.concatenate([
            actual_scores_high.mean(axis=(0, 2)),
            actual_scores_low.mean(axis=(0, 2)),
        ]),
        "Expression": ["High"] * 3020 + ["Low"] * 3020,
    })

    text_y = avg_df["Saliency Score"].max() + avg_df["Saliency Score"].mean()
    op     = _opacity_param("avg_opacity")

    with avg_line_col:
        line = _build_line_chart(
            avg_df,
            title="Average Saliency Map",
            subtitle="Saliency scores averaged across all genes predicted as high / low expressed",
            color_field="Expression",
            color_scale=expr_scale,
            opacity_param=op,
        )
        chart = _composite(_region_layers(), line, text_y)
        st.altair_chart(chart, use_container_width=True, theme=None)
        st.download_button(
            "⬇️ Download average saliency CSV",
            data=_pivot_csv(avg_df, "Expression"),
            file_name=f"avg_saliency_{datetime.now().strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
        )

    with avg_scat_col:
        rows = {"Expression": [], "Sum Saliency Score": [], "P(high expr.)": [], "Gene ID": []}
        for scores, probs, gids, label in [
            (actual_scores_high, p_h, g_h, "High"),
            (actual_scores_low,  p_l, g_l, "Low"),
        ]:
            if not np.isnan(probs[0]):
                for i in range(scores.shape[0]):
                    rows["Expression"].append(label)
                    rows["Sum Saliency Score"].append(scores[i].sum())
                    rows["P(high expr.)"].append(probs[i])
                    rows["Gene ID"].append(gids[i])

        scat_df = pd.DataFrame(rows)
        scat_df = scat_df.rename(columns={"P(high expr.)": "Probability of high expression"})

        scatter = _build_scatter(
            scat_df,
            title="Saliency Score vs. Predicted Probability",
            subtitle="Sum saliency score per gene plotted against probability of high expression",
            color_field="Expression",
            color_scale=expr_scale,
        )
        st.altair_chart(scatter, use_container_width=True, theme=None)

    # ── 2. Per-nucleotide base-type saliency maps ─────────────────────────────
    _section("🔬 Base-Type Saliency Maps")

    df_by_class = {}
    for label, scores in [("High", actual_scores_high), ("Low", actual_scores_low)]:
        df_by_class[label] = pd.DataFrame({
            "Nucleotide Position": np.tile(positions, 4),
            "Saliency Score": np.concatenate([scores.mean(axis=0)[:, i] for i in range(4)]),
            "Base": list(itertools.chain(*[[b] * 3020 for b in ["A", "C", "G", "T"]])),
        })

    y_min = min(df["Saliency Score"].min() for df in df_by_class.values())
    y_max = max(df["Saliency Score"].max() for df in df_by_class.values())

    base_line_col, base_scat_col = st.columns([0.6, 0.4], vertical_alignment="top", gap="medium")

    with base_line_col:
        for label, df in df_by_class.items():
            op_base = _opacity_param(f"base_{label.lower()}_opacity")
            text_y_base = y_max + abs(y_max - y_min) * 0.05

            line = _build_line_chart(
                df,
                title=f"Base-Type Saliency Map — {label} Expressed Genes",
                subtitle=f"Saliency scores averaged per nucleotide for {label.lower()}-expressed genes",
                color_field="Base",
                color_scale=base_scale,
                opacity_param=op_base,
                y_domain=[y_min, y_max],
            )
            chart = _composite(_region_layers(), line, text_y_base)
            st.altair_chart(chart, use_container_width=True, theme=None)
            st.download_button(
                f"⬇️ Download {label} base-type saliency CSV",
                data=_pivot_csv(df, "Base"),
                file_name=f"base_saliency_{label.lower()}_{datetime.now().strftime('%Y-%m-%d')}.csv",
                mime="text/csv",
                key=f"dl_base_{label}",
            )

    with base_scat_col:
        for scores, probs, gids in [
            (actual_scores_high, p_h, g_h),
            (actual_scores_low,  p_l, g_l),
        ]:
            if np.isnan(probs[0]):
                continue
            n = len(probs)
            df = pd.DataFrame({
                "Base": list(itertools.chain(*[[b] * n for b in ["A", "C", "G", "T"]])),
                "Probability of high expression": list(itertools.chain(*[probs] * 4)),
                "Sum Saliency Score": np.concatenate([scores.sum(axis=1)[:, i] for i in range(4)]),
                "Gene ID": list(itertools.chain(*[gids] * 4)),
            })
            scatter = _build_scatter(
                df,
                title="Base-Type Saliency vs. Probability",
                subtitle="Sum saliency score per base vs. probability of high expression",
                color_field="Base",
                color_scale=base_scale,
            )
            st.altair_chart(scatter, use_container_width=True, theme=None)
