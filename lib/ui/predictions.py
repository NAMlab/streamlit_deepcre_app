import streamlit as st
import pandas as pd
from datetime import datetime
import altair as alt

# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(label: str) -> None:
    st.markdown(
        f"""<div style="font-size:1.25rem;font-weight:700;text-transform:uppercase;
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


def _color_enc(palette: list) -> alt.Color:
    """Always return a fresh Color encoding — sharing one instance across charts causes silent failures."""
    return alt.Color(
        "Expression:N",
        scale=alt.Scale(range=palette, domain=["High", "Low"]),
        legend=alt.Legend(title="Expression class", orient="top"),
    )


@st.cache_data
def _to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ── Public entry point ────────────────────────────────────────────────────────

def show_predictions_tab(
    gene_ids, gene_chroms, gene_starts, gene_ends,
    gene_size, gene_gc_cont, preds, color_palette_low_high,
):
    # Use simple Altair-safe column names (no dots, parens, or special chars)
    predictions = pd.DataFrame({
        "Gene ID":        gene_ids,
        "Chromosome":     [f"Chr {c}" for c in gene_chroms],
        "Gene Start":     gene_starts,
        "Gene End":       gene_ends,
        "Gene Size":      gene_size,
        "GC Content":     gene_gc_cont,
        "Probability":    list(preds),
        "Expression":     ["High" if p > 0.5 else "Low" for p in preds],
    })

    # ── Table + per-chromosome gene count ────────────────────────────────────
    _section("📋 Gene Predictions Table")
    tbl_col, chr_col = st.columns([0.65, 0.35], vertical_alignment="top")

    with tbl_col:
        st.dataframe(
            predictions,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Probability": st.column_config.ProgressColumn(
                    "P(high expression)", min_value=0, max_value=1, format="%.3f"
                )
            },
        )
        st.download_button(
            "⬇️ Download table as CSV",
            data=_to_csv(predictions),
            file_name=f"deepcre_predictions_{datetime.now().strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
        )

    with chr_col:
        chart = (
            alt.Chart(predictions, title=_chart_title(
                "Genes per Chromosome", "Number of genes per chromosome"
            ))
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Chromosome:N", axis=alt.Axis(labelAngle=-35, title=None)),
                y=alt.Y("count()", title="Gene count"),
                color=alt.value("#4F1787"),
                tooltip=["Chromosome:N", "count()"],
            )
        )
        st.altair_chart(chart, use_container_width=True, theme=None)

    # ── Expression class by chromosome ───────────────────────────────────────
    _section("📊 Expression Class Distribution")
    dist_col, hist_col = st.columns([0.65, 0.35], vertical_alignment="top", gap="medium")

    with dist_col:
        data_agg = (
            predictions
            .groupby(["Chromosome", "Expression"])
            .size()
            .reset_index(name="Count")
        )
        chart_by_chrom = (
            alt.Chart(data_agg, title=_chart_title(
                "High / Low Predictions per Chromosome",
                "Gene count per chromosome split by predicted expression class",
            ))
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Chromosome:N", axis=alt.Axis(labelAngle=0, title=None)),
                xOffset="Expression:N",
                y=alt.Y("Count:Q", title="Gene count"),
                color=_color_enc(color_palette_low_high),
                tooltip=["Chromosome:N", "Expression:N", "Count:Q"],
            )
        )
        st.altair_chart(chart_by_chrom, use_container_width=True, theme=None)

    with hist_col:
        chart_prob_hist = (
            alt.Chart(predictions, title=_chart_title(
                "Probability Distribution",
                "Histogram of predicted probabilities of high expression",
            ))
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X(
                    "Probability:Q",
                    bin=alt.Bin(extent=[0, 1], step=0.05),
                    title="P(high expression)",
                ),
                y=alt.Y("count()", stack=None, title="Gene count"),
                color=_color_enc(color_palette_low_high),
                tooltip=["Expression:N", "count()"],
            )
        )
        st.altair_chart(chart_prob_hist, use_container_width=True, theme=None)

    # ── Gene size & GC content scatter plots ─────────────────────────────────
    _section("🔬 Gene Properties vs. Predicted Expression")
    size_col, gc_col = st.columns(2, gap="medium")

    with size_col:
        chart_size = (
            alt.Chart(predictions, title=_chart_title(
                "Gene Size vs. Expression Probability",
                "Relationship between gene length and predicted probability of high expression",
            ))
            .mark_circle(size=30, opacity=0.7)
            .encode(
                x=alt.X("Probability:Q", title="P(high expression)", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Gene Size:Q", title="Gene size (bp)"),
                color=_color_enc(color_palette_low_high),
                tooltip=["Gene ID:N", "Expression:N", "Probability:Q", "Gene Size:Q"],
            )
        )
        st.altair_chart(chart_size, use_container_width=True, theme=None)

    with gc_col:
        chart_gc = (
            alt.Chart(predictions, title=_chart_title(
                "GC Content vs. Expression Probability",
                "Relationship between GC content and predicted probability of high expression",
            ))
            .mark_circle(size=30, opacity=0.7)
            .encode(
                x=alt.X("Probability:Q", title="P(high expression)", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("GC Content:Q", title="GC content"),
                color=_color_enc(color_palette_low_high),
                tooltip=["Gene ID:N", "Expression:N", "Probability:Q", "GC Content:Q"],
            )
        )
        st.altair_chart(chart_gc, use_container_width=True, theme=None)
