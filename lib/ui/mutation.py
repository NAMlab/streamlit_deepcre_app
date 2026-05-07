import itertools
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ── Shared style helpers ──────────────────────────────────────────────────────

def _section(label: str) -> None:
    st.markdown(
        f"""<div style="font-size:1.25rem;font-weight:700;letter-spacing:0.04em;
                        color:#4F1787;margin:1.2rem 0 0.5rem 0;
                        padding-bottom:4px;border-bottom:1px solid #ede9f5;">
                {label}
            </div>""",
        unsafe_allow_html=True,
    )


def _chart_title(title: str, subtitle: str = "") -> alt.TitleParams:
    parts = [subtitle] if subtitle else []
    parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return alt.TitleParams(title, subtitle=parts, subtitleColor="#9ca3af",
                           fontSize=13, subtitleFontSize=10)


# ── Genomic region spans (reusable) ──────────────────────────────────────────

def _region_span(x1: int, x2: int, color: str, tip: str) -> alt.Chart:
    return (
        alt.Chart(pd.DataFrame({"x1": [x1], "x2": [x2]}))
        .mark_rect(opacity=0.1)
        .encode(
            x=alt.X("x1", scale=alt.Scale(domain=[1, 3021]), title="Nucleotide Position"),
            x2="x2",
            color=alt.value(color),
            tooltip=alt.value(tip),
        )
    )


def _genomic_spans(utr_len: int, central_pad_size: int) -> list[alt.Chart]:
    return [
        _region_span(0,                                    999,             "grey",          "gUR (gene upstream region)"),
        _region_span(1000,                                 1000 + utr_len,  "red",           "gTUR (transcribed upstream portion)"),
        _region_span(1000 + utr_len + central_pad_size,    2019,            "cornflowerblue","gTDR (transcribed downstream portion)"),
        _region_span(2020,                                 3020,            "grey",          "gDR (gene downstream region)"),
    ]


def _annotation_layer(text_y: float) -> alt.Chart:
    df = pd.DataFrame([
        (1000, text_y, "TSS", "Transcription Start Site"),
        (2020, text_y, "TTS", "Transcription Termination Site"),
    ], columns=["Nucleotide Position", "Saliency Score", "marker", "description"])
    return (
        alt.Chart(df)
        .mark_text(size=13, align="center", fontWeight="bold")
        .encode(
            x=alt.X("Nucleotide Position", scale=alt.Scale(domain=[1, 3021])),
            y=alt.Y("Saliency Score:Q"),
            text="marker",
            tooltip="description",
        )
    )


# ── Analysis type selector ────────────────────────────────────────────────────

def choose_analysis_type() -> str:
    _section("🔧 Analysis Mode")
    col, _ = st.columns([0.25, 0.75])
    with col:
        mode = st.radio(
            "Choose analysis type",
            options=["**manual**", "**VCF**"],
            captions=["Manually edit the CRE sequence", "Apply natural variants from a VCF file"],
            label_visibility="collapsed",
        )
    return mode


# ── Manual mutation UI ────────────────────────────────────────────────────────

def show_manual_mutation(gene_id: str, start: int, end: int,
                         seq: str, utr_len: int, central_pad_size: int):
    # Initialise session state for this gene
    if "current_gene" not in st.session_state or st.session_state.current_gene != gene_id:
        st.session_state.current_gene = gene_id
        st.session_state.mutated_seq  = seq

    _section("🗂️ Select Region to Mutate")
    sel_region = st.radio(
        "Region",
        options=["gUR", "gTUR", "gTDR", "gDR"],
        horizontal=True,
        label_visibility="collapsed",
    )

    region_coords = {
        "gUR":  (1,                                        1,    1000),
        "gDR":  (1001 + (2 * utr_len) + central_pad_size,
                 1001 + (2 * utr_len) + central_pad_size,
                 2000 + (2 * utr_len) + central_pad_size),
        "gTUR": (1001, 1001, 1000 + utr_len),
        "gTDR": (1001 + utr_len + central_pad_size,
                 1001 + utr_len + central_pad_size,
                 1000 + (2 * utr_len) + central_pad_size),
    }
    val, min_val, max_val = region_coords[sel_region]

    slider_col, seq_col = st.columns([0.4, 0.6])

    with slider_col:
        _section("📍 Coordinate Range")
        with st.form("mutation_form", clear_on_submit=False, border=False):
            slider_vals = st.slider(
                "Start / End coordinates",
                min_value=val, max_value=max_val,
                value=(min_val, max_val), step=1,
                label_visibility="collapsed",
            )
            st.form_submit_button("Apply", type="primary")

    mut_reg_start, mut_reg_end = slider_vals
    mut_reg_start -= 1
    if mut_reg_start == mut_reg_end:
        mut_reg_end += 1

    sub_seq = st.session_state.mutated_seq[mut_reg_start:mut_reg_end]
    pad = mut_reg_end - mut_reg_start - len(sub_seq)
    if pad > 0:
        sub_seq += "N" * pad

    if "sub_seq_to_mutate" not in st.session_state:
        st.session_state.sub_seq_to_mutate = sub_seq

    def apply_mutation():
        edited = st.session_state.sub_seq_to_mutate
        if len(edited) != mut_reg_end - mut_reg_start:
            edited += "N" * (mut_reg_end - mut_reg_start - len(edited))
        full = st.session_state.mutated_seq
        st.session_state.mutated_seq = full[:mut_reg_start] + edited + full[mut_reg_end:]

    with seq_col:
        _section(f"✏️ Edit {sel_region} ({mut_reg_start}–{mut_reg_end})")
        edited_seq = st.text_area(
            label=f"{sel_region} sequence",
            value=st.session_state.mutated_seq[mut_reg_start:mut_reg_end],
            max_chars=len(st.session_state.mutated_seq[mut_reg_start:mut_reg_end]),
            height=80,
            key="sub_seq_to_mutate",
            label_visibility="collapsed",
            on_change=apply_mutation,
        )
        if len(edited_seq) != mut_reg_end - mut_reg_start:
            edited_seq += "N" * (mut_reg_end - mut_reg_start - len(edited_seq))

        if st.button("Mutate", type="primary"):
            st.session_state.mutated_seq = st.session_state.mutated_seq[:mut_reg_start] + edited_seq + st.session_state.mutated_seq[mut_reg_end:]
            
    return mut_reg_start, mut_reg_end


# ── Mutation results ──────────────────────────────────────────────────────────

def show_mutation_results(
    gene_id: str,
    pred_probs,
    actual_scores,
    seq: str,
    utr_len: int,
    central_pad_size: int,
    mut_reg_start,
    mut_reg_end,
    mut_markers=None,
):
    positions = np.arange(1, 3021)
    labels    = [gene_id, f"{gene_id} (mutated)"]
    palette   = ["#6b7280", "#33BBC5"]   # grey = original, teal = mutated

    color_scale = alt.Scale(range=palette, domain=labels)
    color_enc   = alt.Color("Gene ID:N", scale=color_scale,
                            legend=alt.Legend(orient="top", direction="vertical",
                                              titleAnchor="middle", labelLimit=0))

    # ── Bar chart: predicted probabilities ───────────────────────────────────
    bar_df = pd.DataFrame({
        "Probability of high expression": pred_probs,
        "Gene ID": labels,
    })
    bar_chart = (
        alt.Chart(bar_df, title=_chart_title("Predicted Expression Probability"))
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Gene ID:N", axis=alt.Axis(labels=False, tickSize=0, title=None)),
            y=alt.Y("Probability of high expression:Q", scale=alt.Scale(domain=[0, 1])),
            color=color_enc,
            tooltip=["Gene ID:N", "Probability of high expression:Q"],
        )
    )

    # ── Saliency line chart ───────────────────────────────────────────────────
    sal_df = pd.DataFrame({
        "Saliency Score":     np.concatenate([actual_scores[0].mean(axis=1), actual_scores[1].mean(axis=1)]),
        "Gene ID":            [labels[0]] * 3020 + [labels[1]] * 3020,
        "Nucleotide Position": np.tile(positions, 2),
    })

    y_min, y_max = sal_df["Saliency Score"].min(), sal_df["Saliency Score"].max()
    text_y       = y_max + abs(y_max - y_min) * 0.08

    line_chart = (
        alt.Chart(sal_df, title=_chart_title(
            "Saliency Map: Original vs. Mutated",
            "Average saliency scores per nucleotide position",
        ))
        .mark_line()
        .encode(
            x=alt.X("Nucleotide Position", scale=alt.Scale(domain=[1, 3021]),
                    axis=alt.Axis(tickCount=10)),
            y=alt.Y("Saliency Score:Q", scale=alt.Scale(domain=[y_min, y_max])),
            color=color_enc,
            opacity=alt.condition(alt.datum["Gene ID"] == gene_id, alt.value(1.0), alt.value(0.65)),
        )
    )

    rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[3, 3], color="black")
        .encode(y="y:Q")
    )

    spans  = _genomic_spans(utr_len, central_pad_size)
    ann    = _annotation_layer(text_y)
    layers = spans + [line_chart, rule, ann]

    # Overlay SNP markers if provided
    if mut_markers is not None:
        snp_df = pd.DataFrame(mut_markers, columns=["Nucleotide Position", "marker", "description"])
        snp_df["Saliency Score"] = text_y
        snp_layer = (
            alt.Chart(snp_df)
            .mark_rule(strokeDash=[2, 2], color="gray", size=2)
            .encode(
                x=alt.X("Nucleotide Position", scale=alt.Scale(domain=[1, 3021])),
                tooltip="description",
            )
        )
        layers.insert(-1, snp_layer)

    # ── Layout ────────────────────────────────────────────────────────────────
    _section("📊 Mutation Results")
    bar_col, sal_col = st.columns([0.18, 0.82])

    with bar_col:
        st.altair_chart(bar_chart, use_container_width=True, theme=None)

    with sal_col:
        st.altair_chart(alt.layer(*layers), use_container_width=True, theme=None)

        delta = float(pred_probs[0] - pred_probs[1])
        if delta != 0:
            direction = "increased" if delta < 0 else "decreased"
            st.info(
                f"**Δ probability = {delta:+.3f}** — Mutation {direction} the predicted "
                f"probability of high expression by **{abs(delta):.3f}**.",
                icon="📉" if delta > 0 else "📈",
            )

    def _reset():
        st.session_state.mutated_seq = seq
        if mut_reg_start is not None and mut_reg_end is not None:
            st.session_state.sub_seq_to_mutate = seq[mut_reg_start:mut_reg_end]

    if mut_markers is None:
        st.button("↺ Reset sequence", type="primary", on_click=_reset)


# ── VCF file uploader ─────────────────────────────────────────────────────────

def show_vcf_input():
    _section("📂 Upload VCF File")
    upload_col, _ = st.columns([0.35, 0.65])
    with upload_col:
        st.caption(
            "Upload a gzip-compressed VCF file (.gz). "
            "Pre-filter to your genes of interest for faster processing."
        )
        vcf_file = st.file_uploader(
            label="VCF file (.gz)",
            accept_multiple_files=False,
            type=[".gz"],
            help="Variant Call Format file, gzip-compressed.",
        )
    return vcf_file
