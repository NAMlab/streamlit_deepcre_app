import hashlib
import itertools
from datetime import datetime
from lib.ui.tutorial import load_tutorial_fasta
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from lib.utils import normalize_sequence, genomic_from_window, window_from_genomic
from lib.ui.sidebar import request_widget_state
from lib.storage import currentModelId

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


def _sentence(text: str) -> str:
    """Upper-case only the first letter (the built-in capitalize would lower-case the 'N' padding base)."""
    return text[:1].upper() + text[1:]


def _chart_title(title: str, subtitle: str = "") -> alt.TitleParams:
    parts = [subtitle] if subtitle else []
    parts.append(f"deepCRE model: {currentModelId()}")
    parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return alt.TitleParams(title, subtitle=parts, subtitleColor="#9ca3af",
                           fontSize=13, subtitleFontSize=10)


# ── Genomic region spans (reusable) ──────────────────────────────────────────

def _inline(df: pd.DataFrame) -> alt.Data:
    """Embed a (small) DataFrame directly in the layer's spec.

    Passing several DataFrames to a layered chart makes Altair register one named
    dataset per layer. Streamlit then fails to repaint the chart when it is replaced
    by a new one (axes only, marks appear on hover). Inline data avoids that.
    """
    return alt.Data(values=df.to_dict("records"))


def _region_span(x1: int, x2: int, color: str, tip: str) -> alt.Chart:
    return (
        alt.Chart(_inline(pd.DataFrame({"x1": [x1], "x2": [x2]})))
        .mark_rect(opacity=0.1)
        .encode(
            x=alt.X("x1:Q", scale=alt.Scale(domain=[1, 3021]), title="Nucleotide Position"),
            x2="x2:Q",
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
        alt.Chart(_inline(df))
        .mark_text(size=13, align="center", fontWeight="bold")
        .encode(
            x=alt.X("Nucleotide Position:Q", scale=alt.Scale(domain=[1, 3021])),
            y=alt.Y("Saliency Score:Q"),
            text="marker:N",
            tooltip="description:N",
        )
    )


# ── Analysis type selector ────────────────────────────────────────────────────

def choose_analysis_type() -> str:
    """Renders the radio button to select between manual and VCF mutation modes."""
    st.subheader("Analysis Mode")

    mode = st.radio(
        label="Select Analysis Mode",
        options=["**manual**", "**VCF**"],
        captions=[
            "Manually edit the CRE sequence",
            "Apply natural variants from a VCF file"
        ],
        label_visibility="collapsed",  # Hides the redundant label above the buttons
        key="analysis_mode_radio"
    )

    return mode

# ── Typing the coordinates instead of dragging them ──────────────────────────

WINDOW_SYSTEM = "deepCRE window (1–3020)"
GENOMIC_SYSTEM = "Genomic"


def _window_regions(utr_len: int, central_pad_size: int) -> dict:
    """1-based window spans of the four editable regions, in window order."""
    return {
        "gUR":  (1, 1000),
        "gTUR": (1001, 1000 + utr_len),
        "gTDR": (1001 + utr_len + central_pad_size,
                 1000 + (2 * utr_len) + central_pad_size),
        "gDR":  (1001 + (2 * utr_len) + central_pad_size,
                 2000 + (2 * utr_len) + central_pad_size),
    }


def _parse_coord(text: str):
    """Read a coordinate the way a user pastes it: 20,135,343 / 20 135 343."""
    cleaned = str(text).strip().replace(",", "").replace("_", "").replace(" ", "")
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def _resolve_typed_range(first: int, second: int, system: str, start: int,
                         end: int, strand: str, utr_len: int,
                         central_pad_size: int):
    """Typed pair -> (window_from, window_to, region, error).

    Returns the error as a sentence for st.warning; on success error is None.
    Genomic input is mapped through window_from_genomic(), so a minus-strand
    gene -- where rising genomic coordinates run backwards through the window --
    needs no special handling here beyond re-ordering the two ends.
    """
    size = 2 * (1000 + utr_len) + central_pad_size
    regions = _window_regions(utr_len, central_pad_size)

    if system == WINDOW_SYSTEM:
        for value in (first, second):
            if not 1 <= value <= size:
                return None, None, None, (
                    f"{value:,} is outside the deepCRE window, which runs from "
                    f"1 to {size:,}.")
        w_from, w_to = first, second
    else:
        prom = (start - 1000 + 1, start + utr_len)
        term = (end - utr_len + 1, end + 1000)
        mapped = []
        for value in (first, second):
            w = window_from_genomic(value, start, end, strand, utr_len,
                                    central_pad_size)
            if w is None:
                return None, None, None, (
                    f"{value:,} is not inside the sequence this gene's window "
                    f"covers. The window holds {prom[0]:,}–{prom[1]:,} "
                    f"(1,000 bp upstream of the gene plus {utr_len} bp into it) "
                    f"and {term[0]:,}–{term[1]:,} ({utr_len} bp before the gene "
                    f"end plus 1,000 bp downstream).")
            mapped.append(w)
        w_from, w_to = mapped

    if w_from > w_to:
        w_from, w_to = w_to, w_from

    pad_from, pad_to = 1001 + utr_len, 1000 + utr_len + central_pad_size
    if central_pad_size and not (w_to < pad_from or w_from > pad_to):
        return None, None, None, (
            f"Window positions {pad_from:,}–{pad_to:,} are the N-padding "
            f"between the two halves of the window and cannot be edited. "
            f"The range given covers {w_from:,}–{w_to:,}.")

    holding = [name for name, (lo, hi) in regions.items()
               if lo <= w_from <= hi and lo <= w_to <= hi]
    if not holding:
        spanned = [name for name, (lo, hi) in regions.items()
                   if not (w_to < lo or w_from > hi)]
        return None, None, None, (
            f"The range {w_from:,}–{w_to:,} spans "
            f"{' and '.join(spanned)}. Only one region can be edited at a time; "
            f"please give a range inside a single region.")

    return w_from, w_to, holding[0], None


def _typed_coordinates(start: int, end: int, strand: str, chrom: str,
                       utr_len: int, central_pad_size: int, sel_region: str,
                       slider_vals) -> None:
    """Text entry for the coordinate range, in window or genomic coordinates.

    The slider above is the quick way; this is the exact way, for a user who has
    a position from a browser, a GFF or a VCF. The typed range is written back
    into the slider (and the region radio) through the pending-widget-state
    queue, because Streamlit does not allow a widget's value to be set after it
    has been instantiated in the same run.
    """
    genomic_label = "Genomic (1-based)"

    # What the slider currently holds, in both systems, so the two are tied
    # together visibly rather than the user having to trust the conversion.
    g_lo = genomic_from_window(slider_vals[0], start, end, strand, utr_len,
                               central_pad_size)
    g_hi = genomic_from_window(slider_vals[1], start, end, strand, utr_len,
                               central_pad_size)
    if g_lo is not None and g_hi is not None:
        lo, hi = sorted((g_lo, g_hi))
        where = f"{chrom}:" if chrom else ""
        st.caption(f"Window {slider_vals[0]:,}–{slider_vals[1]:,} "
                   f"= {where}{lo:,}–{hi:,} ({strand} strand)")

    if st.session_state.get("mut_coord_notice"):
        st.success(st.session_state.pop("mut_coord_notice"), icon="✅")

    with st.expander("Type the coordinates instead"):
        system = st.radio(
            "Coordinate system", (WINDOW_SYSTEM, genomic_label),
            key="mut_coord_system",
            help="deepCRE coordinates run 1–3020 across the model window. "
                 "Genomic coordinates are the positions on "
                 + (f"chromosome {chrom}" if chrom else "the chromosome")
                 + ", as a GFF, a VCF or a genome browser gives them; they are "
                   "converted to the window for this gene.",
        )
        col_from, col_to = st.columns(2)
        from_txt = col_from.text_input("From", key="mut_coord_from",
                                       placeholder="e.g. 1001")
        to_txt = col_to.text_input("To", key="mut_coord_to",
                                   placeholder="e.g. 1500")

        if st.button("Set range", key="mut_coord_set"):
            first, second = _parse_coord(from_txt), _parse_coord(to_txt)
            if first is None or second is None:
                missing = [lab for lab, v in (("From", first), ("To", second))
                           if v is None]
                st.warning(f"Please give a whole number for "
                           f"{' and '.join(missing)}.", icon="⚠️")
                return
            w_from, w_to, region, error = _resolve_typed_range(
                first, second, WINDOW_SYSTEM if system == WINDOW_SYSTEM
                else GENOMIC_SYSTEM,
                start, end, strand, utr_len, central_pad_size)
            if error:
                st.warning(error, icon="⚠️")
                return
            request_widget_state(mut_region_selection=region,
                                 mut_coord_slider=(w_from, w_to))
            note = f"Range set to {w_from:,}–{w_to:,} in {region}."
            if region != sel_region:
                note += f" The region was switched from {sel_region} to {region}."
            if system != WINDOW_SYSTEM:
                note += (f" ({first:,}–{second:,} on {chrom or 'the chromosome'} "
                         f"maps there on the {strand} strand.)")
            st.session_state.mut_coord_notice = note
            st.rerun()


# ── Manual mutation UI ────────────────────────────────────────────────────────

def show_manual_mutation(gene_id: str, start: int, end: int,
                         seq: str, utr_len: int, central_pad_size: int,
                         strand: str = "+", chrom: str = ""):
    # --- Tutorial Shortcut for Manual Mode ---
    st.info("**Tutorial Shortcut:** See the **Tutorial** tab for full context, or load the demo sequence right here.", icon="ℹ️")
    if st.session_state.get("tutorial_mutation_active"):
        if st.button("❌ Clear Tutorial Sequence", key="mut_clear_fasta"):
            st.session_state.tutorial_mutation_active = False
            st.rerun()
    else:
        if st.button("Load Tutorial Sequence (OsACT1)", key="mut_load_fasta"):
            st.session_state.tutorial_mutation_active = True
            st.session_state.tutorial_mut_seq = load_tutorial_fasta("gTUR_Osativa_OsACT1_KP100426-PIG2_5UTR500BP")

            # --- Snap the UI coordinates safely ---
            st.session_state.mut_region_selection = "gTUR"
            st.session_state.mut_coord_slider = (1001, 1500)

            st.rerun()

    # Initialise session state for this gene
    if "current_gene" not in st.session_state or st.session_state.current_gene != gene_id:
        st.session_state.current_gene = gene_id
        st.session_state.mutated_seq  = seq
        st.session_state.mut_notice = None

    _section("Select Region to Mutate")
    sel_region = st.radio(
        "Region",
        options=["gUR", "gTUR", "gTDR", "gDR"],
        horizontal=True,
        label_visibility="collapsed",
        key="mut_region_selection"
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
        _section("Coordinate Range")
        with st.form("mutation_form", clear_on_submit=False, border=False):
            slider_vals = st.slider(
                "Start / End coordinates",
                min_value=val, max_value=max_val,
                value=(min_val, max_val), step=1,
                label_visibility="collapsed",
                key="mut_coord_slider"
            )
            st.form_submit_button("Apply", type="primary")

        _typed_coordinates(start, end, strand, chrom, utr_len, central_pad_size,
                           sel_region, slider_vals)

    mut_reg_start, mut_reg_end = slider_vals
    mut_reg_start -= 1
    if mut_reg_start == mut_reg_end:
        mut_reg_end += 1

    sub_seq = st.session_state.mutated_seq[mut_reg_start:mut_reg_end]
    pad = mut_reg_end - mut_reg_start - len(sub_seq)
    if pad > 0:
        sub_seq += "N" * pad

    # ---: Dynamic sequence update & Tutorial Injection ---
    if "sub_seq_to_mutate" not in st.session_state or st.session_state.get("last_mut_coords") != (mut_reg_start, mut_reg_end):
        st.session_state.sub_seq_to_mutate = sub_seq
        st.session_state.last_mut_coords = (mut_reg_start, mut_reg_end)
        st.session_state.mut_notice = None  # a different region/range: old notice no longer applies

    region_len = mut_reg_end - mut_reg_start

    def _splice(edited_text):
        """Write the edited region back, always keeping the full sequence at its original length."""
        fixed, _ = normalize_sequence(edited_text, region_len)
        full = st.session_state.mutated_seq
        merged = full[:mut_reg_start] + fixed + full[mut_reg_end:]
        # Final safety net: the model needs exactly len(seq) bases.
        merged, _ = normalize_sequence(merged, len(seq))
        return merged, fixed

    def apply_mutation():
        edited_text = st.session_state.sub_seq_to_mutate
        # Remember what had to be corrected: the box is padded below, so by the next
        # run the text already has the right length and would otherwise show no hint.
        _, notes = normalize_sequence(edited_text, region_len)
        st.session_state.mut_notice = (
            f"The {sel_region} region must be exactly {region_len} bp. "
            + _sentence("; ".join(notes)) + "."
        ) if notes else None
        merged, fixed = _splice(edited_text)
        st.session_state.mutated_seq = merged
        # Show the corrected (e.g. padded) text in the box instead of the short one.
        st.session_state.sub_seq_to_mutate = fixed

    # --- Tutorial injection ---
    # This used to write the donor sequence into the text box only. The box then
    # showed the swapped sequence while st.session_state.mutated_seq was still the
    # reference, so the model was handed [reference, reference] and every download
    # contained the reference twice. Splice it into the working sequence as well,
    # i.e. do what pressing "Mutate" does.
    if st.session_state.get("tutorial_mutation_active"):
        donor = st.session_state.tutorial_mut_seq
        merged, fixed = _splice(donor)
        st.session_state.mutated_seq = merged
        st.session_state.sub_seq_to_mutate = fixed
        st.session_state.tutorial_mutation_active = False

    with seq_col:
        _section(f"Edit {sel_region} ({mut_reg_start}–{mut_reg_end})")
        edited_seq = st.text_area(
            label=f"{sel_region} sequence",
            value=st.session_state.sub_seq_to_mutate, # Uses our synced state
            max_chars=region_len,
            height=80,
            key="sub_seq_to_mutate",
            label_visibility="collapsed",
            on_change=apply_mutation,
        )

        # Tell the user (instead of crashing) when the edit does not match the required length.
        _, length_notes = normalize_sequence(edited_seq, region_len)
        if length_notes:
            st.warning(
                f"The {sel_region} region must be exactly {region_len} bp. "
                + _sentence("; ".join(length_notes)) + ".",
                icon="⚠️",
            )
        elif st.session_state.get("mut_notice"):
            # The box was already corrected by the callback; still tell the user what happened.
            st.warning(st.session_state.mut_notice + " The corrected sequence is shown above.", icon="⚠️")

        if st.button("Mutate", type="primary"):
            st.session_state.mutated_seq, _ = _splice(edited_seq)

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
        alt.Chart(_inline(pd.DataFrame({"Saliency Score": [0]})))
        .mark_rule(
            strokeDash=[3, 3],
            color="black",
        )
        .encode(
            y=alt.Y("Saliency Score:Q")
        )
    )

    spans  = _genomic_spans(utr_len, central_pad_size)
    ann    = _annotation_layer(text_y)
    layers = spans + [line_chart, rule, ann]

    # Overlay SNP markers if provided
    if mut_markers is not None:
        snp_df = pd.DataFrame(mut_markers, columns=["Nucleotide Position", "marker", "description"])
        snp_df["Saliency Score"] = text_y
        snp_layer = (
            alt.Chart(_inline(snp_df))
            .mark_rule(strokeDash=[2, 2], color="gray", size=2)
            .encode(
                x=alt.X("Nucleotide Position:Q", scale=alt.Scale(domain=[1, 3021])),
                tooltip="description:N",
            )
        )
        layers.insert(-1, snp_layer)

    # ── Layout ────────────────────────────────────────────────────────────────
    _section("Mutation Results")
    bar_col, sal_col = st.columns([0.18, 0.82])

    # A key that changes with the results makes Streamlit build a fresh chart after each
    # mutation. Without it the chart is updated in place and may not repaint until hovered.
    result_key = hashlib.md5(
        np.asarray(actual_scores[1]).tobytes() + np.asarray(pred_probs, dtype=np.float64).tobytes()
        + str(mut_markers).encode()
    ).hexdigest()[:12]

    with bar_col:
        st.altair_chart(bar_chart, use_container_width=True, theme=None, key=f"mut_bar_{result_key}")

    with sal_col:
        st.altair_chart(alt.layer(*layers), use_container_width=True, theme=None, key=f"mut_sal_{result_key}")

        # Two identical input sequences produce two identical curves and a
        # download whose "mutated" half is a copy of the reference. Say so
        # instead of letting it read as a result.
        if np.array_equal(np.asarray(actual_scores[0]), np.asarray(actual_scores[1])):
            st.warning(
                "No change applied — the edited sequence is identical to the reference, "
                "so both curves and both halves of the downloads are the reference. "
                "Edit the sequence and press **Mutate**.",
            )

        delta = float(pred_probs[0] - pred_probs[1])
        if delta != 0:
            direction = "increased" if delta < 0 else "decreased"
            st.info(
                f"**Δ ( normal - mutated ) probability = {delta:+.3f}** — Mutation {direction} the predicted "
                f"probability of high expression by **{abs(delta):.3f}**.",
                icon="📉" if delta > 0 else "📈",
            )

    def _reset():
        st.session_state.mut_notice = None
        st.session_state.mutated_seq = seq
        if mut_reg_start is not None and mut_reg_end is not None:
            st.session_state.sub_seq_to_mutate = seq[mut_reg_start:mut_reg_end]

    if mut_markers is None:
        st.button("↺ Reset sequence", type="primary", on_click=_reset)




    st.markdown("<br>", unsafe_allow_html=True) # A tiny bit of breathing room

# --- Download Buttons ---
    # The reference and the mutated importance scores are both exported, and in a
    # shape that can be used directly: one row per position with the two series
    # side by side and their difference, rather than the two stacked in a single
    # column. The per-base file keeps the four nucleotide channels the model
    # actually saw, which the averaged file reduces away.
    st.markdown("<br>", unsafe_allow_html=True) # A tiny bit of breathing room
    model_id = currentModelId()
    ref, mut = np.asarray(actual_scores[0]), np.asarray(actual_scores[1])
    imp_df = pd.DataFrame({"Nucleotide Position": positions})
    for i, b in enumerate("ACGT"):
        imp_df[f"reference_{b}"] = ref[:, i]
    for i, b in enumerate("ACGT"):
        imp_df[f"mutated_{b}"] = mut[:, i]
    imp_df["reference_mean"] = ref.mean(axis=1)
    imp_df["mutated_mean"] = mut.mean(axis=1)
    imp_df["delta_mean (mutated - reference)"] = imp_df["mutated_mean"] - imp_df["reference_mean"]
    imp_df["gene"] = gene_id
    imp_df["model"] = model_id

    btn_col2, btn_col1 = st.columns(2)

    with btn_col1:
        st.download_button(
            label="Download Importance Map (CSV)",
            data=imp_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{gene_id}_saliency_impact_{model_id}.csv",
            mime="text/csv",
            help="One row per position: the four base channels for the reference and "
                 "for the mutated sequence, their averages, and the difference.",
            width="stretch"
        )

    with btn_col2:
        # Changed to bar_df
        csv_bar = bar_df.assign(model=currentModelId()).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction Probs (CSV)",
            data=csv_bar,
            file_name=f"{gene_id}_prediction_probs_{currentModelId()}.csv",
            mime="text/csv",
            width="stretch"
        )


# ── VCF file uploader ─────────────────────────────────────────────────────────

def show_vcf_input():
    _section("📂 Upload VCF File")
    upload_col, _ = st.columns([0.35, 0.65])

    with upload_col:
        # --- NEW: Tutorial Shortcut for VCF Mode ---
        if st.session_state.get("tutorial_vcf_active"):
            st.info("**Tutorial VCF Mode Active:** The demo file is pre-loaded. Check the **Tutorial** tab for full context.", icon="ℹ️")
            if st.button("❌ Clear VCF Data", key="mut_clear_vcf"):
                st.session_state.tutorial_vcf_active = False
                st.rerun()
        else:
            st.info("**Tutorial Shortcut:** See the **Tutorial** tab for context, or load the demo VCF directly.", icon="ℹ️")
            if st.button("Load Tutorial VCF (RAP2.12)", key="mut_load_vcf"):
                st.session_state.tutorial_vcf_active = True
                st.session_state.tutorial_vcf_gene = "AT1G53910"
                st.rerun()
            st.markdown("---")
        # -------------------------------------------

        # Only show the physical uploader if the tutorial isn't active
        if not st.session_state.get("tutorial_vcf_active"):
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
        else:
            vcf_file = None # Handled by app.py injection!

    return vcf_file


# --- Tutorial Injection for VCF ---
    if st.session_state.get("tutorial_vcf_active"):
        import io
        import os

        file_path = "tutorial/File_3_deepCRE_tutorial_rap12-2-variant_Supplementary-file-3_dCRE_Peleketal2025.vcf.gz"

        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            # Create a perfect mock of Streamlit's UploadedFile to prevent backend crashes
            class MockUploadedFile(io.BytesIO):
                def __init__(self, buffer, name):
                    super().__init__(buffer)
                    self.name = name
                    self.size = len(buffer)
                    self.type = "application/gzip"

            vcf_file = MockUploadedFile(file_bytes, "tutorial_variants.vcf.gz")

            with upload_col:
                st.success("✅ Tutorial VCF file injected!")
        else:
            with upload_col:
                st.error(f"❌ Error: Could not find the file at `{file_path}`. Please check the spelling!")

    return vcf_file
