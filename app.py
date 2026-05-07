import os
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

from lib.utils import one_hot_to_dna, one_hot_encode, dataframe_with_selections
from lib.ui.about import show_about_tab
from lib.ui.sidebar import show_sidebar
from lib.ui.predictions import show_predictions_tab
from lib.ui.saliency import show_saliency_tab
from lib.ui.license_ref import show_license_ref
from lib.ui.mutation import choose_analysis_type, show_manual_mutation, show_mutation_results, show_vcf_input
from lib.storage import *

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
tf.config.set_visible_devices([], "GPU")

# ── Constants ────────────────────────────────────────────────────────────────
COLOR_PALETTE = ["#4F1787", "#EB3678"]
AVAILABLE_GENOMES = pd.read_csv("genomes/genomes.csv")
SPECIES = AVAILABLE_GENOMES["display_name"].tolist() + ["New"]
MODEL_NAMES = sorted(
    f.split(".")[0] for f in os.listdir("models") if f.endswith(".h5")
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Global typography ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Page header ── */
.deepcre-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.2rem 0 0.4rem 0;
    border-bottom: 2px solid #4F1787;
    margin-bottom: 1.4rem;
}
.deepcre-header .logo {
    font-size: 2rem;
}
.deepcre-header h1 {
    margin: 0;
    font-size: 1.55rem;
    font-weight: 700;
    color: #4F1787;
    line-height: 1.2;
}
.deepcre-header p {
    margin: 0;
    font-size: 1.25rem;
    color: #6b7280;
}

/* ── Sidebar polish ── */
section[data-testid="stSidebar"] {
    background: #fafafa;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stFileUploader label {
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #374151;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 1rem; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 130px;
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-card .label { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
.metric-card .value { font-size: 1.5rem; font-weight: 700; color: #111827; margin-top: 2px; }
.metric-card .sub   { font-size: 0.75rem; color: #9ca3af; margin-top: 1px; }

/* ── Section dividers ── */
.section-header {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4F1787;
    margin: 1.4rem 0 0.5rem 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #ede9f5;
}

/* ── Info banner ── */
.info-banner {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.87rem;
    color: #166534;
    margin-bottom: 1rem;
}
.warn-banner {
    background: #fefce8;
    border-left: 4px solid #eab308;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.87rem;
    color: #713f12;
    margin-bottom: 1rem;
}
</style>
"""


def _gene_index(gene_ids: list, gene_id: str) -> int:
    """Return list index for a gene ID (avoids repeated .index() calls)."""
    return gene_ids.index(gene_id)


def _render_header() -> None:
    st.markdown(
        """
        <div class="deepcre-header">
            <span class="logo">🧬</span>
            <div>
                <h1>deepCRE</h1>
                <p>Predicting gene expression from cis-regulatory elements using deep learning</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dataset_banner(genome, annotation, genes_list, use_example, selected_organism) -> None:
    """Show contextual info/warning banners about the current dataset state."""
    if genome is not None and annotation is not None and genes_list is None:
        if use_example:
            st.markdown(
                f'<div class="warn-banner">⚠️ No gene list uploaded — displaying results for '
                f"100 randomly sampled genes from the <b>{selected_organism}</b> genome.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="info-banner">
                ℹ️ No data uploaded yet. Tick <b>"Use 100 random genes from the genome"</b> in the
                sidebar to explore the tool, or upload your own gene list.
                </div>""",
                unsafe_allow_html=True,
            )


def _handle_manual_mutation(gene_ids, gene_starts, gene_ends, x, progress_marker) -> None:
    gene_col, _ = st.columns([0.3, 0.7])
    with gene_col:
        gene_id = st.selectbox(
            label=":gray[Select] :green[**gene**]", options=gene_ids
        )

    idx = _gene_index(gene_ids, gene_id)
    seq = one_hot_to_dna(x[idx])[0]
    start, end = gene_starts[idx], gene_ends[idx]
    half_span = abs(start - end) // 2
    utr_len = min(500, half_span)
    central_pad_size = 3020 - (1000 + utr_len) * 2

    mut_reg_start, mut_reg_end = show_manual_mutation(
        gene_id, start, end, seq, utr_len, central_pad_size
    )

    seqs = np.array([one_hot_encode(s) for s in [seq, st.session_state.mutated_seq]])
    validateMutationSequences(seqs)

    progress_marker.update(label="Applying mutations…")
    preds = getMutationPredictions()
    actual_scores, pred_probs, _ = getMutationScores(gene_id)

    show_mutation_results(
        gene_id, pred_probs, actual_scores, seq,
        utr_len, central_pad_size, mut_reg_start, mut_reg_end,
    )


def _handle_vcf_mutation(gene_ids, gene_starts, gene_ends, gene_chroms, gene_strands, x, progress_marker) -> None:
    vcf_file = show_vcf_input()
    progress_marker.update(label="Processing VCF file…")
    vcf_df = getVcfContent(vcf_file, gene_starts, gene_ends, gene_chroms)

    if vcf_file is None:
        return

    vcf_col, gene_col, _ = st.columns([0.4, 0.5, 0.1], vertical_alignment="center")

    with vcf_col:
        st.markdown('<div class="section-header">First 50 SNPs in VCF</div>', unsafe_allow_html=True)
        st.dataframe(vcf_df.head(50))

    with gene_col:
        gene_id = st.selectbox(label="Choose gene", options=gene_ids)
        idx = _gene_index(gene_ids, gene_id)

        seq = one_hot_to_dna(x[idx])[0]
        strand = gene_strands[idx]
        start, end = gene_starts[idx], gene_ends[idx]
        chrom = gene_chroms[idx]
        utr_len = min(500, abs(end - start) // 2)
        central_pad_size = 3020 - (1000 + utr_len) * 2

        prom_start, prom_end = start - 1000, start + utr_len
        term_start, term_end = end - utr_len, end + 1000

        def _tag_snps(mask, region_label_plus, region_label_minus):
            df = vcf_df[mask].copy()
            df["Region"] = region_label_plus if strand == "+" else region_label_minus
            return df

        snps_prom = _tag_snps(
            (vcf_df["Pos"] > prom_start) & (vcf_df["Pos"] < prom_end) & (vcf_df["Chrom"] == chrom),
            "Promoter", "Terminator",
        )
        snps_term = _tag_snps(
            (vcf_df["Pos"] > term_start) & (vcf_df["Pos"] < term_end) & (vcf_df["Chrom"] == chrom),
            "Terminator", "Promoter",
        )

        snps_cis = (
            pd.concat([snps_prom, snps_term], axis=0)
            .assign(Strand=strand)
            .sort_values(["Region", "Pos"])
            .reset_index(drop=True)
        )

        if "current_gene" not in st.session_state:
            st.session_state.current_gene = gene_id

        st.markdown(
            f'<div class="section-header">SNPs in cis-regulatory regions of {gene_id}</div>',
            unsafe_allow_html=True,
        )
        selection = dataframe_with_selections(df=snps_cis)
        st.markdown('<div class="section-header">Selected SNPs</div>', unsafe_allow_html=True)
        st.dataframe(selection, use_container_width=True)

    if selection.empty:
        return

    complements = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

    if st.button("Mutate Sequence", type="primary"):
        if "cis_seq" not in st.session_state:
            st.session_state["cis_seq"] = seq
        if st.session_state.current_gene != gene_id:
            st.session_state.current_gene = gene_id
            st.session_state["cis_seq"] = seq

        mut_cis_seq = st.session_state["cis_seq"]
        mut_markers = []

        for _, snp_pos, _, ref_allele, alt_allele, snp_region, snp_strand in selection.values:
            if snp_strand == "+":
                if snp_region == "Promoter":
                    rel = snp_pos - prom_start - 1 if snp_pos != prom_start else 0
                    mapped = rel
                else:
                    rel = snp_pos - term_start - 1 if snp_pos != term_start else 0
                    mapped = (1000 + utr_len + central_pad_size) + rel
            else:
                if snp_region == "Promoter":
                    rel = snp_pos - term_start - 1 if snp_pos != term_start else 0
                    mapped = (1000 + utr_len) - rel - 1
                else:
                    rel = snp_pos - prom_start - 1 if snp_pos != prom_start else 0
                    mapped = 3020 - rel - 1

            mut_markers.append((mapped, "*", f"SNP: {ref_allele} → {alt_allele}"))
            base = alt_allele if snp_strand == "+" else complements[alt_allele]
            mut_cis_seq = mut_cis_seq[:mapped] + base + mut_cis_seq[mapped + 1:]

        seqs = np.array([one_hot_encode(s) for s in [st.session_state["cis_seq"], mut_cis_seq]])
        validateMutationSequences(seqs)
        preds = getMutationPredictions()
        actual_scores, pred_probs, _ = getMutationScores(gene_id)
        show_mutation_results(
            gene_id, pred_probs, actual_scores, seq,
            utr_len, central_pad_size, None, None, mut_markers,
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        layout="wide",
        page_title="deepCRE",
        page_icon="🧬",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    initStorage()

    _render_header()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    selected_organism, genome, annotation, genes_list, selected_model, use_example = show_sidebar(
        available_species=SPECIES,
        available_genomes=AVAILABLE_GENOMES,
        available_models=MODEL_NAMES,
    )
    validateDataset(genome, annotation, genes_list, use_example)
    validateModel(f"models/{selected_model}.h5")

    _render_dataset_banner(genome, annotation, genes_list, use_example, selected_organism)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    progress_marker = st.status("Processing data…", expanded=False)
    home_tab, preds_tab, interpret_tab, mutations_tab, about_tab = st.tabs(
        ["🏠 Home", "📊 Predictions", "🔍 Explanation", "🧪 Mutation", "ℹ️ About"]
    )

    with home_tab:
        show_about_tab(AVAILABLE_GENOMES)
    with about_tab:
        show_license_ref()

    # ── Data pipeline ─────────────────────────────────────────────────────────
    x = None
    if genome is not None and annotation is not None:
        progress_marker.update(label="Loading dataset…")
        (
            x, gene_ids, gene_chroms,
            gene_starts, gene_ends,
            gene_size, gene_gc_cont, gene_strands,
        ) = getDataset()

    if x is None or x.size == 0:
        progress_marker.update(state="complete", label="Awaiting input")
        return

    # ── Predictions ───────────────────────────────────────────────────────────
    progress_marker.update(label="Running predictions…")
    preds = getPredictions()

    with preds_tab:
        n_high = sum(1 for p in preds if p > 0.5)
        n_low  = len(preds) - n_high
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Genes analysed", len(preds))
        m2.metric("High expression", n_high, delta=f"{100*n_high/len(preds):.0f}%")
        m3.metric("Low expression",  n_low,  delta=f"{100*n_low/len(preds):.0f}%")
        m4.metric("Model", selected_model)
        st.divider()
        show_predictions_tab(
            gene_ids, gene_chroms, gene_starts, gene_ends,
            gene_size, gene_gc_cont, preds, COLOR_PALETTE,
        )

    # ── Saliency ──────────────────────────────────────────────────────────────
    progress_marker.update(label="Extracting saliency scores…")
    actual_scores_low, actual_scores_high, g_l, g_h, p_l, p_h = getScores()

    with interpret_tab:
        show_saliency_tab(
            actual_scores_high, actual_scores_low,
            p_h, p_l, COLOR_PALETTE, g_h, g_l,
        )

    # ── Mutations ─────────────────────────────────────────────────────────────
    with mutations_tab:
        mutate_analysis_type = choose_analysis_type()
        if mutate_analysis_type == "**manual**":
            _handle_manual_mutation(gene_ids, gene_starts, gene_ends, x, progress_marker)
        else:
            _handle_vcf_mutation(
                gene_ids, gene_starts, gene_ends,
                gene_chroms, gene_strands, x, progress_marker,
            )

    progress_marker.update(state="complete", label="Done ✓")


if __name__ == "__main__":
    main()
