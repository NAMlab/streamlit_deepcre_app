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
from lib.ui.tutorial import show_tutorial_tab
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
/* ── Global typography (Safely targeted) ── */
html, body, [class*="css"] { 
    font-family: 'Inter', 'Segoe UI', sans-serif !important; 
}

/* Safely target standard markdown text without breaking Streamlit widgets */
.stMarkdown p { 
    color: #333333; 
}

/* ── Harmonized Headers ── */
h1, h2, h3, h4, h5 {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: #4F1787 !important; /* Unified Deep Purple */
    font-weight: 700 !important;
}
h1 { font-size: 2.0rem !important; border-bottom: 2px solid #4F1787; padding-bottom: 10px; margin-bottom: 20px;}
h3 { font-size: 1.3rem !important; margin-top: 1.5rem !important; margin-bottom: 0.5rem !important; }

/* ── Primary Action Buttons (Red-Orange) ── */
button[kind="primary"] {
    background: linear-gradient(90deg, #FF4B4B, #FF8E53) !important;
    border: none !important;
}
button[kind="primary"] p {
    color: #FFFFFF !important; /* Forces text to be white */
    font-weight: 600 !important;
}

/* ── Sidebar polish ── */
section[data-testid="stSidebar"] {
    background: #fafafa;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] label {
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #4F1787 !important; 
}

/* ── Info & Instruction Banners ── */
.stAlert {
    border-radius: 6px !important;
}

/* ── Force Tab Text to be Larger ── */
button[data-baseweb="tab"] p {
    font-size: 1.25rem !important; 
    font-weight: 600 !important;
    color: #333333;
}
button[data-baseweb="tab"] {
    padding-top: 0.8rem !important;
    padding-bottom: 0.8rem !important;
}
/* ── Harmonized Headers ── */
h1, h2, h4, h5 {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: #4F1787 !important; /* Unified Deep Purple for standard titles */
    font-weight: 700 !important;
}
h1 {
    font-size: 2.0rem !important;
    border-bottom: 2px solid #4F1787;
    padding-bottom: 10px;
    margin-bottom: 20px;
}

/* ── Standout Section Headers (st.subheader / h3) ── */
h3 {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    margin-top: 1.8rem !important;
    margin-bottom: 0.8rem !important;
    padding-bottom: 6px;

    /* Vibrant Red-Orange Gradient Text */
    background: linear-gradient(90deg, #FF4B4B, #FF8E53) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    border-bottom: 2px solid #FFEDEA; /* Soft underline matching the gradient */
}
/* ── Navigation Tabs (Framed & Purple Gradient) ── */

/* 1. The Grey Background Frame */
div[data-baseweb="tab-list"] {
    background-color: #F1F5F9 !important; /* Soft, professional slate grey */
    padding: 0.6rem 1rem !important;
    border-radius: 10px !important;
    gap: 0.5rem !important;
}

/* 2. The Tab Button Containers */
button[data-baseweb="tab"] {
    padding: 0.6rem 1.4rem !important;
    border-radius: 8px !important;
    border: none !important;
    background-color: transparent !important;
}

/* 3. Make the "Active" tab pop out like a physical card */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #FFFFFF !important;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.08) !important;
}
/* 4. The Tab Text (Larger + Solid Purple) */
button[data-baseweb="tab"] p {
    font-size: 1.55rem !important; /* Keeps the larger size */
    font-weight: 800 !important;
    margin: 0 !important;
    color: #4F1787 !important; /* Solid Deep Purple */
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
            <span class="logo"/span>
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
            label="Select Gene", options=gene_ids
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
    
    # --- Hide the empty uploader during the tutorial to avoid confusion ---
    if st.session_state.get("tutorial_vcf_active"):
        st.info("💡 **Tutorial VCF Mode Active:** The file `Supplementary-file-3` is pre-loaded in memory. The standard file uploader is hidden.", icon="ℹ️")
        import io
        import os
        file_path = "tutorial/File_3_deepCRE_tutorial_rap12-2-variant_Supplementary-file-3_dCRE_Peleketal2025.vcf.gz"
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vcf_file = io.BytesIO(f.read())
                vcf_file.name = "tutorial_variants.vcf.gz"
                vcf_file.size = os.path.getsize(file_path)
        else:
            st.error(f"❌ Could not find the VCF file at: {file_path}")
            return
    else:
        vcf_file = show_vcf_input()

    if vcf_file is None:
        return

    progress_marker.update(label="Processing VCF file…")
    vcf_df = getVcfContent(vcf_file, gene_starts, gene_ends, gene_chroms)

    # --- NEW: Layout simplified (removed the 50-SNP head table and split columns) ---
    
    # --- Auto-select the RAP2.12 tutorial gene ---
    default_idx = 0
    if st.session_state.get("tutorial_vcf_active"):
        tut_gene = st.session_state.get("tutorial_vcf_gene", "AT1G53910")
        if tut_gene in gene_ids:
            default_idx = gene_ids.index(tut_gene)

    gene_id = st.selectbox(label="Choose gene", options=gene_ids, index=default_idx)
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
    
    # --- NEW: Added explicit instructions for the user ---
    st.info("**Instructions:** Click on the lines (rows) in the table below to select your desired SNPs. Once selected, click the **Mutate Sequence** button below the table to apply the chosen mutations to the sequence.", icon="ℹ️")

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

    # ---Catch the Tutorial Data Override ---
    if st.session_state.get("tutorial_active", False):
        st.info("💡 **Tutorial Demo is active!** Using tutorial gene list and model. Go to the Tutorial tab to clear this data.", icon="ℹ️")
        genes_list = st.session_state.tutorial_genes
        selected_model = st.session_state.tutorial_model
        selected_organism = "Arabidopsis thaliana (TAIR10)" 

    validateDataset(genome, annotation, genes_list, use_example)
    validateModel(f"models/{selected_model}.h5")

    # Hide the "upload data" banners if the tutorial is running
    if not st.session_state.get("tutorial_active", False):
        _render_dataset_banner(genome, annotation, genes_list, use_example, selected_organism)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    progress_marker = st.status("Processing data…", expanded=False)
    home_tab, preds_tab, interpret_tab, mutations_tab, tutorial_tab, about_tab = st.tabs(
        ["Home", "Predictions", "Explanation", "Mutation","Tutorial", "About"]
    )

    with home_tab:
        show_about_tab(AVAILABLE_GENOMES)
    with tutorial_tab:                
        show_tutorial_tab()              
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
