import streamlit as st
import pandas as pd
from io import StringIO

# ── Sidebar section header helper ────────────────────────────────────────────

def _section(label: str) -> None:
    st.sidebar.markdown(
        f"""
        <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.07em;color:#4F1787;margin:1.1rem 0 0.3rem 0;
                    padding-bottom:3px;border-bottom:1px solid #ede9f5;">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── File validation ───────────────────────────────────────────────────────────

def _validate_file(file, label: str):
    """Return the file if non-empty, otherwise show an error and return None."""
    if file.size > 0:
        return file
    st.sidebar.error(f"⚠️ The uploaded **{label}** file is empty. Please verify.", icon="🚨")
    return None


# ── Public entry point ────────────────────────────────────────────────────────

def show_sidebar(available_species: list, available_genomes: pd.DataFrame, available_models: list):
    """
    Render the sidebar and return the user's selections.

    Returns
    -------
    selected_organism : str
    genome            : str | UploadedFile | None
    annotation        : str | UploadedFile | None
    genes_list        : list[str] | None
    selected_model    : str
    use_example       : bool
    """
    # ── Branding ──────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div style="text-align:center;padding:0.6rem 0 0.2rem 0;">
            <span style="font-size:2rem;">🧬</span>
            <div style="font-size:1.05rem;font-weight:700;color:#4F1787;margin-top:2px;">deepCRE</div>
            <div style="font-size:0.72rem;color:#9ca3af;">CRE expression predictor</div>
        </div>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:0.7rem 0;">
        """,
        unsafe_allow_html=True,
    )

    # ── Species / genome ──────────────────────────────────────────────────────
    _section("🌿 Reference Organism")
    selected_organism = st.sidebar.selectbox(
        label="Species",
        options=available_species,
        help='Select a pre-loaded genome or choose "New" to upload your own.',
    )

    genome = annotation = None

    if selected_organism == "New":
        _section("📂 Upload Genome")
        genome_upload = st.sidebar.file_uploader(
            label="Genome (.fa / .fa.gz)",
            accept_multiple_files=False,
            type=[".fa", ".gz"],
            help="FASTA file, preferably gzip-compressed (.fa.gz).",
        )
        if genome_upload is not None:
            genome = _validate_file(genome_upload, "genome")

        annot_upload = st.sidebar.file_uploader(
            label="Annotation (.gtf / .gff3 / .gz)",
            accept_multiple_files=False,
            type=[".gtf", ".gff3", ".gff", ".gz"],
            help="GTF or GFF3 annotation file, preferably gzip-compressed.",
        )
        if annot_upload is not None:
            annotation = _validate_file(annot_upload, "annotation")
    else:
        row = available_genomes.loc[available_genomes["display_name"] == selected_organism]
        genome     = row["assembly_file"].values[0]
        annotation = row["annotation_file"].values[0]

    # ── Gene list ─────────────────────────────────────────────────────────────
    _section("🔬 Gene List")
    genes_upload = st.sidebar.file_uploader(
        label="Gene IDs (.csv / .txt)",
        type=[".csv", ".txt"],
        accept_multiple_files=False,
        help="One gene ID per line, up to 1 000 genes. Excess genes are ignored.",
    )

    genes_list = None
    use_example = False

    if genes_upload is None:
        use_example = st.sidebar.checkbox(
            "Use 100 random genes from the genome",
            value=False,
            help="Randomly sample 100 genes from the selected genome for a quick demo.",
        )
    else:
        validated = _validate_file(genes_upload, "genes list")
        if validated is not None:
            genes_list = (
                pd.read_csv(StringIO(validated.getvalue().decode("utf-8")), header=None)
                .values.ravel()
                .tolist()
            )

    # ── Model ─────────────────────────────────────────────────────────────────
    _section("🤖 deepCRE Model")
    selected_model = st.sidebar.selectbox(
        label="Model",
        options=available_models,
        help="Pre-trained deepCRE models from Peleke et al., 2024.",
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:1.4rem 0 0.5rem 0;">
        <div style="font-size:0.68rem;color:#9ca3af;text-align:center;">
            NAMlab · deepCRE toolkit<br>
            <a href="https://github.com/NAMlab/streamlit_deepcre_app"
               style="color:#4F1787;text-decoration:none;">GitHub ↗</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return selected_organism, genome, annotation, genes_list, selected_model, use_example
