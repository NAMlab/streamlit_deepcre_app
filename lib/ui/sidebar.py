import re

import streamlit as st
import pandas as pd
from io import StringIO

# ── Cross-tab widget control ─────────────────────────────────────────────────
#
# Streamlit refuses to write a widget's session-state entry once that widget has
# been instantiated in the current run, so a button in the Tutorial tab cannot
# set the sidebar's model directly. It queues the value here instead and calls
# st.rerun(); the queue is applied at the top of the next run, before any widget
# is built. This keeps the sidebar the single source of truth for the model, the
# organism and the gene list.

PENDING_KEY = "_pending_widget_state"


def request_widget_state(**values) -> None:
    """Queue widget values to be applied at the start of the next run."""
    pending = st.session_state.get(PENDING_KEY, {})
    pending.update(values)
    st.session_state[PENDING_KEY] = pending


def _apply_pending(available_species: list, available_models: list) -> None:
    """Apply queued widget values, skipping any this deployment does not offer."""
    pending = st.session_state.pop(PENDING_KEY, {})
    options = {"sidebar_organism": available_species, "sidebar_model": available_models}
    for key, value in pending.items():
        if key in options and value not in options[key]:
            continue            # not served here — leave the widget as the user set it
        st.session_state[key] = value


# ── Sidebar section header helper ────────────────────────────────────────────

def _section(label: str) -> None:
    st.sidebar.markdown(
        f"""
        <div style="font-size:1.25rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.07em;color:#4F1787;margin:1.1rem 0 0.3rem 0;
                    padding-bottom:3px;border-bottom:1px solid #ede9f5;">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Model menu ───────────────────────────────────────────────────────────────
#
# 36 models are served: the 26 single-species and 2 multi-species models of this
# manuscript (_v26J), and the 8 models of Peleke et al., 2024, which stay on the
# platform so the tutorial and the published supplementary figures remain
# reproducible. A flat list of 36 is unusable and lets a reader pick a species
# that a given set does not contain, so the menu is set -> species -> tissue and
# resolves to exactly one file.
#
# models/manifest.tsv drives it (file, version, kind, species_or_set, tissue,
# fold_or_held_out, label, sha256). Without the manifest the menu falls back to
# the plain list of .h5 files, so a deployment that has not got it still works.

MODEL_SETS = [
    ("deepCRE 2026 — single species", "v26J", "single-species"),
    ("deepCRE 2026 — multi-species",  "v26J", "multi-species"),
    ("Peleke et al., 2024",           "2024", "single-species"),
]


@st.cache_data
def _load_model_manifest(path: str = "models/manifest.tsv"):
    try:
        man = pd.read_csv(path, sep="\t")
    except Exception:
        return None
    needed = {"file", "version", "kind", "species_or_set", "tissue"}
    return man if needed.issubset(man.columns) else None


def request_model(file_id: str) -> None:
    """Queue the model menu so that it resolves to this model file.

    Used by the Tutorial tab, which has to select a specific model: the 2024
    models for the steps that reproduce published numbers, the _v26J models for
    the promoter swap. A model this deployment does not serve is ignored.
    """
    man = _load_model_manifest()
    if man is None:
        request_widget_state(sidebar_model=file_id)
        return
    fname = file_id if str(file_id).endswith(".h5") else f"{file_id}.h5"
    row = man[man.file == fname]
    if row.empty:
        return
    row = row.iloc[0]
    label = next((lab for lab, v, k in MODEL_SETS
                  if v == row.version and k == row.kind), None)
    if label is None:
        return
    request_widget_state(sidebar_model_set=label,
                         sidebar_species=row.species_or_set,
                         sidebar_tissue=row.tissue)


def _model_picker(available_models: list) -> str:
    """Return the model id (file name without .h5) the user selected."""
    man = _load_model_manifest()
    if man is None:
        return st.sidebar.selectbox(
            label="Model", options=available_models,
            help="models/manifest.tsv not found — showing every model file.",
            key="sidebar_model",
        )

    labels = [lab for lab, v, k in MODEL_SETS
              if not man[(man.version == v) & (man.kind == k)].empty]
    chosen = st.sidebar.radio("Model set", options=labels, key="sidebar_model_set",
                              help="Models of this manuscript, or the published "
                                   "Peleke et al., 2024 models the tutorial uses.")
    version, kind = next((v, k) for lab, v, k in MODEL_SETS if lab == chosen)
    subset = man[(man.version == version) & (man.kind == kind)]

    species = st.sidebar.selectbox(
        "Species" if kind == "single-species" else "Training set",
        options=sorted(subset.species_or_set.unique()), key="sidebar_species")
    rows = subset[subset.species_or_set == species]

    tissues = sorted(rows.tissue.unique())
    tissue = st.sidebar.radio("Tissue", options=tissues, horizontal=True,
                              key="sidebar_tissue")
    row = rows[rows.tissue == tissue].iloc[0]

    st.sidebar.caption(f"→ `{row.file}`")
    return row.file[:-3] if str(row.file).endswith(".h5") else row.file


# ── File validation ───────────────────────────────────────────────────────────

def _validate_file(file, label: str):
    """Return the file if non-empty, otherwise show an error and return None."""
    if file.size > 0:
        return file
    st.sidebar.error(f"⚠️ The uploaded **{label}** file is empty. Please verify.")
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
    _apply_pending(available_species, available_models)

    # ── Branding ──────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div style="text-align:center;padding:0.6rem 0 0.2rem 0;">
            <span style="font-size:2rem;"/span>
            <div style="font-size:1.05rem;font-weight:700;color:#4F1787;margin-top:2px;">deepCRE</div>
            <div style="font-size:0.72rem;color:#9ca3af;">CRE expression predictor</div>
        </div>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:0.7rem 0;">
        """,
        unsafe_allow_html=True,
    )

    # ── Species / genome ──────────────────────────────────────────────────────
    _section("Reference Organism")
    selected_organism = st.sidebar.selectbox(
        label="Species",
        options=available_species,
        help='Select a pre-loaded genome or choose "New" to upload your own.',
        key="sidebar_organism",
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
    _section("📂 Gene List")
    genes_upload = st.sidebar.file_uploader(
        label="Gene IDs (.csv / .txt)",
        type=[".csv", ".txt"],
        accept_multiple_files=False,
        help="One gene ID per line, up to 1 000 genes. Excess genes are ignored.",
    )

    genes_list = None
    use_example = False

    # Typing one or a few gene IDs is the common case -- looking at a single
    # gene of interest -- and writing a file for it is needless work. The
    # placeholder names a gene that exists in the tutorial genome, the Rubisco
    # small subunit RBCS1A, so the box can be tried without looking anything up.
    typed = st.sidebar.text_input(
        label="Gene IDs",
        placeholder="AT1G67090",
        key="sidebar_typed_genes",
        help="Type one or more gene IDs, separated by comma, space or new line. "
             "Example: AT1G67090 (RBCS1A, Rubisco small subunit 1A, A. thaliana). "
             "An uploaded file takes precedence over this box.",
    )
    typed_genes = [g.strip() for g in re.split(r"[,\s;]+", typed or "") if g.strip()]

    if genes_upload is not None:
        validated = _validate_file(genes_upload, "genes list")
        if validated is not None:
            genes_list = (
                pd.read_csv(StringIO(validated.getvalue().decode("utf-8")), header=None)
                .values.ravel()
                .tolist()
            )
        if typed_genes:
            st.sidebar.info("Using the uploaded file; the typed gene IDs are ignored.")
    elif typed_genes:
        genes_list = typed_genes[:1000]
        if len(typed_genes) > 1000:
            st.sidebar.warning("Only the first 1 000 gene IDs are used.")
        st.sidebar.caption(
            f"{len(genes_list)} gene ID{'s' if len(genes_list) != 1 else ''} entered")
    else:
        use_example = st.sidebar.checkbox(
            "Use 100 random genes from the genome",
            value=False,
            help="Randomly sample 100 genes from the selected genome for a quick demo.",
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    _section("deepCRE Model")
    selected_model = _model_picker(available_models)

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
