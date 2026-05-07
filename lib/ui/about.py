import numpy as np
import pandas as pd
import streamlit as st

# ── Style helpers ─────────────────────────────────────────────────────────────

def _section(label: str, icon: str = "") -> None:
    prefix = f"{icon} " if icon else ""
    st.markdown(
        f"""<div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.07em;color:#4F1787;margin:1.6rem 0 0.5rem 0;
                        padding-bottom:5px;border-bottom:2px solid #ede9f5;">
                {prefix}{label}
            </div>""",
        unsafe_allow_html=True,
    )


def _step_badge(n: int, label: str) -> None:
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:10px;margin:1.2rem 0 0.4rem 0;">
                <div style="background:#4F1787;color:#fff;border-radius:50%;width:28px;height:28px;
                            display:flex;align-items:center;justify-content:center;
                            font-size:0.82rem;font-weight:700;flex-shrink:0;">{n}</div>
                <span style="font-weight:700;font-size:0.97rem;color:#111827;">{label}</span>
            </div>""",
        unsafe_allow_html=True,
    )


def _callout(text: str, color: str = "#f0fdf4", border: str = "#22c55e",
             text_color: str = "#166534") -> None:
    st.markdown(
        f"""<div style="background:{color};border-left:4px solid {border};border-radius:6px;
                        padding:0.7rem 1rem;font-size:0.86rem;color:{text_color};margin:0.5rem 0;">
                {text}
            </div>""",
        unsafe_allow_html=True,
    )


def _code_block(text: str) -> None:
    st.markdown(
        f"""<pre style="background:#f8f9fa;border:1px solid #e5e7eb;border-radius:6px;
                        padding:0.8rem 1rem;font-size:0.82rem;overflow-x:auto;
                        color:#374151;line-height:1.6;">{text}</pre>""",
        unsafe_allow_html=True,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def show_about_tab(available_genomes: pd.DataFrame) -> None:
    _, centre, _ = st.columns([0.1, 0.8, 0.1])

    with centre:
        # ── Abstract ─────────────────────────────────────────────────────────
        _section("Abstract", "📄")
        st.markdown(
            """
            Short **cis-regulatory elements (CREs)** significantly shape the relationship between a gene's
            non-coding proximal regulatory sequences and its expression. Deep learning models have been developed
            to elucidate this relationship, providing a template to investigate the effects of naturally occurring
            variation on gene expression.

            **deepCRE** is an interactive web tool that gives users access to trained deep learning models to
            predict gene expression and interpret model predictions. Its interface allows investigation of the
            effect of mutations in silico — either by uploading Variant Call Format (VCF) files or by
            manually editing cis-regulatory sequences.
            """
        )

        # ── Tutorial ─────────────────────────────────────────────────────────
        _section("Tutorial", "📚")
        st.markdown("**Reproducing the deepCRE results — gene promoter characterisation**")
        _callout(
            "This tutorial familiarises users with deepCRE's functions and potential applications. "
            "Source code: <a href='https://github.com/NAMlab/streamlit_deepcre_app' "
            "style='color:#4F1787;'>github.com/NAMlab/streamlit_deepcre_app</a>",
            color="#f5f3ff", border="#4F1787", text_color="#3b0764",
        )

        # ── Step 1 ────────────────────────────────────────────────────────────
        _step_badge(1, "Provide a query and select a deepCRE model")
        st.markdown(
            """
            deepCRE requires a list of gene IDs from one of the available reference organisms
            (see *Available Genomes* below, or Supplementary Table 1).
            """
        )
        st.image("images/Slide1.jpg", use_column_width=True)
        st.markdown(
            "Click **Browse files** and upload *Supplementary File 1* containing gene IDs for "
            "*A. thaliana* TAIR10. Each gene ID should appear on its own line, without version numbers:"
        )
        _code_block("AT1G67090\nAT2G22200\nAT2G37620\nAT2G46800\nAT3G02150")
        st.markdown(
            "Alternatively, tick **Use 100 random genes from the genome** to test the tool without uploading data. "
            "Novel genomes and annotations can be uploaded by selecting **New** in the Species field — use "
            "gzip-compressed files (.gz)."
        )
        st.markdown(
            """
            **1.2** Select a pre-trained deepCRE model (Peleke et al., 2024). Choose
            *Arabidopsis_thaliana_leaf* and later *Arabidopsis_thaliana_root* to reproduce the paper's results.
            Predictions are generated automatically once your selections are complete.
            """
        )

        # ── Step 2 ────────────────────────────────────────────────────────────
        _step_badge(2, "Access Prediction Results")
        st.markdown(
            "Navigate to the **Predictions** tab to view tabular and graphical summaries of gene expression "
            "predictions (high = pink, low = purple). All tables and figures can be downloaded."
        )
        st.image("images/Slide2.jpg", use_column_width=True)
        st.dataframe(pd.read_csv("data/Tutorial_table1_Atleaf.csv", nrows=5), hide_index=True)
        st.image("images/Slide3.jpg", use_column_width=True)
        st.markdown(
            "You can change the deepCRE model mid-session without losing your gene query. "
            "Switch to *Arabidopsis_thaliana_root* to reproduce figures 2b and 2d:"
        )
        st.dataframe(pd.read_csv("data/Tutorial_table2_Atroot.csv", nrows=5), hide_index=True)
        _callout(
            "Additional plots available: chromosome gene distribution, High/Low expression class distribution, "
            "probability histogram, gene size vs. probability, GC content vs. probability."
        )

        # ── Step 3 ────────────────────────────────────────────────────────────
        _step_badge(3, "Access Explanation Results")
        st.markdown(
            """
            Model interpretations use the **DeepSHAP / DeepExplainer** implementation (Lundberg & Lee, 2017),
            computing nucleotide-resolution importance scores averaged across all queried genes.
            Navigate to the **Explanation** tab to reproduce figures 2e and 2f by switching between the
            Arabidopsis leaf and root models.
            """
        )
        st.image("images/Slide4.jpg", use_column_width=True)
        _callout(
            "Available explanation plots: averaged saliency map, sum saliency vs. predicted probability, "
            "base-type saliency map (high & low), sum saliency per base."
        )

        # ── Step 4 ────────────────────────────────────────────────────────────
        _step_badge(4, "Mutate sequences and measure effects")
        st.markdown(
            "The **Mutation** tab lets you edit input sequences and observe changes in predicted probabilities "
            "— either via manual editing or VCF-guided mutations."
        )

        st.markdown("**4.1 Promoter Swaps (Manual mode)**")
        st.markdown(
            "Select a gene and region of interest. The sequence is displayed in an editable text window. "
            "Paste a replacement sequence and click **Mutate** to compare predictions."
        )
        st.image("images/Slide5.jpg", use_column_width=True)
        st.markdown(
            "To reproduce the promoter characterisation results, select gene **AT1G67090** and the "
            "**5′UTR** region (coordinates 1001–1500). Paste the sequence from Supplementary File 2:"
        )
        _code_block(
            ">gTUR_Osativa_OsACT1_KP100426-PIG2_5UTR500BP\n"
            "GCCCTCCCTCCGCTTCCAAAGAAACGCCCCCCATCGCCACTATATACATACCCCCCCTCTCCTCCCATCCCCCAACCCTACCACCACCACCACC\n"
            "ACCACCTCCACCTCCTCCCCCCTCGCTGCCGGACGACGAGCTCCTCCCCCCTCCCCCTCCGCCGCCGCCGCGCCGGTAACCACCCCGCCCCTCT\n"
            "CCTCTTTCTTTCTCCGTTTTTTTTTTCCGTCTCGGTCTCGATCTTTGGCCTTGGTAGTTTGGGTGGGCGAGAGGCGGCTTCGTGCGCGCCCAGA\n"
            "TCGGTGCGCGGGAGGGGCGGGATCTCGCGGCTGGGGCTCTCGCCGGCGTGGATCCGGCCCGGATCTCGCGGGGAATGGGGCTCTCGGATGTAGA\n"
            "TCTGCGATCCGCCGTTGTTGGGGGAGATGATGGGGGGTTTAAAATTTCCGCCATGCTAAACAAGATCAGGAAGAGGGGAAAAGGGCACTATGGT\n"
            "TTATATTTTTATATATTTCTGCTGCTTCGT"
        )
        st.image("images/Slide6.jpg", use_column_width=True)
        st.image("images/Slide7.jpg", use_column_width=True)
        st.markdown(
            "After pasting and clicking **Mutate**, new probabilities and saliency maps are generated "
            "(figure 3g and 3h). Next, select the **Promoter** region and paste the sequence below:"
        )
        _code_block(
            ">gUR_Osativa_OsACT1_KP100426-PIG2_Promoter1000BP\n"
            "GTAATTCCATAAAATTTTTAATGTCCATAATTATAATAAAGAACAATGGATATATATACATATATAATAATAACTTATAAAAAAATATAATATTT\n"
            "TTGGAAAAAAAAAGAATAATAATAAAACTTAAATAAAAAAAACCTATATTAAACTTTGTTTTAAAACCTTGCAAAAGATATCATGTTTTACTTAT\n"
            "GAGTCATCAAATTGAAGTACAAGTAGGTTATATAAGCTTCTAGCATACTCGAGGTCATTCATATGCTTGAGAAGAGAGTCGGGATAGTCCAAAAT\n"
            "AAAACAAAGGTAAGATTACCTGGTCAAAAGTGAAAACATCAGTTAAAAGGTGGTATAAAGTAAAATATCGGTAATAAAAGGTGGCCCAAAGTGAAA\n"
            "TTTACTCTTTTCTACTATTATAAAAATTGAGGATGTTTTTGTCGGTACTTTGATACGTCATTTTTGTATGAATTGGTTTTTAAGTTTATTCGCTTTT\n"
            "GGAAATGCATATCTGTATTTGAGTCGGGTTTTAAGTTCGTTTGCTTTTGTAAATACAGAGGGATTTGTATAAGAAATATCTTTAAAAAAACCCATAT\n"
            "GCTAATTTGACATAATTTTTGAGAAAAATATATATTCAGGCGAATTCTCACAATGAACAATAATAAGATTAAAATAGCTTTCCCCCGTTGCAGCGCA\n"
            "TGGGTATTTTTTCTAGTAAAAATAAAAGATAAACTTAGACTCAAAACATTTACAAAAACAACCCCTAAAGTTCCTAAAGCCCAAAGTGCTATCCACGA\n"
            "TCCATAGCAAGCCCAGCCCAACCCAACCCAACCCAACCCACCCCAGTCCAGCCAACTGGACAATAGTCTCCACACCCCCCCACTATCACCGTGAGTTG\n"
            "TCCGCACGCACCGCACGTCTCGCAGCCAAAAAAAAAAAAAGAAAGAAAAAAAAGAAAAAGAAAAAACAGCAGGTGGGTCCGGGTCGTGGGGGCCGGAA\n"
            "ACGCGAGGAGGATCGCGAGCCAGCGACGAGGCCG"
        )
        st.image("images/Slide8.jpg", use_column_width=True)
        _callout(
            "The final sequence combines the OsACT1 gUT and gTUR regions with the downstream region of "
            "AT1G67090. Click <b>↺ Reset sequence</b> to revert all changes. Additional swap sequences are "
            "available in Supplementary File 2."
        )

        st.markdown("**4.2 Variant Effect Prediction (VCF mode)**")
        st.markdown(
            "Upload a gzip-compressed VCF file and select a gene to evaluate the effect of natural variants "
            "on predicted expression. An example VCF (Supplementary File 4) contains all variants across "
            "ecotypes analysed by Luo et al. Select gene **AT1G53910** (RAP2.12) to follow along."
        )
        st.image("images/Slide9.jpg", use_column_width=True)
        st.markdown(
            "Select variants of interest and click **Mutate Sequence** to generate predictions and "
            "saliency maps. Dotted grey lines mark the positions of selected variants."
        )
        st.image("images/Slide10.png", use_column_width=True)
        st.markdown(
            "To reproduce the I-Cat0 haplotype results, select the following 17 SNPs "
            "(results in a ~12 % decrease in predicted probability):"
        )
        st.dataframe(pd.read_csv("data/Tutorial_table3_icat.csv"), hide_index=True)
        _callout(
            "Using only the 9 key SNPs (tagged 'yes' contributors) results in a 14 % decrease. "
            "Remove the remaining rows and click <b>Mutate Sequence</b> again."
        )
        st.image("images/Slide11.jpg", use_column_width=True)
        st.image("images/Slide12.jpg", use_column_width=True)
        st.markdown("The resulting plots reproduce Figure 4b, c and d.")

        # ── Available Genomes ──────────────────────────────────────────────────
        _section("Available Genomes", "🌍")
        st.caption('To use a genome not listed here, select **"New"** in the Species field on the left.')

        n_per_page = 5
        n_genomes  = len(available_genomes)
        last_page  = int(np.ceil(n_genomes / n_per_page) - 1)

        if "page_number" not in st.session_state:
            st.session_state.page_number = 0

        prev_col, _, next_col = st.columns([1, 10, 1])
        with prev_col:
            if st.button("◀", type="primary"):
                st.session_state.page_number = (
                    last_page if st.session_state.page_number - 1 < 0
                    else st.session_state.page_number - 1
                )
        with next_col:
            if st.button("▶", type="primary"):
                st.session_state.page_number = (
                    0 if st.session_state.page_number + 1 > last_page
                    else st.session_state.page_number + 1
                )

        page    = st.session_state.page_number
        sub_df  = available_genomes.iloc[page * n_per_page : (page + 1) * n_per_page]
        display = sub_df[["display_name", "description"]].rename(
            columns={"display_name": "Genome", "description": "Description"}
        )
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.caption(f"Page {page + 1} of {last_page + 1}  ·  {n_genomes} genomes total")
