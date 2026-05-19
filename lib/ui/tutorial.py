import os
import io
import zipfile
import streamlit as st
import pandas as pd

# ── Helper Functions ─────────────────────────────────────────────────────────

def get_tutorial_zip():
    """Creates a zip archive of the tutorial files in memory."""
    zip_buffer = io.BytesIO()
    tutorial_dir = "tutorial"
    
    files_to_zip = [
        "File_1_deepCRE_tutorial_gene_IDs_Supplementary-file-1_dCRE-Pelekeetal2025.txt",
        "File_2_deepCRE_tutorial_promoter_sequences_Supplementary-file-2_dCRE-Pelekeetal2025.txt",
        "File_3_deepCRE_tutorial_rap12-2-variant_Supplementary-file-3_dCRE_Peleketal2025.vcf.gz",
        "README.md"
    ]
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_name in files_to_zip:
            file_path = os.path.join(tutorial_dir, file_name)
            if os.path.exists(file_path):
                zip_file.write(file_path, arcname=file_name)
                
    return zip_buffer.getvalue()

def load_tutorial_genes():
    """Reads gene IDs directly from File 1."""
    file_path = "tutorial/File_1_deepCRE_tutorial_gene_IDs_Supplementary-file-1_dCRE-Pelekeetal2025.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return ["AT1G67090", "AT2G22200", "AT2G37620", "AT2G46800", "AT3G02150"]

def load_tutorial_fasta(target_header):
    """Reads a specific sequence from the FASTA File 2."""
    file_path = "tutorial/File_2_deepCRE_tutorial_promoter_sequences_Supplementary-file-2_dCRE-Pelekeetal2025.txt"
    seq = ""
    capture = False
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if target_header in line:
                        capture = True
                    elif capture:
                        break  # Found the next header, stop capturing
                elif capture:
                    seq += line
                    
    # Fallback sequence just in case the file goes missing
    if not seq:
        seq = "GCCCTCCCTCCGCTTCCAAAGAAACGCCCCCCATCGCCACTATATACATACCCCCCCTCTCCTCCCATCCCCCAACCCTACCACCACCACCACCACCACCTCCACCTCCTCCCCCCTCGCTGCCGGACGACGAGCTCCTCCCCCCTCCCCCTCCGCCGCCGCCGCGCCGGTAACCACCCCGCCCCTCTCCTCTTTCTTTCTCCGTTTTTTTTTTCCGTCTCGGTCTCGATCTTTGGCCTTGGTAGTTTGGGTGGGCGAGAGGCGGCTTCGTGCGCGCCCAGATCGGTGCGCGGGAGGGGCGGGATCTCGCGGCTGGGGCTCTCGCCGGCGTGGATCCGGCCCGGATCTCGCGGGGAATGGGGCTCTCGGATGTAGATCTGCGATCCGCCGTTGTTGGGGGAGATGATGGGGGGTTTAAAATTTCCGCCATGCTAAACAAGATCAGGAAGAGGGGAAAAGGGCACTATGGTTTATATTTTTATATATTTCTGCTGCTTCGT"
    return seq

# ── Main UI Function ─────────────────────────────────────────────────────────

def show_tutorial_tab():
    _, tut_col, _ = st.columns([0.15, 0.7, 0.15])
    
    with tut_col:
        tutorial_header, _ = st.columns([0.8, 0.2])
        with tutorial_header:
            st.subheader("Tutorial on the deepCRE toolkit:")
            
        st.write("**Reproducing the deepCRE results sections of the gene promoter characterization**")
        st.write("""
        The aim of the Tutorial is to familiarize the users with its functions and potential applications.
        The code for this toolkit: https://github.com/NAMlab/streamlit_deepcre_app
        """)

        # --- DOWNLOAD BUTTON SECTION ---
        st.markdown("### 📥 Download Tutorial Files")
        st.write("You can download all supplementary files required for this tutorial as a single ZIP archive.")
        
        zip_data = get_tutorial_zip()
        st.download_button(
            label="Download All Tutorial Files (.zip)",
            data=zip_data,
            file_name="deepCRE_tutorial_files.zip",
            mime="application/zip",
            type="primary"
        )
        st.markdown("---")

        st.write("""**Contents**""")
        st.write("""
        1. Providing a query for prediction and selecting a deepCRE model\n
        2. Accessing the deepCRE Prediction Results\n
        3. Accessing the deepCRE models Explanation Results\n
        4. Mutating gene sequence and measuring effects""")

        st.write("\n---\n")

        st.write("**1. Providing a query and selecting a deepCRE model**")
        st.write("""
        **1.1** The deepCRE toolkit requires a list of gene ids from one available reference organism. 
        The available reference organisms and models are shown on the Home Tab under Available genomes.
        """)
        
        st.image('images/Slide1.jpg', use_column_width=True)

        st.write("""
        Click on “Browse files” and upload File 1 from the Tutorial containing a list of matching gene ids 
        for A. thaliana TAIR10. The gene ids in the file do not require version numbers and are provided in rows.
        """)
        
        preview_genes = load_tutorial_genes()[:5]
        
        # Format each gene as a proper Markdown bullet point, then join with newlines
        formatted_genes = "\n".join([f"* {gene}" for gene in preview_genes]) + "\n* ..."
        st.markdown(formatted_genes)

        # --- INTERACTIVE DEMO BUTTON 1 ---
        st.markdown("---")
        st.markdown("**Try it out instantly:**")
        if st.button("Load Tutorial Genes & Set Model", type="primary"):
            st.session_state.tutorial_active = True
            st.session_state.tutorial_genes = load_tutorial_genes()
            st.session_state.tutorial_model = "Arabidopsis_thaliana_leaf"
            st.success("✅ Tutorial data loaded! Click the **Predictions** tab above to see the results.")
            
        if st.session_state.get("tutorial_active"):
            if st.button("❌ Clear Tutorial Data", type="secondary", key="clear_genes_btn"):
                st.session_state.tutorial_active = False
                st.rerun()
        st.markdown("---")

        st.write("Alternatively, the user can always select 100 random genes from a selected genome in the sidebar.")
        
        st.write("**2. Accessing the deepCRE Prediction Results**")
        st.write("""
        After the user has provided queries for analyses, the prediction results can be accessed in the tab "Predictions".
        The toolkit provides tabular and graphical output for the query genes, mainly highlighting genes that are 
        predicted to have low and high rates of transcription (pink and purple).
        """)
        st.image('images/Slide2.jpg', use_column_width=True)

        st.write("""All tables and figures can be downloaded.""")
        if os.path.exists('data/Tutorial_table1_Atleaf.csv'):
            st.dataframe(pd.read_csv('data/Tutorial_table1_Atleaf.csv', nrows=5))

        st.write("""
        The toolkit should have produced figures. Options to further process the output should become visible
        by mouse-over. Please save outputs by clicking on the options.
        """)
        st.image('images/Slide3.jpg', use_column_width=True)

        st.write("""
        During the Predictions the chosen deepCRE model can be changed, without the query being lost. Please change
        the deepCRE model from Arabidopsis_thaliana_leaf to the Arabidopsis_thaliana_root.
        """)
        if os.path.exists('data/Tutorial_table2_Atroot.csv'):
            st.dataframe(pd.read_csv('data/Tutorial_table2_Atroot.csv', nrows=5))
            
        st.write("""
        The toolkit should have produced figures. 
        The deepCRE toolkit provides more figures than shown in the results. The users have access to multiple graphical
        output showing the analyses results:\n
        Distribution of genes across the genome\n
        Distribution of Low and High predictions across chromosomes,\n
        Distribution of Predicted probabilities\n
        Gene size vs Predicted probabilities\n
        GC content vs Predicted probabilities\n 
        """)

        st.write("\n---\n")

        st.write('**3. Accessing the deepCRE Explanation Results**')
        st.write("""
        Model interpretations are done using the DeepSHAP/DeepExplainer implementation of (Lundberg & Lee, 2017)
        which computes nucleotide resolution importance scores, highlighting the most salient features of every cis-regulatory
        sequence. These scores are averaged across all genes within the provided list of genes, providing users with an 
        averaged saliency map.
        """)
        st.write("""
        The users can access saliency maps by clicking on the Tab “Saliency Maps”. This is how the user can produce figure
        2e and 2f, switching between the At(leaf) and At(root) models.
        """)
        st.image('images/Slide4.jpg', use_column_width=True)

        st.write("""
        The deepCRE toolkit provides more figures than shown in the results. The users have access to graphical output:\n
        Averaged saliency map\n
        Sum saliency score vs Predicted probabilities\n
        Base-type average saliency map for highly expressed genes\n
        Sum saliency score for highly expressed genes\n
        Base-type average saliency map for lowly expressed genes\n
        Sum saliency score for lowly expressed gene\n
        """)

        st.write("\n---\n")

        st.write("""**4. Mutating gene sequence and measuring effects**""")
        st.write("""
        The "Mutation Analysis" tab provides users the opportunity to specifically edit input sequences and measure changes in predicted probabilities 
        using manual or vcf guided mutations. The user can switch between the two modes of analysis. In both modes 
        users can select a gene of interest. 
        """)
        st.write("**4.1**  Promoter Swaps")
        st.write("""
        In the Manual editing mode the user can display and edit sequences within the webtool. This allows the user to 
        manually change, e.g. copy-paste sequences from different sources and compare the effects to the query sequence 
        measured by change in predicted probability and saliency maps. 
        """)
        st.image('images/Slide5.jpg', use_column_width=True)
        st.write("""
        To reproduce the results of the gene promoter characterization please select the Manual editing mode, gene of 
        interest AT1G67090, and the 5’UTR (gTUR) region as region of interest. 
        """)
        st.image('images/Slide6.jpg', use_column_width=True)
        st.write("""
        The coordinates can be changed that will be on display within the Text editing window after clicking on Submit. 
        After the sequence has been edited, changes are confirmed by clicking onto Mutate. \n
        The coordinates should be set to  1001-1500 after selecting the 5’UTR (gTUR) region. Please open the 
        Tutorial File 2 and copy the sequence of the fasta to the gTUR section:
        """)
        
        st.write("`>gTUR_Osativa_OsACT1_KP100426-PIG2_5UTR500BP`")
        fasta_seq = load_tutorial_fasta("gTUR_Osativa_OsACT1_KP100426-PIG2_5UTR500BP")
        st.write(f"*{fasta_seq[:100]}...*")

        # --- INTERACTIVE DEMO BUTTON 2 ---
        st.markdown("---")
        st.markdown("**Try it out instantly:**")
        if st.button("Load OsACT1 gTUR Sequence into Mutation Tab", type="primary"):
            st.session_state.tutorial_mutation_active = True
            st.session_state.tutorial_mut_seq = fasta_seq
            
            # Set the exact UI state for the Mutation tab
            st.session_state.analysis_mode_radio = "**manual**"
            st.session_state.mut_region_selection = "gTUR"
            st.session_state.mut_coord_slider = (1001, 1500)
            
            st.success("✅ Sequence loaded! Click the **Mutation** tab above, ensure gene **AT1G67090** is selected, and your sequence will be ready in the text box.")
            
        if st.session_state.get("tutorial_mutation_active"):
            if st.button("❌ Clear Mutation Data", type="secondary", key="clear_mut_btn"):
                st.session_state.tutorial_mutation_active = False
                st.rerun()
        st.markdown("---")

        st.write("""
        Paste this sequence into the target window for text editing of the deepCRE toolkit Mutation mode. 
        After clicking onto Mutate new probabilities and saliency maps should be generated. The exchange of the gTUR 
        should result in the generation of figure 3g and 3h.
        """)
        st.image('images/Slide7.jpg', use_column_width=True)

        st.write("""
        The change in predicted probabilities is displayed below the plots. The exact predicted probability for the 
        sequence before (grey) and after (cyan) editing can be read out by mouse-over the barplot. To reproduce the 
        results shown in the deepCRE toolkit manuscript, sequences in the supplementary file 2 were trimmed to sizes that
        can be copied to the webtool. To enable cross evaluation, the different models can be selected without the edited 
        sequence being changed. This accounts also for changes in the other selectable regions of interest. 
        """)
        
        st.write("""
        **4.2**  Variant Effect Prediction\n
        In the VCF editing mode the user can upload a variant call file (VCF) as GNUzipped (.gz) within the webtool and 
        evaluate changes in the predicted probability. This allows the user to analyze variant effects over e.g. population 
        structure. We provide an exemplary vcf file as Tutorial File 3 that contains all variants found in the ecotypes
        analyzed by Luo and colleagues. Please switch to the VCF mode within the toolkit and follow the instructions.\n 
        Please select a new gene of interest for this study: AT1G53910 (RAP2.12). 
        """)
        # --- INTERACTIVE DEMO BUTTON 3 ---
        st.markdown("---")
        st.markdown("**Try it out instantly:**")
        if st.button("Load VCF File & Switch to RAP2.12", type="primary"):
            st.session_state.tutorial_vcf_active = True
            st.session_state.analysis_mode_radio = "**VCF**"
            st.session_state.tutorial_vcf_gene = "AT1G53910" 
            
            # (I removed the auto-mutate flag here!)
            
            st.success("✅ VCF loaded! Click the **Mutation** tab above to select your SNPs.")

        st.markdown("---")
        st.image('images/Slide9.jpg', use_column_width=True)
        st.write("""
        The toolkit displays the uploaded vcf file and all variants found within the selected
        gene regions. From the latter, distinct variants can be tagged and will be displayed in a thief table containing
        your variant selection. Please select all variants available for AT1G53910 by ticking the box above the selection
        column. After loading, please click on Mutate Sequence to perform predictions and explanation for gene variants. 
        """)

        st.image('images/Slide10.png', use_column_width=True)

        st.write("""
        This will generate a plot as output showing the change in predicted probability and the effect on single nucleotide
        importances. The dotted grey lines indicate the position of selected variants within the gene flanking regions. 
        The sequence with the lowest predicted probability belongs to the A. thaliana ecotype I-Cat0. These are the variants
        found for this ecotype compared to A. thaliana col-0. The list of variants is provided as supplementary table 3.
        Please select the following SNPs to generate a sequence similar to the I-Cat0 haplotype. 
        """)
        if os.path.exists('data/Tutorial_table3_icat.csv'):
            st.dataframe(pd.read_csv('data/Tutorial_table3_icat.csv'))
            
        st.write("The selection of the 17 SNPs of I-Cat0 results in a decrease of predicted probabilities of 12%")
        st.image('images/Slide11.jpg', use_column_width=True)
        st.write("""
        The change in predicted probabilities can also be explained with just 9 SNPs of I-Cat0 resulting in a decrease 
        of predicted probabilities of 14%. Please remove the tick from all rows that are tagged as “no” contributors in 
        the table above and click onto mutate.
        """)
        st.image('images/Slide12.jpg', use_column_width=True)
        st.write("""
        The resulting plots should be similar to Figure 4b,c and d.
        """)
