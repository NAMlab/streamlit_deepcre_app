import streamlit as st
import numpy as np
import pandas as pd

def show_about_tab(available_genomes):
    _, abt_col, _ = st.columns([0.15, 0.7, 0.15])
    with abt_col:
        about_header, _ = st.columns([0.2, 0.8])
        with about_header:
            st.subheader('Abstract')
        st.write("""
        The presence of short cis-regulatory elements (CREs) significantly defines the relationship between the 
        expression of a gene and its non-coding proximal regulatory sequences. Deep learning models have been 
        developed to facilitate the elucidation of this relationship, therefore providing us a template to further 
        investigate the effects of other factors, such as, for example, naturally occurring variations on 
        gene expression. Here we present an interactive web-based tool that provides users access to 
        trained deep learning models to predict the expression of a gene and interpret model predictions. 
        Its user-friendly interface also offers users the opportunity to investigate the effect of mutations on
        gene expression in silico, either by uploading variant call format (VCF) files or by manual editing
        of the cis-regulatory sequences.
        """)

        st.write("\n\n\n")

        tutorial_header, _ = st.columns([0.8, 0.2])
        with tutorial_header:
            st.subheader("""
            Tutorial on the deepCRE toolkit:
            """)
        st.write("**Reproducing the deepCRE results sections of the gene promoter characterization**")
        st.write("""
        The aim of the Tutorial is to familiarize the users with its functions and potential applications.
        The code for this toolkit: https://github.com/NAMlab/streamlit_deepcre_app" 
        """)

        st.write("""**Contents**""")
        st.write("""
        1. Providing a query for prediction and selecting a deepCRE model\n
        2. Accessing the deepCRE Prediction Results\n
        3. Accessing the deepCRE models Explanation Results\n
        4. Mutating gene sequence and measuring effects""")

        st.write("**1. Providing a query and selecting a deepCRE model**")
        st.write("""
        **1.1** The deepCRE toolkit requires a list of gene ids from one available reference organism. 
        The available reference organisms and models are shown on the Home Tab under Available genomes or in 
        Supplementary table 1.
        """)
        ## Image 1
        st.image('images/Slide1.jpg', use_column_width=True)

        st.write("""
        Click on “Browse files” and upload Supplementary-file-1 containing a list of matching gene ids 
        for A. thaliana TAIR10. The gene ids in the file do not require version numbers and are provided in rows.
        """)
        st.write("""
        AT1G67090\n
        AT2G22200\n
        AT2G37620\n
        AT2G46800\n
        AT3G02150\n
        """)
        st.write("Alternatively, the user can always select 100 random genes from a selected genome.")
        st.write("""
        Other reference genomes and annotations for non-reference organisms can be uploaded by selecting a “New” species. 
        The user needs to upload a matching genome, annotation files and a list of gene ids. For the upload please use
        gzip compressed files (.gz). 
        """)

        st.write("""
        **1.2**  The deepCRE toolkit provides pre-trained deep learning models described in Peleke et al., 2024. 
        The user can select these independent of the selected Species references.
        """)
        st.write("""
        Select the Arabidopsis_thaliana_leaf and later the Arabidopsis_thaliana_root models to reproduce the results.
        """)
        st.write("""
        The selected queries will be processed accordingly and predictions of gene expression levels will be modelled 
        automatically after the selections are complete. This may take a few seconds. 
        """)

        st.write("**2. Accessing the deepCRE Prediction Results**")
        st.write("""
        After the user has provided queries for analyses, the prediction results can be accessed in the tab "Predictions".
        The toolkit provides tabular and graphical output for the query genes, mainly highlighting genes that are 
        predicted to have low and high rates of transcription (pink and purple).
        """)
        ## Image 2
        st.image('images/Slide2.jpg', use_column_width=True)


        st.write("""All tables and figures can be downloaded.""")
        ## Table
        st.dataframe(pd.read_csv('data/Tutorial_table1_Atleaf.csv', nrows=5))

        st.write("""
        The toolkit should have produced figure 2a and 2c. Options to further process the output should become visible
        by mouse-over. Please save outputs by clicking on the options.
        """)
        ## Image 3
        st.image('images/Slide3.jpg', use_column_width=True)

        st.write("""
        During the Predictions the chosen deepCRE model can be changed, without the query being lost. Please change
        the deepCRE model from Arabidopsis_thaliana_leaf to the Arabidopsis_thaliana_root.
        """)
        ## Image 4: table
        st.dataframe(pd.read_csv('data/Tutorial_table2_Atroot.csv', nrows=5))
        st.write("""
        The toolkit should have produced figure 2b and 2d. 
        The deepCRE toolkit provides more figures than shown in the results. The users have access to multiple graphical
        output showing the analyses results:\n
        Distribution of genes across the genome\n
        Distribution of Low and High predictions across chromosomes,\n
        Distribution of Predicted probabilities\n
        Gene size vs Predicted probabilities\n
        GC content vs Predicted probabilities\n 

        """)

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
        ## Image 5
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
        ## Image 6
        st.image('images/Slide5.jpg', use_column_width=True)
        st.write("""
        To reproduce the results of the gene promoter characterization please select the Manual editing mode, gene of 
        interest AT1G67090, and the 5’UTR (gTUR) region as region of interest. 
        """)
        ## Image 7
        st.image('images/Slide6.jpg', use_column_width=True)
        st.write("""
        The coordinates can be changed that will be on display within the Text editing window after clicking on Submit. 
        After the sequence has been edited, changes are confirmed by clicking onto Mutate. \n
        The coordinates should be set to  1001-1500 after selecting the 5’UTR (gTUR) region. Please open the 
        Supplementary file 2 and copy the sequence of the fasta:
        """)
        st.write("""
        \>gTUR_Osativa_OsACT1_KP100426-PIG2_5UTR500BP\n
        GCCCTCCCTCCGCTTCCAAAGAAACGCCCCCCATCGCCACTATATACATACCCCCCCTCTCCTCCCATCCCCCAACCCTACCACCACCACCACCACCACCTCCACCTCCTCC
        CCCCTCGCTGCCGGACGACGAGCTCCTCCCCCCTCCCCCTCCGCCGCCGCCGCGCCGGTAACCACCCCGCCCCTCTCCTCTTTCTTTCTCCGTTTTTTTTTTCCGTCTCGGT
        CTCGATCTTTGGCCTTGGTAGTTTGGGTGGGCGAGAGGCGGCTTCGTGCGCGCCCAGATCGGTGCGCGGGAGGGGCGGGATCTCGCGGCTGGGGCTCTCGCCGGCGTGGATC
        CGGCCCGGATCTCGCGGGGAATGGGGCTCTCGGATGTAGATCTGCGATCCGCCGTTGTTGGGGGAGATGATGGGGGGTTTAAAATTTCCGCCATGCTAAACAAGATCAGGAA
        GAGGGGAAAAGGGCACTATGGTTTATATTTTTATATATTTCTGCTGCTTCGT
        """)

        st.write("""
        Paste this sequence into the target window for text editing of the deepCRE toolkit Mutation mode. 
        After clicking onto Mutate new probabilities and saliency maps should be generated. The exchange of the gTUR 
        should result in the generation of figure 3g and 3h.
        """)
        ## Image 8
        st.image('images/Slide7.jpg', use_column_width=True)

        st.write("""
        The change in predicted probabilities is displayed below the plots. The exact predicted probability for the 
        sequence before (grey) and after (cyan) editing can be read out by mouse-over the barplot. To reproduce the 
        results shown in the deepCRE toolkit manuscript, sequences in the supplementary file 2 were trimmed to sizes that
        can be copied to the webtool. To enable cross evaluation, the different models can be selected without the edited 
        sequence being changed. This accounts also for changes in the other selectable regions of interest. 
        Please select the gUR (Promoter) region for manual editing after the gTUR has been mutated. Please open 
        Supplementary file 2 and copy the following sequence:
        """)
        st.write("""
        \>gUR_Osativa_OsACT1_KP100426-PIG2_Promoter1000BP\n
        GTAATTCCATAAAATTTTTAATGTCCATAATTATAATAAAGAACAATGGATATATATACATATATAATAATAACTTATAAAAAAATATAATATTTTTGGAAAAAAAAAGAAT
        AATAATAAAACTTAAATAAAAAAAACCTATATTAAACTTTGTTTTAAAACCTTGCAAAAGATATCATGTTTTACTTATGAGTCATCAAATTGAAGTACAAGTAGGTTATATA
        AGCTTCTAGCATACTCGAGGTCATTCATATGCTTGAGAAGAGAGTCGGGATAGTCCAAAATAAAACAAAGGTAAGATTACCTGGTCAAAAGTGAAAACATCAGTTAAAAGGT
        GGTATAAAGTAAAATATCGGTAATAAAAGGTGGCCCAAAGTGAAATTTACTCTTTTCTACTATTATAAAAATTGAGGATGTTTTTGTCGGTACTTTGATACGTCATTTTTGT
        ATGAATTGGTTTTTAAGTTTATTCGCTTTTGGAAATGCATATCTGTATTTGAGTCGGGTTTTAAGTTCGTTTGCTTTTGTAAATACAGAGGGATTTGTATAAGAAATATCTT
        TAAAAAAACCCATATGCTAATTTGACATAATTTTTGAGAAAAATATATATTCAGGCGAATTCTCACAATGAACAATAATAAGATTAAAATAGCTTTCCCCCGTTGCAGCGCA
        TGGGTATTTTTTCTAGTAAAAATAAAAGATAAACTTAGACTCAAAACATTTACAAAAACAACCCCTAAAGTTCCTAAAGCCCAAAGTGCTATCCACGATCCATAGCAAGCCC
        AGCCCAACCCAACCCAACCCAACCCACCCCAGTCCAGCCAACTGGACAATAGTCTCCACACCCCCCCACTATCACCGTGAGTTGTCCGCACGCACCGCACGTCTCGCAGCCA
        AAAAAAAAAAAAGAAAGAAAAAAAAGAAAAAGAAAAAACAGCAGGTGGGTCCGGGTCGTGGGGGCCGGAAACGCGAGGAGGATCGCGAGCCAGCGACGAGGCCG
        """)

        st.write("""
        Paste this sequence into the target window for text editing of the deepCRE toolkit Mutation mode. 
        After clicking onto Mutate new probabilities and saliency maps should be generated.
        """)
        ## Image 8
        st.image('images/Slide8.jpg', use_column_width=True)

        st.write("""
        The sequence of display now consists of the OsACT1 gUT and gTUR regions combined with the downstream region of 
        gene AT1G67090.\n
        All changes to a sequence can be resetted by clicking on Reset.\n 
        More exemplary material for in silico promoter swap experiments are available in Supplementary file 2.\n 
        """)

        st.write("""
        **4.2**  Variant Effect Prediction\n
        In the VCF editing mode the user can upload a variant call file (VCF) as GNUzipped (.gz) within the webtool and 
        evaluate changes in the predicted probability. This allows the user to analyze variant effects over e.g. population 
        structure. We provide an exemplary vcf file as Supplementary file 4 that contains all variants found in the ecotypes
        analyzed by Luo and colleagues. Please switch to the VCF mode within the toolkit and follow the instructions.\n 
        Please select a new gene of interest for this study: AT1G53910 (RAP2.12). 

        """)
        ## Image 8
        st.image('images/Slide9.jpg', use_column_width=True)
        st.write("""
        The toolkit displays the first 50 variants of the uploaded vcf file and all variants found within the selected
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

        avail_gen, _ = st.columns([0.9, 0.1])
        prev_page, _, next_page = st.columns([1, 10, 1])
        n_rows_to_show = 5
        n_genomes = len(available_genomes)
        last_page = np.ceil(n_genomes/n_rows_to_show)-1
        if 'page_number' not in st.session_state:
            st.session_state.page_number = 0


        with avail_gen:
            st.subheader('Available genomes', divider='grey')
            st.write('If you have a novel genome or assembly, please select "New" in the Species field to the left.')

        if next_page.button("⇒", type='primary'):
            if st.session_state.page_number + 1 > last_page:
                st.session_state.page_number = 0
            else:
                st.session_state.page_number += 1

        if prev_page.button("⇐", type='primary'):

            if st.session_state.page_number - 1 < 0:
                st.session_state.page_number = last_page
            else:
                st.session_state.page_number -= 1

        curr_idx = int(st.session_state.page_number * n_rows_to_show)
        end_idx = int((1 + st.session_state.page_number) * n_rows_to_show)

        # Index into the sub dataframe
        sub_df = available_genomes.iloc[curr_idx:end_idx]
        sub_df = sub_df[['display_name', 'description']]
        sub_df.columns = ['Genome', 'Description']
        st.dataframe(sub_df, hide_index=True, use_container_width=True)