import streamlit as st
import numpy as np

def show_about_tab(available_genomes):
    _, abt_col, _ = st.columns([0.15, 0.7, 0.15])
    with abt_col:
        about_header, _ = st.columns([0.2, 0.8])
        with about_header:
            st.subheader('About')
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

        tutorial_header, _ = st.columns([0.2, 0.8])
        with tutorial_header:
            st.subheader('Tutorials')
        st.write("""Here is a short tutorial on what you will find on our tool and how to use it. The tool has 4 main
        sections: Home, Predictions, Saliency Maps and Mutation Analysis. At home you find information about the tool,
        this short tutorial, our available genomes and contact information of our lab for questions.
        """)

        st.write('**Predictions**')
        st.write("""In this section, a user can select a genome from our list of available genomes from the dropdown menu in the
                 upload area. Then, they can either load a list of genes belonging to this genome for analysis or use
                 the checkbox to get a list of 100 random genes selected for the analysis. :red[**If this is your first time using our tool
                 you can use a set of 100 random genes to see what we offer**]. Finally the user can select a deepCRE model for the 
                 downstream analysis. By default, the model currently on display is always used for downstream analysis.""")
        st.image('images/image_2.png')
        st.write("""If the user has a new genome, an older or a newer  version of the genomes we have listed, they also have the
        option to upload a new genome. In this case, they also need to upload a new GTF/GFF3 and a list of genes for their analysis.
        Once the new genome and GTF/GFF3 are loaded, you can also check the box to use a set of random genes to see what our
        models predict.""")
        st.image('images/image_1.png')
        st.write("""Once the data is uploaded and the list of genes uploaded or the check box to use random genes is checked. The selected
        model swiftly makes predictions on the cis-regulatory sequences of these genes and provides results to the users.
        1. The users get a table of genes, their predictions and useful meta-information about these genes and their cis-regulatory
        regions.
        2. Users also get plots such as the predicted probabilities of high expression against GC content, the distribution of
        genes on chromosomes across the genome. The plots can be downloaded any various formats and the tables can be downloaded
        as csv files.""")
        st.image('images/image_3.png')

        st.write('**Saliency Maps**')
        st.write("""Deep learning models have been called black boxes because of the difficulty in interpretation brought about
                by their complexity. Here, we provide users model interpretation using ShAP. This tools provide nucleotide resolution
                importance scores for every sequence. This is average across all genes to give users an overview of the regions our models
                focus on during their predictions.""")
        st.image('images/image_4.png')

        st.write('**Mutation Analysis**')
        st.write("""We also provide users two ways to perform in-silico mutagenesis and obtain predictions. This should give users
        insights into potentially positive and negative variants.""")
        st.write('1. User can choose to either manually mutate selected regions of the sequence or use SNPs from an uploaded VCF')
        st.write('2. User selects a gene of interest for the mutation analysis')
        st.write('3. User selects the region to target during mutation analysis')
        st.write('4. Within the selected region, the user narrows down a subsequence to mutate by selecting a range.')
        st.write('5. The sequence belonging to the selected range is extracted and presented to the user to perform mutations.')
        st.image('images/image_5.png')
        st.image('images/image_6.png')
        st.image('images/image_7.png')
        st.image('images/image_8.png')
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

    # Logo of lab and link
    _, lab_logo, lab_name = st.columns([0.3, 0.4, 0.3], vertical_alignment='bottom', gap='small')

    with lab_logo:
        st.image('images/logos.png', use_column_width=True)
        st.subheader('CONTACT US', divider='grey')
        st.write("Forschungszentrum Jülich GmbH D-52425 Jülich, Germany")
        st.markdown(
            f"Lab: <a style='text-decoration:none; text-align: center; color:#FF4BAB;' href=https://www.szymanskilab.com/>SZYMANSKI LAB</a>",
            unsafe_allow_html=True,
        )
        st.write(f"email: :red[j.szymanski@fz-juelich.de]")


