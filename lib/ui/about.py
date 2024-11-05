import streamlit as st
import numpy as np

def show_about_tab(available_genomes):
    _, abt_col, _ = st.columns([0.15, 0.7, 0.15])
    with abt_col:
        about_header, _ = st.columns([0.2, 0.8])
        with about_header:
            st.subheader('About', divider='grey')
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


