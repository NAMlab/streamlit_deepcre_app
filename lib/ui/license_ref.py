import streamlit as st

def show_license_ref():
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

    st.subheader("**References**")
    st.write("""Peleke, Fritz Forbang, Simon Maria Zumkeller, Mehmet Gültas, Armin Schmitt, and Jędrzej Szymański. 2024. 
            “Deep Learning the Cis-Regulatory Code for Gene Expression in Selected Model Plants.” Nature Communications 15 (1): 3488.""")
    st.write("""
    Lundberg, Scott, and Su-In Lee. 2017. “A Unified Approach to Interpreting Model Predictions.” arXiv [cs.AI]. arXiv. http://arxiv.org/abs/1705.07874.
    """)

    st.divider()
    st.write("""
    Copyright © 2024 Leibniz Institute of Plant Genetics and Crop Plant Research (IPK), Gatersleben, Germany \n
    This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your option) any later version.
    """)
