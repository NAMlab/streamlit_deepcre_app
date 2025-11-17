# This is the full content for lib/ui/annotation.py

import streamlit as st
import pandas as pd
from lib.annotation_helpers import load_jaspar_motifs, plot_weblogo

def show_annotation_tab():
    st.header("Motif Annotation Database")
    
    # --- 1. Load Data ---
    # This uses caching, so it only runs once
    df, matrices = load_jaspar_motifs("data/epm_db.jaspar")
    
    # Handle case where file loading failed
    if df is None or matrices is None:
        st.error("Motif database could not be loaded. Please check the `data/epm_db.jaspar` file.")
        return # Stop execution for this tab

    st.info(f"Loaded {len(df)} motifs from the database.")

    # --- 2. Create Filters ---
    st.subheader("Filter Motifs")
    
    # Create columns for filters
    filt_col1, filt_col2, filt_col3 = st.columns(3)
    
    with filt_col1:
        # Get unique organisms and add "All"
        organisms = ["All"] + sorted(df['Organism'].unique())
        selected_organism = st.selectbox("Filter by Organism", organisms)

    with filt_col2:
        # Get unique experiments and add "All"
        experiments = ["All"] + sorted(df['Experiment'].unique())
        selected_experiment = st.selectbox("Filter by Experiment", experiments)
        
    with filt_col3:
        # Get unique predictors and add "All"
        predictors = ["All"] + sorted(df['Predictor'].unique())
        selected_predictor = st.selectbox("Filter by Predictor", predictors)

    # --- 3. Filter DataFrame ---
    filtered_df = df.copy() # Start with the full dataframe
    
    if selected_organism != "All":
        filtered_df = filtered_df[filtered_df['Organism'] == selected_organism]
        
    if selected_experiment != "All":
        filtered_df = filtered_df[filtered_df['Experiment'] == selected_experiment]
        
    if selected_predictor != "All":
        filtered_df = filtered_df[filtered_df['Predictor'] == selected_predictor]

    # --- 4. Display Table ---
    st.subheader(f"Displaying {len(filtered_df)} Motifs")
    st.dataframe(filtered_df)

    # --- 5. Display Weblogo ---
    st.subheader("Visualize Motif")
    
    if filtered_df.empty:
        st.warning("No motifs match the current filter. Cannot select a motif to visualize.")
    else:
        # Create a dropdown with the MotifIDs from the *filtered* table
        motif_ids_list = filtered_df['MotifID'].tolist()
        selected_motif = st.selectbox("Select MotifID to visualize", motif_ids_list)
        
        if selected_motif:
            # Get the matrix
            pfm_matrix = matrices[selected_motif]
            
            # Generate the plot
            logo_fig = plot_weblogo(pfm_matrix, selected_motif)
            
            # Display in Streamlit
            st.pyplot(logo_fig)
