# Save this as lib/annotation_helpers.py

import streamlit as st
import pandas as pd
import re
import logomaker
import matplotlib.pyplot as plt

@st.cache_data
def load_jaspar_motifs(filepath):
    """
    Parses a JASPAR file and returns a DataFrame of motif info
    and a dictionary of the count matrices.
    """
    motif_data = []
    matrices = {}
    
    # Regex to parse the header: >epm_Atha_S0_p0m01F_2477
    # Groups: 1=Organism, 2=Experiment, 3=Predictor, 4=Motif, 5=Orientation, 6=Count
    header_regex = re.compile(r'>epm_([A-Za-z]+)_([A-Za-z0-9]+)_(p[0-9])(m[0-9]+)([FR])_([0-9]+)')

    try:
        with open(filepath, 'r') as f:
            current_motif_id = None
            current_matrix = []
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # If we were building a matrix, save it
                    if current_motif_id and current_matrix:
                        matrices[current_motif_id] = pd.DataFrame(
                            current_matrix, index=['A', 'C', 'G', 'T']
                        ).astype(int)
                    
                    # Start new motif
                    current_motif_id = line
                    current_matrix = []
                    
                    # Parse the header
                    match = header_regex.match(line)
                    if match:
                        motif_data.append({
                            "MotifID": line,
                            "Organism": match.group(1),
                            "Experiment": match.group(2),
                            "Predictor": match.group(3),
                            "Motif": match.group(4),
                            "Orientation": match.group(5),
                            "Count": int(match.group(6))
                        })
                
                elif line.startswith(('A', 'C', 'G', 'T')):
                    # Extract numbers from [ ... ]
                    counts = [int(n) for n in re.findall(r'\d+', line)]
                    current_matrix.append(counts)
            
            # Save the very last motif
            if current_motif_id and current_matrix:
                matrices[current_motif_id] = pd.DataFrame(
                    current_matrix, index=['A', 'C', 'G', 'T']
                ).astype(int)

    except FileNotFoundError:
        st.error(f"Error: Motif file not found at {filepath}")
        return None, None
    except Exception as e:
        st.error(f"Error parsing motif file: {e}")
        return None, None

    if not motif_data:
        st.warning("No motifs were successfully parsed from the file.")
        return None, None
        
    df = pd.DataFrame(motif_data)
    return df, matrices

def plot_weblogo(pfm_df, motif_id):
    """
    Generates a weblogo from a Position Frequency Matrix (PFM) DataFrame.
    """
    # 1. Convert PFM (counts) to PPM (probabilities)
    ppm_df = pfm_df.apply(lambda x: x / x.sum(), axis=0)
    
    # 2. Convert PPM to Information Content (bits)
    # Logomaker expects the matrix to have columns as positions and rows as bases (A, C, G, T)
    # Our parser already does this, so we just need to transpose for logomaker
    info_df = logomaker.transform_matrix(
        ppm_df.T, 
        from_type='probability', 
        to_type='information'
    )
    
    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(len(info_df) * 0.5, 3))
    logo = logomaker.Logo(info_df, ax=ax)
    
    # Style the plot
    ax.set_title(f"Weblogo for {motif_id}")
    ax.set_xlabel("Position")
    ax.set_ylabel("Bits")
    
    return fig
