import streamlit as st
import pandas as pd
from io import StringIO

def check_file(file, file_type):
    if file.size > 0:
        return_file = file
    else:
        return_file = None
        st.error(f":red[The uploaded {file_type} file is empty. Please verify !]", icon="⚠️")
    return return_file

def show_sidebar(available_species, available_genomes, available_models):
    selected_organism = st.sidebar.selectbox(label=":four_leaf_clover: Species ( Select **New** for a new species )",
                                    options=available_species)
    selected_genes = None
    if selected_organism == "New":
        genome = st.sidebar.file_uploader(label="genome", accept_multiple_files=False, type=['.fa', '.gz'],
                                          help="""upload a genome in FASTA format. File should be in .gz format, for example
                                          Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa.gz""")
        if genome is not None:
            genome = check_file(file=genome, file_type="genome")

        annot = st.sidebar.file_uploader(label="gtf/gff3", accept_multiple_files=False, type=['.gtf', '.gff3', '.gff', '.gz'],
                                         help="""upload a gtf/gff3 file. File should be in .gz format, for example
                                         Zea_mays.Zm-B73-REFERENCE-NAM-5.0.60.gtf.gz""")
        if annot is not None:
            annot = check_file(file=annot, file_type="GTF/GFF3")
    else:
        genome = available_genomes.loc[available_genomes["display_name"] == selected_organism, "assembly_file"].values[0]
        annot = available_genomes.loc[available_genomes["display_name"] == selected_organism, "annotation_file"].values[0]
    genes_list = st.sidebar.file_uploader(label="genes", type=['.csv', '.txt'], accept_multiple_files=False,
                                          help="""upload a list of max 1000 gene IDs.
                                           Each gene ID must be on a new line. If genes are more than 1000, the
                                           first 1000 genes will be analysed.""")
    if genes_list is None:
        use_example = st.sidebar.checkbox("Use 100 random genes from the genome", value=False,
                                        help="""If selected, the tool will run on 100 random genes from the selected genome. 
                                        This gives you an opportunity to test out the tool without uploading your own genes.""")
    else:
        genes_list = check_file(file=genes_list, file_type="genes list")
        if genes_list is not None:
            selected_genes = pd.read_csv(StringIO(genes_list.getvalue().decode("utf-8")), header=None).values.ravel().tolist()
            use_example = False
    deepcre_model = st.sidebar.selectbox(label="Choose deepCRE model", options=available_models, )


    return selected_organism, genome, annot, selected_genes, deepcre_model, use_example