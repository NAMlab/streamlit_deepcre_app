import streamlit as st
import io
from lib.utils import prepare_dataset, make_predictions, extract_scores, prepare_vcf

diagnostic_fields = ['current_genome', 'current_annotation', 'current_selected_genes', 'currently_using_example', 'current_model', 'current_mutation_sequences',
                     'current_vcf_file', 'vcf_content']
depend_on_mutation_sequences = ['mutation_predictions', 'mutation_scores']
depend_on_model = depend_on_mutation_sequences + ['predictions', 'scores']
depend_on_dataset =  depend_on_model + ['dataset']

def initStorage():
    for i in diagnostic_fields + depend_on_dataset:
        if i not in st.session_state:
            st.session_state[i] = None

def validateDataset(genome, annotation, selected_genes, using_example):
    if genome != st.session_state.current_genome or annotation != st.session_state.current_annotation or \
        selected_genes != st.session_state.current_selected_genes or using_example != st.session_state.currently_using_example:
        st.session_state.current_genome = genome
        st.session_state.current_annotation = annotation
        st.session_state.current_selected_genes = selected_genes
        st.session_state.currently_using_example = using_example
        for i in depend_on_dataset:
            st.session_state[i] = None

def validateModel(model):
    if model != st.session_state.current_model:
        st.session_state.current_model = model
        for i in depend_on_model:
            st.session_state[i] = None

def validateMutationSequences(sequences):
    if not (sequences == st.session_state.current_mutation_sequences).all():
        st.session_state.current_mutation_sequences = sequences
        for i in depend_on_mutation_sequences:
            st.session_state[i] = None

def getDataset():
    if st.session_state.dataset is None:
        st.session_state.dataset = prepare_dataset(genome=st.session_state.current_genome, annot=st.session_state.current_annotation,
                                                    gene_list=st.session_state.current_selected_genes,
                                                    use_example=st.session_state.currently_using_example)
    return st.session_state.dataset

def getPredictions():
    if st.session_state.predictions is None:
        st.session_state.predictions = make_predictions(model=st.session_state.current_model, x=getDataset()[0])
    return st.session_state.predictions

def getScores():
    if st.session_state.scores is None:
        st.session_state.scores = extract_scores(seqs=getDataset()[0], pred_probs=getPredictions(), genes=getDataset()[1], model=st.session_state.current_model)
    return st.session_state.scores

def getMutationPredictions():
    if st.session_state.mutation_predictions is None:
        st.session_state.mutation_predictions = make_predictions(model=st.session_state.current_model, x=st.session_state.current_mutation_sequences)
    return st.session_state.mutation_predictions

def getMutationScores(gene_id):
    if st.session_state.mutation_scores is None:
        st.session_state.mutation_scores = extract_scores(seqs=st.session_state.current_mutation_sequences, pred_probs=getMutationPredictions(), genes=[gene_id, f'{gene_id}: Mutated'], model=st.session_state.current_model, separate=False)
    return st.session_state.mutation_scores

def getVcfContent(vcf_file, gene_start, gene_ends, gene_chroms):
    if vcf_file != st.session_state.current_vcf_file:
        st.session_state.current_vcf_file = vcf_file
        if vcf_file is not None:
            vcf_file = io.BytesIO(vcf_file.read())
            st.session_state.vcf_content = prepare_vcf(uploaded_file=vcf_file, gene_starts=gene_start, gene_ends=gene_ends, gene_chroms=gene_chroms)
        else:
            st.session_state.vcf_content = None
    return st.session_state.vcf_content