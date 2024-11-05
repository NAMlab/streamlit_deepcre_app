import os
import numpy as np
import streamlit as st
from utils import prepare_dataset, extract_scores, make_predictions, one_hot_to_dna, one_hot_encode, prepare_vcf
from utils import dataframe_with_selections
import pandas as pd
import tensorflow as tf
import io
from lib.ui.about import show_about_tab
from lib.ui.sidebar import show_sidebar
from lib.ui.predictions import show_predictions_tab
from lib.ui.saliency import show_saliency_tab
from lib.ui.mutation import choose_analysis_type, show_manual_mutation, show_mutation_results, show_vcf_input
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
tf.config.set_visible_devices([], 'GPU')

available_genomes = pd.read_csv("genomes/genomes.csv")
species = available_genomes["display_name"].tolist()
species.append("New")
model_names = sorted([x.split('.')[0] for x in os.listdir('models')])
color_palette_low_high = ['#4F1787', '#EB3678']

def main():
    st.set_page_config(layout="wide", page_title='deepCRE')

    st.subheader(':green[deepCRE: A web-based tool for predicting gene expression from cis-regulatory elements]')

    # Sidebar
    selected_organism, genome, annotation, genes_list, selected_model = show_sidebar(available_species=species,
                                                                                     available_genomes=available_genomes,
                                                                                     available_models=model_names)

    ### Three main Tabs
    about_tab, preds_tab, interpret_tab, mutations_tab = st.tabs(['Home', 'Predictions', 'Saliency Maps', 'Mutation Analysis'])

    with about_tab:
        show_about_tab(available_genomes)

    if genome is not None and annotation is not None:
        with preds_tab:
            if genes_list is None:
                if st.button('use example', type='primary'):
                    st.warning(f":red[No gene list uploaded. Displaying results for 100 random genes from the {selected_organism} genome.]",
                                icon="⚠️")
                    use_example = True
                else:
                    st.info("""Currently you have not uploaded any data for processing. To see how our tool works please click the "Use example" button. 
                    This will run our tool on 100 sampled genes from the selected genome. To use your own genes of interest, 
                    please uploaded a list of genes at the upload section to the left.""", icon="ℹ️")
                    use_example = False
            else:
                use_example = False

        if genome is not None and annotation is not None:
            x, gene_ids, gene_chroms, gene_starts, gene_ends, gene_size, gene_gc_cont, gene_strands = prepare_dataset(genome=genome,
                                                                                                                    annot=annotation,
                                                                                                                    gene_list=st.session_state.selected_genes,
                                                                                                                        use_example=use_example)
            if x is not None and x.size > 0:
                if use_example:
                    st.session_state.selected_genes = gene_ids
                preds = make_predictions(model=f'models/{selected_model}.h5', x=x)
                actual_scores_low, actual_scores_high, g_h, g_l, p_l, p_h = extract_scores(seqs=x, pred_probs=preds,
                                                                                        genes=gene_ids,
                                                                                        model=f'models/{selected_model}.h5')
                with preds_tab:
                    show_predictions_tab(gene_ids, gene_chroms, gene_starts, gene_ends, gene_size, gene_gc_cont, preds, color_palette_low_high)

                with interpret_tab:
                    show_saliency_tab(actual_scores_high, actual_scores_low, p_h, p_l, color_palette_low_high)

                with mutations_tab:
                    mutate_analysis_type = choose_analysis_type()
                    if mutate_analysis_type == '**manual**':
                        gene_col, _ = st.columns([0.3, 0.7])
                        with gene_col:
                            gene_id = st.selectbox(label=':gray[Select] :green[**gene**]', options=gene_ids)
                        seq = one_hot_to_dna(x[gene_ids.index(gene_id)])[0]
                        start, end = gene_starts[gene_ids.index(gene_id)], gene_ends[gene_ids.index(gene_id)]
                        utr_len = 500 if abs(start - end) // 2 > 500 else abs(end - start) // 2
                        central_pad_size = 3020 - (1000 + utr_len) * 2
                        mut_reg_start, mut_reg_end = show_manual_mutation(gene_id, start, end, seq, utr_len, central_pad_size)

                        seqs = np.array([one_hot_encode(i) for i in [seq, st.session_state.mutated_seq]])
                        preds = make_predictions(model=f'models/{selected_model}.h5', x=seqs)
                        actual_scores, pred_probs, gene_names = extract_scores(seqs=seqs, pred_probs=preds,
                                                                            genes=[gene_id, f'{gene_id}: Mutated'],
                                                                            model=f'models/{selected_model}.h5',
                                                                            separate=False)
                        show_mutation_results(gene_id, pred_probs, actual_scores, seq, utr_len, central_pad_size, mut_reg_start, mut_reg_end)


                    else:
                        vcf_file = show_vcf_input()
                        if vcf_file is not None:
                            vcf_file = io.BytesIO(vcf_file.read())
                            vcf_df = prepare_vcf(uploaded_file=vcf_file)
                            vcf_col, gene_col, _ = st.columns([0.4, 0.5, 0.1], vertical_alignment='center')
                            with vcf_col:
                                st.write('Here are your first 50 :green[**SNPs**]')
                                st.dataframe(vcf_df.head(50))
                            with gene_col:
                                gene_id = st.selectbox(label='Choose gene', options=gene_ids)
                                seq = one_hot_to_dna(x[gene_ids.index(gene_id)])[0]
                                strand = gene_strands[gene_ids.index(gene_id)]
                                start, end = gene_starts[gene_ids.index(gene_id)], gene_ends[gene_ids.index(gene_id)]
                                utr_len = 500 if abs(start-end)//2 > 500 else abs(end-start)//2
                                central_pad_size = 3020 - (1000 + utr_len)*2
                                chrom = gene_chroms[gene_ids.index(gene_id)]
                                snps_in_prom = vcf_df[(vcf_df['Pos'] > start - 1000) & (vcf_df['Pos'] < start + utr_len) & (vcf_df['Chrom'] == chrom)]
                                snps_in_term = vcf_df[(vcf_df['Pos'] > end - utr_len) & (vcf_df['Pos'] < end + 1000) & (vcf_df['Chrom'] == chrom)]
                                if strand == '+':
                                    snps_in_prom['Region'] = ['Promoter']*snps_in_prom.shape[0]
                                    snps_in_term['Region'] = ['Terminator']*snps_in_term.shape[0]
                                else:
                                    snps_in_prom['Region'] = ['Terminator'] * snps_in_prom.shape[0]
                                    snps_in_term['Region'] = ['Promoter'] * snps_in_term.shape[0]
                                snps_cis_regions = pd.concat([snps_in_prom, snps_in_term], axis=0)
                                snps_cis_regions['Strand'] = [strand]*snps_cis_regions.shape[0]
                                snps_cis_regions.sort_values(by=['Region', 'Pos'], ascending=True, inplace=True)
                                snps_cis_regions.reset_index(drop=True, inplace=True)
                                if 'current_gene' not in st.session_state:
                                    st.session_state.current_gene = gene_id
                                st.write(f'These are the SNPs in the cis-regulatory regions of ' + f':green[**{gene_id}**]')
                                selection = dataframe_with_selections(df=snps_cis_regions)
                                st.write('Here are your selected SNPs')
                                st.dataframe(selection, use_container_width=True)
                            if not selection.empty:
                                prom_start, prom_end = start-1000, start+utr_len
                                term_start, term_end = end-utr_len, end+1000
                                complements = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}

                                if st.button('Mutate Sequence', type='primary'):
                                    # Initialize session cis-regulatory sequence ---------------------------
                                    if "cis_seq" not in st.session_state:
                                        st.session_state['cis_seq'] = seq
                                    if st.session_state.current_gene != gene_id:
                                        st.session_state.current_gene = gene_id
                                        st.session_state['cis_seq'] = seq
                                    mut_cis_seq = st.session_state['cis_seq']
                                    mut_markers = []
                                    for _, snp_pos, _, ref_allele, alt_allele, _, snp_region, snp_strand in selection.values:

                                        if snp_strand == '+':
                                            if snp_region == 'Promoter':
                                                snp_pos = snp_pos - prom_start - 1 if snp_pos != prom_start else 0
                                                snp_pos = 0 + snp_pos
                                            else:
                                                snp_pos = snp_pos - term_start - 1 if snp_pos != term_start else 0
                                                snp_pos = (1000+utr_len+central_pad_size) + snp_pos
                                        else:
                                            if snp_region == 'Promoter':
                                                snp_pos = snp_pos - term_start - 1 if snp_pos != term_start else 0
                                                snp_pos = (1000+utr_len) - snp_pos - 1

                                            else:
                                                snp_pos = snp_pos - prom_start - 1 if snp_pos != prom_start else 0
                                                snp_pos = 3020 - snp_pos - 1
                                        mut_markers.append((snp_pos, '*', f'single nucleotide polymorphism ( {ref_allele} - {alt_allele} )'))

                                        if snp_strand == '+':
                                            mut_cis_seq = mut_cis_seq[:snp_pos] + alt_allele + mut_cis_seq[snp_pos+1:]

                                        else:
                                            mut_cis_seq = mut_cis_seq[:snp_pos] + complements[alt_allele] + mut_cis_seq[snp_pos + 1:]

                                    seqs = np.array([one_hot_encode(i) for i in [st.session_state['cis_seq'], mut_cis_seq]])
                                    preds = make_predictions(model=f'models/{selected_model}.h5', x=seqs)
                                    actual_scores, pred_probs, gene_names = extract_scores(seqs=seqs, pred_probs=preds,
                                                                                        genes=[gene_id, f'{gene_id}: Mutated'],
                                                                                        model=f'models/{selected_model}.h5',
                                                                                        separate=False)
                                    show_mutation_results(gene_id, pred_probs, actual_scores, seq, utr_len, central_pad_size, None, None, mut_markers)


if __name__ == '__main__':
    main()