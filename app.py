import os
from datetime import datetime
import numpy as np
import streamlit as st
from utils import prepare_dataset, extract_scores, make_predictions, one_hot_to_dna, one_hot_encode, prepare_vcf
from utils import dataframe_with_selections
import pandas as pd
import itertools
import altair as alt
import tensorflow as tf
import io
from lib.ui.about import show_about_tab
from lib.ui.sidebar import show_sidebar
from lib.ui.predictions import show_predictions_tab
from lib.ui.saliency import show_saliency_tab
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
    selected_organism, genome, annotation, genes_list, selected_model, use_example = show_sidebar(available_species=species, available_genomes=available_genomes,
                                                            available_models=model_names)

    if genome is not None and annotation is not None:
        if genes_list is None:
            if use_example:
                st.warning(f":red[No gene list uploaded. Displaying results for 100 random genes from the {selected_organism} genome.]",
                            icon="⚠️")
            else:
                st.info("""Currently you have not uploaded any data for processing. To see how our tool works please 
                            check the "Use 100 random genes from the genome" box. This will run our tool on 100 sampled genes from the selected 
                            genome. To use your own genes of interest, please uploaded a list of genes at the
                            upload section to the left.
                            """, icon="ℹ️")
    ### Three main Tabs
    about_tab, preds_tab, interpret_tab, mutations_tab = st.tabs(['About', 'Predictions', 'Saliency Maps', 'Mutation Analysis'])

    with about_tab:
        show_about_tab(available_genomes)


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
            mut_col, _ = st.columns([0.2, 0.8])
            with mut_col:
                mutate_analysis_type = st.radio(':gray[Choose] :green[**analysis type**]',
                                                options=['**manual**', '**VCF**'],
                                                captions=[
                                                    "Manually mutate sequence",
                                                    "Use natural variants from VCF"
                                                ])
            if mutate_analysis_type == '**manual**':
                gene_col, _ = st.columns([0.3, 0.7])
                with gene_col:
                    gene_id = st.selectbox(label=':gray[Select] :green[**gene**]', options=gene_ids)
                    seq = one_hot_to_dna(x[gene_ids.index(gene_id)])[0]
                    start, end = gene_starts[gene_ids.index(gene_id)], gene_ends[gene_ids.index(gene_id)]
                    utr_len = 500 if abs(start - end) // 2 > 500 else abs(end - start) // 2
                    central_pad_size = 3020 - (1000 + utr_len) * 2
                    if 'current_gene' not in st.session_state or st.session_state.current_gene != gene_id:
                        st.session_state.current_gene = gene_id
                        st.session_state.mutated_seq = seq

                # Initialize session state promoter and terminator sequences ---------------------------

                sel_region = st.radio(label='Select :green[**region**] to mutate',
                                        options=["promoter", "5'UTR", "3'UTR", "terminator"])
                region_to_coords = {'promoter': [0, 0, 1000],
                                    'terminator': [2019, 2019, 3020],
                                    "5'UTR": [1000, 1000, 1000 + utr_len],
                                    "3'UTR": [1000 + utr_len + central_pad_size, 1000 + utr_len + central_pad_size,
                                                2019]}
                start_to_mut, end_to_mut, _ = st.columns([0.1, 0.1, 0.8])
                val, min_val, max_val = region_to_coords[sel_region]
                slider_col, extracted_seq_col = st.columns([0.4, 0.6])

                with slider_col:
                    with st.form('mutation_form', clear_on_submit=False, border=False):
                        slider_vals = st.slider('Select :green[**Start**] and :green[**End**] coordinates',
                                                min_value=val, max_value=max_val, value=(min_val, max_val), step=1)
                        st.form_submit_button('submit', type="primary")

                mut_reg_start, mut_reg_end = slider_vals
                sub_seq_to_mutate = st.session_state.mutated_seq[mut_reg_start:mut_reg_end]
                len_sub_seq = len(sub_seq_to_mutate)
                if len_sub_seq != mut_reg_end - mut_reg_start:
                    sub_seq_to_mutate = sub_seq_to_mutate + 'N' * (mut_reg_end - mut_reg_start - len_sub_seq)
                if "sub_seq_to_mutate" not in st.session_state:
                    st.session_state.sub_seq_to_mutate = sub_seq_to_mutate
                with extracted_seq_col:
                    extracted_sub_seq = st.text_area(label=f'Target {sel_region} region ({mut_reg_start} - {mut_reg_end})',
                                                        value=st.session_state.mutated_seq[mut_reg_start:mut_reg_end],
                                                        max_chars=len(st.session_state.mutated_seq[mut_reg_start:mut_reg_end]),
                                                        height=50, key="sub_seq_to_mutate")
                    if len(extracted_sub_seq) != mut_reg_end - mut_reg_start:
                        extracted_sub_seq = extracted_sub_seq + 'N'*(mut_reg_end - mut_reg_start - len(extracted_sub_seq))
                    if st.button('Mutate', type="primary"):
                        st.session_state.mutated_seq = st.session_state.mutated_seq[:mut_reg_start] + extracted_sub_seq + st.session_state.mutated_seq[mut_reg_end:]

                seqs = np.array([one_hot_encode(i) for i in [seq, st.session_state.mutated_seq]])
                preds = make_predictions(model=f'models/{selected_model}.h5', x=seqs)
                actual_scores, pred_probs, gene_names = extract_scores(seqs=seqs, pred_probs=preds,
                                                                    genes=[gene_id, f'{gene_id}: Mutated'],
                                                                    model=f'models/{selected_model}.h5',
                                                                    separate=False)

                pred_chart = alt.Chart(pd.DataFrame({'Probability of high expression':pred_probs,
                                                    'Gene ID': [gene_id, f'{gene_id}: Mutated']})).mark_bar()\
                    .encode(x='Gene ID:N',
                            y=alt.Y('Probability of high expression:Q', scale=alt.Scale(domain=[0, 1])),
                            color=alt.Color('Gene ID:N', scale=alt.Scale(range=['grey', '#33BBC5'],
                                                    domain=[gene_id, f'{gene_id}: Mutated'])))

                mut_probs_col, mut_sal_map_col = st.columns([0.2, 0.9])
                with mut_probs_col:
                    st.altair_chart(pred_chart, use_container_width=True, theme=None)
                with mut_sal_map_col:
                    mut_df = pd.DataFrame({
                        'Saliency Score': np.concatenate([actual_scores[0].mean(axis=1), actual_scores[1].mean(axis=1)]),
                        'Gene ID': list(itertools.chain(*[[gene_id] * 3020, [f'{gene_id}: Mutated'] * 3020])),
                        'Nucleotide Position': np.concatenate([np.arange(1, 3021) for _ in range(2)], axis=0)
                    })

                    chart_title = alt.TitleParams(
                        f"Average Saliency Map for mutated sequence compared to the original sequence",
                        subtitle=[
                            """Saliency scores are averaged across all sequences predicted per nucleotide""",
                            f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
                        subtitleColor='grey'
                    )

                    base = alt.Chart(mut_df, title=chart_title)

                    saliency_chart_mut = base.mark_line(line=False, point=False).encode(
                        x=alt.X('Nucleotide Position', scale=alt.Scale(domain=[1, 3021]),
                                axis=alt.Axis(tickCount=10)),
                        y=alt.Y('Saliency Score:Q', scale=alt.Scale(
                            domain=[mut_df['Saliency Score'].min(), mut_df['Saliency Score'].max()])),
                        color=alt.Color('Gene ID:N',
                                        scale=alt.Scale(range=['grey', '#33BBC5'],
                                                        domain=[gene_id, f'{gene_id}: Mutated'])),
                        opacity=alt.condition(alt.datum['Gene ID'] == gene_id, alt.value(1), alt.value(0.7))
                    )
                    max_saliency = mut_df['Saliency Score'].max()
                    mean_pos_saliency = mut_df['Saliency Score'].mean()
                    text_y = max_saliency + mean_pos_saliency

                    annotations = [
                        (1000, text_y, "TSS", "Transcription Start Site"),
                        (2020, text_y, "TTS", "Transcription Termination site"),
                    ]
                    annotations_df = pd.DataFrame(
                        annotations, columns=["Nucleotide Position", "Saliency Score", "marker", "description"]
                    )
                    annotation_layer = (
                        alt.Chart(annotations_df)
                        .mark_text(size=12, dx=0, dy=0, align="center")
                        .encode(x=alt.X("Nucleotide Position", scale=alt.Scale(domain=[1, 3021])),
                                y=alt.Y("Saliency Score:Q"), text="marker",
                                tooltip="description")
                    )

                    rule = base.mark_rule(strokeDash=[2, 2]).encode(
                        y=alt.datum(0),
                        color=alt.value("black")
                    )

                    span_prom = alt.Chart(pd.DataFrame({'x1': [0], 'x2': [999]})).mark_rect(
                        opacity=0.1,
                    ).encode(
                        x=alt.X('x1', scale=alt.Scale(domain=[1, 3021]), title='Nucleotide Position'),
                        x2='x2',
                        color=alt.value('grey'),
                        tooltip=alt.value("promoter")
                    )


                    span_5utr = alt.Chart(pd.DataFrame({'x1': [1000], 'x2': [1000+utr_len]})).mark_rect(
                        opacity=0.1,
                    ).encode(
                        x=alt.X('x1', scale=alt.Scale(domain=[1, 3021]), title='Nucleotide Position'),
                        x2='x2',  # alt.datum(2019),
                        color=alt.value('red'),
                        tooltip=alt.value("5' UTR")
                    )

                    span_3utr = alt.Chart(pd.DataFrame({'x1': [1000+utr_len+central_pad_size], 'x2': [2019]})).mark_rect(
                        opacity=0.1,
                    ).encode(
                        x=alt.X('x1', scale=alt.Scale(domain=[1, 3021]), title='Nucleotide Position'),
                        x2='x2',  # alt.datum(2019),
                        color=alt.value('cornflowerblue'),
                        tooltip=alt.value("3' UTR")
                    )

                    span_term = alt.Chart(pd.DataFrame({'x1': [2020], 'x2': [3020]})).mark_rect(
                        opacity=0.1,
                    ).encode(
                        x=alt.X('x1', scale=alt.Scale(domain=[1, 3021]), title='Nucleotide Position'),
                        x2='x2',
                        color=alt.value('grey'),
                        tooltip=alt.value("terminator")
                    )

                    saliency_chart_mut = span_prom + span_5utr + span_3utr + span_term + saliency_chart_mut + annotation_layer + rule
                    st.altair_chart(saliency_chart_mut, use_container_width=True, theme=None)

                def reset_seq():
                    st.session_state.mutated_seq = seq
                    st.session_state.sub_seq_to_mutate = seq[mut_reg_start:mut_reg_end]

                st.button('Reset', type="primary", on_click=reset_seq)

            else:
                file_upload, _ = st.columns([0.3, 0.7])
                with file_upload:
                    vcf_file = st.file_uploader(label='VCF file', accept_multiple_files=False, type=['.gz'],
                                                help="""upload a VCF file. File should be in .gz format""")
                if vcf_file is not None:
                    if vcf_file.name.endswith('.gz'):
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
                                pred_chart = alt.Chart(pd.DataFrame({'Probability of high expression': pred_probs,
                                                                    'Gene ID': [gene_id,
                                                                                f'{gene_id}: Mutated']})).mark_bar() \
                                    .encode(x='Gene ID:N',
                                            y=alt.Y('Probability of high expression:Q', scale=alt.Scale(domain=[0, 1])),
                                            color=alt.Color('Gene ID:N', scale=alt.Scale(range=['grey', '#33BBC5'],
                                                                                        domain=[gene_id,
                                                                                                f'{gene_id}: Mutated'])))
                                mut_probs_col, mut_sal_map_col = st.columns([0.2, 0.9])
                                with mut_probs_col:
                                    st.altair_chart(pred_chart, use_container_width=True, theme=None)
                                with mut_sal_map_col:
                                    mut_df = pd.DataFrame({
                                        'Saliency Score': np.concatenate(
                                            [actual_scores[0].mean(axis=1), actual_scores[1].mean(axis=1)]),
                                        'Gene ID': list(
                                            itertools.chain(*[[gene_id] * 3020, [f'{gene_id}: Mutated'] * 3020])),
                                        'Nucleotide Position': np.concatenate([np.arange(1, 3021) for _ in range(2)],
                                                                            axis=0)
                                    })

                                    chart_title = alt.TitleParams(
                                        f"Average Saliency Map for mutated sequence compared to the original sequence",
                                        subtitle=[
                                            """Saliency scores are averaged across all sequences predicted per nucleotide""",
                                            f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
                                        subtitleColor='grey'
                                    )

                                    base = alt.Chart(mut_df, title=chart_title)
                                    saliency_chart_vcf = base.mark_line(line=False, point=False).encode(
                                        x=alt.X('Nucleotide Position', scale=alt.Scale(domain=[1, 3021]),
                                                axis=alt.Axis(tickCount=10)),
                                        y=alt.Y('Saliency Score:Q', scale=alt.Scale(
                                            domain=[mut_df['Saliency Score'].min(), mut_df['Saliency Score'].max()])),
                                        color=alt.Color('Gene ID:N',
                                                        scale=alt.Scale(range=['grey', '#33BBC5'],
                                                                        domain=[gene_id, f'{gene_id}: Mutated'])),
                                        opacity=alt.condition(alt.datum['Gene ID'] == gene_id, alt.value(1), alt.value(0.7))
                                    )
                                    max_saliency = mut_df['Saliency Score'].max()
                                    mean_pos_saliency = mut_df['Saliency Score'].mean()
                                    text_y = max_saliency + mean_pos_saliency

                                    annotations = [
                                        (1000, text_y, "TSS", "Transcription Start Site"),
                                        (2020, text_y, "TTS", "Transcription Termination site"),
                                    ]
                                    annotations_df = pd.DataFrame(
                                        annotations,
                                        columns=["Nucleotide Position", "Saliency Score", "marker", "description"]
                                    )
                                    annotation_layer = (
                                        alt.Chart(annotations_df)
                                        .mark_text(size=15, dx=0, dy=0, align="center")
                                        .encode(x=alt.X("Nucleotide Position", scale=alt.Scale(domain=[1, 3021])),
                                                y=alt.Y("Saliency Score:Q"), text="marker",
                                                tooltip="description")
                                    )

                                    rule = base.mark_rule(strokeDash=[2, 2]).encode(
                                        y=alt.datum(0),
                                        color=alt.value("black")
                                    )

                                    # SNP marking
                                    snp_annotation_df = pd.DataFrame(mut_markers,
                                                                    columns=["Nucleotide Position", "marker", "description"])
                                    snp_annotation_df["Saliency Score"] = [text_y]*len(mut_markers)

                                    snp_annotation_layer = alt.Chart(snp_annotation_df).mark_rule(strokeDash=[2, 2]).encode(
                                        x=alt.X('Nucleotide Position', scale=alt.Scale(domain=[1, 3021])),
                                        text='marker:N',
                                        color=alt.value("gray"),
                                        tooltip="description",
                                        size=alt.value(3)
                                        )
                                    span_prom = alt.Chart(pd.DataFrame({'x1': [0], 'x2': [999]})).mark_rect(
                                        opacity=0.1,
                                    ).encode(
                                        x=alt.X('x1', scale=alt.Scale(domain=[1, 3021]),
                                                title='Nucleotide Position'),
                                        x2='x2',
                                        color=alt.value('grey'),
                                        tooltip=alt.value("promoter")
                                    )

                                    span_5utr = alt.Chart(
                                        pd.DataFrame({'x1': [1000], 'x2': [1000 + utr_len]})).mark_rect(
                                        opacity=0.1,
                                    ).encode(
                                        x=alt.X('x1', scale=alt.Scale(domain=[1, 3021]),
                                                title='Nucleotide Position'),
                                        x2='x2',  # alt.datum(2019),
                                        color=alt.value('red'),
                                        tooltip=alt.value("5' UTR")
                                    )

                                    span_3utr = alt.Chart(pd.DataFrame(
                                        {'x1': [1000 + utr_len + central_pad_size], 'x2': [2019]})).mark_rect(
                                        opacity=0.1,
                                    ).encode(
                                        x=alt.X('x1', scale=alt.Scale(domain=[1, 3021]),
                                                title='Nucleotide Position'),
                                        x2='x2',  # alt.datum(2019),
                                        color=alt.value('cornflowerblue'),
                                        tooltip=alt.value("3' UTR")
                                    )

                                    span_term = alt.Chart(pd.DataFrame({'x1': [2020], 'x2': [3020]})).mark_rect(
                                        opacity=0.1,
                                    ).encode(
                                        x=alt.X('x1', scale=alt.Scale(domain=[1, 3021]),
                                                title='Nucleotide Position'),
                                        x2='x2',
                                        color=alt.value('grey'),
                                        tooltip=alt.value("terminator")
                                    )

                                    saliency_chart_vcf = span_prom + span_5utr + span_3utr + span_term + snp_annotation_layer + saliency_chart_vcf + rule + annotation_layer
                                    st.altair_chart(saliency_chart_vcf, use_container_width=True, theme=None)

                    else:
                        st.write(':red[Warning: Please upload a .gz file]')


if __name__ == '__main__':
    main()