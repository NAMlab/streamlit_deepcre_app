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
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
tf.config.set_visible_devices([], 'GPU')


def main():
    st.set_page_config(layout="wide", page_title='deepCRE')
    color_palette_low_high = ['#4F1787', '#EB3678']
    species = sorted([x.split(".")[0] for x in os.listdir('species')])
    species.append("New")
    model_names = sorted([x.split('.')[0] for x in os.listdir('models')])

    organism = st.sidebar.selectbox(label=":four_leaf_clover: Species ( Select **New** for a new species )",
                                    options=species)
    # Logo of lab and link
    lab_logo, lab_name, _ = st.columns([0.1, 0.1, 0.9], vertical_alignment='bottom', gap='small')
    with lab_logo:
        st.image('images/szymasnki_lab_logo.png', use_column_width=True)
    with lab_name:
        st.markdown(
            f"<a style='text-decoration:none; text-align: center; color:#FF4BAB;' href=https://www.szymanskilab.com/>SZYMANSKI LAB</a>",
            unsafe_allow_html=True,
        )
    st.subheader(':green[DeepCRE: Gene expression prediction]')

    ### Three main Tabs
    preds_tab, interpret_tab, mutations_tab = st.tabs(['Predictions', 'Saliency Maps', 'Mutation Analysis'])

    if organism == "New":
        genome = st.sidebar.file_uploader(label="genome",
                                          help="""upload a genome in FASTA format. File should be in .gz format, for example
                                          Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa.gz""")
        annot = st.sidebar.file_uploader(label="gtf/gff3",
                                         help="""upload a gtf/gff3 file. File should be in .gz format, for example
                                         Zea_mays.Zm-B73-REFERENCE-NAM-5.0.60.gtf.gz""")
        new = True
    else:
        genome = [x for x in os.listdir('species') if x.startswith(organism)][0]
        genome = f"species/{genome}"
        annot = [x for x in os.listdir('annotations') if x.startswith(organism)][0]
        annot = f"annotations/{annot}"
        new = False
    genes_list = st.sidebar.file_uploader(label="genes",
                                          help="""upload a csv file of max 1000 gene IDs.
                                           Each gene ID must be on a new line. If genes are more than 1000, the
                                           first 1000 genes will be analysed.""")
    deepcre_model = st.sidebar.selectbox(label="Choose deepCRE model", options=model_names, )
    if genome is not None and annot is not None and genes_list is not None:
        x, gene_ids, gene_chroms, gene_starts, gene_ends, gene_size, gene_gc_cont, gene_strands = prepare_dataset(genome=genome,
                                                                                                                  annot=annot,
                                                                                                                  gene_list=genes_list,
                                                                                                                  new=new)
        preds = make_predictions(model=f'models/{deepcre_model}.h5', x=x)
        with preds_tab:
            predictions = pd.DataFrame(data={'Gene ID': gene_ids, 'Chromosome': [f'Chr: {i}' for i in gene_chroms],
                                             'Gene Start': gene_starts,
                                             'Gene End': gene_ends, 'Gene size': gene_size,
                                             'Probability of high expression': preds,
                                             'GC Content': gene_gc_cont,
                                             'Expressed': ['High' if i > 0.5 else 'Low' for i in preds]})

            # Predictions
            col_preds, col_preds_hist = st.columns([0.7, 0.3], vertical_alignment='top')
            with col_preds:
                st.dataframe(predictions, hide_index=True, use_container_width=True)
            with col_preds_hist:
                chart_title = alt.TitleParams(
                    "Distribution of genes across the genome",
                    subtitle=[
                        """Number of genes per chromosome""",
                        f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
                    subtitleColor='grey'
                )
                chart = alt.Chart(predictions, title=chart_title).mark_bar().encode(
                    alt.X("Chromosome:N"),
                    y="count()").configure_mark(
                    color='#3FA2F6')
                st.altair_chart(chart, use_container_width=True, theme=None)


            # SHAP saliency maps
            col_pred_by_chrom, hist_pred = st.columns([0.7, 0.3], vertical_alignment='top', gap='medium')
            with col_pred_by_chrom:
                num_chroms = len(predictions['Chromosome'].unique())
                num_chroms = num_chroms if num_chroms%2==0 else num_chroms+1
                n_rows = 3 if num_chroms%3 == 0 else 2
                n_cols = num_chroms//n_rows
                data_agg = predictions.groupby(['Chromosome', 'Expressed']).agg({'Expressed': 'count'})
                data_agg.rename(columns={'Expressed': 'Count'}, inplace=True)
                data_agg.reset_index(inplace=True)

                chart_title = alt.TitleParams(
                    "Distribution of Low and High predictions across Chromosomes",
                    subtitle=[
                        """Number of genes predicted as high/low expression for each chromosome""",
                        f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
                    subtitleColor='grey'
                )
                chart_preds_pre_chrom = alt.Chart(data_agg, title=chart_title).mark_bar().encode(
                    x=alt.X('Chromosome:N', axis=alt.Axis(labelAngle=0)),
                    xOffset='Expressed',
                    y=alt.Y('Count'),
                    color=alt.Color('Expressed:N',
                                    scale=alt.Scale(range=color_palette_low_high,
                                                    domain=['High', 'Low']))
                )
                st.altair_chart(chart_preds_pre_chrom, use_container_width=True, theme=None)


            with hist_pred:
                chart_title = alt.TitleParams(
                    "Distribution of Predicted probabilities",
                    subtitle=[
                        """Predicted probabilities of high and low expressed genes.""",
                        f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
                    subtitleColor='grey'
                )
                chart_hist_pred_prob = alt.Chart(predictions, title=chart_title).mark_bar().encode(
                    alt.X('Probability of high expression:Q').bin(maxbins=20),
                    alt.Y('count()').stack(None),
                    color=alt.Color('Expressed:N', scale=alt.Scale(range=color_palette_low_high,
                                                                   domain=['High', 'Low']))
                )
                st.altair_chart(chart_hist_pred_prob, use_container_width=True, theme=None)

            pred_to_size, preds_to_gc = st.columns([0.5, 0.5], vertical_alignment='top', gap='medium')
            with pred_to_size:
                chart_title = alt.TitleParams(
                    "Gene size vs Predicted probabilities",
                    subtitle=[
                        """Relationship between the size of genes and their probability of being highly expressed""",
                        f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
                    subtitleColor='grey'
                )
                chart_hist_size = alt.Chart(predictions, title=chart_title).mark_circle(size=25).encode(
                    x='Probability of high expression:Q',
                    y='Gene size:Q',
                    color=alt.Color('Expressed:N',
                                    scale=alt.Scale(range=color_palette_low_high,
                                                    domain=['High', 'Low'])),
                )
                st.altair_chart(chart_hist_size, use_container_width=True, theme=None)

            with preds_to_gc:
                chart_title = alt.TitleParams(
                    "GC Content vs Predicted probabilities",
                    subtitle=[
                        """Relationship between the GC content of genes and their probability of being highly expressed""",
                        f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
                    subtitleColor='grey'
                )
                chart_hist_gc = alt.Chart(predictions, title=chart_title).mark_circle(size=25).encode(
                    x='Probability of high expression:Q',
                    y='GC Content:Q',
                    color=alt.Color('Expressed:N',
                                    scale=alt.Scale(range=color_palette_low_high,
                                                    domain=['High', 'Low'])),
                )
                st.altair_chart(chart_hist_gc, use_container_width=True, theme=None)

        with interpret_tab:
            actual_scores_low, actual_scores_high, g_h, g_l, p_l, p_h = extract_scores(seqs=x, pred_probs=preds,
                                                                                       genes=gene_ids,
                                                                                       model=f'models/{deepcre_model}.h5')

            sal_line, sal_scat = st.columns([0.6, 0.4], vertical_alignment='top', gap='medium')
            with sal_line:
                avg_saliency = pd.DataFrame(data={
                    'Saliency Score': np.concatenate([actual_scores_high.mean(axis=(0, 2)),
                                                actual_scores_low.mean(axis=(0, 2))], axis=0),
                    'Expressed': list(itertools.chain(*[['High'] * 3020, ['Low'] * 3020])),
                    'Nucleotide Position': np.concatenate([np.arange(1, 3021), np.arange(1, 3021)], axis=0)
                })
                chart_title = alt.TitleParams(
                    "Average Saliency map",
                    subtitle=[
                        """Saliency scores are averaged across all sequences predicted as either high/low expressed""",
                        f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
                    subtitleColor='grey'
                )
                saliency_chart = alt.Chart(avg_saliency, title=chart_title).mark_area(line=True, point=False).encode(
                    x=alt.X('Nucleotide Position', scale=alt.Scale(domain=[1, 3021]),
                            axis=alt.Axis(tickCount=10)),
                    y='Saliency Score:Q',
                    color=alt.Color('Expressed:N',
                                        scale=alt.Scale(range=color_palette_low_high,
                                                        domain=['High', 'Low']))
                )
                max_saliency = avg_saliency['Saliency Score'].max()
                mean_pos_saliency = avg_saliency[avg_saliency['Expressed']=='High']['Saliency Score'].mean()
                text_y = max_saliency+mean_pos_saliency

                annotations = [
                    (1000, text_y, "TSS", "Transcription Start Site"),
                    (2020, text_y, "TTS", "Transcription Termination site"),
                ]
                annotations_df = pd.DataFrame(
                    annotations, columns=["Nucleotide Position", "Saliency Score", "marker", "description"]
                )
                annotation_layer = (
                    alt.Chart(annotations_df)
                    .mark_text(size=15, dx=-10, dy=0, align="center")
                    .encode(x=alt.X("Nucleotide Position", scale=alt.Scale(domain=[1, 3021])),
                            y=alt.Y("Saliency Score:Q"), text="marker",
                            tooltip="description")
                )
                saliency_chart = saliency_chart + annotation_layer
                st.altair_chart(saliency_chart, use_container_width=True, theme=None)

            with sal_scat:
                sum_saliency_score, pred_prob, expressed = [], [], []
                for idx in range(actual_scores_high.shape[0]):
                    sum_saliency_score.append(actual_scores_high[idx].sum())
                    pred_prob.append(p_h[idx])
                    expressed.append('High')
                for idx in range(actual_scores_low.shape[0]):
                    sum_saliency_score.append(actual_scores_low[idx].sum())
                    pred_prob.append(p_l[idx])
                    expressed.append('Low')
                data_sal_scat = pd.DataFrame(data={'Expressed':expressed,
                                                   'Sum Saliency Score':sum_saliency_score,
                                                   'Probability of high expression':pred_prob})
                chart_title = alt.TitleParams(
                    "Saliency score vs Predicted probabilities",
                    subtitle=["""Saliency scores are summed per gene and plotted against probabilities of high expression""",
                              f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
                    subtitleColor='grey'
                )
                saliency_scat = alt.Chart(data_sal_scat, title=chart_title).mark_circle(size=25).encode(
                    x=alt.X('Probability of high expression:Q'),
                    y=alt.Y('Sum Saliency Score:Q'),
                    color = alt.Color('Expressed:N',
                                      scale=alt.Scale(range=color_palette_low_high,
                                                      domain=['High', 'Low']))
                )
                st.altair_chart(saliency_scat, use_container_width=True, theme=None)

            sal_line_nucl, sal_scat_nucl = st.columns([0.6, 0.4], vertical_alignment='top', gap='medium')
            with sal_line_nucl:
                df_low = pd.DataFrame(data={
                    'Nucleotide Position': np.concatenate([np.arange(1, 3021) for _ in range(4)], axis=0),
                    'Saliency Score': np.concatenate([actual_scores_low.mean(axis=0)[:, i] for i in range(4)]),
                    'Base': list(itertools.chain(*[['A']*3020, ['C']*3020, ['G']*3020, ['T']*3020])),
                })
                df_high = pd.DataFrame(data={
                    'Nucleotide Position': np.concatenate([np.arange(1, 3021) for _ in range(4)], axis=0),
                    'Saliency Score': np.concatenate([actual_scores_high.mean(axis=0)[:, i] for i in range(4)]),
                    'Base': list(itertools.chain(*[['A'] * 3020, ['C'] * 3020, ['G'] * 3020, ['T'] * 3020])),
                })
                max_ylimit = max(df_low['Saliency Score'].max(), df_high['Saliency Score'].max())
                min_ylimit = min(df_low['Saliency Score'].min(), df_high['Saliency Score'].min())
                for df_name, df in zip(['High', 'Low'], [df_high, df_low]):
                    chart_title = alt.TitleParams(
                        f"Base-type average Saliency map for {df_name} expressed genes",
                        subtitle=[
                            """Saliency scores are averaged across all sequences predicted per nucleotide""",
                            f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
                        subtitleColor='grey'
                    )
                    base = alt.Chart(df, title=chart_title)
                    saliency_chart_high = base.mark_line(line=False, point=False).encode(
                        x=alt.X('Nucleotide Position', scale=alt.Scale(domain=[1, 3021]),
                                axis=alt.Axis(tickCount=10)),
                        y=alt.Y('Saliency Score:Q', scale=alt.Scale(domain=[min_ylimit, max_ylimit])),
                        color=alt.Color('Base:N',
                                        scale=alt.Scale(range=['green', 'cornflowerblue', 'darkorange',  'red'],
                                                        domain=['A', 'C', 'G', 'T']))
                    )
                    max_saliency = df_high['Saliency Score'].max()
                    mean_pos_saliency = avg_saliency['Saliency Score'].mean()
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
                        .mark_text(size=15, dx=-10, dy=0, align="center")
                        .encode(x=alt.X("Nucleotide Position", scale=alt.Scale(domain=[1, 3021])),
                                y=alt.Y("Saliency Score:Q"), text="marker",
                                tooltip="description")
                    )

                    rule = base.mark_rule(strokeDash=[2, 2]).encode(
                        y=alt.datum(0),
                        color=alt.value("black")
                    )

                    saliency_chart_high = saliency_chart_high + annotation_layer + rule
                    st.altair_chart(saliency_chart_high, use_container_width=True, theme=None)

            with sal_scat_nucl:
                for scores_arr, probs in zip([actual_scores_high, actual_scores_low], [p_h, p_l]):
                    n_genes = len(probs)
                    df = pd.DataFrame(data={
                        'Base':list(itertools.chain(*[['A'] * n_genes , ['C'] * n_genes,
                                                      ['G'] * n_genes, ['T'] * n_genes])),
                        'Probability of high expression': list(itertools.chain(*[probs for _ in range(4)])),
                        'Sum Saliency Score': np.concatenate([scores_arr.sum(axis=1)[:, i] for i in range(4)]),
                    })

                    chart_title = alt.TitleParams(
                        "Saliency score vs Predicted probabilities",
                        subtitle=[
                            """Base-type""",
                            f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
                        subtitleColor='grey'
                    )
                    saliency_scat = alt.Chart(df, title=chart_title).mark_circle(size=25).encode(
                        x=alt.X('Probability of high expression:Q', scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y('Sum Saliency Score:Q'),
                        color=alt.Color('Base:N',
                                        scale=alt.Scale(range=['green', 'cornflowerblue', 'darkorange',  'red'],
                                                        domain=['A', 'C', 'G', 'T']))
                    )
                    st.altair_chart(saliency_scat, use_container_width=True, theme=None)


        with mutations_tab:
            mut_col, _ = st.columns([0.1, 0.9])
            with mut_col:
                mutate_analysis_type = st.selectbox('Type of mutation analysis', options=['manual', 'VCF'])
            if mutate_analysis_type == 'manual':
                gene_col, _ = st.columns([0.3, 0.7])
                with gene_col:
                    gene_id = st.selectbox(label='Choose gene', options=gene_ids)
                    seq = one_hot_to_dna(x[gene_ids.index(gene_id)])[0]
                    if 'current_gene' not in st.session_state:
                        st.session_state.current_gene = gene_id

                # Initialize session state promoter and terminator sequences ---------------------------

                if "prom_seq" not in st.session_state:
                    st.session_state['prom_seq'] = seq[:1500]
                if "term_seq" not in st.session_state:
                    st.session_state['term_seq'] = seq[1520:]
                if st.session_state.current_gene != gene_id:
                    st.session_state['prom_seq'] = seq[:1500]
                    st.session_state['term_seq'] = seq[1520:]
                    st.session_state.current_gene = gene_id

                prom_col, term_col = st.columns([0.5, 0.5])
                with prom_col:
                    st.subheader("Mutate Promoter")
                    with st.form('promoter_coords', clear_on_submit=False, border=False):
                        row = st.columns([0.1, 0.1, 0.8])
                        with row[0]:
                            prom_mut_start = st.number_input(label='Start', value=0, min_value=0, max_value=1500)
                        with row[1]:
                            prom_mut_end = st.number_input(label='End', value=1500,  min_value=0, max_value=1500)
                        st.form_submit_button('submit')

                    prom_slider = [prom_mut_start, prom_mut_end]
                    prom_subseq = st.text_area(label='Target promoter region',
                                               value=seq[prom_slider[0]:prom_slider[1]],
                                               max_chars=len(seq[prom_slider[0]:prom_slider[1]]),
                                               height=50, key="prom_sub_seq")

                    # Resetting and Mutating
                    reset_cols = st.columns([0.2, 0.2, 0.6])
                    def reset_promoter_seq():
                        st.session_state["prom_sub_seq"] = seq[prom_slider[0]:prom_slider[1]]
                        st.session_state.prom_seq = seq[:1500]

                    with reset_cols[0]:
                        if st.button('Mutate promoter', type="primary"):
                            promoter = st.session_state['prom_seq'][:prom_slider[0]] + prom_subseq + st.session_state['prom_seq'][prom_slider[1]:1500]
                            st.session_state['prom_seq'] = promoter
                        else:
                            promoter = st.session_state['prom_seq']
                    with reset_cols[1]:
                        st.button('Reset promoter', type="primary", on_click=reset_promoter_seq)
                with term_col:
                    st.subheader("Mutate Terminator")
                    with st.form('terminator_coords', clear_on_submit=False, border=False):
                        row = st.columns([0.1, 0.1, 0.8])
                        with row[0]:
                            term_mut_start = st.number_input(label='Start', value=1520, min_value=1520, max_value=3020)
                        with row[1]:
                            term_mut_end = st.number_input(label='End', value=3020,  min_value=1520, max_value=3020)
                        st.form_submit_button('submit')
                    term_slider = [term_mut_start, term_mut_end]
                    term_subseq = st.text_area(label='Target terminator region',
                                               value=seq[term_slider[0]:term_slider[1]],
                                               max_chars=len(seq[term_slider[0]:term_slider[1]]),
                                               height=50, key="term_sub_seq")

                    # Resetting and Mutating buttons
                    reset_cols = st.columns([0.2, 0.2, 0.6])
                    def reset_terminator_seq():
                        st.session_state['term_sub_seq'] = seq[term_slider[0]:term_slider[1]]
                        st.session_state.term_seq = seq[1520:]

                    with reset_cols[0]:
                        if st.button('Mutate terminator', type="primary"):
                            terminator = st.session_state['term_seq'][(1520-1520):(term_slider[0]-1520)] + term_subseq + st.session_state['term_seq'][(term_slider[1]-1520):(3020-1520)]
                            st.session_state['term_seq'] = terminator
                        else:
                            terminator = st.session_state['term_seq']
                    with reset_cols[1]:
                        st.button("Reset terminator", type="primary", on_click=reset_terminator_seq)

                n_pad = 3020 - len(promoter) - len(terminator)
                new_seq = promoter + 'N'*n_pad + terminator
                seqs = np.array([one_hot_encode(i) for i in [seq, new_seq]])
                preds = make_predictions(model=f'models/{deepcre_model}.h5', x=seqs)
                actual_scores, pred_probs, gene_names = extract_scores(seqs=seqs, pred_probs=preds,
                                                                       genes=[gene_id, f'{gene_id}: Mutated'],
                                                                       model=f'models/{deepcre_model}.h5',
                                                                       separate=False)

                pred_chart = alt.Chart(pd.DataFrame({'Probability of high expression':pred_probs,
                                                     'Gene ID': [gene_id, f'{gene_id}: Mutated']})).mark_bar()\
                    .encode(x='Gene ID:N', y='Probability of high expression',
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
                    saliency_chart_high = base.mark_line(line=False, point=False).encode(
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

                    saliency_chart_high = saliency_chart_high + annotation_layer + rule
                    st.altair_chart(saliency_chart_high, use_container_width=True, theme=None)

                def reset_seq():
                    st.session_state["prom_sub_seq"] = seq[prom_slider[0]:prom_slider[1]]
                    st.session_state['term_sub_seq'] = seq[term_slider[0]:term_slider[1]]
                    st.session_state.prom_seq = seq[:1500]
                    st.session_state.term_seq = seq[1520:]
                st.button('Reset', type="primary", on_click=reset_seq)

            else:
                file_upload, _ = st.columns([0.3, 0.7])
                with file_upload:
                    vcf_file = st.file_uploader(label='VCF file', accept_multiple_files=False)
                if vcf_file is not None:
                    if vcf_file.name.endswith('.gz'):
                        vcf_file = io.BytesIO(vcf_file.read())
                        vcf_df = prepare_vcf(uploaded_file=vcf_file)
                        vcf_col, gene_col, _ = st.columns([0.4, 0.5, 0.1], vertical_alignment='center')
                        with vcf_col:
                            st.write('Here are your first 50 SNPs')
                            st.dataframe(vcf_df.head(50))
                        with gene_col:
                            gene_id = st.selectbox(label='Choose gene', options=gene_ids)
                            seq = one_hot_to_dna(x[gene_ids.index(gene_id)])[0]
                            strand = gene_strands[gene_ids.index(gene_id)]
                            start, end = gene_starts[gene_ids.index(gene_id)], gene_ends[gene_ids.index(gene_id)]
                            chrom = gene_chroms[gene_ids.index(gene_id)]
                            snps_in_prom = vcf_df[(vcf_df['Pos'] > start - 1000) & (vcf_df['Pos'] < start + 500) & (vcf_df['Chrom'] == chrom)]
                            snps_in_term = vcf_df[(vcf_df['Pos'] > end - 500) & (vcf_df['Pos'] < end + 1000) & (vcf_df['Chrom'] == chrom)]
                            if strand == '+':
                                snps_in_prom['Region'] = ['Promoter']*snps_in_prom.shape[0]
                                snps_in_term['Region'] = ['Terminator']*snps_in_term.shape[0]
                            else:
                                snps_in_prom['Region'] = ['Terminator'] * snps_in_prom.shape[0]
                                snps_in_term['Region'] = ['Promoter'] * snps_in_term.shape[0]
                            snps_cis_regions = pd.concat([snps_in_prom, snps_in_term], axis=0)
                            snps_cis_regions['Strand'] = [strand]*snps_cis_regions.shape[0]
                            snps_cis_regions.sort_values(by='Region', ascending=True, inplace=True)
                            snps_cis_regions.reset_index(drop=True, inplace=True)
                            if 'current_gene' not in st.session_state:
                                st.session_state.current_gene = gene_id
                            st.write(f'These are the SNPs in the cis-regulatory regions of ' + f':blue[{gene_id}]')
                            selection = dataframe_with_selections(df=snps_cis_regions)
                            st.write('Here is your selected SNP')
                            st.dataframe(selection, use_container_width=True)
                        if not selection.empty:
                            prom_start, prom_end = start-1000, start+500
                            term_start, term_end = end-500, end+1000
                            snp_pos, snp_region = selection['Pos'].values[0], selection['Region'].values[0]
                            ref_allele, alt_allele = selection['Ref'].values[0], selection['Alt'].values[0]
                            complements = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N'}

                            # Initialize session cis-regulatory sequence ---------------------------

                            if "cis_seq" not in st.session_state:
                                st.session_state['cis_seq'] = seq
                            if st.session_state.current_gene != gene_id:
                                st.session_state.current_gene = gene_id
                                st.session_state['cis_seq'] = seq
                            if strand == '+':
                                if snp_region == 'Promoter':
                                    snp_pos = snp_pos - prom_start - 1 if snp_pos != prom_start else 0
                                    snp_pos = 0+snp_pos
                                    ref_pos = st.session_state['cis_seq'][snp_pos]
                                    st.write(ref_pos)
                                else:
                                    snp_pos = snp_pos - term_start - 1 if snp_pos != term_start else 0
                                    snp_pos = 1520+snp_pos
                                    ref_pos = st.session_state['cis_seq'][snp_pos]
                                    st.write(ref_pos)
                            else:
                                if snp_region == 'Promoter':
                                    snp_pos = snp_pos - term_start - 1 if snp_pos != term_start else 0
                                    snp_pos = 1500-snp_pos-1

                                else:
                                    snp_pos = snp_pos - prom_start - 1 if snp_pos != prom_start else 0
                                    snp_pos = 3020-snp_pos-1

                            if st.button('Mutate Sequence', type='primary'):
                                if strand == '+':
                                    mut_cis_seq = st.session_state['cis_seq'][:snp_pos]+alt_allele+st.session_state['cis_seq'][snp_pos+1:]

                                else:
                                    mut_cis_seq = st.session_state['cis_seq'][:snp_pos] + complements[alt_allele] + st.session_state['cis_seq'][snp_pos + 1:]

                                seqs = np.array([one_hot_encode(i) for i in [st.session_state['cis_seq'], mut_cis_seq]])
                                preds = make_predictions(model=f'models/{deepcre_model}.h5', x=seqs)
                                actual_scores, pred_probs, gene_names = extract_scores(seqs=seqs, pred_probs=preds,
                                                                                       genes=[gene_id, f'{gene_id}: Mutated'],
                                                                                       model=f'models/{deepcre_model}.h5',
                                                                                       separate=False)
                                pred_chart = alt.Chart(pd.DataFrame({'Probability of high expression': pred_probs,
                                                                     'Gene ID': [gene_id,
                                                                                 f'{gene_id}: Mutated']})).mark_bar() \
                                    .encode(x='Gene ID:N', y='Probability of high expression',
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
                                    saliency_chart_high = base.mark_line(line=False, point=False).encode(
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
                                        #(snp_pos, text_y, f'{ref_allele} - {alt_allele}', 'single nucleotide polymorphism')
                                    ]
                                    annotations_df = pd.DataFrame(
                                        annotations,
                                        columns=["Nucleotide Position", "Saliency Score", "marker", "description"]
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

                                    # SNP marking
                                    snp_annotation_df = pd.DataFrame([(snp_pos, text_y, f'{ref_allele} - {alt_allele}', 'single nucleotide polymorphism')],
                                                                     columns=["Nucleotide Position", "Saliency Score", "marker", "description"])

                                    snp_annotation_layer = (
                                        alt.Chart(snp_annotation_df)
                                        .mark_text(size=15, dx=0, dy=0, align="center", color='red')
                                        .encode(x=alt.X("Nucleotide Position", scale=alt.Scale(domain=[1, 3021])),
                                                y=alt.Y("Saliency Score:Q"), text="marker",
                                                tooltip="description"))

                                    snp_rule = base.mark_rule(strokeDash=[2, 2]).encode(
                                        x=alt.datum(snp_pos),
                                        color=alt.value("silver")
                                    )
                                    saliency_chart_high = saliency_chart_high + annotation_layer + rule + snp_rule + snp_annotation_layer
                                    st.altair_chart(saliency_chart_high, use_container_width=True, theme=None)

                    else:
                        st.write(':red[Warning: Please upload a .gz file]')





if __name__ == '__main__':
    main()