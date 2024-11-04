import os
from datetime import datetime
import numpy as np
import streamlit as st
from utils import prepare_dataset, extract_scores, make_predictions, one_hot_to_dna, one_hot_encode, prepare_vcf
from utils import dataframe_with_selections, check_file
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

    available_genomes = pd.read_csv("genomes/genomes.csv")
    species = available_genomes["display_name"].tolist()
    species.append("New")
    model_names = sorted([x.split('.')[0] for x in os.listdir('models')])

    organism = st.sidebar.selectbox(label=":four_leaf_clover: Species ( Select **New** for a new species )",
                                    options=species)

    st.subheader(':green[deepCRE: A web-based tool for predicting gene expression from cis-regulatory elements]')

    ### Three main Tabs
    about_tab, preds_tab, interpret_tab, mutations_tab = st.tabs(['About', 'Predictions', 'Saliency Maps', 'Mutation Analysis'])

    # About section-------------------------------------
    with about_tab:
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



    if organism == "New":
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
        genome = available_genomes.loc[available_genomes["display_name"] == organism, "assembly_file"].values[0]
        annot = available_genomes.loc[available_genomes["display_name"] == organism, "annotation_file"].values[0]
    genes_list = st.sidebar.file_uploader(label="genes", type=['.csv', '.txt'], accept_multiple_files=False,
                                          help="""upload a list of max 1000 gene IDs.
                                           Each gene ID must be on a new line. If genes are more than 1000, the
                                           first 1000 genes will be analysed.""")
    if genes_list is not None:
        genes_list = check_file(file=genes_list, file_type="genes list")
    deepcre_model = st.sidebar.selectbox(label="Choose deepCRE model", options=model_names, )
    if genome is not None and annot is not None:
        if genes_list is None:
            with preds_tab:
                if st.button('Use example', type='primary'):
                    use_example = True
                    st.warning(f":red[No gene list uploaded. Displaying results for 100 random genes from the {organism}.]",
                               icon="⚠️")
                else:
                    st.info("""Currently you have not uploaded any data for processing. To see how our tool works please 
                                click on the "use example" button. This will run our tool on 100 sampled genes from the selected 
                                genome. To use your own genes of interest, please uploaded a list of genes at the
                                upload section to the left.
                                """, icon="ℹ️")
                    use_example = False
        else:
            use_example = False

        x, gene_ids, gene_chroms, gene_starts, gene_ends, gene_size, gene_gc_cont, gene_strands = prepare_dataset(genome=genome,
                                                                                                                  annot=annot,
                                                                                                                  gene_list=genes_list,
                                                                                                                  use_example=use_example)
        if x is not None and x.size > 0:
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
                        alt.X('Probability of high expression:Q', bin=alt.Bin(extent=[0, 1], step=0.05)),
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
                if actual_scores_low.shape[0] == 0:
                    actual_scores_low = np.zeros_like(actual_scores_high)
                    g_l = [np.nan]
                    p_l = [np.nan]
                if actual_scores_high.shape[0] == 0:
                    actual_scores_high = np.zeros_like(actual_scores_low)
                    g_h = [np.nan]
                    p_h = [np.nan]

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
                        pd.DataFrame({'x1': [1000], 'x2': [1499]})).mark_rect(
                        opacity=0.1,
                    ).encode(
                        x=alt.X('x1', scale=alt.Scale(domain=[1, 3021]),
                                title='Nucleotide Position'),
                        x2='x2',  # alt.datum(2019),
                        color=alt.value('red'),
                        tooltip=alt.value("5' UTR")
                    )

                    span_3utr = alt.Chart(pd.DataFrame(
                        {'x1': [1519], 'x2': [2019]})).mark_rect(
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
                    saliency_chart = span_prom + span_5utr + span_3utr + span_term + saliency_chart + annotation_layer
                    st.altair_chart(saliency_chart, use_container_width=True, theme=None)

                with sal_scat:
                    sum_saliency_score, pred_prob, expressed = [], [], []
                    if  not np.isnan(p_h[0]):
                        for idx in range(actual_scores_high.shape[0]):
                            sum_saliency_score.append(actual_scores_high[idx].sum())
                            pred_prob.append(p_h[idx])
                            expressed.append('High')
                    if not np.isnan(p_l[0]):
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
                            pd.DataFrame({'x1': [1000], 'x2': [1499]})).mark_rect(
                            opacity=0.1,
                        ).encode(
                            x=alt.X('x1', scale=alt.Scale(domain=[1, 3021]),
                                    title='Nucleotide Position'),
                            x2='x2',  # alt.datum(2019),
                            color=alt.value('red'),
                            tooltip=alt.value("5' UTR")
                        )

                        span_3utr = alt.Chart(pd.DataFrame(
                            {'x1': [1519], 'x2': [2019]})).mark_rect(
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

                        saliency_chart_high = span_prom + span_5utr + span_3utr + span_term + saliency_chart_high + annotation_layer + rule
                        st.altair_chart(saliency_chart_high, use_container_width=True, theme=None)

                with sal_scat_nucl:
                    for scores_arr, probs in zip([actual_scores_high, actual_scores_low], [p_h, p_l]):
                        n_genes = len(probs)
                        if n_genes == scores_arr.shape[0]:
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
                        if 'mutated_seq' not in st.session_state or st.session_state.mutated_seq != seq:
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
                    preds = make_predictions(model=f'models/{deepcre_model}.h5', x=seqs)
                    actual_scores, pred_probs, gene_names = extract_scores(seqs=seqs, pred_probs=preds,
                                                                        genes=[gene_id, f'{gene_id}: Mutated'],
                                                                        model=f'models/{deepcre_model}.h5',
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
                                st.write('Here are your first 50 SNPs')
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
                                st.write(f'These are the SNPs in the cis-regulatory regions of ' + f':blue[{gene_id}]')
                                selection = dataframe_with_selections(df=snps_cis_regions)
                                st.write('Here is your selected SNP')
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
                                    preds = make_predictions(model=f'models/{deepcre_model}.h5', x=seqs)
                                    actual_scores, pred_probs, gene_names = extract_scores(seqs=seqs, pred_probs=preds,
                                                                                        genes=[gene_id, f'{gene_id}: Mutated'],
                                                                                        model=f'models/{deepcre_model}.h5',
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

                                        snp_annotation_layer = (
                                            alt.Chart(snp_annotation_df)
                                            .mark_text(size=15, dx=0, dy=0, align="center", color='red')
                                            .encode(x=alt.X("Nucleotide Position", scale=alt.Scale(domain=[1, 3021])),
                                                    y=alt.Y("Saliency Score:Q"), text="marker",
                                                    tooltip="description"))

                                        for nucl_pos in snp_annotation_df['Nucleotide Position'].values.tolist():
                                            snp_rule = base.mark_rule(strokeDash=[2, 2]).encode(
                                                x=alt.datum(nucl_pos, scale=alt.Scale(domain=[1, 3021])),
                                                color=alt.value("silver")
                                            )
                                            saliency_chart_vcf = saliency_chart_vcf + snp_rule
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

                                        saliency_chart_vcf = span_prom + span_5utr + span_3utr + span_term + saliency_chart_vcf + snp_annotation_layer + rule + annotation_layer
                                        st.altair_chart(saliency_chart_vcf, use_container_width=True, theme=None)

                        else:
                            st.write(':red[Warning: Please upload a .gz file]')


if __name__ == '__main__':
    main()