import streamlit as st
import pandas as pd
from datetime import datetime
import altair as alt

def show_predictions_tab(gene_ids, gene_chroms, gene_starts, gene_ends, gene_size, gene_gc_cont, preds, color_palette_low_high):
                predictions = pd.DataFrame(data={'Gene ID': gene_ids, 'Chromosome': [f'Chr: {i}' for i in gene_chroms],
                                                 'Gene Start': gene_starts,
                                                 'Gene End': gene_ends, 'Gene size': gene_size,
                                                 'GC Content': gene_gc_cont,
                                                 'Probability of high expression': preds,
                                                 'Predicted Expression Class': ['High' if i > 0.5 else 'Low' for i in preds]})

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
                    data_agg = predictions.groupby(['Chromosome', 'Predicted Expression Class']).agg({'Predicted Expression Class': 'count'})
                    data_agg.rename(columns={'Predicted Expression Class': 'Count'}, inplace=True)
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
                        xOffset='Predicted Expression Class',
                        y=alt.Y('Count'),
                        color=alt.Color('Predicted Expression Class:N',
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
                        color=alt.Color('Predicted Expression Class:N', scale=alt.Scale(range=color_palette_low_high,
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
                        color=alt.Color('Predicted Expression Class:N',
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
                        color=alt.Color('Predicted Expression Class:N',
                                        scale=alt.Scale(range=color_palette_low_high,
                                                        domain=['High', 'Low'])),
                    )
                    st.altair_chart(chart_hist_gc, use_container_width=True, theme=None)