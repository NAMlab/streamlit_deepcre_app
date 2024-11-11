import numpy as np
import streamlit as st
from datetime import datetime
import itertools
import pandas as pd
import altair as alt

def show_saliency_tab(actual_scores_high, actual_scores_low, p_h, p_l, color_palette_low_high):
    if actual_scores_low.shape[0] == 0:
        actual_scores_low = np.zeros_like(actual_scores_high)
        p_l = [np.nan]
    if actual_scores_high.shape[0] == 0:
        actual_scores_high = np.zeros_like(actual_scores_low)
        p_h = [np.nan]

    sal_line, sal_scat = st.columns([0.6, 0.4], vertical_alignment='top', gap='medium')
    with sal_line:
        avg_saliency = pd.DataFrame(data={
            'Saliency Score': np.concatenate([actual_scores_high.mean(axis=(0, 2)),
                                        actual_scores_low.mean(axis=(0, 2))], axis=0),
            'Predicted Expression Class': list(itertools.chain(*[['High'] * 3020, ['Low'] * 3020])),
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
            color=alt.Color('Predicted Expression Class:N',
                                scale=alt.Scale(range=color_palette_low_high,
                                                domain=['High', 'Low']))
        )
        max_saliency = avg_saliency['Saliency Score'].max()
        mean_pos_saliency = avg_saliency[avg_saliency['Predicted Expression Class']=='High']['Saliency Score'].mean()
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
        data_sal_scat = pd.DataFrame(data={'Predicted Expression Class':expressed,
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
            color = alt.Color('Predicted Expression Class:N',
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
