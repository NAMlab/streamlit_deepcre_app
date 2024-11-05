import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import itertools
from datetime import datetime

def choose_analysis_type():
    mut_col, _ = st.columns([0.2, 0.8])
    with mut_col:
        mutate_analysis_type = st.radio(':gray[Choose] :green[**analysis type**]',
                                        options=['**manual**', '**VCF**'],
                                        captions=[
                                            "Manually mutate sequence",
                                            "Use natural variants from VCF"
                                        ])
    return mutate_analysis_type
    
def show_manual_mutation(gene_id, start, end, seq, utr_len, central_pad_size):
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
    return mut_reg_start, mut_reg_end

def show_mutation_results(gene_id, pred_probs, actual_scores, seq, utr_len, central_pad_size, mut_reg_start, mut_reg_end, mut_markers = None):
 
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

        # SNP marking
        if mut_markers is not None:
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
 
        if mut_markers is None:
            saliency_chart_mut = span_prom + span_5utr + span_3utr + span_term + saliency_chart_mut + annotation_layer + rule
        else:
            saliency_chart_mut = span_prom + span_5utr + span_3utr + span_term + snp_annotation_layer + saliency_chart_mut + rule + annotation_layer
        st.altair_chart(saliency_chart_mut, use_container_width=True, theme=None)
 
    def reset_seq():
        st.session_state.mutated_seq = seq
        st.session_state.sub_seq_to_mutate = seq[mut_reg_start:mut_reg_end]
 
    if mut_markers is None:
        st.button('Reset', type="primary", on_click=reset_seq)

def show_vcf_input():
    file_upload, _ = st.columns([0.3, 0.7])
    with file_upload:
        vcf_file = st.file_uploader(label='VCF file', accept_multiple_files=False, type=['.gz'],
                                    help="""upload a VCF file. File should be in .gz format""")
    return vcf_file