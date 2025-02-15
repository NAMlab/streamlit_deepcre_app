import numpy as np
import pandas as pd
from typing import Any
from deeplift.dinuc_shuffle import dinuc_shuffle
import shap
import streamlit as st
from tensorflow.keras.models import load_model
import gzip
from lib.readers.annotation import read_gene_models
from lib.readers.genome import read_genome

def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
    """One-hot encode sequence."""
    def to_uint8(string):
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
    sequence = sequence.upper()
    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]


def one_hot_to_dna(one_hot):
    one_hot = np.expand_dims(one_hot, axis=0) if len(one_hot.shape) == 2 else one_hot
    bases = np.array(["A", "C", "G", "T", "N"])
    one_hot_inds = np.tile(one_hot.shape[2], one_hot.shape[:2])
    batch_inds, seq_inds, base_inds = np.where(one_hot)
    one_hot_inds[batch_inds, seq_inds] = base_inds
    seq_array = bases[one_hot_inds]
    return ["".join(seq) for seq in seq_array]


def compute_gc(enc_seq):
    c_count = np.where(enc_seq[:, [1, 2]] == 1)[0]
    return len(c_count)/enc_seq.shape[0]

def prepare_dataset(genome, annot, gene_list, upstream=1000, downstream=500, use_example=False):
    genome = read_genome(genome) 
    gene_models = read_gene_models(annot)

    if use_example:
        # If no genes are provided, we just show results for 100 random ones
        gene_models_overlap = gene_models.sample(n=100)
    elif gene_list is not None:
        genes = gene_list
        if len(genes) > 1000:
            st.warning(":red[You uploaded more than 1000 genes. Only the first 1000 genes will be considered for the analysis.]")
            genes = genes[:1000]
        gene_models_overlap = gene_models[gene_models['gene_id'].isin(genes)]
        if gene_models_overlap.empty:
            st.error("None of the genes in your list were found in the genome annotation. " + 
                    "Please check you're using the correct formatting for gene IDs. " +
                    "Here are 8 random genes from the genome: " + 
                    ', '.join(np.random.choice(gene_models['gene_id'].values, 8)))
            return None, None, None, None, None, None, None, None
    else:
        return None, None, None, None, None, None, None, None
    expected_final_size = 2 * (upstream + downstream) + 20

    x, gene_ids, gene_sizes, gene_chroms, gene_starts, gene_ends, gene_gc_content, gene_strand = [], [], [], [], [], [], [], []
    for chrom, start, end, strand, gene_id in gene_models_overlap.values:
        gene_size = end - start
        extractable_downstream = downstream if gene_size // 2 > downstream else gene_size // 2
        prom_start, prom_end = start - upstream, start + extractable_downstream
        term_start, term_end = end - extractable_downstream, end + upstream

        promoter = one_hot_encode(str(genome[chrom][prom_start:prom_end].seq))
        terminator = one_hot_encode(str(genome[chrom][term_start:term_end].seq))
        extracted_size = promoter.shape[0] + terminator.shape[0]
        central_pad_size = expected_final_size - extracted_size

        pad_size = central_pad_size

        if strand == '+':
            seq = np.concatenate([
                promoter,
                np.zeros(shape=(pad_size, 4)),
                terminator
            ])
        else:
            seq = np.concatenate([
                terminator[::-1, ::-1],
                np.zeros(shape=(pad_size, 4)),
                promoter[::-1, ::-1]
            ])

        if seq.shape[0] == expected_final_size:
            x.append(seq)
            gene_ids.append(gene_id)
            gene_chroms.append(chrom)
            gene_starts.append(start)
            gene_ends.append(end)
            gene_sizes.append(gene_size)
            gene_gc_content.append(compute_gc(seq))
            gene_strand.append(strand)

    x = np.array(x)
    return x, gene_ids, gene_chroms, gene_starts, gene_ends, gene_sizes, gene_gc_content, gene_strand


def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example, seed=1234):
    assert len(list_containing_input_modes_for_an_example) == 1
    onehot_seq = list_containing_input_modes_for_an_example[0]
    rng = np.random.RandomState(seed)
    to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(10)])

    return [to_return]


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    for l in range(len(mult)):
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape) == 2
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:, i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None, :, :] - bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference * mult[l]
            projected_hypothetical_contribs[:, :, i] = np.sum(hypothetical_contribs, axis=-1)
        to_return.append(np.mean(projected_hypothetical_contribs, axis=0))
    return to_return

def compute_actual_hypothetical_scores(x, model):
    model = load_model(model)
    shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
    shap.explainers.deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers.deep.deep_tf.linearity_1d(0)
    dinuc_shuff_explainer = shap.DeepExplainer(
        (model.input, model.layers[-2].output[:, 0]),
        data=dinuc_shuffle_several_times,
        combine_mult_and_diffref=combine_mult_and_diffref)
    hypothetical_scores = dinuc_shuff_explainer.shap_values(x)
    actual_scores = hypothetical_scores * x
    return actual_scores


def extract_scores(seqs, pred_probs, genes, model, separate=True):
    if separate:
        x_low, x_high, g_low, g_high, probs_low, probs_high = [], [], [], [], [], []
        for idx, pred in enumerate(pred_probs):
            if pred > 0.5:
                x_high.append(seqs[idx])
                g_high.append(genes[idx])
                probs_high.append(pred_probs[idx])
            else:
                x_low.append(seqs[idx])
                g_low.append(genes[idx])
                probs_low.append(pred_probs[idx])

        x_low, x_high = np.array(x_low), np.array(x_high)
        actual_scores_low = compute_actual_hypothetical_scores(x=x_low, model=model)
        actual_scores_high = compute_actual_hypothetical_scores(x=x_high, model=model)
        return actual_scores_low, actual_scores_high, g_low, g_high, probs_low, probs_high
    else:
        actual_scores = compute_actual_hypothetical_scores(x=seqs, model=model)
        return actual_scores, pred_probs, genes

def make_predictions(model, x):
    model = load_model(model)
    preds = model.predict(x).ravel()
    return preds

def prepare_vcf(uploaded_file, gene_starts, gene_ends, gene_chroms):
    iter_lines = pd.read_csv(uploaded_file, delimiter='\t', comment='#', usecols=[0, 1, 2, 3, 4],
                             names=['Chrom', 'Pos', 'ID', 'Ref', 'Alt'],
                             dtype={'Chrom': 'category', 'Pos': 'int', 'Ref': 'category', 'Alt': 'category'},
                             compression='gzip',
                             chunksize=100000,
                             header=None,
                             iterator=True)
    lines = pd.concat([chunk[(chunk['Ref'].str.len() == 1) & (chunk['Alt'].str.len() == 1) &
                             (((chunk['Pos'] > s-1000) & (chunk['Pos'] < s+500) & (chunk['Chrom'] == c)) |
                              ((chunk['Pos'] > e-500) & (chunk['Pos'] < e+1000) & (chunk['Chrom'] == c)))]
                       for chunk in iter_lines
                       for s, e, c in zip(gene_starts, gene_ends, gene_chroms)])
    return lines


def dataframe_with_selections(df):#
    event = st.dataframe(df,
                         on_select='rerun',
                         selection_mode='multi-row',
                         use_container_width=True)
    selection_info = event['selection']
    return df.loc[selection_info['rows']]

def check_file(file, file_type):
    if file.size > 0:
        return_file = file
    else:
        return_file = None
        st.error(f":red[The uploaded {file_type} file is empty. Please verify !]", icon="⚠️")
    return return_file
