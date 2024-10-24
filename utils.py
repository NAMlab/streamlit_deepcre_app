from email.policy import default

import numpy as np
import pandas as pd
from typing import Any
from Bio import SeqIO
from io import StringIO, BytesIO
from deeplift.dinuc_shuffle import dinuc_shuffle
import shap
import streamlit as st
from tensorflow.keras.models import load_model
import gzip
from mimetypes import guess_type
from functools import partial
import re

def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
    """One-hot encode sequence."""
    def to_uint8(string):
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]


@st.cache_data
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

@st.cache_data
def prepare_dataset(genome, annot, gene_list, upstream=1000, downstream=500, new=False):
    if new:
        if genome.name.endswith('.gz'):
            genome = BytesIO(genome.read())
            with gzip.open(filename=genome, mode='rt') as f:
                genome = SeqIO.to_dict(SeqIO.parse(f, format='fasta'))
        else:
            genome = SeqIO.to_dict(SeqIO.parse(StringIO(genome.getvalue().decode("utf-8")), format='fasta'))
    else:
        encoding = guess_type(genome)[1]  # uses file extension
        _open = partial(gzip.open, mode='rt') if encoding == 'gzip' else open
        with _open(genome) as f:
            genome = SeqIO.to_dict(SeqIO.parse(f, format='fasta'))

    genes = pd.read_csv(StringIO(gene_list.getvalue().decode("utf-8")), header=None).values.ravel().tolist()[-1000:]
    if new:
        if annot.name.endswith(('gtf', 'gtf.gz')):
            gene_models = read_gtf(annot, new=new)
            gene_models = gene_models[gene_models['Feature'] == 'gene']
            gene_models = gene_models[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
        else:
            gene_models = read_gff3(annot, new=new)
            gene_models = gene_models[gene_models['Feature'] == 'gene']
            gene_models = gene_models[['Chromosome', 'Start', 'End', 'Strand', 'ID']]
    else:
        if annot.endswith(('gtf', 'gtf.gz')):
            gene_models = read_gtf(annot, new=new)
            gene_models = gene_models[gene_models['Feature'] == 'gene']
            gene_models = gene_models[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
        else:
            gene_models = read_gff3(annot, new=new)
            gene_models = gene_models[gene_models['Feature'] == 'gene']
            gene_models = gene_models[['Chromosome', 'Start', 'End', 'Strand', 'ID']]

    gene_models.columns = ['Chromosome', 'Start', 'End', 'Strand', 'gene_id']
    gene_models = gene_models[gene_models['gene_id'].isin(genes)]

    expected_final_size = 2 * (upstream + downstream) + 20

    x, gene_ids, gene_sizes, gene_chroms, gene_starts, gene_ends, gene_gc_content, gene_strand = [], [], [], [], [], [], [], []
    for chrom, start, end, strand, gene_id in gene_models.values:
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

@st.cache_data
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


@st.cache_resource
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

@st.cache_resource
def make_predictions(model, x):
    model = load_model(model)
    preds = model.predict(x).ravel()
    return preds


# ------------- Thanks to pyranges package: https://github.com/pyranges/pyranges/blob/master/pyranges/readers.py#L419
# The gtf and gff3 readers code was copied from the pyranges package. Slightly modified
_ordered_gtf_columns = [
    "seqname",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attribute",
]
_ordered_gff3_columns = [
    "seqname",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "phase",
    "attribute",
]

def rename_core_attrs(df, ftype, rename_attr=False):
    if ftype == "gtf":
        core_cols = _ordered_gtf_columns
    elif ftype == "gff3":
        core_cols = _ordered_gff3_columns

    dupe_core_cols = list(set(df.columns) & set(core_cols))

    # if duplicate columns were found
    if len(dupe_core_cols) > 0:
        print(f"Found attributes with reserved names: {dupe_core_cols}.")
        if not rename_attr:
            raise ValueError
        else:
            print("Renaming attributes with suffix '_attr'")
            dupe_core_dict = dict()
            for c in dupe_core_cols:
                dupe_core_dict[c] = f"{c}_attr"
            df.rename(dupe_core_dict, axis=1, inplace=True)

    return df


def parse_kv_fields(line):
    return [kv.replace('""', '"NA"').replace('"', "").split(None, 1) for kv in line.rstrip("; ").split("; ")]


def to_rows(anno, ignore_bad: bool = False):
    rowdicts = []
    try:
        line = anno.head(1)
        for line in line:
            line.replace('"', "").replace(";", "").split()
    except AttributeError:
        raise Exception(
            "Invalid attribute string: {line}. If the file is in GFF3 format, use pr.read_gff3 instead.".format(
                line=line
            )
        )

    try:
        for line in anno:
            rowdicts.append({k: v for k, v in parse_kv_fields(line)})
    except ValueError:
        if not ignore_bad:
            print(f"The following line is not parseable as gtf:\n{line}\n\nTo ignore bad lines use ignore_bad=True.")
            raise

    return pd.DataFrame.from_records(rowdicts)


def to_rows_gff3(anno):
    rowdicts = []

    for line in list(anno):
        # stripping last white char if present
        lx = (it.split("=") for it in line.rstrip("; ").split(";"))
        rowdicts.append({k: v for k, v in lx})

    return pd.DataFrame.from_records(rowdicts).set_index(anno.index)


def read_gtf(file_name, new=False):
    names = "Chromosome Source Feature Start End Score Strand Frame Attribute".split()
    if new:
        if file_name.name.endswith('.gz'):
            gtf_file = BytesIO(file_name.read())
            df_iter = pd.read_csv(gtf_file, header=None, comment='#', sep='\t', compression='gzip',
                                  dtype={"Chromosome": "category", "Feature": "category", "Strand": "category"},
                                  names=names, chunksize=int(1e5))
        else:
            df_iter = pd.read_csv(StringIO(file_name.getvalue().decode("utf-8")), header=None, comment='#', sep='\t',
                                  dtype={"Chromosome": "category", "Feature": "category", "Strand": "category"},
                                  names=names, chunksize=int(1e5))

    else:
        df_iter = pd.read_csv(file_name, header=None, comment='#', sep='\t',
                              dtype={"Chromosome": "category", "Feature": "category", "Strand": "category"},
                              names=names, chunksize=int(1e5))

    dfs = []
    for df in df_iter:
        extra = to_rows(df.Attribute, ignore_bad=False)
        df = df.drop("Attribute", axis=1)
        extra.set_index(df.index, inplace=True)
        ndf = pd.concat([df, extra], axis=1, sort=False)
        dfs.append(ndf)

    df = pd.concat(dfs, sort=False)
    df.loc[:, "Start"] = df.Start - 1

    df = rename_core_attrs(df, ftype="gtf", rename_attr=False)
    return df


def read_gff3(file_name, new=False):
    names = "Chromosome Source Feature Start End Score Strand Frame Attribute".split()
    if new:
        if file_name.name.endswith('.gz'):
            gtf_file = BytesIO(file_name.read())
            df_iter = pd.read_csv(gtf_file, header=None, comment='#', sep='\t', compression='gzip',
                                  dtype={"Chromosome": "category", "Feature": "category", "Strand": "category"},
                                  names=names, chunksize=int(1e5))
        else:
            df_iter = pd.read_csv(StringIO(file_name.getvalue().decode("utf-8")), header=None, comment='#', sep='\t',
                                  dtype={"Chromosome": "category", "Feature": "category", "Strand": "category"},
                                  names=names, chunksize=int(1e5))

    else:
        df_iter = pd.read_csv(file_name, header=None, comment='#', sep='\t',
                              dtype={"Chromosome": "category", "Feature": "category", "Strand": "category"},
                              names=names, chunksize=int(1e5))
    dfs = []
    for df in df_iter:
        extra = to_rows_gff3(df.Attribute.astype(str))
        df = df.drop("Attribute", axis=1)
        extra.set_index(df.index, inplace=True)
        ndf = pd.concat([df, extra], axis=1, sort=False)
        dfs.append(ndf)

    df = pd.concat(dfs, sort=False)

    df.loc[:, "Start"] = df.Start - 1
    df['ID'] = df['ID'].apply(lambda x: re.sub(r'^gene:', '', x))

    return df

@st.cache_data
def prepare_vcf(uploaded_file):
    lines = []
    with gzip.open(filename=uploaded_file, mode='rt') as fin:
        for line in fin.readlines():
            if not line.startswith('#'):
                lines.append(line.split('\n')[0].split('\t')[:5])
    lines = pd.DataFrame(lines)
    lines[5] = ['SNP' if len(x[3]) == len(x[4]) == 1 else 'INDEL' for x in lines.values]
    lines = lines[lines[5] == 'SNP']
    lines.columns = ['Chrom', 'Pos', 'ID', 'Ref', 'Alt', 'Annot']
    lines['Pos'] = lines['Pos'].astype('int')
    lines.reset_index(inplace=True, drop=True)
    return lines


def dataframe_with_selections(df):#
    event = st.dataframe(df,
                         on_select='rerun',
                         selection_mode='single-row',
                         use_container_width=True)
    selection_info = event['selection']
    return df.loc[selection_info['rows']]