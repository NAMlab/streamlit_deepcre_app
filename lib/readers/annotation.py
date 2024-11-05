# ------------- Thanks to pyranges package: https://github.com/pyranges/pyranges/blob/master/pyranges/readers.py#L419
# The gtf and gff3 readers code was copied from the pyranges package. Slightly modified

import streamlit as st
import re
from io import StringIO, BytesIO
import pandas as pd

@st.cache_data(max_entries=5)
def read_gene_models(annotation_file):
    if isinstance(annotation_file, st.runtime.uploaded_file_manager.UploadedFile):
        format = 'gtf' if annotation_file.name.endswith(('gtf', 'gtf.gz')) else 'gff3'
        if annotation_file.name.endswith('.gz'):
            compression = 'gzip'
            pd_input = BytesIO(annotation_file.read())
        else:
            compression = None
            pd_input = StringIO(annotation_file.getvalue().decode("utf-8"))
    else:
        format = 'gtf' if annotation_file.endswith(('gtf', 'gtf.gz')) else 'gff3'
        compression = 'gzip' if annotation_file.endswith('.gz') else None
        pd_input = "genomes/annotation/" + annotation_file

    names = ["Chromosome", "Source", "Feature", "Start", "End", "Score", "Strand", "Frame", "Attribute"]
    df_iter = pd.read_csv(pd_input, header=None, comment='#', sep='\t', compression=compression,
                            dtype={"Chromosome": "category", "Feature": "category", "Strand": "category"},
                            names=names, chunksize=int(1e5))

    dfs = []
    for df in df_iter:
        extra = parse_row_gtf(df.Attribute, ignore_bad=False) if format == "gtf" else parse_row_gff3(df.Attribute)
        df = df.drop("Attribute", axis=1)
        extra.set_index(df.index, inplace=True)
        ndf = pd.concat([df, extra], axis=1, sort=False)
        dfs.append(ndf)

    df = pd.concat(dfs, sort=False)
    df.loc[:, "Start"] = df.Start - 1

    df = df[df['Feature'] == 'gene']
    if format == "gtf":
        df = rename_core_attrs(df, ftype="gtf", rename_attr=False)
        df = df[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
    else:
        df['ID'] = df['ID'].apply(lambda x: re.sub(r'^gene:', '', x))
        df = df[['Chromosome', 'Start', 'End', 'Strand', 'ID']]

    df.columns = ['Chromosome', 'Start', 'End', 'Strand', 'gene_id']
    return df

def parse_row_gtf(anno, ignore_bad: bool = False):
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

def parse_row_gff3(anno):
    rowdicts = []

    for line in list(anno):
        # stripping last white char if present
        lx = (it.split("=") for it in line.rstrip("; ").split(";"))
        rowdicts.append({k: v for k, v in lx})

    return pd.DataFrame.from_records(rowdicts).set_index(anno.index)

def parse_kv_fields(line):
    return [kv.replace('""', '"NA"').replace('"', "").split(None, 1) for kv in line.rstrip("; ").split("; ")]

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
