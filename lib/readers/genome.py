from Bio import SeqIO
from io import StringIO, BytesIO
import streamlit as st
import gzip

# cache_resource keeps the parsed genome in memory as-is; cache_data would
# pickle and copy the whole genome on every cache hit, which is very slow.
@st.cache_resource(max_entries=5, show_spinner=False)
def read_genome(genome_file):
    if isinstance(genome_file, st.runtime.uploaded_file_manager.UploadedFile):
        if genome_file.name.endswith('.gz'):
            compressed = True
            seqio_input = BytesIO(genome_file.read())
        else:
            compressed = False
            seqio_input = StringIO(genome_file.getvalue().decode("utf-8"))
    else:
        compressed = genome_file.endswith('.gz')
        seqio_input = "genomes/assembly/" + genome_file

    if compressed:
        with gzip.open(seqio_input, mode='rt') as f:
            genome = SeqIO.to_dict(SeqIO.parse(f, format='fasta'))
    else:
        genome = SeqIO.to_dict(SeqIO.parse(seqio_input, format='fasta'))
    
    return genome