## Selectable Genomes
In this folder, you can provide genome assemblies and structural annotations that users can select in the app without having to upload them themselves.
Please put the chromosome fasta files in gzipped format into the `assembly` folder and the structural annotations in GFF3 or GTF format in the `annotation` folder.
Then you can define which genomes the user can select from in the `genomes.csv`.
For example, let's say you want to provide the *Arabidopsis thaliana* TAIR10 genome.
After adding the chromosome fasta file (let's call it `Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz`) and annotation file (`Arabidopsis_thaliana.TAIR10.60.gff3.gz`) in their respective folders, you can add the following entry to the `genomes.csv` file:

```
"Arabidopsis thaliana (TAIR10)","This is the latest version of the Arabidopsis thaliana reference genome, aqcuired from Ensembl Plants on <date>","Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz","Arabidopsis_thaliana.TAIR10.60.gff3.gz"
```

It will then show up as "Arabidopsis thaliana (TAIR10) in the list and carry the description you entered.
If you have different structural annotations referencing the same sequence fasta file, you can also refer to the same files in multiple entries.