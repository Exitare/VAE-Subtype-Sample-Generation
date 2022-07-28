VAE-Subtype-Sample-Generation operations
Goal is to generate TCGA cohort samples
    based on encoding of all TCGA sample
    use gene expression data type only

next actions:
    clone into Exacloud
    upload gene expression extraction from molecular file scripts
    read direct from GDAN TMP dir (?)

# 2022-07-28
# 25 is latent space column dimension
# v03 is result dir name
sbatch sample_gen.sh 25 v03 data/trn_v00.tsv data/val_v00.tsv data/tst_v00.tsv


