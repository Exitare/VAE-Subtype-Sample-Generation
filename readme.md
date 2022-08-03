VAE-Subtype-Sample-Generation operations
Goal is to generate TCGA cohort samples
    based on encoding of all TCGA sample
    use gene expression data type only

2022-08-03
file prepped with 8009 samples and 99 subtypes
labels encoded
to /VAE-Subtype-Sample-Generation/data on Exacloud
sbatch sample_gen_data.sh 1000 v0.2 data/cmbnd_25_lbl_xfrm.tsv
* expecting out_of_memory fail

2022-08-02
sbatch sample_gen.sh -lt 1000 -p v00 --data data/BRCA_feat_ntrsct_25.tsv
sbatch sample_gen.sh 1000 v0.1 path_to_file
sbatch sample_gen_data.sh 1000 v0.1 data/BRCA_feat_ntrsct_25.tsv

next actions:
    clone into Exacloud
    upload gene expression extraction from molecular file scripts
    read direct from GDAN TMP dir (?)

2022-07-28

output and error report directories setup

25 is latent space column dimension
v03 is result dir name
sbatch sample_gen.sh 25 v03 data/trn_v00.tsv data/val_v00.tsv data/tst_v00.tsv


