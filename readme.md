# VAE-Subtype-Sample-Generation operations
### Goal: Generate TCGA cohort samples
    based on encoding of all TCGA samples
    use gene expression data type only

pip install -r requirements.txt  
python3 -m venv venv  
source venv/bin/activate  

### 2022-10-04  
VAE_smpl_gen_04.ipynb pushed, connected to UMAP now  

2022-08-08
test on 1000 x 100 make_classification fabricated data
Use latent space dimension of 10 b/c order mag < 100
sbatch fbrctd.sh 10 fbrctd_00 data/1000_100_X.tsv

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


