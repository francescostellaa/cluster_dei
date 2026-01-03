#!/bin/bash
#SBATCH -J Evo2_full_genome_emb
#SBATCH -o evo2_test_emb/evo2_test_emb_output/output_test_emb_%j.txt
#SBATCH -e evo2_test_emb/evo2_test_emb_err/errors_test_emb_%j.txt
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH --mail-user francesco.stella.1@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH -p allgroups
#SBATCH --mem 48G
#SBATCH --gres=gpu:l40s


PROJECT_DIR="/nfsd/biogenomica/stellafran"

cd $PROJECT_DIR 
# This is to save the downloaded model 
mkdir -p ${PROJECT_DIR}/huggingface

echo "Job started on node: $(hostname)"
echo "Running Evo2 7 bilion test to generate embeddings..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# /usr/local/cuda/compat/lib/libcuda.so.1
# Define sungularity execution command
srun singularity exec --nv \
--bind ${PROJECT_DIR}:/workdir \
--pwd /workdir \
--bind ${PROJECT_DIR}/huggingface:/workdir/huggingface \
--bind /usr/lib64/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1 \
--bind /usr/lib64/libcuda.so.1:/usr/local/cuda/lib64/stubs/libcuda.so.1 \
--env HF_HOME=/workdir/huggingface \
evo2-basic.sif \
python3 evo2_test_emb/evo2_emb.py --data data --output output_emb --window_size 32768 --stride 32768
#python3 evo2_test_emb/evo2_test_emb.py
#--bind ${PROJECT_DIR}/data:/workdir/data \
echo "Job finished."
