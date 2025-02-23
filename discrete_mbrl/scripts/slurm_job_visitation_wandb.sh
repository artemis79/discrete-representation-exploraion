#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=1 
#SBATCH --gpus-per-node=1 
#SBATCH --mem=16G 
#SBATCH --time=0-2:59
#SBATCH --array=1-30
#SBATCH --output=model_free/delayed_visitation_count_%j.out



echo "Starting task $SLURM_ARRAY_TASK_ID"
# SOCKS5 Proxy
if [ "$SLURM_TMPDIR" != "" ]; then
    echo "Setting up SOCKS5 proxy..."
    ssh -q -N -T -f -D 8888 `echo $SSH_CONNECTION | cut -d " " -f 3`
    export ALL_PROXY=socks5h://localhost:8888
fi
 
module load python/3.10 StdEnv/2023 gcc opencv/4.8.1 swig

cd $SLURM_TMPDIR

export ALL_PROXY=socks5h://localhost:8888

# Clone project
git config --global http.proxy 'socks5://127.0.0.1:8888'
git clone https://github.com/artemis79/discrete-representation-exploraion

#Install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh
cd discrete-representation-exploraion/

# Install packages
uv venv $SLURM_TMPDIR/.venv --python 3.10
source $SLURM_TMPDIR/.venv/bin/activate
uv pip install -r pyproject.toml --cache-dir $SLURM_TMPDIR/uv/cache
uv pip install requests[socks] 

cd discrete_mbrl/model_free

python train.py --wandb --count --ae_recon_loss --ae_er_train --log_pos --device "cuda" --log_freq 10000 --learning_rate 0.0003 --batch_size 256 \
                --mf_steps 1000000 --embedding_dim 64 --env_max_steps 1000 --n_ae_updates 8 --ppo_batch_size 64 --ppo_value_coef 0.5 \
                --ppo_iters 10 --ppo_clip 0.2 --rl_start_step 500000 --ae_model_type "vqvae" --filter_size 6 --codebook_size 256 \
                --beta 0.01 --env_name "minigrid-door-key-stochastic" --stochastic "categorical" --rm_reward

wandb sync --sync-all

