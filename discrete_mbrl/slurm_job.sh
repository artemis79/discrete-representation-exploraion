#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=1 
#SBATCH --gpus-per-node=1 
#SBATCH --mem=16G 
#SBATCH --time=0-2:59
#SBATCH --array=1-30


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
uv venv
source .venv/bin/activate

uv pip sync pyproject.toml
uv add 'requests[socks]' 

cd discrete_mbrl/

if [[ "$1" == "ae_door_key" ]]; then
   python comet_sweep.py --sweep_id "new" --config "sweep_configs/comet_ml/model_free_experiments/delayed_vanilla_mf/ae_door_key.json"
elif [[ "$1" == "softmax_ae_door_key" ]]; then
   python comet_sweep.py --sweep_id "new" --config "sweep_configs/comet_ml/model_free_experiments/delayed_vanilla_mf/softmax_ae_door_key.json" 
elif [[ "$1" == "fta_ae_door_key" ]]; then
   python comet_sweep.py --sweep_id "new" --config "sweep_configs/comet_ml/model_free_experiments/delayed_vanilla_mf/fta_ae_door_key.json"
elif [[ "$1" == "vqvae_door_key" ]]; then
   python comet_sweep.py --sweep_id "new" --config "sweep_configs/comet_ml/model_free_experiments/delayed_vanilla_mf/vqvae_door_key.json"
elif [[ "$1" == "vqvae_count_door_key" ]]; then
   python comet_sweep.py --sweep_id "new" --config "sweep_configs/comet_ml/model_free_experiments/delayed_vanilla_mf/vqvae_count_door_key.json"
elif [[ "$1" == "vqvae_count_door_key_sweep" ]]; then
   python comet_sweep.py --sweep_id "new" --config "sweep_configs/comet_ml/model_free_experiments/delayed_vanilla_mf_sweep/vqvae_count_door_key.json"
else
   echo "Not a valid sweep"
fi



