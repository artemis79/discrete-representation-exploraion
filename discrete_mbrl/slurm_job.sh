#!/bin/bash
#SBATCH --account=def-mbowling
#SBATCH --cpus-per-task=1 
#SBATCH --gpus-per-node=1 
#SBATCH --mem=16G 
#SBATCH --time=0-2:59'

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
uv add requests[socks]
uv pip sync pyproject.toml


cd discrete_mbrl/
