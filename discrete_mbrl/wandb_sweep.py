import argparse
import ast
import copy
import os
import signal
import subprocess
import time
import yaml


import wandb



# Create args
parser = argparse.ArgumentParser()

parser.add_argument('-s', '--sweep_id', default=None)
parser.add_argument('-c', '--count', type=int, default=1)
parser.add_argument('-p', '--config', type=str, nargs='*', default=None)

def create_sweep(config_path):
    with open(config_path, 'r') as file:
        sweep_config = yaml.safe_load(file)
        return wandb.sweep(sweep_config)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.config:
        config_ids = []
        config_names = []
        for config in args.config:
        # Get file name from path
            config_names.append(config.split('/')[-1])
            new_id = create_sweep(config)
            # config_ids.extend(new_ids)
        
        print('Created sweeps with ids:\n', ', '.join(config_ids))
        print('From configs:\n', ', '.join(config_names))


    if args.sweep_id is not None:
        if args.sweep_id == 'new':
            args.sweep_id = new_id
        for _ in range(args.count):
            wandb.agent(args.sweep_id, count=args.count)


  