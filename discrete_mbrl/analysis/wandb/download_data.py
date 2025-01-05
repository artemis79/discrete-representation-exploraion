import argparse
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import os
import wandb


DEFAULT_METRICS = [
    'n_step', 'eval_policies', 'random_state_distrib_kl_div', 
    'random_state_distrib_kl_div_mean', 'goal_state_distrib_kl_div', 
    'goal_state_distrib_kl_div_mean', 'explore_right_state_distrib_kl_div', 
    'explore_right_state_distrib_kl_div_mean'
]
DEFAULT_PARAMS = [
    'codebook_size', 'filter_size', 'eval_policy', 'env_name', 
    'ae_model_type'
]


def run_to_rows(run, metrics_names, param_names, index='step'):
    """
    Converts a wandb run to rows of data containing metrics and parameters.

    Args:
        run (wandb.ApiRun): A single wandb run.
        metrics_names (list): List of metric names to fetch.
        param_names (list): List of parameter names to fetch.
        index (str): The column to use as an index (default is 'step').

    Returns:
        list[dict]: A list of dictionaries containing the combined data.
    """
    # Fetch history (metrics over time)


    # Gather parameter values
    metrics_names = metrics_names
    metric_data = run.scan_history(keys=metrics_names)
    param_data = run.config

    # Gather parameter values
    run_id = run.id
    param_vals = {name: param_data.get(name, None) for name in param_names}
    param_vals.update({'run_id': run_id})
 

    # Coalesce metric data into rows
    rows = []
    for row in metric_data:
        row.update(param_vals)
        rows.append(row)


    return rows
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', type=str, required=True, help='W&B entity (user or team).')
    parser.add_argument('--project', type=str, required=True, help='W&B project name.')
    parser.add_argument('--history_vars', nargs='*', type=str, default=None, help='Metrics to fetch.')
    parser.add_argument('--params', nargs='*', type=str, default=None, help='Parameters to fetch.')
    parser.add_argument('--output_path', type=str, default=None)
    # parser.add_argument('--rm_reward', action='store_true')
    args = parser.parse_args()

    # Initialize wandb API
    api = wandb.Api(timeout=39)

    print('Fetching runs...')
    runs = api.runs(f"{args.entity}/{args.project}")
    print(f'Found {len(runs)} runs.')

    # Use default metrics and params if not provided
    metrics = args.history_vars if args.history_vars else DEFAULT_METRICS
    params = args.params if args.params else DEFAULT_PARAMS
    columns = metrics + params + ['run_id']
    
    output_path = f"analysis/wandb/{args.project}_data.csv"

# Create or overwrite the CSV file with headers
    with open(output_path, 'w') as f:
        pd.DataFrame(columns=metrics + ['beta', 'run_id']).to_csv(f, index=False)

    n_valid_runs = 0
    for run in tqdm(runs):
        try:
            # Process the run and convert it into rows
            new_rows = run_to_rows(run, metrics, params, index='step')
            if new_rows:
                n_valid_runs += 1

                # Append the new rows to the CSV file
                pd.DataFrame(new_rows).to_csv(output_path, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error fetching data for run {run.id}: {e}")

    print(f'{n_valid_runs}/{len(runs)} runs saved.')
    print(f'Data saved to {output_path}.')