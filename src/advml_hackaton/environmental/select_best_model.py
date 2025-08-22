"""
This module selects the best model from the Pareto front based on a specified metric and size constraint. 
It reads the Pareto front data, filters models by size, and selects the one with the optimal metric value.

Main functionalities:
- Filtering models based on size constraints.
- Selecting the best model based on a specified metric.
- Exporting the parameters of the selected model to a YAML file.
"""

import argparse
import pandas as pd
import yaml
import os

def main(args):
    """
    Main function to select the best model from the Pareto front.

    Steps:
    - Loads the Pareto front data from a CSV file.
    - Filters models based on the maximum size constraint.
    - Selects the model with the optimal value for the specified metric.
    - Exports the parameters of the selected model to a YAML file.

    Args:
        args (argparse.Namespace): Command-line arguments containing paths, metric, and size constraints.

    Returns:
        None
    """

    # load pareto front data
    pareto_quant_df = pd.read_csv(os.path.join(args.artifacts_path, "pareto_front.csv"))


    # find trial with the lowest val_mae under the size threshold
    # order the data by val_mae
    filtered_df = pareto_quant_df[pareto_quant_df['tflite_kb'] < args.max_model_size_kb]
    best_trial = filtered_df.loc[filtered_df[args.metric].idxmin()]

    
    # write yaml with the parameters of the best model to artifacts
    with open(os.path.join(args.artifacts_path, "best_model_params.yaml"), "w") as f:
        yaml.dump(best_trial.to_dict(), f)


if __name__ == "__main__":

    conf = yaml.safe_load(open("config/config.yaml"))

    parser = argparse.ArgumentParser(description="Select the best model from the Pareto front.")
    parser.add_argument("--artifacts_path", type=str, default=conf["artifacts_path"], 
                            help="Path to the artifacts directory.")
    parser.add_argument("--metric", type=str, default="val_mae_tflite", help="The metric to optimize.")
    parser.add_argument("--max_model_size_kb", type=float, default=conf['max_model_size_kb'], help="The maximum model size (KB) for the selected metric.")
    args = parser.parse_args()

    main(args)