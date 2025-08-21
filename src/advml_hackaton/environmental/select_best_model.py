import argparse
import pandas as pd
import yaml
import os

def main(args):

    # load pareto front data
    pareto_quant_df = pd.read_csv(os.path.join(args.artifacts_path, "pareto_front_quantized.csv"))


    # find trial with the lowest val_mae under the size threshold
    # order the data by val_mae
    filtered_df = pareto_quant_df[pareto_quant_df['tflite_kb'] < args.max_model_size_kb]
    best_trial = filtered_df.loc[filtered_df[args.metric].idxmin()]

    
    # write yaml with the parameters of the best model to artifacts
    with open(os.path.join(args.artifacts_path, "best_model_params.yaml"), "w") as f:
        yaml.dump(best_trial.to_dict(), f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Select the best model from the Pareto front.")
    parser.add_argument("--artifacts_path", type=str, default="artifacts/", 
                            help="Path to the artifacts directory.")
    parser.add_argument("--metric", type=str, default="val_mae_tflite", help="The metric to optimize.")
    parser.add_argument("--max_model_size_kb", type=float, default=16, help="The maximum model size (KB) for the selected metric.")
    args = parser.parse_args()

    main(args)