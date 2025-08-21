from data_utils import get_data
import argparse
import os
import csv
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.linear_model import LinearRegression
import yaml
import optuna

from model_utils import (set_seed,build_base_model,count_dense_macs,representative_ds_fn,to_int8_tflite)

# silence warnings
import warnings
warnings.filterwarnings("ignore")

# --------------- HPO (Optuna) ---------------
def run_optuna_hpo(
    train, val, input_dim,
    n_trials=40, pretrain_epochs=10, qat_epochs=10,
    batch_size=64, float_lr=1e-3, qat_lr=3e-4,
    use_qat=True,  # turn QAT on/off
    optimize_on_tflite_mae=True  # use INT8 val MAE as objective if True
):
    rng = np.random.default_rng(0)
    tf.keras.utils.set_random_seed(0)

    def count_dense_macs(model: tf.keras.Model) -> int:
        macs = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense) and layer.kernel is not None:
                in_dim, out_dim = layer.kernel.shape
                macs += int(in_dim) * int(out_dim)
        return macs

    def tflite_val_mae(tfl_bytes: bytes) -> float:
        interp = tf.lite.Interpreter(model_content=tfl_bytes)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
        in_scale, in_zp = inp["quantization"]; out_scale, out_zp = out["quantization"]

        Xv = val["X"].astype(np.float32)
        yv = val["y"].astype(np.float32).ravel()
        mae_sum = 0.0
        for i in range(Xv.shape[0]):
            x = Xv[i:i+1]
            x_q = np.round(x / in_scale + in_zp).astype(np.int8)
            interp.set_tensor(inp["index"], x_q)
            interp.invoke()
            y_q = interp.get_tensor(out["index"])
            y_pred = (y_q.astype(np.int32) - out_zp) * out_scale
            mae_sum += abs(float(y_pred.ravel()[0]) - float(yv[i]))
        return mae_sum / Xv.shape[0]

    def objective(trial: optuna.Trial):
        layers = trial.suggest_int("layers", 1, 6)
        units  = trial.suggest_int("units", 4, 64)

        # 1) Float pre-train
        float_model = build_base_model(input_dim, layers, units, out_act='tanh')
        float_model.compile(optimizer=tf.keras.optimizers.Adam(float_lr), loss="mse", metrics=["mae"])
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        float_model.fit(
            train["X"], train["y"],
            validation_data=(val["X"], val["y"]),
            epochs=pretrain_epochs, batch_size=batch_size, verbose=0, callbacks=[es]
        )

        if use_qat and qat_epochs > 0:
            # 2) QAT fine-tune
            qat_model = make_qat_model(float_model)
            qat_model.compile(optimizer=tf.keras.optimizers.Adam(qat_lr), loss="mse", metrics=["mae"])
            es_q = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            qat_model.fit(
                train["X"], train["y"],
                validation_data=(val["X"], val["y"]),
                epochs=qat_epochs, batch_size=batch_size, verbose=0, callbacks=[es_q]
            )
            model_for_export = qat_model
        else:
            model_for_export = float_model

        # Complexity metric
        macs = int(count_dense_macs(model_for_export))
        params = int(model_for_export.count_params())

        # Objective MAE (choose Keras or TFLite-INT8)
        if optimize_on_tflite_mae:
            tfl = to_int8_tflite(model_for_export, rep_ds=lambda: representative_ds_fn(train["X"]))
            val_mae_tfl = float(tflite_val_mae(tfl))
            val_mae_keras = float(model_for_export.evaluate(val["X"], val["y"], verbose=0, return_dict=True)["mae"])
        else:
            res = model_for_export.evaluate(val["X"], val["y"], verbose=0, return_dict=True)
            val_mae_keras = float(res["mae"])
            tfl = to_int8_tflite(model_for_export, rep_ds=lambda: representative_ds_fn(train["X"]))
            val_mae_tfl = float(tflite_val_mae(tfl))

        # Stash extras for CSV
        trial.set_user_attr("params", params)
        trial.set_user_attr("val_mae_keras", val_mae_keras)
        trial.set_user_attr("val_mae_tflite", val_mae_tfl)
        trial.set_user_attr("tflite_kb", len(tfl) / 1024.0)

        # Multi-objective: minimize (MAE target, MACs)
        return (val_mae_tfl if optimize_on_tflite_mae else val_mae_keras), macs

    study = optuna.create_study(directions=["minimize", "minimize"], study_name="ops_vs_mae_qat")
    study.optimize(objective, n_trials=n_trials)

    # --- Dump Pareto front ---
    pf = study.best_trials
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    out_csv = Path("artifacts/pareto_front.csv")
    if pf:
        import csv
        rows = []
        for t in pf:
            rows.append({
                "trial": t.number,
                "obj_mae": t.values[0],
                "macs": t.values[1],
                "layers": t.params.get("layers"),
                "units": t.params.get("units"),
                "params": t.user_attrs.get("params"),
                "val_mae_keras": t.user_attrs.get("val_mae_keras"),
                "val_mae_tflite": t.user_attrs.get("val_mae_tflite"),
                "tflite_kb": t.user_attrs.get("tflite_kb"),
                "quantization": "uint8"
            })
        rows.sort(key=lambda r: (r["obj_mae"], r["macs"]))
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Saved Pareto front ({len(rows)} models) -> {out_csv}")
    else:
        print("No Pareto-optimal trials found.")

    return study


def main(args):
    # get dataset
    train, val, test, data_params = get_data(args.data_path)


    # ---------- Optuna HPO + Pareto dump ----------
    study = run_optuna_hpo(
        train, val, input_dim=train["X"].shape[1],
        n_trials=args.hpo_trials,
        pretrain_epochs=args.pretrain_epochs,
        qat_epochs=args.qat_epochs,
        use_qat=args.use_qat,
        optimize_on_tflite_mae=args.optimize_on_tflite_mae
    )



if __name__ == "__main__":

    conf = yaml.safe_load(open("config/config.yaml"))

    parser = argparse.ArgumentParser(description="Power Consumption Training")
    parser.add_argument("--data_path", type=str, default=conf["data_path"],
                        help="Path to the data directory")
    parser.add_argument("--model_registry_path", type=str, default=conf["model_registry_path"],
                        help="Path to the model registry directory")

    # NEW: HPO controls
    parser.add_argument("--use_qat", action="store_true", help="Enable Quantization Aware Training")
    parser.add_argument("--pretrain_epochs", type=int, default=conf['pretrain_epochs'])
    parser.add_argument("--qat_epochs", type=int, default=conf['qat_epochs'])
    parser.add_argument("--hpo_trials", type=int, default=conf['hpo_trials'])
    parser.add_argument("--optimize_on_tflite_mae", action="store_true",
                        help="Use TFLite-INT8 validation MAE as the HPO objective")

    args = parser.parse_args()
    main(args)