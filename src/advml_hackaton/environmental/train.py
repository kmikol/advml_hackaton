import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

import tensorflow_model_optimization as tfmot


from data_utils import get_data, get_data_meta
from model_utils import (set_seed,
                            build_base_model,
                            count_dense_macs,
                            representative_ds_fn,
                            to_int8_tflite,
                            write_c_array_hex,
                            write_normalization_header)



def tflite_mae(tfl_bytes: bytes, X: np.ndarray, y: np.ndarray) -> float:
    interp = tf.lite.Interpreter(model_content=tfl_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    X = X.astype(np.float32)
    y = y.astype(np.float32).ravel()
    mae = 0.0
    for i in range(X.shape[0]):
        x1 = X[i:i+1]
        x_q = np.round(x1 / in_scale + in_zp).astype(np.int8)
        interp.set_tensor(inp["index"], x_q)
        interp.invoke()
        y_q = interp.get_tensor(out["index"]).astype(np.int32)
        y_pred = (y_q - out_zp) * out_scale
        mae += abs(float(y_pred.ravel()[0]) - float(y[i]))
    return mae / X.shape[0]





# -------------------------- Main Train --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/power_consumption_tetouan")
    parser.add_argument("--params_path", type=str, default="artifacts/best_model_params.yaml")
    parser.add_argument("--out_dir", type=str, default="model_registry/env")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # Load data
    train, val, test, data_params = get_data(args.data_path)
    input_dim = int(train["X"].shape[1])

    # Load YAML
    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f) or {}

    layers          = int(params.get("layers", 2))
    units           = int(params.get("units", 16))
    learning_rate   = float(params.get("learning_rate", 3e-4))
    epochs          = int(params.get("epochs", 30))
    batch_size      = int(params.get("batch_size", 64))
    use_qat         = bool(params.get("use_qat", True))
    out_act         = str(params.get("output_activation", "linear"))
    patience        = int(params.get("patience", 8))

    # Build model
    base = build_base_model(input_dim, layers, units, out_act)

    if use_qat:
        if tfmot is None:
            raise RuntimeError(
                "best_model_params.yaml sets use_qat: true, "
                "but tensorflow-model-optimization is not installed. "
                "pip install tensorflow-model-optimization"
            )
        model = tfmot.quantization.keras.quantize_model(base)
    else:
        model = base

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    hist = model.fit(
        train["X"], train["y"],
        validation_data=(val["X"], val["y"]),
        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[es]
    )

    # Evaluate (Keras)
    val_res  = model.evaluate(val["X"],  val["y"],  verbose=0, return_dict=True)
    test_res = model.evaluate(test["X"], test["y"], verbose=0, return_dict=True)

    # Convert to fully-INT8 TFLite & evaluate
    rep = lambda: representative_ds_fn(train["X"], n=256, seed=args.seed)
    tfl = to_int8_tflite(model, rep)
    val_mae_tfl  = tflite_mae(tfl,  val["X"],  val["y"])
    test_mae_tfl = tflite_mae(tfl,  test["X"], test["y"])

    # Complexity
    macs   = int(count_dense_macs(model))
    params_count = int(model.count_params())
    tfl_kb = len(tfl) / 1024.0

    # Save artifacts
    # 1) Keras model
    model_path = out_dir / "model.keras"
    model.save(model_path)

    # 2) TFLite
    tfl_path = out_dir / "model.tflite"
    tfl_path.write_bytes(tfl)

    # 3) Arduino C array
    cc_path, h_path = write_c_array_hex(tfl, out_dir=out_dir)

    write_normalization_header(stats=get_data_meta(args.data_path),
                               out_dir=out_dir,
                               filename="normalization_data.h",
                               namespace="norm")

    # 4) Metrics
    metrics = {
        "layers": layers,
        "units": units,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "use_qat": use_qat,
        "output_activation": out_act,
        "patience": patience,
        "val_loss_keras": float(val_res["loss"]),
        "val_mae_keras": float(val_res["mae"]),
        "test_loss_keras": float(test_res["loss"]),
        "test_mae_keras": float(test_res["mae"]),
        "val_mae_tflite": float(val_mae_tfl),
        "test_mae_tflite": float(test_mae_tfl),
        "macs": macs,
        "params": params_count,
        "tflite_kb": tfl_kb,
        "artifacts": {
            "model.keras": str(model_path),
            "model.tflite": str(tfl_path),
            "model_data.cc": cc_path,
            "model_data.h": h_path,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 5) Copy params used
    (out_dir / "best_params_used.yaml").write_text(yaml.safe_dump(params))

    # Console summary
    print("\n=== Training complete ===")
    print(f"Val  MAE (Keras): {metrics['val_mae_keras']:.5f} | Val  MAE (TFLite INT8): {metrics['val_mae_tflite']:.5f}")
    print(f"Test MAE (Keras): {metrics['test_mae_keras']:.5f} | Test MAE (TFLite INT8): {metrics['test_mae_tflite']:.5f}")
    print(f"Params: {params_count:,} | MACs: {macs:,} | TFLite size: {tfl_kb:.1f} KB")
    print(f"Artifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()