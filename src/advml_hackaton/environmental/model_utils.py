"""
This module provides utility functions for building, quantizing, and exporting machine learning models, 
as well as generating normalization headers for embedded systems.

Main functionalities:
- Setting random seeds for reproducibility.
- Building TensorFlow models with customizable layers and activations.
- Quantizing models to INT8 format for deployment on resource-constrained devices.
- Writing model data and normalization parameters to C/C++ headers for embedded use.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import re

def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility in both NumPy and TensorFlow.

    Args:
        seed (int): The random seed value. Default is 42.

    Returns:
        None
    """
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_base_model(input_dim: int, n_layers: int, n_units: int, out_act: str):
    """
    Builds a sequential TensorFlow model with specified input dimensions, layers, and activation functions.

    Args:
        input_dim (int): The number of input features.
        n_layers (int): The number of hidden layers.
        n_units (int): The number of units in each hidden layer.
        out_act (str): The activation function for the output layer.

    Returns:
        tf.keras.Sequential: The constructed TensorFlow model.
    """
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,), name="x")])
    for _ in range(n_layers):
        m.add(tf.keras.layers.Dense(n_units, activation="relu"))
    # NOTE: if your target isn't scaled to [-1, 1], "linear" is usually better than "tanh"
    m.add(tf.keras.layers.Dense(1, activation=out_act))
    return m

def count_dense_macs(model: tf.keras.Model) -> int:
    """
    Calculates the Multiply-Accumulate Operations (MACs) for Dense layers in a TensorFlow model.

    Args:
        model (tf.keras.Model): The TensorFlow model to analyze.

    Returns:
        int: The total number of MACs for Dense layers.
    """
    macs = 0
    for layer in model.layers:
        base = getattr(layer, "layer", layer)
        if isinstance(base, tf.keras.layers.Dense) and getattr(base, "kernel", None) is not None:
            in_dim, out_dim = base.kernel.shape
            macs += int(in_dim) * int(out_dim)
    return macs


def representative_ds_fn(X, n=256, seed=0):
    """
    Generates a representative dataset for TensorFlow Lite quantization.

    Args:
        X (array-like): The input data.
        n (int): The number of samples to include in the representative dataset. Default is 256.
        seed (int): Random seed for sampling. Default is 0.

    Yields:
        list: A batch of input samples for quantization.
    """
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if X.shape[0] > n:
        X = X[rng.choice(X.shape[0], n, replace=False)]
    for i in range(X.shape[0]):
        yield [X[i:i+1]]


def to_int8_tflite(model: tf.keras.Model, rep_ds):
    """
    Converts a TensorFlow model to an INT8-quantized TensorFlow Lite model.

    Args:
        model (tf.keras.Model): The TensorFlow model to convert.
        rep_ds (function): A representative dataset function for quantization.

    Returns:
        bytes: The INT8-quantized TensorFlow Lite model as a byte array.
    """
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_ds
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()





def write_c_array(tflite_bytes: bytes, var="g_model", out_dir=Path("artifacts")):
    """
    Writes a TensorFlow Lite model as a C array for embedded deployment.

    Args:
        tflite_bytes (bytes): The TensorFlow Lite model in byte format.
        var (str): The variable name for the C array. Default is "g_model".
        out_dir (Path): The output directory for the generated files. Default is "artifacts".

    Returns:
        tuple: Paths to the generated .cc and .h files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h_path = out_dir / "model.h"
    cc_path = out_dir / "model.cc"
    h_path.write_text(
        "#pragma once\n#include <cstdint>\n\n"
        f"extern const unsigned char {var}[];\n"
        f"extern const int {var}_len;\n"
    )
    lines = [f'#include "model_data.h"\n', f"alignas(8) const unsigned char {var}[] = {{"]
    B = tflite_bytes
    for i in range(0, len(B), 12):
        lines.append("  " + ", ".join(str(b) for b in B[i:i+12]) + ",")
    lines.append("};")
    lines.append(f"const int {var}_len = {len(B)};")
    cc_path.write_text("\n".join(lines))
    return str(cc_path), str(h_path)

def write_c_array_hex(tflite_bytes: bytes, var="g_model",
                      out_dir=Path("artifacts"), bytes_per_line=16, uppercase=True):
    """
    Writes a TensorFlow Lite model as a C array in hexadecimal format for embedded deployment.

    Args:
        tflite_bytes (bytes): The TensorFlow Lite model in byte format.
        var (str): The variable name for the C array. Default is "g_model".
        out_dir (Path): The output directory for the generated files. Default is "artifacts".
        bytes_per_line (int): The number of bytes per line in the output. Default is 16.
        uppercase (bool): Whether to use uppercase hexadecimal letters. Default is True.

    Returns:
        tuple: Paths to the generated .cc and .h files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h_path = out_dir / "model.h"
    cc_path = out_dir / "model.cc"

    # header: no align needed here (it only matters on the definition)
    h_path.write_text(
        "#pragma once\n#include <cstdint>\n\n"
        f"extern const uint8_t {var}[];\n"
        f"extern const int {var}_len;\n"
    )

    fmt = "0x{b:02X}" if uppercase else "0x{b:02x}"
    lines = [
        '#include "model.h"\n',
        f"alignas(16) const uint8_t {var}[] = {{"
    ]
    B = tflite_bytes
    for i in range(0, len(B), bytes_per_line):
        chunk = ", ".join(fmt.format(b=b) for b in B[i:i+bytes_per_line])
        lines.append(f"  {chunk},")
    lines.append("};")
    lines.append(f"const int {var}_len = {len(B)};")
    cc_path.write_text("\n".join(lines))
    return str(cc_path), str(h_path)




def write_normalization_header(stats: dict,
                               out_dir: Path = Path("artifacts"),
                               filename: str = "normalization_data.h",
                               namespace: str = "norm"):
    """
    Generates a C++ header file containing normalization parameters and helper functions.

    Args:
        stats (dict): A dictionary containing normalization statistics (e.g., means, stds).
        out_dir (Path): The output directory for the header file. Default is "artifacts".
        filename (str): The name of the header file. Default is "normalization_data.h".
        namespace (str): The C++ namespace for the generated code. Default is "norm".

    Returns:
        str: The path to the generated header file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    # --- validate & pull values ---
    x_feats = stats.get("x_features") or []
    y_feats = stats.get("y_features") or []
    x_mean  = stats.get("x_mean") or []
    x_std   = stats.get("x_std") or []
    y_mean  = stats.get("y_mean") or []
    y_std   = stats.get("y_std") or []

    if not (len(x_feats) == len(x_mean) == len(x_std)):
        raise ValueError("x_features, x_mean, x_std must have equal lengths")
    if len(y_mean) != 1 or len(y_std) != 1:
        # header supports multiple Ys, but most MCU regressors have one
        pass

    def c_ident(s: str) -> str:
        s = s.strip()
        s = re.sub(r"[^0-9A-Za-z_]", "_", s)
        if re.match(r"^[0-9]", s):
            s = "_" + s
        return s

    # numeric formatting (compact but precise)
    def f(v: float) -> str:
        return format(float(v), ".17g")

    k_num_x = len(x_feats)
    k_num_y = len(y_mean)

    # --- build header text ---
    lines = []
    lines.append("// Auto-generated normalization parameters")
    lines.append("#pragma once")
    lines.append("#include <cstddef>")
    lines.append("")
    lines.append(f"namespace {namespace} {{")
    lines.append(f"constexpr std::size_t kNumX = {k_num_x};")
    lines.append(f"constexpr std::size_t kNumY = {k_num_y};")
    lines.append("")

    # Optional: feature indices enum for readability
    if x_feats:
        lines.append("enum XFeatureIndex {")
        for i, name in enumerate(x_feats):
            lines.append(f"  X_{c_ident(name).upper()} = {i}, // {name}")
        lines.append("};")
        lines.append("")

    # Arrays (constexpr so header-only is safe)
    if x_mean:
        lines.append(f"constexpr float kXMean[kNumX] = {{ {', '.join(f(v) for v in x_mean)} }};")
    if x_std:
        lines.append(f"constexpr float kXStd[kNumX]  = {{ {', '.join(f(v) for v in x_std)} }};")

    if y_mean:
        lines.append(f"constexpr float kYMean[{k_num_y}] = {{ {', '.join(f(v) for v in y_mean)} }};")
    if y_std:
        lines.append(f"constexpr float kYStd[{k_num_y}]  = {{ {', '.join(f(v) for v in y_std)} }};")

    lines.append("")

    # Inline helpers
    lines.append("// Normalize inputs: out[i] = (in[i] - kXMean[i]) / max(kXStd[i], 1e-12)")
    lines.append("inline void NormalizeX(const float in[kNumX], float out[kNumX]) {")
    lines.append("  for (std::size_t i = 0; i < kNumX; ++i) {")
    lines.append("    const float s = (kXStd[i] == 0.0f) ? 1.0f : kXStd[i];")
    lines.append("    out[i] = (in[i] - kXMean[i]) / s;")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append("// Normalize a single Y (index 0); extend if you add more outputs")
    lines.append("inline float NormalizeY(float y, std::size_t idx = 0) {")
    lines.append("  const float s = (kYStd[idx] == 0.0f) ? 1.0f : kYStd[idx];")
    lines.append("  return (y - kYMean[idx]) / s;")
    lines.append("}")
    lines.append("")
    lines.append("// Denormalize Y back to original units")
    lines.append("inline float DenormalizeY(float y_norm, std::size_t idx = 0) {")
    lines.append("  return y_norm * kYStd[idx] + kYMean[idx];")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace " + namespace)
    lines.append("")

    path.write_text("\n".join(lines))
    return str(path)