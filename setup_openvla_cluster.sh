#!/bin/bash
# setup_openvla_cluster.sh -- one-time setup of OpenVLA on the LFC-HPC cluster (everything goes into your home folder).
#
# Before you run it, upload to your home folder (Files menu):
#     patch_openvla.py        and        1.0.0.zip   (the zip of the dataset folder 1.0.0)
#
# Run it so it keeps going if the browser tab closes:
#     nohup bash setup_openvla_cluster.sh > setup.log 2>&1 &
#     tail -f setup.log            (Ctrl+C stops only the tail, not the setup)
# It takes about 20-40 minutes. The last line of the log says "SETUP FINISHED" when everything worked.
set -e
cd "$HOME"

echo "=== 1/6 Miniforge (conda) ==="
if [ ! -d "$HOME/miniforge3" ]; then
  curl -L -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3.sh -b -p "$HOME/miniforge3"
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"

echo "=== 2/6 Python 3.10 environment 'openvla' ==="
conda env list | grep -q '^openvla ' || conda create -n openvla python=3.10 -y
conda activate openvla
python --version

echo "=== 3/6 OpenVLA code + drone patch ==="
[ -d "$HOME/openvla" ] || git clone https://github.com/openvla/openvla.git "$HOME/openvla"
python "$HOME/patch_openvla.py" "$HOME/openvla"

echo "=== 4/6 OpenVLA and its packages (torch 2.2.0, tensorflow 2.15, ...) ==="
cd "$HOME/openvla"
pip install -e .
pip install packaging ninja
ninja --version
# Package pins found necessary on the cluster: the newest tensorflow-metadata needs a newer protobuf
# than TensorFlow 2.15 allows. pip prints a harmless conflict warning after the second line.
pip install "tensorflow-metadata<1.17"
pip install "protobuf==4.25.9"

echo "=== 5/6 Flash Attention (optional: LoRA fine-tuning also works without it) ==="
pip install "flash-attn==2.5.5" --no-build-isolation || echo "flash-attn could not be installed -- skipped, this is fine"

echo "=== 6/6 Dataset and model download ==="
mkdir -p "$HOME/tensorflow_datasets/drone_dataset"
if [ -f "$HOME/1.0.0.zip" ]; then            # zip that contains the folder 1.0.0
  python -m zipfile -e "$HOME/1.0.0.zip" "$HOME/tensorflow_datasets/drone_dataset"
elif [ -f "$HOME/drone_dataset.zip" ]; then  # zip that contains drone_dataset/1.0.0
  python -m zipfile -e "$HOME/drone_dataset.zip" "$HOME/tensorflow_datasets"
else
  echo "ERROR: no 1.0.0.zip or drone_dataset.zip in your home folder"; exit 1
fi
ls -la "$HOME/tensorflow_datasets/drone_dataset/1.0.0/"
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('openvla/openvla-7b'))"

echo "=== Check: torch sees the GPU? ==="
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

echo "=== Check: the dataset reads with the OpenVLA versions of tensorflow_datasets? ==="
python - <<'PYEOF' 2>&1 | grep -v -E "^(I0000|W0000|E0000|WARNING: All log|To enable|.*cuda|.*TF-TRT|.*oneDNN)" || true
import os
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
b = tfds.builder_from_directory(os.path.expanduser("~/tensorflow_datasets/drone_dataset/1.0.0"))
for s in ("train", "val"):
    n_ep = n_st = 0
    for ep in b.as_dataset(split=s):
        n_ep += 1; n_st += len(list(ep["steps"]))
    print(s, n_ep, "episodes", n_st, "steps")
PYEOF
echo "(expected: train 45 episodes 6048 steps / val 5 episodes 826 steps)"
echo "SETUP FINISHED"
