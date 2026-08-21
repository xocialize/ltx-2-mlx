#!/bin/bash
set -e
REF=/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx/.work/ltx25-ref
UP=/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-upstream-v120
cd $REF
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch torchaudio "transformers==5.14.1" safetensors accelerate "scipy>=1.14" einops numpy av tqdm pillow "cloudpickle>=3.1" mps-sdpa
uv pip install --python .venv/bin/python --no-deps -e $UP/packages/ltx-core -e $UP/packages/ltx-pipelines
.venv/bin/python -c "import torch, transformers, ltx_core; print('REF ENV OK: torch', torch.__version__, 'mps', torch.backends.mps.is_available(), '| transformers', transformers.__version__)"
echo "SETUP DONE"
