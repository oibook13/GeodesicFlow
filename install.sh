# https://github.com/bghira/SimpleTuner/blob/39c05a7355ff0de5f3579b24b103c3151eee6823/documentation/quickstart/SD3.md
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda create -y --name RiemannianRF python=3.11
conda activate RiemannianRF
pip install -U poetry
poetry install

pip install huggingface_hub
pip install wandb
# https://github.com/microsoft/DeepSpeed/issues/2772
# edit export CUDA_HOME=/usr/local/cuda-12.2 in train.sh
conda install -c nvidia cuda-compiler
pip install torch torchvision torchaudio
pip install 'triton==3.0.0' --force-reinstall
pip install bitsandbytes==0.45.5

# for evaluating FID, IS, and CLIP-Score
pip install matplotlib
pip install torch-fidelity
pip install git+https://github.com/openai/CLIP.git

# if we come across any issues with diffusers version
# It works with transformers==4.51.3
pip install --force-reinstall diffusers==0.31.0
pip install --force-reinstall torchao==0.5.0

pip install dagshub
pip install datasets
pip install ipywidgets