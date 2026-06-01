pip install -e ./verl
pip install -e ./deepscaler
pip install wandb
pip install pip install --no-deps vllm==0.6.3
pip install outlines==0.0.46 xformers==0.0.27.post2  torchvision==0.19 torch==2.4.0
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install lm-format-enforcer==0.10.6 gguf==0.10.0 pyzmq partial-json-parser msgspec mistral-common 
pip uninstall -y vllm-flash-attn