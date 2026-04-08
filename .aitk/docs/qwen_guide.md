# Convert CPU model

```
cd Qwen-Qwen3.5-4B\builtin
uv venv -p 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install onnxruntime==1.26.0.dev20260407005 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple
uv pip install onnxruntime-genai==0.13.0.dev20260402 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple
uv run optimize.py --config-dir cpu_and_mobile --device cpu
```

# Run

```
uv run inference.py --prompt "What is the capital of France?"
```

# Convert GPU model

```
cd Qwen-Qwen3.5-9B\builtin
uv venv -p 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install onnxruntime-gpu==1.26.0.dev20260407002 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple
uv pip install onnxruntime-genai-cuda==0.12.0.dev20260205 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple
uv run optimize.py --config-dir cuda --device gpu
```

# Run

```
uv run inference.py --prompt "What is the capital of France?"
```
