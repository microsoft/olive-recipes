# Requirement Special Commands (WIP)

We support special commands to enable better venv setup. So when one installs requirements manually, one should check these commands first to see if there is any command should be done before install requirements file.

## uvpip

It is in the format of `# uvpip:COMMANDS;pre|post`, for example:

`# uvpip:install onnxruntime-genai-winml==0.8.3 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple --no-deps;post`

It means after installing requirements, run `uv pip install onnxruntime-genai-winml==0.8.3 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple --no-deps`

It could help install individual packages from different sources or with different options.

## download

It is in the format of `# download:XXX.whl`. It will download file from [release](https://github.com/microsoft/windows-ai-studio-templates/releases/tag/model-lab-runtime-0.0.1) and save to [cache](C:\Users\USER\.aitk\bin\model_lab_runtime\cache). The cache folder is the working directory of uv, so downloaded whl could be installed directly.
