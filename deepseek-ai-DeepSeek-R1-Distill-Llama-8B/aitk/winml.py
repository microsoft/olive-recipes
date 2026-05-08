def register_execution_providers():
    import os
    import ctypes
    import importlib.util

    # Locate onnxruntime package path without importing it first
    ort_spec = importlib.util.find_spec("onnxruntime")
    ort_package_path = os.path.dirname(ort_spec.origin)
    ort_capi_dir = os.path.join(ort_package_path, "capi")
    ort_dll_path = os.path.join(ort_capi_dir, "onnxruntime.dll")

    # Load the onnxruntime DLL because "C:\Windows\System32\onnxruntime.dll" may be exist and loaded first
    ctypes.WinDLL(ort_dll_path)

    import subprocess
    import json
    import sys
    import onnxruntime_genai as og

    worker_script = os.path.abspath('winml.py')
    result = subprocess.check_output([sys.executable, worker_script], text=True)
    paths = json.loads(result)
    for item in paths.items():
        try:
            og.register_execution_provider_library(item[0], item[1])
            print(f"Successfully registered execution provider {item[0]} from {item[1]}")
        except Exception as e:
            print(f"Failed to register execution provider {item[0]} from {item[1]}: {e}")


def _get_ep_paths() -> dict[str, str]:
    from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
        InitializeOptions,
        initialize
    )
    import winui3.microsoft.windows.ai.machinelearning as winml
    eps = {}
    with initialize(options = InitializeOptions.ON_NO_MATCH_SHOW_UI):
        catalog = winml.ExecutionProviderCatalog.get_default()
        providers = catalog.find_all_providers()
        for provider in providers:
            provider.ensure_ready_async().get()
            eps[provider.name] = provider.library_path
            # DO NOT call provider.try_register in python. That will register to the native env.
    return eps


if __name__ == "__main__":
    import json
    eps = _get_ep_paths()
    print(json.dumps(eps))
