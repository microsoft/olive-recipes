import json


def register_execution_providers():
    import ctypes
    import importlib.util
    from pathlib import Path

    # Locate onnxruntime package path without importing it first
    ort_spec = importlib.util.find_spec("onnxruntime")
    assert ort_spec is not None and ort_spec.origin is not None
    ort_package_path = Path(ort_spec.origin).parent
    ort_capi_dir = ort_package_path / "capi"
    ort_dll_path = ort_capi_dir / "onnxruntime.dll"

    # Load the onnxruntime DLL because "C:\Windows\System32\onnxruntime.dll" may be exist and loaded first
    ctypes.WinDLL(str(ort_dll_path))

    import subprocess
    import sys

    import onnxruntime as ort

    worker_script = __file__
    result = subprocess.check_output([sys.executable, worker_script], text=True)
    paths = json.loads(result)
    for item in paths.items():
        try:
            ort.register_execution_provider_library(item[0], item[1])
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
    eps = _get_ep_paths()
    print(json.dumps(eps))
