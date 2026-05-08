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

    from windowsml import EpCatalog
    import onnxruntime as ort

    with EpCatalog() as catalog:
        for ep in catalog.find_all_providers():
            try:
                # Ensure the provider is ready (downloads/installs if needed)
                ep.ensure_ready()
                ort.register_execution_provider_library(ep.name, ep.library_path)
                print(f"Successfully registered execution provider {ep.name} from {ep.library_path}")
            except Exception as e:
                print(f"Failed to register execution provider {ep.name} from {ep.library_path}: {e}")
