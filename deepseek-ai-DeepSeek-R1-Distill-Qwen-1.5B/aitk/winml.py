# https://pypi.org/project/windowsml/

def register_execution_providers(ep: str | None = None):
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
    ctypes.WinDLL(ort_dll_path)

    from windowsml import EpCatalog, EpReadyState
    import onnxruntime_genai as og

    with EpCatalog() as catalog:
        for provider in catalog.find_all_providers():
            if ep is not None and provider.name != ep:
                continue

            try:
                provider.ensure_ready()
                og.register_execution_provider_library(provider.name, provider.library_path)
                print(f"Successfully registered execution provider {provider.name} from {provider.library_path}")
            except Exception as e:
                print(
                    f"Execution provider '{provider.name}' is unavailable. Status: {provider.ready_state}; error code: {e}"
                )
