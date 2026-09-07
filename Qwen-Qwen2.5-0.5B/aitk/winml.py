# https://pypi.org/project/windowsml/


def _get_ep_paths(ep: str | None = None) -> dict[str, str]:
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

    from windowsml import EpCatalog, EpReadyState

    eps = {}
    with EpCatalog() as catalog:
        providers = catalog.find_all_providers()
        for provider in providers:
            if ep is not None and provider.name != ep:
                continue
            try:
                provider.ensure_ready()
            except Exception as e:
                print(f"Execution provider '{provider.name}' is unavailable. Error code: {e}")
                continue
            if provider.ready_state == EpReadyState.Ready:
                eps[provider.name] = provider.library_path
            else:
                print(f"Execution provider '{provider.name}' is unavailable. Status: {provider.ready_state}")
    return eps


def register_execution_providers(ep: str | None = None):
    paths = _get_ep_paths(ep)

    import onnxruntime_genai as og

    for item in paths.items():
        try:
            og.register_execution_provider_library(item[0], item[1])  # pyright: ignore[reportAttributeAccessIssue]
            print(f"Successfully registered execution provider {item[0]} from {item[1]}")
        except Exception as e:
            print(f"Failed to register execution provider {item[0]} from {item[1]}: {e}")
