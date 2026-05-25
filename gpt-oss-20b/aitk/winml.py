# https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers?tabs=python#production-app-example


def _get_ep_paths() -> dict[str, str]:
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

    # remove the msvcp140.dll from the winrt-runtime package.
    # So it does not cause issues with other libraries.
    from importlib import metadata

    site_packages_path = Path(str(metadata.distribution("winrt-runtime").locate_file("")))
    dll_path = site_packages_path / "winrt" / "msvcp140.dll"
    if dll_path.exists():
        dll_path.unlink()

    import winui3.microsoft.windows.ai.machinelearning as winml
    from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import InitializeOptions, initialize

    eps = {}
    with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
        catalog = winml.ExecutionProviderCatalog.get_default()
        providers = catalog.find_all_providers()
        for provider in providers:
            result = provider.ensure_ready_async().get()
            if result.status == winml.ExecutionProviderReadyResultState.SUCCESS:
                eps[provider.name] = provider.library_path
            else:
                print(
                    f"Execution provider '{provider.name}' is unavailable. Status: {result.status}; reason: {result.diagnostic_text}; error code: {result.extended_error.value}"
                )
    return eps


def register_execution_providers():
    paths = _get_ep_paths()

    import onnxruntime_genai as og

    for item in paths.items():
        try:
            og.register_execution_provider_library(item[0], item[1])  # pyright: ignore[reportAttributeAccessIssue]
            print(f"Successfully registered execution provider {item[0]} from {item[1]}")
        except Exception as e:
            print(f"Failed to register execution provider {item[0]} from {item[1]}: {e}")
