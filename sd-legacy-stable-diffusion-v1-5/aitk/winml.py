def register_execution_providers():
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
