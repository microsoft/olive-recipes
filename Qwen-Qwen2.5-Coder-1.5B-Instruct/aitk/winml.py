import json

def _get_ep_paths() -> dict[str, str]:
    from windowsml import EpCatalog
    eps = {}
    with EpCatalog() as catalog:
        for ep in catalog.find_all_providers():
            # Ensure the provider is ready (downloads/installs if needed)
            ep.ensure_ready()
            eps[ep.name] = ep.library_path
    return eps

if __name__ == "__main__":
    eps = _get_ep_paths()
    print(json.dumps(eps))
