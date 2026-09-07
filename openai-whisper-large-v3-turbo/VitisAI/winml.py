# MIT License
#
# Copyright (C) 2026, Advanced Micro Devices, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (C) [2026] Advanced Micro Devices, Inc. All Rights Reserved.

import json

def _get_ep_paths() -> dict[str, str]:
    from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
        InitializeOptions,
        initialize
    )
    import winui3.microsoft.windows.ai.machinelearning as winml
    eps = {}
    with initialize(options = InitializeOptions.ON_NO_MATCH_SHOW_UI):
        pass
        catalog = winml.ExecutionProviderCatalog.get_default()
        providers = catalog.find_all_providers()
        for provider in providers:
            if provider.name != 'VitisAIExecutionProvider':
                continue
            if provider.ready_state == winml.ExecutionProviderReadyState.READY:
                eps[provider.name] = provider.library_path
                continue
            elif provider.ready_state == winml.ExecutionProviderReadyState.NOT_PRESENT:
                # print(f"Downloading and ensuring EP {provider.name} is ready. This may take a while...")
                result = provider.ensure_ready_async().get()
                if result.status != winml.ExecutionProviderReadyResultState.SUCCESS:
                    print(f"Failed to ensure EP {provider.name} is ready. Status: {result.status}")
            elif provider.ready_state == winml.ExecutionProviderReadyState.NOT_READY:
                # print(f"Ensuring EP {provider.name} is ready.")
                result = provider.ensure_ready_async().get()
                if result.status != winml.ExecutionProviderReadyResultState.SUCCESS:
                    print(f"Failed to ensure EP {provider.name} is ready. Status: {result.status}")
            else:
                print(f"EP {provider.name} is in unexpected state {provider.ready_state}")
            eps[provider.name] = provider.library_path
            # DO NOT call provider.try_register in python. That will register to the native env.
    return eps

if __name__ == "__main__":
    eps = _get_ep_paths()
    print(json.dumps(eps))
