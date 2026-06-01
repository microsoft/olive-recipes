# How to setup Intel Profiling Tool

To see op-level profiling data when running on Intel (OpenVINO) NPU/GPU, you need to do the following:

- Download [Intel® Unified Telemetry](https://www.intel.com/content/www/us/en/developer/tools/intel-unified-telemetry.html)
- Extract the content to a path like C:\Users\XXX\Downloads\ut-tool-ext-v0.2.0-beta1.1
- In Foundry Toolkit settings, set "Model Lab Intel Unified Telemetry Path" to C:\Users\XXX\Downloads\ut-tool-ext-v0.2.0-beta1.1\bin
    - Point to the bin folder where ut.exe exists

Intel Unified Telemetry needs admin permission to run. So if you are running VS Code in non-admin mode, during profiling, it will ask for an elevated permission and a new terminal window will pop up showing the progressing of running it. Please wait and do not close that window.

If your VS Code is already running in admin mode, no ask happened and no additional window.
