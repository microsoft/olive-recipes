def main():
    import argparse
    import onnxruntime as ort
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--ep")
    parser.add_argument("--model")
    args = parser.parse_args()

    ExecutionProvider="VitisAIExecutionProvider"
    ep_dll = os.path.join(args.ep, "onnxruntime_vitisai_ep.dll")
    custom_dll = os.path.join(args.ep, "onnxruntime_providers_ryzenai.dll")
    cache_dir = os.path.join(args.model, "cache")
    model_file = os.path.join(args.model, "replaced.onnx")

    ort.register_execution_provider_library(ExecutionProvider, ep_dll)
    print("Registered!")

    # Create ONNX runtime session
    def add_ep_for_device(session_options, ep_name, device_type, ep_options=None):
        ep_devices = ort.get_ep_devices()
        for ep_device in ep_devices:
            if ep_device.ep_name == ep_name and ep_device.device.type == device_type:
                print(f"Adding {ep_name} for {device_type}")
                session_options.add_provider_for_devices([ep_device], {} if ep_options is None else ep_options)
                break

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(custom_dll)
    print("Registered custom dll!")
    session_options.add_session_config_entry("dd_cache",cache_dir)
    add_ep_for_device(session_options, ExecutionProvider, ort.OrtHardwareDeviceType.NPU, {"target": "SD"})

    print("Setup!")
    session = ort.InferenceSession(
        model_file,
        sess_options=session_options,
    )
    print("Loaded!")


if __name__ == "__main__":
    main()

# 2026-07-08 11:38:35.5641183 [W:onnxruntime:Default, onnxruntime_pybind_module.cc:45 onnxruntime::python::CreateOrtEnv] Init provider bridge failed.
# WARNING: Logging before InitGoogleLogging() is written to STDERR
# I20260708 11:38:35.686201 30764 register_dynamicdispatch.cpp:49] Running DynamicDispatchOpRegister::register_ops
# I20260708 11:38:35.686201 30764 register_castavx.cpp:48] Running CastAvxOpRegister::register_ops
# Adding VitisAIExecutionProvider for OrtHardwareDeviceType.NPU
# 2026-07-08 11:38:35.7665003 [W:onnxruntime:, session_state.cc:1359 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
# 2026-07-08 11:38:35.7718621 [W:onnxruntime:, session_state.cc:1361 onnxruntime::VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
# Failed to initialize fusion runtime for node 'NhwcConv_0-conv_inConv': [C:\Users\z1aiebuild\dod\src\passes\analyze_buffer_reqs.cpp:441] Invoking OpInterface::get_buffer_reqs() failed !!
# Details:
#   Op Name: NhwcConv_0-/conv_in/Conv
#   Op Type: SDConv
#   Error: Transaction not found: sd_fastpm_conv2d_a16bfw16bfpacc16bf_2_320_4_64_64_64_64_3_3_param2026-07-08 11:38:36.6053108 [E:onnxruntime:, inference_session.cc:2684 onnxruntime::InferenceSession::Initialize::<lambda_7>::operator ()] Exception during initialization: [C:\Users\z1aiebuild\dod\src\passes\analyze_buffer_reqs.cpp:441] Invoking OpInterface::get_buffer_reqs() failed !!
# Details:
#   Op Name: NhwcConv_0-/conv_in/Conv
#   Op Type: SDConv
#   Error: Transaction not found: sd_fastpm_conv2d_a16bfw16bfpacc16bf_2_320_4_64_64_64_64_3_3_param
# Traceback (most recent call last):
#   File "olive-recipes\sd-legacy-stable-diffusion-v1-5\test_sd.py", line 41, in <module>
#     main()
#   File "olive-recipes\sd-legacy-stable-diffusion-v1-5\test_sd.py", line 33, in main
#     session = ort.InferenceSession(
#               ^^^^^^^^^^^^^^^^^^^^^
#   File ".aitk\bin\model_lab_runtime\Python-WCR-win32-x64-3.12.9\Lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 529, in __init__
#     self._create_inference_session(providers, provider_options, disabled_optimizers)
#   File ".aitk\bin\model_lab_runtime\Python-WCR-win32-x64-3.12.9\Lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 635, in _create_inference_session
#     sess.initialize_session(providers, provider_options, disabled_optimizers)
# onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Exception during initialization: [C:\Users\z1aiebuild\dod\src\passes\analyze_buffer_reqs.cpp:441] Invoking OpInterface::get_buffer_reqs() failed !!
# Details:
#   Op Name: NhwcConv_0-/conv_in/Conv
#   Op Type: SDConv
#   Error: Transaction not found: sd_fastpm_conv2d_a16bfw16bfpacc16bf_2_320_4_64_64_64_64_3_3_param

