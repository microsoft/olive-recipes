# ==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2025 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
# ==============================================================================
import os
import subprocess


def get_total_file_size(directory, filename_suffix):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(filename_suffix):
                fp = os.path.join(dirpath, filename)
                total_size += os.path.getsize(fp)
    return total_size

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size_bytes < 1024.0:
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} {unit}"

def run_profile_viewer(profile_viewer_bin_path, profiling_logs_dirpath, log_file_name):
    return subprocess.run(
        [
            profile_viewer_bin_path,
            "--input_log",
            f"{profiling_logs_dirpath}/{log_file_name}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

def get_inference_time(profile_viewer_bin_path, profiling_logs_dirpath, log_file_name):
    # Run profile viewer
    result = run_profile_viewer(
        profile_viewer_bin_path, profiling_logs_dirpath, log_file_name
    )

    logs = result.stdout.splitlines()
    average_stats = False
    for line in logs:
        if line.strip() == "Execute Stats (Average):":
            average_stats = True
        elif average_stats and "Backend (QNN accelerator (execute) time):" in line:
            inference_time = line.strip().split(":")
            return int(inference_time[1].strip().split(" ")[0].strip())
    raise ValueError(f"No Inference logs found in {log_file_name}!")

def get_adapter_switch_time(
    profile_viewer_bin_path, profiling_logs_dirpath, log_file_name
):
    # Run profile viewer
    result = run_profile_viewer(
        profile_viewer_bin_path, profiling_logs_dirpath, log_file_name
    )

    logs = result.stdout.splitlines()
    average_stats = False
    for line in logs:
        if line.strip() == "ApplyBinarySection Stats (Average):":
            average_stats = True
        elif (
            average_stats
            and "Backend (QNN accelerator (ApplyBinarySection) time):" in line
        ):
            adapter_switch_time = line.strip().split(":")
            return int(adapter_switch_time[1].strip().split(" ")[0].strip())
    raise ValueError(f"No adapter logs found in {log_file_name}!")
