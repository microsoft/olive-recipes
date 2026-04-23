# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Example modified from: https://docs.openvino.ai/2023.3/notebooks/225-stable-diffusion-text-to-image-with-output.html
# --------------------------------------------------------------------------
import inspect
import os
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import onnxruntime as ort
import torch

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion import OnnxStableDiffusionPipeline
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from sd_utils.onnx_patch import PatchedOnnxRuntimeModel


class OVStableDiffusionPipeline(OnnxStableDiffusionPipeline):
    def __call__(
        self,
        prompt: Union[str, list[str]] = None,
        height: Optional[int] = 768,
        width: Optional[int] = 768,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, list[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        # check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = np.random

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * np.float32(self.scheduler.init_noise_sigma)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = next(
            (unet_input.type for unet_input in self.unet.model.get_inputs() if unet_input.name == "timestep"),
            "tensor(float)",
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        if do_classifier_free_guidance:
            splits = np.split(prompt_embeds, 2)
            neg_embeds, text_embeds = splits[0], splits[1]

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)

            if do_classifier_free_guidance:
                # Note that in QDQ, we need to use static dimensions (batch is fixed to 1), so we need to split
                unet_input = {"sample": latent_model_input, "timestep": timestep, "encoder_hidden_states": neg_embeds}
                noise_pred_uncond = self.unet(**unet_input)
                noise_pred_uncond = noise_pred_uncond[0]

                unet_input = {"sample": latent_model_input, "timestep": timestep, "encoder_hidden_states": text_embeds}
                noise_pred_text = self.unet(**unet_input)
                noise_pred_text = noise_pred_text[0]

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                unet_input = {
                    "sample": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                }
                noise_pred = self.unet(**unet_input)
                noise_pred = noise_pred[0]

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        latents = 1 / 0.18215 * latents
        # image = self.vae_decoder(latent_sample=latents)[0]
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate([self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])])
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def update_ov_config(config: dict):
    config["passes"] = {
        "ov_convert": config["passes"]["ov_convert"],
        "ov_io_update": config["passes"]["ov_io_update"],
        "ov_encapsulation": config["passes"]["ov_encapsulation"]
        }

    config["search_strategy"] = False
    config["systems"]["local_system"]["accelerators"][0]["device"] = "cpu"
    config["systems"]["local_system"]["accelerators"][0]["execution_providers"] = ["CPUExecutionProvider"]

    del config["evaluators"]
    del config["evaluator"]
    del config["data_configs"]
    return config


def save_optimized_ov_submodel(workflow_output, submodel, optimized_model_dir, optimized_model_path_map):
    print(f"Saving optimized {submodel} model...")
    print(f"Best candidate for {submodel}: {workflow_output.get_best_candidate()}")
    output_model_dir = workflow_output.get_best_candidate().model_path
    folder = Path(output_model_dir)
    onnx_files = sorted(folder.glob("*.onnx"))
    if onnx_files:
        first_file = onnx_files[0]
        target = folder / "model.onnx"
        first_file.rename(target)
        print(f"Renamed {first_file.name} -> {target.name}")
    optimized_model_path_map[submodel] = str(output_model_dir)


def add_ep_for_device(session_options, ep_name, device_type, ep_options=None):
    ep_devices = ort.get_ep_devices()
    for ep_device in ep_devices:
        if ep_device.ep_name == ep_name and ep_device.device.type == device_type:
            print(f"Adding {ep_name} for {device_type}")
            session_options.add_provider_for_devices([ep_device], {} if ep_options is None else ep_options)
            break

def register_execution_providers():
    import json
    import subprocess
    import sys

    worker_script = os.path.abspath('winml.py')
    result = subprocess.check_output([sys.executable, worker_script], text=True)
    paths = json.loads(result)
    for item in paths.items():
        try:
            ort.register_execution_provider_library(item[0], item[1])
        except Exception as e:
            print(f"Failed to register execution provider {item[0]}: {e}")

def get_ov_pipeline(common_args, ov_args, optimized_model_dir):
    if common_args.test_unoptimized:
        return StableDiffusionPipeline.from_pretrained(common_args.model_id)

    register_execution_providers()

    print("Loading models into ORT session...")
    sess_options = ort.SessionOptions()
    provider_options = [{}]

    ep_name = "OpenVINOExecutionProvider"

    device = ov_args.device
    device_map = {
        "CPU": ort.OrtHardwareDeviceType.CPU,
        "GPU": ort.OrtHardwareDeviceType.GPU,
        "NPU": ort.OrtHardwareDeviceType.NPU,
    }

    add_ep_for_device(sess_options, ep_name, device_map[device])

    pipeline = OVStableDiffusionPipeline.from_pretrained(
        optimized_model_dir, provider=ep_name, sess_options=sess_options, provider_options=provider_options
    )

    return pipeline


def save_ov_model_info(model_info, optimized_model_dir, pipeline):
    onnx_pipeline = OnnxStableDiffusionPipeline(
        vae_encoder=PatchedOnnxRuntimeModel.from_pretrained(model_info["vae_encoder"], is_ov_save=True),
        vae_decoder=PatchedOnnxRuntimeModel.from_pretrained(model_info["vae_decoder"], is_ov_save=True),
        text_encoder=PatchedOnnxRuntimeModel.from_pretrained(model_info["text_encoder"], is_ov_save=True),
        tokenizer=pipeline.tokenizer,
        unet=PatchedOnnxRuntimeModel.from_pretrained(model_info["unet"], is_ov_save=True),
        scheduler=pipeline.scheduler,
        safety_checker=None,
        feature_extractor=pipeline.feature_extractor,
        requires_safety_checker=False,
    )

    print("Saving optimized models...")
    onnx_pipeline.save_pretrained(optimized_model_dir)
    return
