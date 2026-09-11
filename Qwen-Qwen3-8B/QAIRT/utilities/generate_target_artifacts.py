#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import pickle
import warnings

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, LCMScheduler
from diffusers.utils.torch_utils import randn_tensor


def save_file(output_fname, data):
    with open(output_fname, "wb") as f:
        pickle.dump(data, f)
    print(f"data saved in {output_fname}")


def generate_target_artifacts(text_encoder, unet, controlnet, tokenizer, scheduler, config, diffusion_steps=[20, 50], guidance_scales=[5.0, 7.5, 10.0], seed_list=[10,50,100], min_seed=0, max_seed=2229135949491605, generator_seeds=[0],save_path = './_exports_', torch_device='cuda'):
    os.makedirs(save_path, exist_ok=True)
    if not isinstance(seed_list, list):
        raise Exception("seed_list should be a list of one or more integers, indicating how many random init latents to generate")
    if not isinstance(diffusion_steps, list):
        raise Exception("diffusion_steps should be a list of one or more integers, indicating the number of diffusion steps to use when exporting scheduler and unet binaries")
    batch_size = 1
    unet.to(torch_device)
    unet.eval()
    if controlnet is not None:
        controlnet.to(torch_device)
        controlnet.eval()
    if isinstance(text_encoder, list):
        for te in text_encoder:
            te.to(torch_device)
            te.eval()
    elif text_encoder != None:
        text_encoder.to(torch_device)
        text_encoder.eval()


    # Export the tokenizer
    if isinstance(tokenizer, list):
        for idx, tok in enumerate(tokenizer):
            tokenizer_root_dir = os.path.join(save_path, f"tokenizer_{idx}")
            tok.save_pretrained(tokenizer_root_dir)
            print(f"data saved in {tokenizer_root_dir}")
    elif tokenizer != None:
        tokenizer_root_dir = os.path.join(save_path, f"tokenizer")
        tokenizer.save_pretrained(tokenizer_root_dir)
        print(f"data saved in {tokenizer_root_dir}")


    # Export scheduler hyperparameters
    scheduler_root = os.path.join(save_path, 'scheduler')
    os.makedirs(scheduler_root, exist_ok=True)
    if isinstance(scheduler, DPMSolverMultistepScheduler):
        lambdas_path = os.path.join(scheduler_root, "lambdas.bin")
        scheduler.lambda_t.detach().numpy().astype(np.float32).tofile(lambdas_path)
        print(f"data saved in {lambdas_path}")
        betas_path = os.path.join(scheduler_root, "betas.bin")
        scheduler.betas.detach().numpy().astype(np.float32).tofile(betas_path)
        print(f"data saved in {betas_path}")
    elif isinstance(scheduler, LCMScheduler):
        unet_out_shape = (batch_size, unet.config.out_channels, unet.config.sample_size, unet.config.sample_size)
        for seed_idx, seed in enumerate(generator_seeds):
            skip_first = True
            generator = torch.Generator(torch_device).manual_seed(int(seed))
            for idx in range(-1, diffusion_steps[0]-1):
                noise_path = os.path.join(scheduler_root, f"noise_seed_idx_{seed_idx}_timestep_{idx}.bin")
                random_noise = randn_tensor(unet_out_shape, generator=generator, device=generator.device).detach().cpu().numpy()
                # first randn_tensor matches initial Unet latent, further randn_tensor(s) are the ones matching LCM noise
                if skip_first:
                    skip_first = False
                    continue # discard first generated random_noise
                random_noise.astype(np.float32).tofile(noise_path)
        betas_path = os.path.join(scheduler_root, "betas.bin")
        scheduler.betas.detach().numpy().astype(np.float32).tofile(betas_path)
        print(f"data saved in {betas_path}")
    elif not isinstance(scheduler,None):
        warnings.warn(f'Support for scheduler of type `{type(scheduler)}` has not been implemented')


    # Saving init random latents
    random_init_latents_root = os.path.join(save_path, 'random_latent_init')
    os.makedirs(random_init_latents_root, exist_ok=True)

    for n_seeds in seed_list:
        seeds = np.linspace(min_seed, max_seed, n_seeds).astype('long').tolist()
        random_init = {}
        for current_seed in seeds:
            generator = torch.manual_seed(current_seed)    # Seed generator to create the inital latent noise
            latents = torch.randn((batch_size, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size), generator=generator)
            random_init[str(current_seed)] = latents.cpu().numpy()

        random_init_path = os.path.join(random_init_latents_root, f'random_init_{n_seeds}.pkl')
        save_file(random_init_path, random_init)
    latents.to(device=unet.device, dtype=unet.dtype)


    # Saving unconditional embedding
    if not isinstance(tokenizer, list):
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0].detach().cpu().data.numpy()
        uncond_emb_path = os.path.join(save_path, 'unconditional_text_emb.pkl')
        save_file(uncond_emb_path, uncond_embeddings)

    # Prepare dirs for saving unet and scheduler exports
    time_step_embeddings_root = os.path.join(save_path, 'time_step_embeddings')
    os.makedirs(time_step_embeddings_root, exist_ok=True)

    # Generate exports for all diffusion steps
    for num_steps in diffusion_steps:
        scheduler.set_timesteps(num_steps)

        # Get scheduler constants & unet time-step embeddings
        time_steps = scheduler.timesteps.data.numpy()
        scheduler_time_steps_path = os.path.join(scheduler_root, f'scheduler_time_steps_{num_steps}.pkl')
        save_file(scheduler_time_steps_path, time_steps)

        unet_has_time_embedding = getattr(unet, "get_time_embedding", None)
        if callable(unet_has_time_embedding):
            unet_time_step_embedding = {}
            for i, t in enumerate(scheduler.timesteps):
                time_emb = unet.get_time_embedding(t, 1)
                unet_time_step_embedding[str(i)] = time_emb.detach().cpu().numpy()
            unet_time_step_embeddings_path = os.path.join(time_step_embeddings_root, f'unet_time_step_embeddings_{num_steps}.pkl')
            save_file(unet_time_step_embeddings_path, unet_time_step_embedding)

        controlnet_has_time_embedding = getattr(controlnet, "get_time_embedding", None)
        if callable(controlnet_has_time_embedding):
            controlnet_time_step_embedding = {}
            for i, t in enumerate(scheduler.timesteps):
                time_emb = controlnet.get_time_embedding(t, 1)
                controlnet_time_step_embedding[str(i)] = time_emb.detach().cpu().numpy()
            controlnet_time_step_embeddings_path = os.path.join(time_step_embeddings_root, f'controlnet_time_step_embeddings_{num_steps}.pkl')
            save_file(controlnet_time_step_embeddings_path, controlnet_time_step_embedding)

        unet_has_get_time_embed = getattr(unet, "get_time_embed", None)
        unet_has_time_embedding = getattr(unet, "time_embedding", None)
        if callable(unet_has_get_time_embed) and callable(unet_has_time_embedding):
            unet_time_step_embedding = {}
            for i, t in enumerate(scheduler.timesteps):
                t_emb = unet.get_time_embed(sample=latents, timestep=t).to(dtype=unet.dtype, device=unet.device)
                emb = unet.time_embedding(t_emb, condition=None)
                unet_time_step_embedding[str(i)] = emb.detach().cpu().numpy()
            unet_time_step_embeddings_path = os.path.join(time_step_embeddings_root, f'unet_timestep_embeddings_{num_steps}.pkl')
            save_file(unet_time_step_embeddings_path, unet_time_step_embedding)

    # Generate guidance embeddings
    unet_has_guidance_embedding = getattr(unet, "get_guidance_embedding", None)
    controlnet_has_guidance_embedding = getattr(controlnet, "get_guidance_embedding", None)

    if callable(unet_has_guidance_embedding) or callable(controlnet_has_guidance_embedding):
        guidance_embeddings_root = os.path.join(save_path, 'guidance_embeddings')
        os.makedirs(guidance_embeddings_root, exist_ok=True)

    if callable(unet_has_guidance_embedding):
        unet_guidance_embeddings = {}
        for guidance_scale in guidance_scales:
            guidance_ip = torch.tensor([guidance_scale], dtype=latents.dtype).to(dtype=unet.dtype, device=unet.device)
            guidance_emb = unet.get_guidance_embedding(guidance_ip, 1)
            unet_guidance_embeddings[guidance_scale] = guidance_emb.detach().cpu().numpy()
        unet_guidance_embeddings_path = os.path.join(guidance_embeddings_root, f'unet_guidance_embeddings.pkl')
        save_file(unet_guidance_embeddings_path, unet_guidance_embeddings)

    if callable(controlnet_has_guidance_embedding):
        controlnet_guidance_embeddings = {}
        for guidance_scale in guidance_scales:
            guidance_ip = torch.tensor([guidance_scale], dtype=latents.dtype).to(dtype=unet.dtype, device=unet.device)
            guidance_emb = controlnet.get_guidance_embedding(guidance_ip, 1)
            controlnet_guidance_embeddings[guidance_scale] = guidance_emb.detach().cpu().numpy()
        controlnet_guidance_embeddings_path = os.path.join(guidance_embeddings_root, f'controlnet_guidance_embeddings.pkl')
        save_file(controlnet_guidance_embeddings_path, controlnet_guidance_embeddings)
