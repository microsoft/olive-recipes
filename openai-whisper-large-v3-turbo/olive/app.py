# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
import torch
from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp, chunk_and_resample_audio
from qai_hub_models.models._shared.hf_whisper.model import (
    CHUNK_LENGTH,
    SAMPLE_RATE,
)
from transformers import WhisperProcessor


def infer_audio(app, model_id, audio_file, save_data):
    audio_dict = np.load(audio_file, allow_pickle=True).item()

    audio = audio_dict["audio"]["array"]
    sample_rate = audio_dict["audio"]["sampling_rate"]
    audio_name = os.path.splitext(os.path.basename(audio_file))[0] if save_data else None

    processor = WhisperProcessor.from_pretrained(model_id)
    reference = processor.tokenizer._normalize(audio_dict["text"])
    print("Reference: ", reference)

    # Perform transcription
    transcription = app.transcribe(audio, sample_rate, audio_name, save_data)
    print("done transcription")
    prediction = processor.tokenizer._normalize(transcription)
    print("Prediction:", prediction)


class HfWhisperAppWithSave(HfWhisperApp):
    def __init__(
        self,
        encoder,
        decoder,
        hf_model_id: str,
        execution_provider: str = "CPUExecutionProvider",
        provider_options: dict = None,
        sample_rate: int = SAMPLE_RATE,
        max_audio_seconds: int = CHUNK_LENGTH,
    ):
        super().__init__(None, None, hf_model_id, sample_rate, max_audio_seconds)
        options = ort.SessionOptions()

        self.encoder = ort.InferenceSession(
            encoder, sess_options=options, providers=[execution_provider], provider_options=[provider_options]
        )

        self.decoder = ort.InferenceSession(
            decoder, sess_options=options, providers=[execution_provider], provider_options=[provider_options]
        )

    def transcribe_tokens(self, audio, sample_rate, audio_name, save_data=False) -> list[int]:
        out_chunked_tokens = []
        for ind, x in enumerate(chunk_and_resample_audio(audio, sample_rate)):
            out_chunked_tokens.append(self._transcribe_single_chunk(x, audio_name, ind, save_data))

        out_tokens: list[int] = []
        for chunk_tokens in out_chunked_tokens:
            out_tokens.extend(chunk_tokens)
        return out_tokens

    def transcribe(self, audio, sample_rate, audio_name, save_data=False) -> str:
        tokens = self.transcribe_tokens(audio, sample_rate, audio_name, save_data)
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def _transcribe_single_chunk(
        self, audio: np.ndarray, audio_name=None, chunk_number=None, save_data=False
    ) -> list[int]:
        # feature
        input_features = self.feature_extractor(audio, sampling_rate=self.sample_rate, return_tensors="np")[
            "input_features"
        ]

        # encoder
        output_names_encoder = [output.name for output in self.encoder.get_outputs()]
        # kv_cache_cross = self.encoder(input_features)
        input_features_feed = {"input_features": input_features}

        if save_data:
            input_features_save_path = os.path.join(save_data, audio_name, f"{chunk_number}_input_features.npy")
            os.makedirs(os.path.dirname(input_features_save_path), exist_ok=True)
            np.save(input_features_save_path, input_features_feed)

        kv_cache_cross_numpy = self.encoder.run(output_names_encoder, input_features_feed)
        kv_cache_cross = [torch.from_numpy(arr) for arr in kv_cache_cross_numpy]
        if not isinstance(kv_cache_cross, tuple):
            kv_cache_cross = (kv_cache_cross,)
        if not isinstance(kv_cache_cross[0], (tuple, list)):
            kv_cache_cross = (kv_cache_cross,)

        sot = self.config.decoder_start_token_id
        num_decoder_blocks = self.config.decoder_layers
        attention_dim = self.config.d_model
        num_decoder_heads = self.config.decoder_attention_heads
        mask_neg = self.config.mask_neg
        eot = self.config.eos_token_id

        # decoder
        output_ids = torch.tensor([[sot]])  # Start of transcript
        output_logits = []
        output_length = output_ids.shape[1]

        position_ids = torch.tensor([0], dtype=torch.int32)
        attention_mask = torch.full(
            (1, 1, 1, self.mean_decode_len),
            mask_neg,
            dtype=torch.float32,
        )

        # init kv_cache_self
        k_cache_self = torch.zeros(
            (
                num_decoder_heads,
                1,
                attention_dim // num_decoder_heads,
                self.mean_decode_len - 1,
            ),
            dtype=torch.float32,
        )
        v_cache_self = torch.zeros(
            (
                num_decoder_heads,
                1,
                self.mean_decode_len - 1,
                attention_dim // num_decoder_heads,
            ),
            dtype=torch.float32,
        )
        kv_cache_self = tuple((k_cache_self, v_cache_self) for _ in range(num_decoder_blocks))

        for n in range(self.mean_decode_len - 1):
            # get current token
            input_ids = output_ids[:, n : n + 1].to(torch.int32)

            # update attention_mask
            attention_mask[:, :, :, self.mean_decode_len - n - 1] = 0.0

            # flattened kv caches input
            flattened_kv_cache_self = tuple(item for sublist in kv_cache_self for item in sublist)
            flattened_kv_cache_cross = tuple(item for sublist in kv_cache_cross for item in sublist)

            # decode and update kv_cache_self
            decoder_input = (
                (input_ids, attention_mask) + flattened_kv_cache_self + flattened_kv_cache_cross + (position_ids,)
            )

            # print("decoder_input: ", decoder_input)
            input_names_decoder = [input.name for input in self.decoder.get_inputs()]
            output_names_decoder = [output.name for output in self.decoder.get_outputs()]

            # decoder_input_feed = dict(zip(input_names_decoder, decoder_input))
            decoder_input_feed = {
                name: tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                for name, tensor in zip(input_names_decoder, decoder_input)
            }

            if save_data:
                decoder_input_save_path = os.path.join(save_data, audio_name, f"{chunk_number}_{n}_decoder_input.npy")
                os.makedirs(os.path.dirname(decoder_input_save_path), exist_ok=True)
                np.save(decoder_input_save_path, decoder_input_feed)

            decoder_output_numpy = self.decoder.run(output_names_decoder, decoder_input_feed)
            decoder_output = [torch.from_numpy(arr) for arr in decoder_output_numpy]
            # decoder_output = self.decoder(*decoder_input)
            if isinstance(decoder_output, tuple) and len(decoder_output) == 2:
                logits, kv_cache_self = decoder_output
            else:
                logits = decoder_output[0]
                kv_cache_self = tuple(decoder_output[i : i + 2] for i in range(1, len(decoder_output), 2))

            # update output_logits
            output_logits.append(logits.detach().clone())

            # update output_ids
            output_id = torch.argmax(logits, 1).squeeze(0)
            # end of transcript
            if len(output_logits) == (self.mean_decode_len - 1) or output_id == eot:
                output_ids = torch.cat((output_ids, output_id), -1)
                break
            if n >= output_length - 1:
                output_ids = torch.cat((output_ids, output_id), -1)

            # update position_ids
            position_ids += 1

        return output_ids[0].tolist()
