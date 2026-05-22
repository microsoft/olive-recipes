#!/bin/bash

olive optimize --precision fp16 --provider CUDAExecutionProvider -m mistralai/Mistral-7B-v0.1 -o mistral_fp16_out
