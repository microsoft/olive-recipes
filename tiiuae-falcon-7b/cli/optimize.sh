#!/bin/bash

olive optimize --precision fp16 --provider CUDAExecutionProvider -m tiiuae/falcon-7b -o falcon_out
