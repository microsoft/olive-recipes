#!/bin/bash

olive optimize --precision int4 --provider CUDAExecutionProvider -m facebook/opt-125m -o opt_125_out
