#!/bin/bash

hf_endpoint=$HF_ENDPOINT
echo "HF_ENDPOINT value: $hf_endpoint"

if [hf_endpoint == "huggingface.co"]
then
    echo "设置huggingface国内镜像"
    hf_endpoint="https://hf-mirror.com"
    export HF_ENDPOINT=hf_endpoint
