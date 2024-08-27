@echo off

set "hf_endpoint=%HF_ENDPOINT%"
echo HF_ENDPOINT value: %hf_endpoint%

if "%hf_endpoint%"=="huggingface.co" (
    echo Setting up Hugging Face mirror in China
    set "hf_endpoint=https://hf-mirror.com"
    set "HF_ENDPOINT=%hf_endpoint%"
)
