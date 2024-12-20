import os

import modal
import torch
import transformers
from dotenv import load_dotenv
from faster_whisper import WhisperModel, download_model
from piper.voice import PiperVoice
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
whisper_vol = modal.Volume.from_name("whisper", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf", create_if_missing=True)
voice_vol = modal.Volume.from_name("voice", create_if_missing=True)
huggingface_model_path = "/hf"


def load_models(whisper_model_path: str, voice_model_path: str, device: str = "cpu"):
    """Load necessary models for STT, TTS, and text generation."""
    # Load Whisper model for speech-to-text
    model_path = download_model("large-v3", cache_dir=whisper_model_path)
    whisper_model = WhisperModel(model_path, device=device, compute_type="float16")

    # Load Hugging Face model for text generation
    llama_model_text = "unsloth/Llama-3.2-1B-Instruct"

    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_model_text,
        load_in_4bit=True,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(llama_model_text)

    llama_pipeline = transformers.pipeline(
        "text-generation",
        model=llama_model,
        tokenizer=tokenizer,
        device_map=device,
        max_new_tokens=2048,
    )

    # # Load voice model for text-to-speech
    if voice_model_path is not None:
        model_filename = os.path.join(voice_model_path, "en_GB-cori-medium.onnx")
        voice_model = PiperVoice.load(
            model_filename,
            use_cuda=torch.cuda.is_available(),
        )
    else:
        voice_model = None

    return whisper_model, llama_pipeline, voice_model


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .env(
        {
            "HUGGINGFACE_ACCESS_TOKEN": os.environ["HUGGINGFACE_ACCESS_TOKEN"],
            "HF_HOME": huggingface_model_path,
            "STRETCHLAB_VECTOR_DB": os.environ["STRETCHLAB_VECTOR_DB"],
            "STRETCHLAB_VECTOR_TOKEN": os.environ["STRETCHLAB_VECTOR_TOKEN"],
        },
    )
    .pip_install(
        "faster-whisper==1.0.3",
        "piper_tts==1.2.0",
        "transformers==4.46.1",
        "python-dotenv==1.0.1",
        "upstash-vector==0.6.0",
    )
    .run_commands(
        "pip uninstall -y torch torchvision torchaudio",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install -U httpx==0.27.0",
        "pip install -U ctranslate2==4.4.0",
        "pip install bitsandbytes==0.44.1",
        "pip install accelerate==0.33.0",
        "apt-get update",
        "apt-get install -y software-properties-common",
        "add-apt-repository ppa:git-core/ppa",
        "apt-get update",
        "apt-get install -y git",
        "apt-get install -y ffmpeg",
        "pip install git+https://github.com/SWivid/F5-TTS.git",
    )
)
