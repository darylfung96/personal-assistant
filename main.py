import base64
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from beam import Image, Volume, endpoint, env
from piper.voice import PiperVoice

from llama import generate_text
from stt import stt
from tts import generate_audio

BEAM_VOLUME_PATH = "./cached_models"
HF_VOLUME_PATH = "./hf_models"
VOICE_VOLUME_PATH = "./voice"

# These packages will be installed in the remote container
if env.is_remote():
    from faster_whisper import WhisperModel, download_model


def load_models():
    model_path = download_model("large-v3", cache_dir=BEAM_VOLUME_PATH)
    model = WhisperModel(model_path, device="cuda", compute_type="float16")

    access_token = os.environ["HUGGINGFACE_ACCESS_TOKEN"]
    hf_home = os.environ["HF_HOME"]
    llama_model_text = "chuanli11/Llama-3.2-3B-Instruct-uncensored"
    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_model_text,
        load_in_4bit=True,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(llama_model_text)
    llama_pipeline = transformers.pipeline(
        "text-generation",
        model=llama_model,
        tokenizer=tokenizer,
        device_map="cuda",
        max_new_tokens=512,
    )

    voice_model = "./voice/en_GB-cori-high.onnx"
    voice_model = PiperVoice.load(voice_model, use_cuda=torch.cuda.is_available())

    return model, llama_pipeline, voice_model


@endpoint(
    on_start=load_models,
    name="faster-whisper",
    secrets=["HUGGINGFACE_ACCESS_TOKEN", "HF_HOME"],
    cpu=2,
    memory="8Gi",
    gpu="T4",
    image=Image(
        base_image="nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04",
        python_version="python3.10",
        python_packages=[
            "git+https://github.com/SYSTRAN/faster-whisper.git",
            "piper_tts==1.2.0",
            "transformers==4.46.1",
        ],
    ).add_commands(
        [
            "pip uninstall -y torch torchvision torchaudio",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "pip install -U httpx==0.27.0",
            "pip install -U ctranslate2==4.4.0",
            "pip install bitsandbytes==0.44.1",
            "pip install accelerate==0.33.0",
        ]
    ),
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        ),
        Volume(
            name="hf",
            mount_path=HF_VOLUME_PATH,
        ),
        Volume(name="voice", mount_path=VOICE_VOLUME_PATH),
    ],
)
def main(context, **inputs):
    model, llama_pipeline, voice_model = context.on_start_value

    audio_base64 = inputs.get("audio_base64")
    text = stt(model, audio_base64)
    print(text)

    if text == "":
        return {"audio_base64": ""}

    all_text = [text]

    all_text.insert(
        0,
        "You will receive a question. You will be in an interview. Please answer it like how a human would and limit to 50 words. ONLY 50 WORDS and please provide just pure text without any symbols.",
    )
    print(" ".join(all_text))
    response = generate_text(llama_pipeline, " ".join(all_text))
    response = " ".join(response.split(" ")[:50])
    audio, audio_filepath = generate_audio(voice_model, response)

    with open(audio_filepath, "rb") as f:
        audio_data = f.read()
    output_audio_base64 = base64.b64encode(audio_data).decode("utf-8")
    return {"audio_base64": output_audio_base64}
