import base64
import os

import modal

from stretchlab.image import (
    hf_vol,
    huggingface_model_path,
    image,
    load_models,
    voice_vol,
    whisper_vol,
)
from stretchlab.lib.f5tts import F5TTS
from stretchlab.llama import generate_text
from stretchlab.stt import stt
from stretchlab.tts import generate_audio
from stretchlab.vector_db.similar_index import query

app = modal.App("personal-interviewer", image=image)
voice_model_path = "/voice"
whisper_model_path = "/whisper"


@app.cls(
    volumes={
        voice_model_path: voice_vol,
        whisper_model_path: whisper_vol,
        huggingface_model_path: hf_vol,
    },
    gpu="A10G",
)
class Model:
    @modal.enter()
    def load(self):
        self.whisper_model, self.llama_pipeline, self.voice_model = load_models(
            whisper_model_path,
            voice_model_path,
            "cuda",
        )

        with open(os.path.join(voice_model_path, "ref_text.txt"), "r") as f:
            ref_text = f.read()
        self.f5tts = F5TTS(
            os.path.join(voice_model_path, "ref_audio.mp3"),
            ref_text,
            voice_model_path,
        )
        self.user_states = []

    @modal.method()
    def generate_audio_from_text(self, text: str):
        audio_filepath = generate_audio(self.f5tts, text)
        with open(audio_filepath, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        return {"audio_base64": audio_base64}

    @modal.method()
    def generate(self, audio_base64: str, customer_name: str):
        text = stt(self.whisper_model, audio_base64)

        print("Recognized speech:")
        print(text)
        print("@@@@@@@@@@")

        # check whether the text is similar to the index
        context = query(text)

        self.user_states.append(
            {
                "role": "user",
                "content": f"{customer_name}: {text} \n Context: {context}",
            }
        )

        if text == "":
            return {"audio_base64": ""}

        # Generate response
        response = generate_text(
            self.llama_pipeline, self.user_states, customer_name=customer_name
        )
        self.user_states.append({"role": "assistant", "content": response})
        print("Generated response:")
        print(response)
        print("@@@@@@@@@@")

        # Generate audio from response
        audio_filepath = generate_audio(self.f5tts, response)

        with open(audio_filepath, "rb") as f:
            audio_data = f.read()

        output_audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        return {"audio_base64": output_audio_base64}
