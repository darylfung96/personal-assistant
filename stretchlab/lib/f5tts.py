import os
import re

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT


class F5TTS:
    def __init__(self, ref_audio, ref_text, asset_path):
        self.voices, self.vocoder, self.vocoder_name, self.ema_model = (
            self.initialize_tts(ref_audio, ref_text, asset_path)
        )

    def initialize_tts(self, ref_audio, ref_text, asset_path):
        vocoder_name = "vocos"
        vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, local_path="")
        model_cls = DiT
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
        repo_name = "F5-TTS"
        exp_name = "F5TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(
            cached_path(
                f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"
            )
        )
        ema_model = load_model(
            model_cls,
            model_cfg,
            ckpt_file,
            mel_spec_type=vocoder_name,
            vocab_file=os.path.join(asset_path, "vocab.txt"),
        )
        config = tomli.load(open(os.path.join(asset_path, "basic.toml"), "rb"))

        main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
        if "voices" not in config:
            voices = {"main": main_voice}
        else:
            voices = config["voices"]
            voices["main"] = main_voice
        for voice in voices:
            voices[voice]["ref_audio"], voices[voice]["ref_text"] = (
                preprocess_ref_audio_text(
                    voices[voice]["ref_audio"], voices[voice]["ref_text"]
                )
            )
            print("Voice:", voice)
            print("Ref_audio:", voices[voice]["ref_audio"])
            print("Ref_text:", voices[voice]["ref_text"])
        return voices, vocoder, vocoder_name, ema_model

    def infer_tts(
        self,
        text_gen,
        remove_silence=False,
        speed=0.9,
        output_filename="/tmp/output.wav",
    ):
        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, text_gen)
        reg2 = r"\[(\w+)\]"
        for text in chunks:
            if not text.strip():
                continue
            match = re.match(reg2, text)
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in self.voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text = re.sub(reg2, "", text)
            gen_text = text.strip()
            ref_audio = self.voices[voice]["ref_audio"]
            ref_text = self.voices[voice]["ref_text"]
            print(f"Voice: {voice}")
            audio, final_sample_rate, spectragram = infer_process(
                ref_audio,
                ref_text,
                gen_text,
                self.ema_model,
                self.vocoder,
                mel_spec_type=self.vocoder_name,
                speed=speed,
            )
            generated_audio_segments.append(audio)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)

            sf.write(output_filename, final_wave, final_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(output_filename)
            return output_filename
        return None
