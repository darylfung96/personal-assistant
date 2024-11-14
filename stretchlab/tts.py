# import wave


from personal_modal.runtime import timing_decorator


@timing_decorator
def generate_audio(f5tts, text: str):
    wav_filepath = "/tmp/output.wav"
    wav_filepath = f5tts.infer_tts(text)
    return wav_filepath
