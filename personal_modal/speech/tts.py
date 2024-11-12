import wave

from personal_modal.runtime import timing_decorator


@timing_decorator
def generate_audio(voice, text: str):
    wav_filepath = "/tmp/output.wav"
    wav_file = wave.open(wav_filepath, "w")
    audio = voice.synthesize(text, wav_file)
    return audio, wav_filepath
