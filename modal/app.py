import base64
import io

import simpleaudio as sa
from func import Model, app
from record import record_audio


@app.local_entrypoint()
def main():
    model = Model()

    while True:
        audio_data = record_audio()

        base64_encoded = base64.b64encode(
            audio_data,
        ).decode("utf-8")
        audio_base64 = model.generate.remote(base64_encoded)["audio_base64"]

        if audio_base64 == "":
            continue

        audio_data = base64.b64decode(audio_base64)
        audio_io = io.BytesIO(audio_data)
        audio = sa.WaveObject.from_wave_file(audio_io)
        # Play the audio
        play_obj = audio.play()
        play_obj.wait_done()
