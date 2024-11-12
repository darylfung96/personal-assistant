import base64
import io
import os

import requests
import simpleaudio as sa

from personal_modal.record import record_audio

# Set up the headers
headers = {
    "Authorization": f"Bearer {os.environ['BEAM_API_KEY']}",
    "Content-Type": "application/json",  # Optional, depending on the API
}


if __name__ == "__main__":
    while True:
        audio_data = record_audio()

        base64_encoded = base64.b64encode(
            audio_data,
        ).decode("utf-8")
        data = {"audio_base64": base64_encoded}
        print("getting response")
        response = requests.post(
            os.environ["BEAM_API_URL"],
            headers=headers,
            json=data,
        )

        audio_base64 = response.json()["audio_base64"]

        if audio_base64 == "":
            continue

        audio_data = base64.b64decode(audio_base64)
        audio_io = io.BytesIO(audio_data)
        audio = sa.WaveObject.from_wave_file(audio_io)
        # Play the audio
        play_obj = audio.play()
        play_obj.wait_done()
