import base64
from tempfile import NamedTemporaryFile

from runtime import timing_decorator


@timing_decorator
def stt(model, audio_base64: str) -> list[str]:
    binary_data = base64.b64decode(audio_base64.encode("utf-8"))
    text = ""

    with NamedTemporaryFile() as temp:
        try:
            # Write the audio data to the temporary file
            temp.write(binary_data)
            temp.flush()

            segments, _ = model.transcribe(temp.name, beam_size=5, language="en")

            for segment in segments:
                text += segment.text + " "

            print(text)
            return text

        except Exception as e:
            return {"error": f"Something went wrong: {e}"}


# while True:
#     data = stream.read(CHUNK, exception_on_overflow=False)
#     frames.append(data)

#     if keyboard.is_pressed('r'):
#         print('loading...')
#         wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(p.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(frames))
#         wf.close()

#         # response = send_audio_file(WAVE_OUTPUT_FILENAME, "https://api.runpod.ai/v2/x8k8uvgi1imi9k/run")

#         print(response)
#         generate_audio(response)
#         play_audio("output.wav")

#         frames = []
