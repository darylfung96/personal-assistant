import base64
from tempfile import NamedTemporaryFile

from personal_modal.runtime import timing_decorator


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
