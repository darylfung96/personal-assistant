import base64

from rich.console import Console
from rich.markdown import Markdown

from personal_modal.record import record_audio
from personal_modal.text.func import TextModel, app


@app.local_entrypoint()
def main():
    model = TextModel()

    while True:
        audio_data = record_audio()

        base64_encoded = base64.b64encode(
            audio_data,
        ).decode("utf-8")
        text = model.generate.remote(base64_encoded)["text"]

        if text == "":
            continue

        # Create a console object
        console = Console()
        markdown = Markdown(text)
        console.print(markdown)
