import io
import wave

import keyboard
import pyaudio

# Set up audio format parameters
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate (Hz)
CHUNK = 1024  # Buffer size


def record_audio() -> bytes:
    """Record audio from the microphone and return it as bytes.

    The function records audio until the 'q' key is pressed.
    The recorded audio is saved as a WAV file in memory and returned as bytes.

    Returns:
        bytes: The recorded audio data in WAV format.

    """
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("Recording... Press 'q' to stop.")

    frames = []

    # Record until 'q' is pressed
    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        # Check if 'q' has been pressed
        if keyboard.is_pressed("q"):
            break

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    audio_buffer.seek(0)
    return audio_buffer.getvalue()
