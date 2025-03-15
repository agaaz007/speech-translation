import numpy as np
import sounddevice as sd
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model

# Load the processor and model from Hugging Face.
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# Audio parameters.
SAMPLERATE = 16000  # Expected sample rate (Hz)
CHANNELS = 1        # Mono audio

# Global list to collect recorded audio frames.
audio_frames = []

def audio_callback(indata, frames, time, status):
    """Callback function to collect audio chunks."""
    if status:
        print(status)
    # Append a copy of the recorded chunk.
    audio_frames.append(indata.copy())

def record_until_enter():
    """Record audio continuously until the user presses Enter."""
    global audio_frames
    audio_frames = []  # Clear any previous recordings.
    print("Recording... Press Enter to stop recording.")
    # Start the input stream with our callback.
    with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, callback=audio_callback):
        input()  # Wait until Enter is pressed.
    # After recording, concatenate all audio frames.
    recorded_audio = np.concatenate(audio_frames, axis=0)
    # Squeeze to remove any singleton dimensions.
    return np.squeeze(recorded_audio)

def play_audio(audio_array, samplerate):
    """Play a NumPy array as audio."""
    sd.play(audio_array, samplerate)
    sd.wait()

def translate_audio(audio_segment):
    """
    Process the recorded audio segment, generate translated speech to English using
    SEAMLESSM4T-v2, and play the translated audio.
    """
    # Process the raw audio segment using the processor.
    audio_inputs = processor(audios=audio_segment, sampling_rate=SAMPLERATE, return_tensors="pt")
    
    print("Translating speech to English...")
    # Generate the translated audio. The first element is assumed to be the waveform.
    outputs = model.generate(**audio_inputs, tgt_lang="eng")
    translated_audio = outputs[0].cpu().numpy().squeeze()
    
    print("Playing translated audio...")
    play_audio(translated_audio, SAMPLERATE)

def main():
    # Record until Enter is pressed.
    audio_segment = record_until_enter()
    # Process and play the translated audio.
    translate_audio(audio_segment)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
