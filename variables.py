from units import TimeUnit

BIRDNET_AUDIO_DURATION = TimeUnit(3)
BIRDNET_SAMPLE_RATE = 48_000
NOISE_LABEL = "Noise"

# LOWERCASE (!) extensions priority used for retrieving the audio files in decreasing order.
# The audio file types have to be accepted by FFmpeg.
AUDIO_EXTENSION_PRIORITY = [
    "wav", "flac", "mp3", "ogg", "aiff"
]
