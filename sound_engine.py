"""
neuro — Neural Audio Synthesis Engine
Generates brainwave entrainment audio using AM modulation,
harmonic drones, and spectrally shaped noise.

Ambient mode uses a dedicated lo-fi synthesis path:
chord pads + vinyl crackle + bit crushing.
"""

import numpy as np
import io
import wave

SAMPLE_RATE = 44100
CHUNK_DURATION = 4  # seconds per chunk — balance between latency and bandwidth

# ─── Sci-Fi Ambient Synthesis ───────────────────────────────────────────────
#
# Sound design goals (sci-fi ambient for deep focus/relaxation):
#   • Cold, metallic pads — open 5th voicings, sawtooth-like harmonics
#   • Deep sub-bass drone — very low rumble, slow evolution
#   • Space wind — multi-band resonant noise with independent LFOs
#   • Crystal transients — sparse high-frequency sine bursts (distant signals)
#   • Gentle alpha entrainment — low depth to preserve the atmosphere
#
# Chord progression: open 5ths cycle (20s each), no major/minor — alien, spacious
#   Em5 → Cm5 → Am5 → Bsus → (repeat)

_SCIFI_CHORDS = [
    [82.41, 123.47, 164.81, 246.94, 329.63],  # Em(open5): E2 B2 E3 B3 E4
    [65.41,  98.00, 130.81, 196.00, 261.63],  # Cm(open5): C2 G2 C3 G3 C4
    [55.00,  82.41, 110.00, 164.81, 220.00],  # Am(open5): A1 E2 A2 E3 A3
    [61.74,  92.50, 123.47, 164.81, 246.94],  # Bsus4:     B1 F#2 B2 E3 B3
]
_SCIFI_CHORD_DUR  = 20.0
_SCIFI_CYCLE_DUR  = _SCIFI_CHORD_DUR * len(_SCIFI_CHORDS)  # 80s


def _scifi_get_chord(t_offset: float) -> list:
    idx = int(t_offset / _SCIFI_CHORD_DUR) % len(_SCIFI_CHORDS)
    return _SCIFI_CHORDS[idx]


def _scifi_pad(chord_hz: list, n: int, sr: int, t0: float) -> np.ndarray:
    """
    Cold metallic pad using sawtooth-like harmonic series.
    Open 5th voicings, detuned pairs for wide stereo-like beating.
    Slow amplitude and pitch LFOs for organic movement.
    """
    t = np.linspace(t0, t0 + n / sr, n, endpoint=False, dtype=np.float64)
    out = np.zeros(n, dtype=np.float32)

    for i, freq in enumerate(chord_hz):
        note_amp = 0.45 / (i + 1) ** 0.65

        # Slow breath LFO (0.018–0.05 Hz, different phase per note)
        lfo = 0.65 + 0.35 * np.sin(2 * np.pi * (0.018 + i * 0.008) * t + i * 2.1)

        # Detuned pair: slight beating between two oscillators (±3 cents)
        det = 1.0 + 0.00173  # ~3 cents in ratio
        vib = 1.0 + 0.0008 * np.sin(2 * np.pi * 0.22 * t + i * 1.7)  # subtle vibrato

        # Sawtooth harmonics (1/k rolloff) up to k=10 — metallic quality
        for k in range(1, 11):
            h_amp = note_amp / k
            out += (h_amp * lfo * (
                np.sin(2 * np.pi * freq * k * det   * vib * t) +
                np.sin(2 * np.pi * freq * k / det   * vib * t) * 0.7
            )).astype(np.float32)

    # Cold high-pass character: attenuate sub-bass on pad itself
    mx = np.max(np.abs(out))
    return out / mx if mx > 0 else out


def _sub_bass_drone(fund_hz: float, n: int, sr: int, t0: float) -> np.ndarray:
    """
    Deep sub-bass drone on the fundamental of the current chord.
    Very slow amplitude modulation for a "breathing engine" feel.
    """
    t = np.linspace(t0, t0 + n / sr, n, endpoint=False, dtype=np.float64)
    lfo = 0.55 + 0.45 * np.sin(2 * np.pi * 0.012 * t)  # ~83s breath cycle
    drone = lfo * (
        0.80 * np.sin(2 * np.pi * fund_hz * t) +
        0.35 * np.sin(2 * np.pi * fund_hz * 2 * t) +
        0.12 * np.sin(2 * np.pi * fund_hz * 3 * t)
    )
    return drone.astype(np.float32)


def _space_wind(n: int, sr: int, t0: float) -> np.ndarray:
    """
    Multi-band resonant noise — simulates electromagnetic space wind.
    Each band has an independent slow LFO on amplitude,
    creating a constantly shifting atmospheric texture.
    """
    t = np.linspace(t0, t0 + n / sr, n, endpoint=False, dtype=np.float64)
    white = np.random.randn(n).astype(np.float64)
    fft_w = np.fft.rfft(white)
    freqs  = np.fft.rfftfreq(n, d=1.0 / sr)
    out = np.zeros(n, dtype=np.float32)

    bands = [
        # (center_hz, bandwidth_hz, lfo_hz, lfo_phase)
        (180,  60, 0.007, 0.00),
        (420, 110, 0.011, 1.57),
        (900, 180, 0.017, 3.14),
        (2200, 400, 0.009, 4.71),
        (5000, 900, 0.014, 2.30),
    ]

    for center, bw, lfo_hz, phase in bands:
        gain = np.exp(-0.5 * ((freqs - center) / bw) ** 2)
        band = np.fft.irfft(fft_w * gain, n=n).astype(np.float32)
        mx = np.max(np.abs(band))
        if mx > 0: band /= mx
        lfo = 0.40 + 0.60 * np.abs(np.sin(np.pi * lfo_hz * t + phase))
        out += (0.015 * lfo * band).astype(np.float32)

    return out


def _crystal_transients(n: int, sr: int) -> np.ndarray:
    """
    Sparse high-frequency sine bursts — distant signals, metallic pings.
    Between 0 and 3 events per 4-second chunk.
    """
    # Pentatonic minor scale in the 5th–7th octave (eerie but musical)
    crystal_freqs = [1760, 2093, 2637, 3136, 3520, 4186]
    out = np.zeros(n, dtype=np.float32)

    n_events = np.random.choice([0, 0, 0, 1, 1, 2, 3], p=[0.30, 0.20, 0.15, 0.15, 0.12, 0.05, 0.03])
    for _ in range(n_events):
        pos = np.random.randint(0, max(1, n - sr))
        freq = np.random.choice(crystal_freqs)
        dur  = np.random.randint(int(sr * 0.25), int(sr * 1.2))
        dur  = min(dur, n - pos)
        if dur <= 0: continue

        t_e = np.arange(dur, dtype=np.float64) / sr
        amp = np.random.uniform(0.04, 0.09)
        # Fast attack, slow exponential decay
        attack = min(int(sr * 0.01), dur)
        env = amp * np.exp(-t_e * np.random.uniform(2.5, 6.0))
        env[:attack] *= np.linspace(0, 1, attack)
        out[pos:pos + dur] += (env * np.sin(2 * np.pi * freq * t_e)).astype(np.float32)

    return out


def _generate_ambient_chunk(intensity: float, t_offset: float) -> bytes:
    """
    Sci-fi ambient synthesis path.

    Layers:
      1. Cold metallic pad    — open 5th chords, sawtooth harmonics, detuned pairs
      2. Sub-bass drone       — fundamental of current chord, very slow LFO
      3. Space wind           — multi-band resonant noise, independent amplitude LFOs
      4. Crystal transients   — sparse high-frequency sine bursts
      5. Alpha AM             — 10 Hz entrainment, low depth (atmospheric)
    """
    n  = int(SAMPLE_RATE * CHUNK_DURATION)
    sr = SAMPLE_RATE

    chord    = _scifi_get_chord(t_offset)
    fund_hz  = chord[0]

    # 1. Metallic pad
    pad = _scifi_pad(chord, n, sr, t_offset)

    # 2. Sub-bass drone
    sub = _sub_bass_drone(fund_hz, n, sr, t_offset)

    # 3. Space wind
    wind = _space_wind(n, sr, t_offset)

    # 4. Crystal transients
    crystal = _crystal_transients(n, sr)

    # Scale by intensity
    s = 0.30 + 0.70 * intensity
    mixed = (
        0.50 * s * pad
      + 0.28 * s * sub
      + 1.00 * wind        # wind level is already scaled internally
      + 0.90 * crystal
    )

    # 5. Alpha AM entrainment — gentle, preserves atmosphere
    depth = 0.08 * (0.2 + 0.8 * intensity)
    mixed = am_modulate(mixed, 10.0, depth, sr, t_offset)

    # Soft clip + normalize
    mixed = np.tanh(mixed * 1.7) / 1.7
    mx = np.max(np.abs(mixed))
    if mx > 0:
        mixed = mixed / mx * 0.87

    _apply_chunk_envelope(mixed)
    pcm = (np.clip(mixed, -1.0, 1.0) * 32767).astype(np.int16)
    return _pcm_to_wav(pcm)

MODES = {
    'focus': {
        'name': 'Focus',
        'emoji': '◎',
        'tagline': 'Sustained concentration',
        'band': 'β', 'band_name': 'Beta', 'band_range': '14–30 Hz',
        'entrainment_hz': 18.0,
        'base_freqs': [55, 110, 165, 220],
        'noise': 'pink', 'noise_vol': 0.28, 'drone_vol': 0.48, 'mod_depth': 0.38,
        'color': '#4a8fff',
    },
    'hyper_focus': {
        'name': 'Hyper Focus',
        'emoji': '⬡',
        'tagline': 'Deep work & flow state',
        'band': 'γ', 'band_name': 'Gamma', 'band_range': '25–100 Hz',
        'entrainment_hz': 30.0,
        'base_freqs': [70, 140, 210, 280],
        'noise': 'pink', 'noise_vol': 0.18, 'drone_vol': 0.55, 'mod_depth': 0.48,
        'color': '#7c4dff',
    },
    'adhd': {
        'name': 'ADHD Focus',
        'emoji': '⚡',
        'tagline': 'Extra neural stimulation',
        'band': 'β', 'band_name': 'Beta', 'band_range': '20–25 Hz',
        'entrainment_hz': 22.0,
        'base_freqs': [80, 160, 240],
        'noise': 'pink', 'noise_vol': 0.25, 'drone_vol': 0.50, 'mod_depth': 0.55,
        'color': '#ff6b35',
    },
    'creativity': {
        'name': 'Creativity',
        'emoji': '◈',
        'tagline': 'Alpha–theta creative flow',
        'band': 'α/θ', 'band_name': 'Alpha–Theta', 'band_range': '7–10 Hz',
        'entrainment_hz': 8.0,
        'base_freqs': [52, 104, 156, 208],
        'noise': 'pink', 'noise_vol': 0.32, 'drone_vol': 0.40, 'mod_depth': 0.28,
        'color': '#ff4081',
    },
    'relax': {
        'name': 'Relax',
        'emoji': '◯',
        'tagline': 'Calm alertness, stress relief',
        'band': 'α', 'band_name': 'Alpha', 'band_range': '8–12 Hz',
        'entrainment_hz': 10.0,
        'base_freqs': [48, 96, 144],
        'noise': 'brown', 'noise_vol': 0.38, 'drone_vol': 0.35, 'mod_depth': 0.25,
        'color': '#00c9a7',
    },
    'meditate': {
        'name': 'Meditate',
        'emoji': '◉',
        'tagline': 'Deep theta mindfulness',
        'band': 'θ', 'band_name': 'Theta', 'band_range': '4–8 Hz',
        'entrainment_hz': 6.0,
        'base_freqs': [40, 80, 120],
        'noise': 'brown', 'noise_vol': 0.32, 'drone_vol': 0.42, 'mod_depth': 0.22,
        'color': '#ab47bc',
    },
    'sleep': {
        'name': 'Sleep',
        'emoji': '◌',
        'tagline': 'Delta waves, deep sleep',
        'band': 'δ', 'band_name': 'Delta', 'band_range': '0.5–4 Hz',
        'entrainment_hz': 2.0,
        'base_freqs': [30, 60, 90],
        'noise': 'brown', 'noise_vol': 0.55, 'drone_vol': 0.22, 'mod_depth': 0.18,
        'color': '#1976d2',
    },
    'nap': {
        'name': 'Power Nap',
        'emoji': '◐',
        'tagline': 'Light sleep recovery',
        'band': 'δ/θ', 'band_name': 'Delta–Theta', 'band_range': '2–6 Hz',
        'entrainment_hz': 4.0,
        'base_freqs': [35, 70, 105],
        'noise': 'brown', 'noise_vol': 0.48, 'drone_vol': 0.28, 'mod_depth': 0.20,
        'color': '#0288d1',
    },
    'ambient': {
        'name': 'Sci-Fi Ambient',
        'emoji': '⊹',
        'tagline': 'Deep space atmosphere',
        'band': 'α', 'band_name': 'Alpha', 'band_range': '8–12 Hz',
        'entrainment_hz': 10.0,
        'base_freqs': [],
        'noise': 'pink', 'noise_vol': 0.0, 'drone_vol': 0.0, 'mod_depth': 0.0,
        'color': '#00d4cc',
        'is_ambient': True,
    },
}


# ─── Noise Generators ───────────────────────────────────────────────────────

def pink_noise(n: int) -> np.ndarray:
    """FFT-based pink noise (1/f spectrum). Warmer than white, brighter than brown."""
    white = np.random.randn(n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    fft *= 1.0 / np.sqrt(freqs)
    fft[0] = 0
    out = np.fft.irfft(fft, n=n).astype(np.float32)
    mx = np.max(np.abs(out))
    return out / mx if mx > 0 else out


def brown_noise(n: int) -> np.ndarray:
    """FFT-based brown noise (1/f² spectrum). Deep, warm, very low frequency-heavy."""
    white = np.random.randn(n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    fft *= 1.0 / freqs
    fft[0] = 0
    out = np.fft.irfft(fft, n=n).astype(np.float32)
    mx = np.max(np.abs(out))
    return out / mx if mx > 0 else out


# ─── Drone Synthesis ────────────────────────────────────────────────────────

def harmonic_drone(base_freqs: list, n: int, sr: int, t0: float = 0.0) -> np.ndarray:
    """
    Generate a rich harmonic drone from a list of base frequencies.
    Each partial has a slow LFO on amplitude for organic movement.
    Phase is continuous across chunks via t0 offset.
    """
    t = np.linspace(t0, t0 + n / sr, n, endpoint=False, dtype=np.float64)
    out = np.zeros(n, dtype=np.float32)

    for i, f in enumerate(base_freqs):
        # Harmonic rolloff — higher partials are quieter
        amp = 1.0 / (i + 1) ** 0.85
        # Micro-detuning for warmth and beating
        f_det = f * (1.0 + i * 0.0018)
        # Independent slow LFO per partial (0.04–0.12 Hz)
        lfo_hz = 0.04 + i * 0.028
        lfo = 0.80 + 0.20 * np.sin(2 * np.pi * lfo_hz * t)
        out += (amp * lfo * np.sin(2 * np.pi * f_det * t)).astype(np.float32)

    mx = np.max(np.abs(out))
    return out / mx if mx > 0 else out


# ─── Entrainment Modulation ─────────────────────────────────────────────────

def am_modulate(audio: np.ndarray, hz: float, depth: float,
                sr: int, t0: float = 0.0) -> np.ndarray:
    """
    Apply amplitude modulation at the entrainment frequency.
    The modulator oscillates between (1-depth) and 1.0.
    Phase-continuous via t0.
    """
    t = np.linspace(t0, t0 + len(audio) / sr, len(audio), endpoint=False, dtype=np.float64)
    # Sine modulator: ranges from (1-depth) to 1.0
    mod = (1.0 - depth * 0.5 * (1.0 - np.sin(2 * np.pi * hz * t - np.pi / 2))).astype(np.float32)
    return audio * mod


# ─── Chunk Generator ────────────────────────────────────────────────────────

def _pcm_to_wav(pcm: np.ndarray) -> bytes:
    """Convert int16 PCM array to WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _apply_chunk_envelope(audio: np.ndarray, fade: int = 2205) -> np.ndarray:
    """Fade in/out at chunk boundaries to prevent clicks."""
    audio[:fade]  *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return audio


# ─── Sci-Fi Ambient + Rain Synthesis ───────────────────────────────────────
#
# Open-5th chord cycle: Em5 → Cm5 → Am5 → Bsus4 (20s each, 80s total)
# Cold metallic pads + deep sub + electromagnetic space wind +
# crystal transients + constant background rain

_SCIFI_CHORDS = [
    [82.41, 123.47, 164.81, 246.94, 329.63],  # Em5:  E2 B2 E3 B3 E4
    [65.41,  98.00, 130.81, 196.00, 261.63],  # Cm5:  C2 G2 C3 G3 C4
    [55.00,  82.41, 110.00, 164.81, 220.00],  # Am5:  A1 E2 A2 E3 A3
    [61.74,  92.50, 123.47, 164.81, 246.94],  # Bsus: B1 F#2 B2 E3 B3
]
_SCIFI_CHORD_DUR = 20.0
_SCIFI_CYCLE_DUR = _SCIFI_CHORD_DUR * len(_SCIFI_CHORDS)  # 80s


def _scifi_get_chord(t_offset: float) -> list:
    idx = int(t_offset / _SCIFI_CHORD_DUR) % len(_SCIFI_CHORDS)
    return _SCIFI_CHORDS[idx]


def _scifi_pad(chord_hz: list, n: int, sr: int, t0: float) -> np.ndarray:
    """
    Cold metallic pad — sawtooth harmonic series, detuned pairs.
    Open 5th voicings give an alien, spacious quality.
    """
    t = np.linspace(t0, t0 + n / sr, n, endpoint=False, dtype=np.float64)
    out = np.zeros(n, dtype=np.float32)

    for i, freq in enumerate(chord_hz):
        note_amp = 0.45 / (i + 1) ** 0.65
        lfo  = 0.65 + 0.35 * np.sin(2 * np.pi * (0.018 + i * 0.008) * t + i * 2.1)
        det  = 1.00173          # +3 cents detuning
        vib  = 1.0 + 0.0008 * np.sin(2 * np.pi * 0.22 * t + i * 1.7)

        for k in range(1, 11):  # sawtooth series → metallic
            h = note_amp / k
            out += (h * lfo * (
                np.sin(2 * np.pi * freq * k * det / det * vib * t) +
                np.sin(2 * np.pi * freq * k * det       * vib * t) * 0.7
            )).astype(np.float32)

    mx = np.max(np.abs(out))
    return out / mx if mx > 0 else out


def _sub_bass_drone(fund_hz: float, n: int, sr: int, t0: float) -> np.ndarray:
    """Deep sub-bass drone on the chord fundamental. ~83s breath cycle."""
    t = np.linspace(t0, t0 + n / sr, n, endpoint=False, dtype=np.float64)
    lfo = 0.55 + 0.45 * np.sin(2 * np.pi * 0.012 * t)
    d = lfo * (
        0.80 * np.sin(2 * np.pi * fund_hz       * t) +
        0.35 * np.sin(2 * np.pi * fund_hz * 2.0 * t) +
        0.12 * np.sin(2 * np.pi * fund_hz * 3.0 * t)
    )
    return d.astype(np.float32)


def _space_wind(n: int, sr: int, t0: float) -> np.ndarray:
    """
    Multi-band resonant noise — electromagnetic space wind.
    5 independent bands with slow LFO on each amplitude.
    """
    t     = np.linspace(t0, t0 + n / sr, n, endpoint=False, dtype=np.float64)
    white = np.random.randn(n).astype(np.float64)
    fw    = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    out   = np.zeros(n, dtype=np.float32)

    bands = [
        (180,  60, 0.007, 0.00),
        (420, 110, 0.011, 1.57),
        (900, 180, 0.017, 3.14),
        (2200, 400, 0.009, 4.71),
        (5000, 900, 0.014, 2.30),
    ]
    for center, bw, lfo_hz, phase in bands:
        gain = np.exp(-0.5 * ((freqs - center) / bw) ** 2)
        band = np.fft.irfft(fw * gain, n=n).astype(np.float32)
        mx = np.max(np.abs(band))
        if mx > 0: band /= mx
        lfo = 0.40 + 0.60 * np.abs(np.sin(np.pi * lfo_hz * t + phase))
        out += (0.015 * lfo * band).astype(np.float32)

    return out


def _crystal_transients(n: int, sr: int) -> np.ndarray:
    """Sparse high-frequency sine bursts — distant metallic signals."""
    freqs_c = [1760, 2093, 2637, 3136, 3520, 4186]
    out = np.zeros(n, dtype=np.float32)
    n_ev = np.random.choice([0,0,0,1,1,2,3], p=[0.30,0.20,0.15,0.15,0.12,0.05,0.03])
    for _ in range(n_ev):
        pos  = np.random.randint(0, max(1, n - sr))
        freq = float(np.random.choice(freqs_c))
        dur  = min(np.random.randint(int(sr*0.25), int(sr*1.2)), n - pos)
        if dur <= 0: continue
        t_e  = np.arange(dur, dtype=np.float64) / sr
        amp  = np.random.uniform(0.04, 0.09)
        atk  = min(int(sr * 0.01), dur)
        env  = amp * np.exp(-t_e * np.random.uniform(2.5, 6.0))
        env[:atk] *= np.linspace(0, 1, atk)
        out[pos:pos+dur] += (env * np.sin(2 * np.pi * freq * t_e)).astype(np.float32)
    return out


def _rain(n: int, sr: int, t0: float) -> np.ndarray:
    """
    Constant background rain synthesis — heard from indoors.

    Layers:
      1. Base rain hiss  — shaped noise peaking at ~1.5 kHz (primary rain spectrum)
      2. Splash layer    — lower band 300–700 Hz (drop impacts)
      3. Drop transients — ~15 discrete drops/sec for texture
      4. Sheet modulation— very slow amplitude variation (0.07 Hz)
    """
    t     = np.linspace(t0, t0 + n / sr, n, endpoint=False, dtype=np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    # Layer 1: main rain hiss
    w1  = np.random.randn(n).astype(np.float64)
    fw1 = np.fft.rfft(w1)
    hiss_shape = (
        0.4 / np.sqrt(np.where(freqs > 0, freqs, 1.0)) +
        2.0 * np.exp(-0.5 * ((freqs - 1500) /  900) ** 2) +
        0.8 * np.exp(-0.5 * ((freqs - 3800) / 1400) ** 2)
    )
    hiss_shape[0] = 0
    # Gentle roll-off above 8 kHz (glass/indoor muffling)
    hiss_shape = np.where(
        freqs > 8000,
        hiss_shape * np.exp(-(freqs - 8000) / 3000),
        hiss_shape
    )
    hiss = np.fft.irfft(fw1 * hiss_shape, n=n).astype(np.float32)
    mx = np.max(np.abs(hiss)); hiss = hiss / mx if mx > 0 else hiss

    # Layer 2: splash / impact (300–700 Hz)
    w2  = np.random.randn(n).astype(np.float64)
    fw2 = np.fft.rfft(w2)
    splash_shape = np.exp(-0.5 * ((freqs - 500) / 200) ** 2)
    splash_shape[0] = 0
    splash = np.fft.irfft(fw2 * splash_shape, n=n).astype(np.float32)
    mx = np.max(np.abs(splash)); splash = splash / mx if mx > 0 else splash

    # Layer 3: individual drop transients
    drops = np.zeros(n, dtype=np.float32)
    for _ in range(int(n / sr * 15)):
        pos = np.random.randint(0, max(1, n - 400))
        dur = min(np.random.randint(40, 220), n - pos)
        if dur <= 0: continue
        amp = min(np.random.exponential(0.045), 0.20)
        env = amp * np.exp(-np.arange(dur, dtype=np.float32) * 18 / dur)
        drops[pos:pos+dur] += np.random.randn(dur).astype(np.float32) * env

    # Sheet modulation — gentle intensity sway
    sheet = 0.88 + 0.12 * np.sin(2 * np.pi * 0.07 * t)
    out   = sheet * (0.62 * hiss + 0.22 * splash + 0.16 * drops)

    mx = np.max(np.abs(out))
    return (out / mx if mx > 0 else out).astype(np.float32)


def _generate_ambient_chunk(intensity: float, t_offset: float) -> bytes:
    """
    Sci-fi ambient synthesis — cold space + constant rain.

    Layers:
      1. Metallic pad      — open-5th chords, sawtooth harmonics
      2. Sub-bass drone    — fundamental, ~83s breath LFO
      3. Space wind        — 5-band resonant noise, shifting amplitudes
      4. Crystal pings     — sparse high-freq sine transients
      5. Rain              — constant indoor rainfall (shaped noise + drops)
      6. Alpha AM          — 10 Hz entrainment, low depth
    """
    n  = int(SAMPLE_RATE * CHUNK_DURATION)
    sr = SAMPLE_RATE

    chord   = _scifi_get_chord(t_offset)
    fund_hz = chord[0]

    pad     = _scifi_pad(chord, n, sr, t_offset)
    sub     = _sub_bass_drone(fund_hz, n, sr, t_offset)
    wind    = _space_wind(n, sr, t_offset)
    crystal = _crystal_transients(n, sr)
    rain    = _rain(n, sr, t_offset)

    s = 0.30 + 0.70 * intensity
    mixed = (
        0.42 * s * pad
      + 0.22 * s * sub
      + 1.00     * wind
      + 0.90     * crystal
      + 0.28     * rain      # rain sits slightly behind the pads
    )

    depth = 0.08 * (0.2 + 0.8 * intensity)
    mixed = am_modulate(mixed, 10.0, depth, sr, t_offset)

    mixed = np.tanh(mixed * 1.7) / 1.7
    mx = np.max(np.abs(mixed))
    if mx > 0: mixed = mixed / mx * 0.87

    _apply_chunk_envelope(mixed)
    pcm = (np.clip(mixed, -1.0, 1.0) * 32767).astype(np.int16)
    return _pcm_to_wav(pcm)




# ─── Chunk Generator ────────────────────────────────────────────────────────

def generate_chunk(mode_key: str, intensity: float = 0.5, t_offset: float = 0.0) -> bytes:
    """
    Generate a WAV chunk for the given mode.

    Args:
        mode_key: One of the keys in MODES
        intensity: 0.0–1.0, scales the modulation depth and noise/drone balance
        t_offset: Time offset in seconds for phase continuity between chunks

    Returns:
        WAV file as bytes
    """
    m = MODES[mode_key]

    # Ambient mode uses its own dedicated synthesis path
    if m.get('is_ambient'):
        return _generate_ambient_chunk(intensity, t_offset)

    n  = int(SAMPLE_RATE * CHUNK_DURATION)
    sr = SAMPLE_RATE

    # Scale parameters with intensity
    nv = m['noise_vol'] * (0.35 + 0.65 * intensity)
    dv = m['drone_vol'] * (0.35 + 0.65 * intensity)
    md = m['mod_depth'] * (0.15 + 0.85 * intensity)

    # Generate noise layer
    noise_fn = pink_noise if m['noise'] == 'pink' else brown_noise
    n_audio  = noise_fn(n)

    # Generate drone layer
    d_audio = harmonic_drone(m['base_freqs'], n, sr, t_offset)

    # Mix layers and apply entrainment AM modulation
    mixed     = nv * n_audio + dv * d_audio
    modulated = am_modulate(mixed, m['entrainment_hz'], md, sr, t_offset)

    # Soft clip + normalize
    modulated = np.tanh(modulated * 1.9) / 1.9
    mx = np.max(np.abs(modulated))
    if mx > 0:
        modulated = modulated / mx * 0.88

    _apply_chunk_envelope(modulated)
    pcm = (np.clip(modulated, -1.0, 1.0) * 32767).astype(np.int16)
    return _pcm_to_wav(pcm)
