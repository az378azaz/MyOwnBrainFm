# neuro — Neural Audio for Focus & Rest

A personal Brain.fm-inspired web app that generates brainwave entrainment audio
using AM modulation, harmonic drones, and spectrally-shaped noise.

## Modes
| Mode | Band | Hz |
|---|---|---|
| Focus | β Beta | 18 Hz |
| Hyper Focus | γ Gamma | 30 Hz |
| ADHD Focus | β Beta | 22 Hz |
| Creativity | α/θ Alpha–Theta | 8 Hz |
| Relax | α Alpha | 10 Hz |
| Meditate | θ Theta | 6 Hz |
| Sleep | δ Delta | 2 Hz |
| Power Nap | δ/θ Delta–Theta | 4 Hz |

## Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open http://localhost:5000
```

To share with someone on the same network: replace `localhost` with your local IP (e.g. `192.168.1.x:5000`).

---

## Deploy to Railway (free, accessible anywhere)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "initial commit"
   gh repo create neuro-sound --private --push
   ```

2. **Create Railway project**
   - Go to [railway.app](https://railway.app) → New Project
   - Select **Deploy from GitHub repo** → choose `neuro-sound`
   - Railway auto-detects Python and uses the `Procfile`

3. **Get your URL**
   - Railway gives you a public URL like `neuro-sound-production.up.railway.app`
   - Share this with your girlfriend — no install required

4. **Free tier**: 500 hours/month — more than enough for personal use.

---

## Architecture

```
Browser (Web Audio API)         Flask Server
─────────────────────           ─────────────────────
selectMode('focus')             GET /chunk/focus?t=0&intensity=0.5
                                  ↓
NeuroAudioEngine                sound_engine.generate_chunk()
  └── fetch chunk                  ├── pink_noise(n)
  └── decodeAudioData              ├── harmonic_drone(freqs, n, t0)
  └── schedule BufferSource        ├── am_modulate(audio, 18Hz, depth)
  └── connect to AnalyserNode      └── returns WAV bytes
  └── visualize waveform
```

## Sound Design

Each mode mixes three layers:
1. **Harmonic drone** — detuned sine oscillators with slow LFO modulation
2. **Shaped noise** — pink (focus modes) or brown (rest modes) noise via FFT
3. **AM entrainment** — amplitude modulation at the target brainwave frequency

Phase continuity is maintained across chunks via `t_offset`, ensuring seamless infinite playback.
