"""
neuro — Flask Server
Serves the web UI and streams audio chunks on demand.
"""

import json
import os
from flask import Flask, Response, render_template, request
from sound_engine import generate_chunk, MODES, CHUNK_DURATION

app = Flask(__name__)


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main UI with mode data injected."""
    modes_json = json.dumps({
        k: {
            'name': v['name'],
            'emoji': v['emoji'],
            'tagline': v['tagline'],
            'band': v['band'],
            'band_name': v['band_name'],
            'band_range': v['band_range'],
            'entrainment_hz': v['entrainment_hz'],
            'color': v['color'],
        }
        for k, v in MODES.items()
    })
    return render_template('index.html', modes_json=modes_json, chunk_duration=CHUNK_DURATION)


@app.route('/chunk/<mode_key>')
def chunk(mode_key):
    """
    Generate and return a single WAV audio chunk.
    
    Query params:
        t         — time offset in seconds (for phase continuity)
        intensity — 0.0 to 1.0 (entrainment depth + mix balance)
    """
    if mode_key not in MODES:
        return 'Unknown mode', 404

    try:
        t_offset = float(request.args.get('t', 0.0))
        intensity = float(request.args.get('intensity', 0.5))
        intensity = max(0.0, min(1.0, intensity))
    except ValueError:
        return 'Invalid parameters', 400

    wav_bytes = generate_chunk(mode_key, intensity, t_offset)

    return Response(
        wav_bytes,
        mimetype='audio/wav',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Content-Length': str(len(wav_bytes)),
            'Access-Control-Allow-Origin': '*',
        }
    )


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
