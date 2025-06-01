from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment
import soundfile as sf
import torch
import torchaudio.functional as F
import tempfile
import os
import requests
import logging
from .aligner import align_lyrics, load_model
from lyric_alignment_api import utils

app = FastAPI()
load_model()

class AlignRequest(BaseModel):
    audio_path: str  # url mp3 public
    lyrics: str

def mp3_to_wav(mp3_path, wav_path):
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")

@app.post("/align")
async def align(req: AlignRequest):
    logging.info(f"Received audio URL: {req.audio_path}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mp3_path = os.path.join(tmpdir, "audio.mp3")
            wav_path = os.path.join(tmpdir, "audio.wav")
            resp = requests.get(req.audio_path, stream=True, timeout=15)
            if resp.status_code != 200:
                return JSONResponse(status_code=400, content={"error": "Failed to download audio"})

            total_bytes = 0
            with open(mp3_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        total_bytes += len(chunk)
                        if total_bytes > 20 * 1024 * 1024:
                            return JSONResponse(status_code=400, content={"error": "File too large"})
                        f.write(chunk)
            mp3_to_wav(mp3_path, wav_path)
            waveform_np, sr = sf.read(wav_path)
            if waveform_np.ndim > 1:
                waveform_np = waveform_np.mean(axis=1)

            waveform = torch.tensor(waveform_np).unsqueeze(0)  # shape: [1, time]
            if sr != 16000:
                waveform = F.resample(waveform, orig_freq=sr, new_freq=16000)

            lyric_segments = utils.parse_lyrics(req.lyrics)
            result = align_lyrics(waveform, lyric_segments)

    except Exception as e:
        logging.exception("Error processing audio")
        return JSONResponse(status_code=500, content={"error": f"Processing error: {str(e)}"})

    return JSONResponse(content=result)
