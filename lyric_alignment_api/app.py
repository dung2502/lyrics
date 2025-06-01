# app.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
import shutil
import os
from align import align_audio_and_lyrics

app = FastAPI()

@app.post("/align")
async def align(audio: UploadFile, lyrics: str = Form(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_audio_path = tmp.name

    try:
        # Gọi hàm align từ model gốc
        alignment_result = align_audio_and_lyrics(tmp_audio_path, lyrics)
        return JSONResponse(content=alignment_result)
    finally:
        os.remove(tmp_audio_path)
