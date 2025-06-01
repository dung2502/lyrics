# inference_helper.py

import torch
from alignment import align_one
from whisper import load_model

# Load mô hình Whisper khi khởi động server
whisper_model = load_model("small", device="cuda" if torch.cuda.is_available() else "cpu")

def run_for_file(audio_path: str, lyric_text: str):
    """
    Gọi align_one từ repo lyric-alignment để xử lý file audio + lời bài hát.
    """
    result = align_one(audio_path, lyric_text, whisper_model)
    return result
