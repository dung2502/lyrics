from pydub import AudioSegment

def mp3_to_wav(mp3_path, wav_path):
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")

# Ví dụ
mp3_to_wav("E:/School/project/lyrics/pythonProject/audio/NguoiTaDauThuongEmTest.mp3", "E:/School/project/lyrics/pythonProject/audio/NguoiTaDauThuongEmOutputTest.wav")

