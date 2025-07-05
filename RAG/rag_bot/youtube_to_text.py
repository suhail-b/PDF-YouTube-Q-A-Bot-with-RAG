import whisper, yt_dlp, tempfile, os, subprocess

def youtube_to_text(url, model_size="tiny"):
    tmp = tempfile.mkdtemp()
    audio_mp4 = os.path.join(tmp, "audio.m4a")
    audio_mp3 = os.path.join(tmp, "audio.mp3")

    yt_dlp.YoutubeDL({'format': 'bestaudio', 'outtmpl': audio_mp4, 'quiet': True}).download([url])
    subprocess.run(["ffmpeg", "-y", "-i", audio_mp4, "-vn", "-acodec", "libmp3lame", audio_mp3],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    text = whisper.load_model(model_size).transcribe(audio_mp3)["text"]

    for f in [audio_mp4, audio_mp3]: os.remove(f)
    os.rmdir(tmp)
    return text
