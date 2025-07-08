import whisper

model = whisper.load_model("base")
result = model.transcribe("audio1174385774.m4a")
print(result["text"])