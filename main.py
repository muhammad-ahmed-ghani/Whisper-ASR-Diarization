import torch
from speechbox import ASRDiarizationPipeline
from flask import Flask ,request, Response
from flask_cors import CORS
from asgiref.wsgi import WsgiToAsgi
import soundfile as sf
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
asgi_app = WsgiToAsgi(app)

MODEL_NAME = "openai/whisper-large-v2"

device = 0 if torch.cuda.is_available() else "cpu"
print("Torch running on device: ", "GPU:0" if device==0 else "CPU")

pipe = ASRDiarizationPipeline.from_pretrained(
    asr_model=MODEL_NAME,
    diarizer_model = "pyannote/speaker-diarization@2.1",
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
    device=device,
)

def tuple_to_string(start_end_tuple):
    start = start_end_tuple[0]
    end = start_end_tuple[1]
    start_hours = int(start / 3600)
    start_minutes = int((start - start_hours * 3600) / 60)
    start_seconds = int(start - start_hours * 3600 - start_minutes * 60)
    start_milliseconds = int((start - start_hours * 3600 - start_minutes * 60 - start_seconds) * 1000)
    end_hours = int(end / 3600)
    end_minutes = int((end - end_hours * 3600) / 60)
    end_seconds = int(end - end_hours * 3600 - end_minutes * 60)
    end_milliseconds = int((end - end_hours * 3600 - end_minutes * 60 - end_seconds) * 1000)
    return str((start_hours, start_minutes, start_seconds, start_milliseconds)) + " --> " + str((end_hours, end_minutes, end_seconds, end_milliseconds))

def format_as_srt(raw_segments):
    srt = ""
    for i, chunk in enumerate(raw_segments):
        srt += str(i+1) + "\n"
        srt += tuple_to_string(chunk["timestamp"]) + "\n"
        srt += chunk["speaker"] + "\n"
        srt += chunk["text"] + "\n\n"
    return srt

def format_as_csv(raw_segments):
    df = pd.DataFrame(raw_segments)
    df["start"] = df["timestamp"].apply(lambda x: x[0])
    df["end"] = df["timestamp"].apply(lambda x: x[1])
    df = df.drop("timestamp", axis=1)
    return df.to_csv(index=False)

def transcribe(file_upload, format_type="txt"):
    raw_segments = pipe(file_upload)
    if format_type == "srt":
        transcription = format_as_srt(raw_segments)
    elif format_type == "csv":
        transcription = format_as_csv(raw_segments)
    else:
        transcription = " ".join([chunk["text"] for chunk in raw_segments])
    return transcription

@app.route("/")
def main():
  return "ASR+Diarization Server is running!"

@app.route("/api/v1/transcribe", methods=["POST"])
def transcribe_endpoint():
    if request.method == "POST":
        if "file" not in request.files:
            return Response("Error: No 'file' param part in the request", mimetype="text/plain", status=400)
        if "return_format" not in request.form:
            return Response("Error: No 'return_format' param in the request", mimetype="text/plain", status=400)
        
        file_upload = request.files["file"]
        format_type = request.form.get("return_format")

        if format_type not in ["txt", "csv", "srt"]:
            return Response("Error: 'return_format' param must be one of: txt, csv, srt", mimetype="text/plain", status=400)
        if file_upload.filename == "":
            return Response("Error: No file selected", mimetype="text/plain", status=400)
        
        print("Received file: ", file_upload.filename)
        audio, sr = sf.read(file_upload)
        sf.write(file_upload.filename, audio, sr)
        try:
            transcription = transcribe(file_upload.filename, format_type=format_type)
        except Exception as e:
            print(e)
            return Response("Error: " + str(e), mimetype="text/plain", status=500)
        if format_type=="txt":
            return Response(transcription, mimetype="text/plain", status=200)
        elif format_type=="csv":
            return Response(transcription, mimetype="text/csv", status=200)
        else:
            return Response(transcription, mimetype="text/plain", status=200)
        