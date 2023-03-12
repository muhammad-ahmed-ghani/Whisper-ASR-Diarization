import torch
from speechbox import ASRDiarizationPipeline
import pandas as pd
import os
from dotenv import load_dotenv
import gradio as gr
load_dotenv()

MODEL_NAME = "openai/whisper-large-v2"

device = 0 if torch.cuda.is_available() else "cpu"
if device == 0:
    print("""
    GPU: {}
    GPU Memory: {} GB
    CUDA Version: {}\n""".format(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9, torch.version.cuda))
else:
    print("Using CPU...")

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
    return df

def transcribe(file_upload, format_type="txt"):
    raw_segments = pipe(file_upload)
    if format_type == "srt":
        transcription = format_as_srt(raw_segments)
        return gr.update(visible=False), gr.update(visible=True, value=transcription), gr.update(visible=False)
    elif format_type == "csv":
        transcription = format_as_csv(raw_segments)
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=transcription)
    else:
        transcription = " ".join([chunk["text"] for chunk in raw_segments])
        return gr.update(visible=True, value=transcription), gr.update(visible=False), gr.update(visible=False)

interface = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(source="upload", type="filepath"),
        gr.Radio(["txt", "srt", "csv"], label="Format", value="txt"),
    ],
    outputs=[gr.Textbox(label="Transcription"), gr.Textbox(label="ASR+Diarization", visible=False), gr.Dataframe(label="ASR+Diarization", visible=False)],
    layout="horizontal",
    theme="huggingface",
    title="Whisper Speaker Diarization: Transcribe Audio",
    description=(
        "Transcribe audio files with speaker diarization using [🤗 Speechbox](https://github.com/huggingface/speechbox/). "
        "Demo uses the pre-trained checkpoint [Whisper large-v2](https://huggingface.co/openai/whisper-large-v2) for the ASR "
        "transcriptions and [pyannote.audio](https://huggingface.co/pyannote/speaker-diarization) to label the speakers."
    ),
    allow_flagging="never",
)

interface.launch(debug=True)