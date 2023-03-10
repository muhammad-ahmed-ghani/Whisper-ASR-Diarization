# Whisper-ASR-Diarization

## System Requirements
1. Ubuntu 18.04 or higher
2. Conda
3. Nvidia GPU 8GB or higher with CUDA 11.0 or higher


## Setup
1. Create a virtual environment and install dependencies
```bash
conda create -n asr-diarization python=3.9 -y
conda activate asr-diarization
pip install -r requirements.txt
```
2. Create a .env file in the root directory and add the following line
```HUGGINGFACE_TOKEN=<your_huggingface_token>```
3. Run the server using ```uvicorn main:asgi_app --reload``` **OR** ```python app.py```
4. Test the server using the following command
```bash
curl -F "file=@<path_to_audio_file>" -F "return_format=<txt/csv/srt>" http://localhost:8000/api/v1/transcribe
```
Here **file** is the audio file absolute path and return_format is the format in which you want the transcription. It can be **txt**, **csv** or **srt**.