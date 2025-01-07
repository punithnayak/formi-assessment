import os
import json
import base64
import asyncio
import audioop
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer
from langchain_ollama import OllamaLLM
from gtts import gTTS
import tempfile
import subprocess


load_dotenv()

# Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
URL = os.getenv('URL')
PORT = int(os.getenv('PORT', 5050))
MODEL_PATH = "/home/punith/Desktop/realtime-twilio-outbound/vosk-live-transcription/model"  # Specify the path to the Vosk model directory
OLLAMA_MODEL_NAME = "llama3"  # Update with the desired model
CL = '\x1b[0K'
BS = '\x08'

app = FastAPI()
vosk_model = Model(MODEL_PATH)
ollama = OllamaLLM(model=OLLAMA_MODEL_NAME)

@app.get("/", response_class=HTMLResponse)
async def index_page():
    return "<html><body><h1>Twilio Media Stream Server is running!</h1></body></html>"

@app.post("/make-call")
async def make_call(request: Request):
    """Make an outgoing call to the specified phone number."""
    data = await request.json()
    to_phone_number = data.get("to")
    if not to_phone_number:
        return {"error": "Phone number is required"}

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call = client.calls.create(
        url=f"{URL}/outgoing-call",
        to=to_phone_number,
        from_=TWILIO_PHONE_NUMBER
    )
    return {"call_sid": call.sid}

@app.api_route("/outgoing-call", methods=["GET", "POST"])
async def handle_outgoing_call(request: Request):
    """Handle outgoing call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    response.say("Please wait while we connect your call to the AI voice assistant...")
    response.pause(length=1)
    response.say("O.K. you can start talking!")
    connect = Connect()
    connect.stream(url=f'wss://{request.url.hostname}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections for media stream."""
    await websocket.accept()
    print("WebSocket connection accepted.")
    rec = KaldiRecognizer(vosk_model, 16000)

    try:
        async for message in websocket.iter_text():
            print(f"Received message: {message}")
            packet = json.loads(message)

            if packet['event'] == 'start':
                print('Streaming is starting')
            elif packet['event'] == 'stop':
                print('\nStreaming has stopped')
            elif packet['event'] == 'media':
                print("Processing 'media' event.")
                # Decode incoming audio
                try:
                    audio = base64.b64decode(packet['media']['payload'])
                    print("Audio decoded successfully.")
                    audio = audioop.ulaw2lin(audio, 2)
                    audio = audioop.ratecv(audio, 2, 1, 8000, 16000, None)[0]
                    print("Audio resampled successfully.")
                except Exception as e:
                    print(f"Error decoding or resampling audio: {e}")
                    continue

                # Transcription
                if rec.AcceptWaveform(audio):
                    print("Waveform accepted.")
                    r = json.loads(rec.Result())
                    user_text = r['text']
                    print(f"User text: {user_text}")

                    # Generate AI response
                    try:
                        ai_response = ollama.invoke(user_text)
                        ai_text = ai_response
                        print(f"Ollama AI Response: {ai_text}")
                    except Exception as e:
                        print(f"Error invoking AI: {e}")
                        ai_text = "Sorry, I couldn't process that."

                    # Convert AI response to audio
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                            print(f"Generating TTS for AI text: {ai_text}")
                            tts = gTTS(ai_text)
                            tts.save(temp_wav.name)
                            print(f"TTS saved to temp file: {temp_wav.name}")

                            # Convert to mulaw/8000
                            mulaw_audio_path = tempfile.NamedTemporaryFile(suffix=".raw", delete=False).name
                            print(f"Converting audio to mulaw format.")
                            subprocess.run([
                                "ffmpeg", "-y", "-i", temp_wav.name,
                                "-ar", "8000", "-f", "mulaw", mulaw_audio_path
                            ], check=True)
                            print(f"Audio converted to mulaw: {mulaw_audio_path}")

                            with open(mulaw_audio_path, "rb") as f:
                                mulaw_audio = f.read()

                        # Base64 encode the raw audio
                        payload = base64.b64encode(mulaw_audio).decode('utf-8')
                        print("Audio base64 encoded successfully.")
                    except Exception as e:
                        print(f"Error during TTS or audio conversion: {e}")
                        continue

                    # Construct and send the media payload to Twilio
                    try:
                        media_message = {
                            "event": "media",
                            "streamSid": packet["streamSid"],
                            "media": {
                                "payload": payload
                            }
                        }
                        print(f"Sending media message: {media_message}")
                        await websocket.send_json(media_message)
                        print("Media message sent to Twilio.")
                    except Exception as e:
                        print(f"Error sending media message: {e}")
                else:
                    print("Waveform not accepted.")
                    r = json.loads(rec.PartialResult())
                    print(f"Partial result: {r['partial']}")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    to_phone_number = input("Please enter the phone number to call: ")
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    try:
        call = client.calls.create(
            url=f"{URL}/outgoing-call",
            to=to_phone_number,
            from_=TWILIO_PHONE_NUMBER
        )
        print(f"Call initiated with SID: {call.sid}")
    except Exception as e:
        print(f"Error initiating call: {e}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
