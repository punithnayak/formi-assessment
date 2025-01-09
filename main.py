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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from typing import Dict
import numpy as np
from scipy.signal import resample

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
escalation_bool = False
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
    response.say("Press button zero whenever you want to escalate to human agent")
    response.pause(length=1)
    response.say("O.K. you can start talking!")
    connect = Connect()
    connect.stream(url=f'wss://{request.url.hostname}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

# Add CORS for the web page
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep track of WebSocket connections
active_connections: Dict[str, WebSocket] = {}

@app.websocket("/web-page")
async def handle_web_page(websocket: WebSocket):
    """Handle WebSocket connections from the web page (human agent)."""
    await websocket.accept()
    client_id = f"{websocket.client[0]}:{websocket.client[1]}"
    active_connections[client_id] = websocket
    print(f"Web page connected: {client_id}")

    try:
        while True:
            # Listen for audio responses from the human agent
            message = await websocket.receive_text()

            try:
                response_packet = json.loads(message)
                if response_packet["event"] == "media":
                    stream_sid = response_packet["streamSid"]
                    base64_audio = response_packet["payload"]

                    # Decode the Base64 payload to raw audio
                    raw_audio = base64.b64decode(base64_audio)

                    # Convert audio to mulaw/8000 for Twilio
                    mulaw_audio = audioop.lin2ulaw(raw_audio, 2)  # Convert PCM to mulaw

                    # Encode mulaw audio back to Base64
                    encoded_audio = base64.b64encode(mulaw_audio).decode('utf-8')

                    # Construct the Twilio-compatible media message
                    twilio_message = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": encoded_audio},
                    }
                    print(f'Sending from browser to twilio:{twilio_message}')
                    # Send the processed audio back to Twilio
                    for client_id, client_ws in active_connections.items():
                        try:
                            await client_ws.send_json(twilio_message)
                            print(f"Forwarded audio to Twilio for Stream SID: {stream_sid}")
                        except Exception as e:
                            print(f"Error forwarding audio to Twilio: {e}")
            except Exception as e:
                print(f"Error processing audio from web client {client_id}: {e}")

    except Exception as e:
        print(f"WebSocket error for web client {client_id}: {e}")
    finally:
        active_connections.pop(client_id, None)
        print(f"Web page disconnected: {client_id}")

templates = Jinja2Templates(directory="templates")

@app.get("/web-page", response_class=HTMLResponse)
async def web_page(request: Request):
    """
    Serve the HTML page dynamically using Jinja2 templates.
    """
    websocket_url = f"wss://{request.url.hostname}/web-page"
    return templates.TemplateResponse(
        "web_page.html",
        {"request": request, "websocket_url": websocket_url},
    )

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections for media stream."""
    global escalation_bool
    await websocket.accept()
    print("WebSocket connection accepted.")
    rec = KaldiRecognizer(vosk_model, 16000)

    try:
        async for message in websocket.iter_text():
            # print(f"Received message: {message}")
            packet = json.loads(message)
            audio_file_path = "/tmp/twilio_audio.raw"
            if packet['event'] == 'start':
                print('Streaming is starting')
            elif packet['event'] == 'stop':
                print('\nStreaming has stopped')
            elif packet['event'] == 'dtmf':
                # Handle DTMF event directly from the media stream
                dtmf_digit = packet['dtmf']['digit']
                print(f"DTMF digit received: {dtmf_digit}")

                if dtmf_digit == "0":
                    escalation_bool = True
                    print("User pressed 0. Escalating to human agent...")
                    for client_id, client_ws in active_connections.items():
                        try:
                            await client_ws.send_json({"event": "escalation", "message": "User pressed 0 for escalation"})
                            print(f"Notified web page: {client_id}")
                        except Exception as e:
                            print(f"Error notifying web page {client_id}: {e}")
            elif escalation_bool == True and packet['event'] == 'media':
                # Decode audio payload from Twilio and send to web clients
                payload = packet["media"]["payload"]
                stream_sid = packet["streamSid"]
                raw_audio = base64.b64decode(payload)

                # Resample from 8000 Hz to 44100 Hz
                pcm_audio_np = np.frombuffer(raw_audio, dtype=np.int16)
                num_samples = int(len(pcm_audio_np) * (44100 / 8000))
                resampled_audio_np = resample(pcm_audio_np, num_samples).astype(np.int16)
                resampled_audio = resampled_audio_np.tobytes()

                # Encode resampled PCM audio to Base64
                resampled_payload = base64.b64encode(resampled_audio).decode('utf-8')

                # Relay audio to human agent
                for client_id, client_ws in active_connections.items():
                    try:
                        await client_ws.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "payload": resampled_payload,
                        })
                        print(f"Relayed audio payload to web client {client_id}")
                    except Exception as e:
                        print(f"Error relaying audio to web client {client_id}: {e}")
                    
            elif escalation_bool  == False and packet['event'] == 'media':
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
