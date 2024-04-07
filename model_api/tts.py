import os
import time
import asyncio
import aiohttp        
import aiofiles
import requests
import websockets
import ssl
import certifi
import edge_tts
import edge_tts.constants
import playsound
from typing import AsyncGenerator, Dict, Any
import json
import subprocess
import shlex
import wave

from configs import (
    logger, SPEECH_DIR, DEFAULT_VOICE, DEFAULT_VOICE_LOCAL,
    CONST_SPEECH_MAP, LOCAL_TTS_SERVER, USING_LOCAL_TTS, AUDIO_FILE_SUFFIX,
)

# Override microsoft edge-tts Communicate class, otherwise aiohttp websocket connecting is too slow
class CommunicateEx(edge_tts.Communicate):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

    async def stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Streams audio and metadata from the service."""

        async def send_command_request() -> None:
            """Sends the request to the service."""

            # Prepare the request to be sent to the service.
            #
            # Note sentenceBoundaryEnabled and wordBoundaryEnabled are actually supposed
            # to be booleans, but Edge Browser seems to send them as strings.
            #
            # This is a bug in Edge as Azure Cognitive Services actually sends them as
            # bool and not string. For now I will send them as bool unless it causes
            # any problems.
            #
            # Also pay close attention to double { } in request (escape for f-string).
            await websocket.send(
                f"X-Timestamp:{edge_tts.communicate.date_to_string()}\r\n"
                "Content-Type:application/json; charset=utf-8\r\n"
                "Path:speech.config\r\n\r\n"
                '{"context":{"synthesis":{"audio":{"metadataoptions":{'
                '"sentenceBoundaryEnabled":false,"wordBoundaryEnabled":true},'
                '"outputFormat":"audio-24khz-48kbitrate-mono-mp3"'
                "}}}}\r\n"
            )

        async def send_ssml_request() -> bool:
            """Sends the SSML request to the service."""

            # Get the next string from the generator.
            text = next(texts, None)

            # If there are no more strings, return False.
            if text is None:
                return False

            # Send the request to the service and return True.
            await websocket.send(
                edge_tts.communicate.ssml_headers_plus_data(
                    edge_tts.communicate.connect_id(),
                    edge_tts.communicate.date_to_string(),
                    edge_tts.communicate.mkssml(text, self.voice, self.rate, self.volume, self.pitch),
                )
            )
            return True

        def parse_metadata():
            for meta_obj in json.loads(data)["Metadata"]:
                meta_type = meta_obj["Type"]
                if meta_type == "WordBoundary":
                    current_offset = meta_obj["Data"]["Offset"] + offset_compensation
                    current_duration = meta_obj["Data"]["Duration"]
                    return {
                        "type": meta_type,
                        "offset": current_offset,
                        "duration": current_duration,
                        "text": meta_obj["Data"]["text"]["Text"],
                    }
                elif meta_type in ("SessionEnd",):
                    continue
                else:
                    raise edge_tts.exceptions.UnknownResponse(f"Unknown metadata type: {meta_type}")

        while True:
            logger.info('Edge-TTS websockets connecting...')

            # Split the text into multiple strings if it is too long for the service.
            texts = edge_tts.communicate.split_text_by_byte_length(
                edge_tts.communicate.escape(edge_tts.communicate.remove_incompatible_characters(self.text)),
                edge_tts.communicate.calc_max_mesg_size(self.voice, self.rate, self.volume, self.pitch),
            )

            # Keep track of last duration + offset to calculate the offset
            # upon word split.
            last_duration_offset = 0

            # Current offset compensations.
            offset_compensation = 0

            # Create a new connection to the service.
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            try:
                async with websockets.connect(
                    f"{edge_tts.constants.WSS_URL}&ConnectionId={edge_tts.communicate.connect_id()}",
                    extra_headers={
                        "Pragma": "no-cache",
                        "Cache-Control": "no-cache",
                        "Origin": "chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Accept-Language": "en-US,en;q=0.9",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        " (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36 Edg/91.0.864.41",
                    },
                    ssl=ssl_ctx,
                ) as websocket:
                    try:
                        logger.info('Edge-TTS websockets connected.')
                        # audio_was_received indicates whether we have received audio data
                        # from the websocket. This is so we can raise an exception if we
                        # don't receive any audio data.
                        audio_was_received = False

                        # Send the request to the service.
                        logger.info('Edge-TTS send_command_request ...')
                        await send_command_request()

                        # Send the SSML request to the service.
                        logger.info('Edge-TTS send_ssml_request ...')
                        await send_ssml_request()

                        logger.info('Edge-TTS start to receive messages ...')
                        async for received in websocket:
                            if type(received) == str:
                                parameters, data = edge_tts.communicate.get_headers_and_data(received)
                                path = parameters.get(b"Path")
                                if path == b"audio.metadata":
                                    # Parse the metadata and yield it.
                                    parsed_metadata = parse_metadata()
                                    yield parsed_metadata

                                    # Update the last duration offset for use by the next SSML request.
                                    last_duration_offset = (
                                        parsed_metadata["offset"] + parsed_metadata["duration"]
                                    )
                                elif path == b"turn.end":
                                    # Update the offset compensation for the next SSML request.
                                    offset_compensation = last_duration_offset

                                    # Use average padding typically added by the service
                                    # to the end of the audio data. This seems to work pretty
                                    # well for now, but we might ultimately need to use a
                                    # more sophisticated method like using ffmpeg to get
                                    # the actual duration of the audio data.
                                    offset_compensation += 8_750_000

                                    # Send the next SSML request to the service.
                                    if not await send_ssml_request():
                                        break
                                elif path in (b"response", b"turn.start"):
                                    pass
                                else:
                                    raise edge_tts.exceptions.UnknownResponse(
                                        "The response from the service is not recognized.\n"
                                        + received
                                    )
                            elif type(received) == bytes:
                                if len(received) < 2:
                                    raise edge_tts.exceptions.UnexpectedResponse(
                                        "We received a binary message, but it is missing the header length."
                                    )

                                header_length = int.from_bytes(received[:2], "big")
                                if len(received) < header_length + 2:
                                    raise edge_tts.exceptions.UnexpectedResponse(
                                        "We received a binary message, but it is missing the audio data."
                                    )

                                audio_was_received = True
                                yield {
                                    "type": "audio",
                                    "data": received[header_length + 2 :],
                                }

                        if not audio_was_received:
                            raise edge_tts.exceptions.NoAudioReceived(
                                "No audio was received. Please verify that your parameters are correct."
                            )
                        break
                    except websockets.ConnectionClosed as e:
                        logger.error(f'edge-tts websocket close: {e}, reconnect...')
                        continue
            except Exception as e:
                logger.info(f'edge-tts websocket connect error: {e}, reconnect...')
                continue

RATE = "+0%"
VOLUMN = "+0%"
PITCH = "+0Hz"

async def create_audio_file(text, voice, output_file):
    if USING_LOCAL_TTS:
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)  # Setting the number of channels
            wf.setsampwidth(2)  # Setting the sample width
            wf.setframerate(32000)  # Setting the sample rate
            async for chunk in local_tts_audio_stream(text, voice):
                wf.writeframes(chunk)
    else:
        communicate = CommunicateEx(text, voice, rate=RATE, volume=VOLUMN, pitch=PITCH)
        await communicate.save(output_file)

async def edge_tts_audio_stream(text, voice):
    communicate = CommunicateEx(text, voice, rate=RATE, volume=VOLUMN, pitch=PITCH)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

async def local_tts_audio_stream(text, voice):
    urlencoded_text = requests.utils.quote(text)
    async with aiohttp.ClientSession() as session:
        payload = {
            "cha_name": voice,
            "character_emotion": "default",
            "text": urlencoded_text,
            "text_language": "auto",
            "batch_size": 10,
            "speed": 1,
            "top_k": 6,
            "top_p": 0.8,
            "temperature": 0.8,
            "stream": "True",
            "cut_method": "auto_cut_25",
            "seed": -1,
            "save_temp": "False"
        }
        async with session.post(LOCAL_TTS_SERVER, json=payload) as resp:
            if resp.status == 200:
                async for data in resp.content.iter_chunked(1024):
                    yield data
            else:
                content = await resp.text()
                logger.error(f'connect to {LOCAL_TTS_SERVER} fail: {content}')

async def play_audio_stream(text, voice):
    ffplay_process = "ffplay -autoexit -nodisp -i pipe:0"
    player = subprocess.Popen(shlex.split(ffplay_process), stdin=subprocess.PIPE)

    if USING_LOCAL_TTS:
        async_audio_stream = local_tts_audio_stream
    else:
        async_audio_stream = edge_tts_audio_stream

    async for audio_data in async_audio_stream(text, voice):
        # 将二进制MP3声音数据传送到ffplay进行播放
        player.stdin.write(audio_data)
    
    player.stdin.close()
    await asyncio.to_thread(player.wait)
    logger.info('play_audio_stream done')

def play_audio_file_in_stream(file_path, block=True):
    logger.info(f'play_audio_file_in_stream: {file_path}')
    ffplay_process = "ffplay -autoexit -nodisp -i pipe:0"
    player = subprocess.Popen(shlex.split(ffplay_process), stdin=subprocess.PIPE)
    chunk_size = 1024
    with open(file_path, 'rb') as fp:
        # 将二进制MP3声音数据传送到ffplay进行播放
        while True:
            data = fp.read(chunk_size)
            if not data:
                break
            player.stdin.write(data)
    
    player.stdin.close()
    if block:
        player.wait()

def play_audio(output_file, block=True):
    logger.info(f'play_audio: {output_file}')
    playsound.playsound(output_file, block=block)

async def text_to_audio(text, voice):
    output_file = os.path.join(SPEECH_DIR, f'{int(time.time())}.{AUDIO_FILE_SUFFIX}')
    await create_audio_file(text, voice, output_file)
    return output_file

async def async_play_audio(output_file, block=True):
    if block:
        await asyncio.to_thread(play_audio, output_file, block=block)
    else:
        play_audio(output_file, block=block)

async def play_text(text, voice, block=True):
    output_file = await text_to_audio(text, voice)
    await async_play_audio(output_file, block=block)

async def create_const_speech():
    voice = DEFAULT_VOICE_LOCAL if USING_LOCAL_TTS else DEFAULT_VOICE
    for x in CONST_SPEECH_MAP.values():
        if os.path.exists(x['file']):
            continue
        await create_audio_file(x['txt'], voice, x['file'])

def play_const_audio(key: str, block=True):
    if key not in CONST_SPEECH_MAP:
        return
    const_speech = CONST_SPEECH_MAP[key]
    if not os.path.exists(const_speech['file']):
        return
    # play_audio(BOOTED_SPEECH_FILE, block=False)
    play_audio_file_in_stream(const_speech['file'], block=block)
