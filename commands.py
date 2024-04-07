import argparse

from db.base import Base, engine
from db.models import *
from configs import logger, NLTK_DATA_PATH, WHISPER_MODEL_PATH

import nltk
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

def create_tables():
    Base.metadata.create_all(bind=engine)

def reset_tables():
    Base.metadata.drop_all(bind=engine)
    create_tables()

def test_faiss_cache():
    import time, random
    from pprint import pprint
    import threading
    from knowledge_base.kb_cache.base import load_local_embeddings
    from knowledge_base.kb_cache.faiss_cache import kb_faiss_pool

    kb_names = ["vs1", "vs2", "vs3"]
    # for name in kb_names:
    #     memo_faiss_pool.load_vector_store(name)

    def worker(vs_name: str, name: str):
        vs_name = "samples"
        time.sleep(random.randint(1, 5))
        embeddings = load_local_embeddings()
        r = random.randint(1, 3)

        with kb_faiss_pool.load_vector_store(vs_name).acquire(name) as vs:
            if r == 1: # add docs
                ids = vs.add_texts([f"text added by {name}"], embeddings=embeddings)
                pprint(ids)
            elif r == 2: # search docs
                docs = vs.similarity_search_with_score(f"{name}", k=3, score_threshold=1.0)
                pprint(docs)
        if r == 3: # delete docs
            logger.warning(f"clear {vs_name} by {name}")
            cache = kb_faiss_pool.get(vs_name);
            if cache:
                cache.clear()

    threads = []
    for n in range(1, 30):
        t = threading.Thread(target=worker,
                             kwargs={"vs_name": random.choice(kb_names), "name": f"worker {n}"},
                             daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

def test_doc_loaders():
    # from document_loaders.FilteredCSVloader import FilteredCSVLoader
    # loader = FilteredCSVLoader(file_path="./samples/langchain-ChatGLM_closed.csv", columns_to_read=[
    #     'title', 'file', 'url', 'detail', 'id',
    # ])
    # docs = loader.load()
    # logger.info(f'csv: {docs[0]}')

    from document_loaders.mydocloader import RapidOCRDocLoader
    loader = RapidOCRDocLoader(file_path="./samples/ocr_test.docx")
    docs = loader.load()
    logger.info(f'docx: {docs[0]}')

    from document_loaders.mypptloader import RapidOCRPPTLoader
    loader = RapidOCRPPTLoader(file_path="./samples/ocr_test.pptx")
    docs = loader.load()
    logger.info(f'ppt: {docs[0]}')

    from document_loaders.myimgloader import RapidOCRLoader
    loader = RapidOCRLoader(file_path="./samples/ocr_test.jpg")
    docs = loader.load()
    logger.info(f'img: {docs[0]}')

    from document_loaders.mypdfloader import RapidOCRPDFLoader
    loader = RapidOCRPDFLoader(file_path="./samples/ocr_test.pdf")
    docs = loader.load()
    logger.info(f'pdf: {docs[0]}')

def test_llm():
    from configs import DEFAULT_CHAT_MODEL, ChatModels
    from model_api.cached_conversation import CachedConversation
    conversation = CachedConversation(DEFAULT_CHAT_MODEL)
    
    query = 'Do you know someting about Japan? Could you act as a guide to Japan?'
    logger.info('start simple_chat ...')
    result = conversation.simple_chat(query)
    logger.info(f'simple_chat result: [{result}]')

    query = 'Talk someting about the history of this country, please.'
    logger.info('start simple_chat ...')
    result = conversation.simple_chat(query)
    logger.info(f'simple_chat result: [{result}]')

    query = 'What files can playsound module play?'
    logger.info('start simple_chat ...')
    result = conversation.simple_chat(query)
    logger.info(f'simple_chat result: [{result}]')

    query = 'What files can playsound module play?'
    reference = """
        playsound module instructions:
        The playsound module contains only a single function named playsound().
        It requires one argument: the path to the file with the sound we have to play. It can be a local file, or a URL.
        There's an optional second argument, block, which is set to True by default. We can set it to False for making the function run asynchronously.
        It works with both WAV and MP3 files.
    """
    logger.info('start kb_chat ...')
    result = conversation.kb_chat(query, reference)
    logger.info(f'kb_chat result: [{result}]')

def test_whisper_transcribe():
    from whisper import Whisper
    try:
        whisper = Whisper(
            model_path=WHISPER_MODEL_PATH, 
            # n_threads=4, 
            # best_of=2,
            print_progress=False,
            print_realtime=False,
            print_colors=True,
        )
        lang, result, prob, logprob_min, elapsed = whisper.transcribe_wav(
            # file_path='./cache/speech/record/20240323_193832.wav', 
            file_path='./samples/jfk.wav', 
            lang='auto', 
        )
        logger.info(f'lang: {lang}, prob: {prob:.04f}, logprob_min: {logprob_min:.04f}, elapsed {elapsed:.04f} seconds')
        logger.info(f'result: [{result}]')

        del whisper
    except Exception as e:
        logger.error(f'whisper error: [{e}]', exc_info=e)

def test_whisper_cmd():
    import time
    import asyncio
    from whisper import Whisper
    from whisper.audio_recorder import AudioRecorder, pack_audio_chunks
    from model_api.tts import create_const_speech

    async def _main():
        await create_const_speech()
    
    try:
        asyncio.run(_main())
        whisper = Whisper(
            model_path=WHISPER_MODEL_PATH, 
            # n_threads=4, 
            # best_of=2,
            print_progress=False,
            print_realtime=False,
            print_colors=True,
        )

        audio_recorder = AudioRecorder()
        time.sleep(1)
        audio_recorder.start()

        while audio_recorder.is_active():
            sentences = audio_recorder.fetch_sentences()
            if len(sentences) == 0:
                time.sleep(0.01)
                continue

            for chunks in sentences:
                raw_data = pack_audio_chunks(chunks)
        
                whisper.handle_command_workflow(
                    raw_data=raw_data, 
                    sample_width=AudioRecorder.WIDTH,
                    lang='auto', 
                )

        del whisper
    except Exception as e:
        logger.error(f'whisper error: [{e}]', exc_info=e)

def test_whisper_wav_cmd():
    import time
    import asyncio
    from whisper import Whisper
    from whisper.audio_recorder import AudioRecorder

    whisper = Whisper(
        model_path=WHISPER_MODEL_PATH, 
        # n_threads=4, 
        # best_of=2,
        print_progress=False,
        print_realtime=False,
        print_colors=True,
    )

    prompt_wav = './cache/speech/record/20240406_203416.wav'
    cmd_wav = './cache/speech/record/20240406_203421.wav'
    pcmf32_prompt, _ = whisper._read_wav(prompt_wav, stereo=False)
    pcmf32_cmd, _ = whisper._read_wav(cmd_wav, stereo=False)

    whisper.process_commands(pcmf32_prompt)

    pcmf32_cmd = pcmf32_prompt + pcmf32_cmd
    whisper.process_commands(pcmf32_cmd)

def test_upload_file():
    from communication.api.kb_doc_api import upload_file
    file_name = 'finite_diff_fomulars.pdf'
    kb_name = 'kb_3'
    rsp = upload_file(file_name, kb_name)
    logger.info(f'rsp: {rsp}')

def test_tts():
    from model_api.tts import play_audio_file_in_stream, play_audio_stream
    import asyncio
    import time
    from configs import DEFAULT_VOICE, DEFAULT_VOICE_LOCAL, USING_LOCAL_TTS

    text = """
        A native of Oakland, California, the 38-year-old Shanahan is a philanthropist with a long history of donating to Democrat and left-leaning causes, including supporting President Biden in his 2020 election bid before switching to Kennedy when he launched his own run for the Democrat nomination last year.
    """
    voice = DEFAULT_VOICE_LOCAL if USING_LOCAL_TTS else DEFAULT_VOICE

    async def _async_main():
        # await play_text(text, voice)
        # print('play finished.')
        await play_audio_stream(text, voice)
        # input('press any key...')
    
    asyncio.run(_async_main())

    # audio_file_1 = './cache/speech/booted_speech.mp3'
    # audio_file_2 = './cache/speech/connected_speech.mp3'
    # while True:
    #     play_audio_file_in_stream(audio_file_1)
    #     play_audio_file_in_stream(audio_file_2)

def test_edge():
    from model_api.tts import edge_tts_audio_stream
    from configs import DEFAULT_VOICE
    import asyncio
    import time
    import wonderwords
    from io import BytesIO

    voice = DEFAULT_VOICE
    
    async def _async_main():
        first_max_tm = 0
        first_min_tm = 0
        max_tm = 0
        min_tm = 0
        while True:
            s = wonderwords.RandomSentence()
            text = ''
            for _ in range(20):
                if text:
                    text += ' '
                text += s.sentence()
            logger.info(f'text: {text}')

            audio_len = 0
            start_tm = time.time()
            got_first_chunk = False
            with BytesIO() as bio:
                async for audio_data in edge_tts_audio_stream(text, voice):
                    if not got_first_chunk:
                        elaspsed_tm = time.time() - start_tm
                        if elaspsed_tm < first_min_tm or first_min_tm <= 0:
                            first_min_tm = elaspsed_tm
                        if elaspsed_tm > first_max_tm:
                            first_max_tm = elaspsed_tm
                        logger.info(f'got_first_chunk, elaspsed_tm({elaspsed_tm:.02f}s), first_min_tm({first_min_tm:.2f}s), first_max_tm({first_max_tm:.2f}s)')
                        got_first_chunk = True
                    audio_len += len(audio_data)
                    bio.write(audio_data)
                    if audio_len > (1 * 1000 * 1000):
                        dump_file_path = f'./cache/{int(time.time())}.audio'
                        with open(dump_file_path, 'wb') as fp:
                            fp.write(bio.getvalue())
            
            elaspsed_tm = time.time() - start_tm
            if elaspsed_tm < min_tm or min_tm <= 0:
                min_tm = elaspsed_tm
            if elaspsed_tm > max_tm:
                max_tm = elaspsed_tm
            logger.info(f'audio_len: {audio_len}, elaspsed_tm({elaspsed_tm:.02f}s), min_tm({min_tm:.2f}s), max_tm({max_tm:.2f}s)')

            time.sleep(3)
    
    asyncio.run(_async_main())


def test_simple_vad():
    import wave
    from io import BytesIO
    import array
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import List

    from whisper.utils import high_pass_filter
    
    sample_rate = 16000
    sample_width = 2

    def _read_wav(file_path):
        pcmf32 = []
        with wave.open(file_path, "rb") as wavfile:
            params = wavfile.getparams()
            if params.nchannels != 1 and params.nchannels != 2:
                raise Exception(f"WAV file '{file_path}' must be mono or stereo")
            if params.framerate != sample_rate:
                raise Exception(f"WAV file '{file_path}' must be {sample_rate/1000} kHz")
            if params.sampwidth != sample_width:
                raise Exception(f"WAV file '{file_path}' must be 16-bit")
            
            bs = wavfile.readframes(params.nframes)
            logger.info(f'{params}, readframes: {len(bs)}')           

            with BytesIO(bs) as bio:
                frames_org = [int.from_bytes(bio.read(2), byteorder='little', signed=True) for _ in range(params.nframes * params.nchannels)]

                mx_sp = max(abs(i) for i in frames_org)
                MAXIMUM = 32767
                frames_normal = []
                for sp in frames_org:
                    frames_normal.append(sp * MAXIMUM / mx_sp)

                frames = frames_org
                # frames = frames_normal

                if params.nchannels == 1:
                    pcmf32 = [ float(frames[x]) / 32768.0 for x in range(params.nframes) ]
                else:
                    pcmf32 = [ float(frames[2*x] + frames[2*x+1]) / 65536.0 for x in range(params.nframes) ]

            wave_data_org = np.array(frames_org)
            wave_data_normal = np.array(frames_normal)
            wave_time = np.arange(0, params.nframes) * (1.0 / params.framerate)

            return pcmf32, wave_data_org, wave_data_normal, wave_time
        
    def _vad_simple(pcmf32: List[float], sample_rate: int, last_ms: int, freq_thold: float):
        n_samples = len(pcmf32)
        n_samples_last = (sample_rate * last_ms) // 1000
        
        energy_all_tt = 0.0
        energy_last_tt = 0.0
        for i in range(n_samples):
            energy_all_tt += abs(pcmf32[i])
            if i >= n_samples - n_samples_last:
                energy_last_tt += abs(pcmf32[i])
        
        if freq_thold > 0.0:
            high_pass_filter(pcmf32, freq_thold, sample_rate)
        
        energy_all  = 0.0
        energy_last = 0.0

        for i in range(n_samples):
            energy_all += abs(pcmf32[i])
            if i >= n_samples - n_samples_last:
                energy_last += abs(pcmf32[i])

        energy_all  /= n_samples
        energy_last /= n_samples_last

        if energy_all <= 0:
            return 1
        
        return energy_last / energy_all
    
    file_path = './test.wav'
    pcmf32, wave_data_org, wave_data_normal, wave_time = _read_wav(file_path)
    total_samples = len(pcmf32)
    time_window_ms = 3000
    last_ms = 1500
    window_sz = int(sample_rate * time_window_ms / 1000)
    vad_thold = 0.3
    freq_thold = 100.0
    chunk_ms = 30
    chunk_sz = int(sample_rate * chunk_ms / 1000)

    if total_samples <= window_sz:
        ratio = _vad_simple(
            pcmf32,
            sample_rate=sample_rate, 
            last_ms=last_ms, 
            freq_thold=freq_thold, 
        )
        ratio_arr = np.linspace(1, ratio, len(wave_time))
        
    else:
        ratio_bounds = []
        pcmf32_len = len(pcmf32)
        start_idx = 0
        stop_idx = 0
        last_ratio = 1
        for i in range(0, total_samples - window_sz, chunk_sz):
            stop_idx = i + window_sz
            ratio = _vad_simple(
                pcmf32[i : i + window_sz],
                sample_rate=sample_rate, 
                last_ms=last_ms, 
                freq_thold=freq_thold, 
            )
            ratio_bounds.append((start_idx, stop_idx, last_ratio, ratio))
            last_ratio = ratio
            start_idx = stop_idx
        
        ratio_chunks = []
        for x in ratio_bounds:
            ratio_chunks.append(np.linspace(x[2], x[3], x[1] - x[0]))
        
        if stop_idx < pcmf32_len:
            ratio = _vad_simple(
                pcmf32[-window_sz:],
                sample_rate=sample_rate, 
                last_ms=last_ms, 
                freq_thold=freq_thold, 
            )
            ratio_chunks.append(np.linspace(last_ratio, ratio, pcmf32_len - stop_idx))
        
        # print(len(ratio_chunks))
        ratio_arr = np.concatenate(ratio_chunks)
        # print(len(ratio_arr))
        # print(len(wave_time))
    
    vad_thold_arr = [vad_thold] * len(wave_time)

    fig = plt.figure(figsize=(10, 8), dpi=100)
    axes1 = fig.add_subplot(311)
    axes1.plot(wave_time, wave_data_org, c="g")
    axes1.set_ylabel('wave samples')

    axes2 = fig.add_subplot(312)
    axes2.plot(wave_time, wave_data_normal, c="b")
    axes2.set_ylabel('normal wave samples')

    axes3 = fig.add_subplot(313)
    axes3.plot(wave_time, ratio_arr, c="g")
    axes3.plot(wave_time, vad_thold_arr, c="b")
    axes3.set_ylabel('wave energy varification')
    axes3.set_xlabel(f'time(seconds), vad_thold({vad_thold}), time_window_ms({time_window_ms})')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Node application commands.")

    parser.add_argument(
        "--create-tables",
        action="store_true",
        help=("create empty tables if not existed")
    )
    parser.add_argument(
        "--clear-tables",
        action="store_true",
        help=("create empty tables, or drop the database tables before recreate vector stores")
    )
    parser.add_argument(
        "--test-faiss-cache",
        action="store_true",
        help=("test faiss cache by adding docs, searching docs and deleting docs in multiple threads.")
    )
    parser.add_argument(
        "--test-doc-loader",
        action="store_true",
        help=("test document loaders: CSV, DOCX, PPT, IMAGE, PDF.")
    )
    parser.add_argument(
        "--test-llm",
        action="store_true",
        help=("test llm.")
    )
    parser.add_argument(
        "--test-whisper-transcribe",
        action="store_true",
        help=("test whisper transcribe.")
    )
    parser.add_argument(
        "--test-whisper-cmd",
        action="store_true",
        help=("test whisper command.")
    )
    parser.add_argument(
        "--test-whisper-wav-cmd",
        action="store_true",
        help=("test whisper wav command.")
    )
    parser.add_argument(
        "--load-node-info",
        action="store_true",
        help=("load node info.")
    )
    parser.add_argument(
        "--test-solana",
        action="store_true",
        help=("test solana.")
    )
    parser.add_argument(
        "--test-upload-file",
        action="store_true",
        help=("test upload file.")
    )
    parser.add_argument(
        "--test-tts",
        action="store_true",
        help=("test tts.")
    )
    parser.add_argument(
        "--test-edge",
        action="store_true",
        help=("test edge.")
    )
    parser.add_argument(
        "--test-vad",
        action="store_true",
        help=("test vad.")
    )

    args = parser.parse_args()

    if args.create_tables:
        create_tables()
    elif args.clear_tables:
        reset_tables()
    elif args.test_faiss_cache:
        test_faiss_cache()
    elif args.test_doc_loader:
        test_doc_loaders()
    elif args.test_llm:
        test_llm()
    elif args.test_whisper_transcribe:
        test_whisper_transcribe()
    elif args.test_whisper_cmd:
        test_whisper_cmd()
    elif args.test_whisper_wav_cmd:
        test_whisper_wav_cmd()
    elif args.load_node_info:
        from communication.manager.local_manager import g_local_manager
        g_local_manager.load_node_info()
    elif args.test_solana:
        from communication.manager.solana_util import generate_keypair, restore_keypair
        kp = generate_keypair()
        print(f'kp: {kp}')
        print(f'pubkey: {str(kp.pubkey())}')
        print(f'secret: {kp.secret().hex()}')

        base58 = str(kp)
        print(f'kp base58: {base58}')

        new_kp = restore_keypair(base58)
        print(f'new_kp: {new_kp}')
        print(f'new_kp base58: {str(new_kp)}')
    elif args.test_upload_file:
        test_upload_file()
    elif args.test_tts:
        test_tts()
    elif args.test_edge:
        test_edge()
    elif args.test_vad:
        test_simple_vad()

