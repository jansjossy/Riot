"""
Phase 2: Audio Intelligence Module (English Mode)
English Whisper + Aggression Detection + Music Detection
"""

import subprocess
import numpy as np
import threading
import time
import queue
import urllib.parse
import torch
import warnings
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

warnings.filterwarnings('ignore')
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import whisper
    from transformers import pipeline
    from thefuzz import fuzz
    import imageio_ffmpeg
    import sounddevice as sd
except ImportError as e:
    print(f"[ERROR] Missing library: {e}")
    sys.exit(1)

# ============================================================================
# ENGLISH PROFANITY LIST (add/remove words as needed)
# ============================================================================
PROFANITY_WORDS = [
    # Common English abuse
    "fuck", "fucked", "fucking", "fucker", "fuckoff", "fuckup",
    "shit", "shitty", "bullshit",
    "bitch", "bastard", "asshole", "ass",
    "damn", "crap",
    "idiot", "moron", "stupid", "dumbass",
    "kill", "murder", "die", "dead",
    "fight", "attack", "beat", "punch", "hit",
    # Common abusive phrases (checked as substrings)
    "shut up", "go to hell", "get out",
    "son of a bitch", "son of a",
]

# AST labels that indicate aggression/violence (language-independent)
AGGRESSION_LABELS = [
    'shout', 'scream', 'yell', 'screaming', 'shouting',
    'battle cry', 'grunt', 'slap', 'smack', 'whack',
    'punch', 'bang', 'crash', 'glass breaking', 'explosion',
]

class AudioIntelligence:
    def __init__(self, source="mic", rtsp_url="", chunk_duration=1.5, sample_rate=16000):
        self.source = source
        self.rtsp_url = rtsp_url
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)

        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"

        # Standard Whisper base - perfect for English, loads in seconds
        print("[INFO] Loading Whisper (English - tiny, fast)...")
        self.whisper_model = whisper.load_model("tiny", device=self.device_str)
        print("[INFO] Whisper loaded!")

        print("[INFO] Loading Audio Classifier (AST)...")
        self.audio_clf = pipeline(
            "audio-classification",
            model="MIT/ast-finetuned-audioset-10-10-0.4593",
            device=0 if self.device_str == "cuda" else -1
        )
        print("[INFO] AST loaded!")

        self.audio_queue = queue.Queue(maxsize=10)
        self.running = False

        # Public state for Phase 3 fusion
        self.profanity_detected = False
        self.music_detected = False
        self.aggression_detected = False
        self.last_transcript = ""

    # ---- SOURCE 1: PC Microphone ----
    def mic_audio_reader(self):
        print(f"[INFO] Mic active ({self.chunk_duration}s chunks @ {self.sample_rate}Hz)")
        self.running = True
        self.mic_buffer = np.array([], dtype=np.float32)

        def callback(indata, frames, time_info, status):
            self.mic_buffer = np.append(self.mic_buffer, indata.flatten())
            if len(self.mic_buffer) >= self.chunk_samples:
                chunk = self.mic_buffer[:self.chunk_samples]
                self.mic_buffer = self.mic_buffer[self.chunk_samples:]
                if self.audio_queue.full():
                    self.audio_queue.get()
                self.audio_queue.put(chunk)

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', callback=callback):
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"[ERROR] Mic: {e}")

    # ---- SOURCE 2: RTSP Camera Audio ----
    def ffmpeg_audio_reader(self):
        print("[INFO] Connecting to RTSP audio stream...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg_exe, '-i', self.rtsp_url,
               '-f', 's16le', '-acodec', 'pcm_s16le',
               '-ar', str(self.sample_rate), '-ac', '1',
               '-vn', '-sn', '-dn', '-']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
        self.running = True
        chunk_bytes = self.chunk_samples * 2
        while self.running:
            raw = proc.stdout.read(chunk_bytes)
            if not raw:
                time.sleep(0.1)
                continue
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if self.audio_queue.full():
                self.audio_queue.get()
            self.audio_queue.put(audio)
        proc.terminate()

    # ---- Whisper in a thread ----
    def _run_whisper(self, audio_chunk, results):
        try:
            result = self.whisper_model.transcribe(
                audio_chunk,
                language="en",
                fp16=torch.cuda.is_available(),
                task="transcribe",
                no_speech_threshold=0.6,   # Skip low-confidence silence
                condition_on_previous_text=False
            )
            results['text'] = result['text'].lower().strip()
        except Exception as e:
            results['text'] = ""

    # ---- AST in a thread ----
    def _run_ast(self, audio_chunk, results):
        try:
            results['ast'] = self.audio_clf(audio_chunk, top_k=5, sampling_rate=self.sample_rate)
        except Exception:
            results['ast'] = []

    # ---- Main AI Processing (Parallel) ----
    def process_audio(self):
        while self.running:
            if self.audio_queue.empty():
                time.sleep(0.05)
                continue

            audio_chunk = self.audio_queue.get()
            
            # ---- ENERGY-BASED VAD ----
            # Calculate RMS energy of the chunk. If too quiet, skip Whisper entirely.
            # This prevents hallucinated transcripts from background noise AND saves CPU.
            rms_energy = np.sqrt(np.mean(audio_chunk ** 2))
            
            if rms_energy < 0.002:  # Very quiet — skip everything
                self.profanity_detected = False
                self.aggression_detected = False
                self.last_transcript = ""
                continue
            
            results = {'text': '', 'ast': []}
            
            if rms_energy < 0.008:
                # Low energy — probably background noise, only run AST (cheap)
                self._run_ast(audio_chunk, results)
                text = ""
            else:
                # Sufficient energy — run both Whisper + AST in parallel
                wt = threading.Thread(target=self._run_whisper, args=(audio_chunk, results))
                at = threading.Thread(target=self._run_ast, args=(audio_chunk, results))
                wt.start(); at.start()
                wt.join(); at.join()
                text = results['text']

            ast_results = results['ast']
            self.last_transcript = text

            # Skip profanity check if no meaningful text
            if len(text) <= 2:
                self.profanity_detected = False
            else:
                # ---- PROFANITY CHECK ----
                self.profanity_detected = False
                matched_word = ""

                # A: Exact substring
                for pw in PROFANITY_WORDS:
                    if pw in text:
                        self.profanity_detected = True
                        matched_word = pw
                        break

                # B: Fuzzy match on individual words (4+ chars)
                if not self.profanity_detected:
                    for spoken_word in text.split():
                        if len(spoken_word) < 4:
                            continue
                        for pw in PROFANITY_WORDS:
                            if len(pw) < 4:
                                continue
                            if fuzz.ratio(spoken_word, pw) > 80:
                                self.profanity_detected = True
                                matched_word = f"{spoken_word} ~ {pw}"
                                break
                        if self.profanity_detected:
                            break

            # ---- MUSIC + AGGRESSION CHECK ----
            self.music_detected = False
            self.aggression_detected = False
            music_label = ""
            aggression_label = ""

            music_keywords = ['music', 'song', 'singing', 'guitar', 'piano', 'drum',
                              'bass', 'hip hop', 'pop', 'rock', 'jazz', 'electronic',
                              'dance', 'beat', 'rap', 'instrument', 'orchestra', 'choir',
                              'male singing', 'female singing', 'vocal', 'soundtrack',
                              'mantra', 'lullaby', 'jingle', 'disco', 'reggae']

            for pred in ast_results:
                label_lower = pred['label'].lower()
                # Music: lowered to 3% because speech dominates AST when someone is talking over music
                if any(kw in label_lower for kw in music_keywords) and pred['score'] > 0.03:
                    self.music_detected = True
                    music_label = f"{pred['label']} ({pred['score']*100:.0f}%)"
                if any(kw in label_lower for kw in AGGRESSION_LABELS) and pred['score'] > 0.10:
                    self.aggression_detected = True
                    aggression_label = f"{pred['label']} ({pred['score']*100:.0f}%)"

            # ---- OUTPUT ----
            top_preds = [f"{p['label']}({p['score']*100:.0f}%)" for p in ast_results[:3]] if ast_results else []

            print("\n" + "="*55)
            print(f"  [Transcript] : {text}")
            if top_preds:
                print(f"  [AST]        : {', '.join(top_preds)}")
            if self.profanity_detected:
                print(f"  [!! PROFANITY]  : DETECTED! Word: '{matched_word}'")
            if self.aggression_detected:
                print(f"  [!! AGGRESSION] : {aggression_label}")
            if self.music_detected:
                print(f"  [~~ MUSIC]      : {music_label}")
            print("="*55, flush=True)

    def start(self):
        """Non-blocking start - for embedding into another script (Phase 3 fusion)"""
        self.running = True
        if self.source == "mic":
            reader = threading.Thread(target=self.mic_audio_reader, daemon=True)
        else:
            reader = threading.Thread(target=self.ffmpeg_audio_reader, daemon=True)
        processor = threading.Thread(target=self.process_audio, daemon=True)
        reader.start()
        processor.start()
        src = "PC Microphone" if self.source == "mic" else "RTSP Camera"
        print(f"[AUDIO] Started | Source: {src}")
    
    def stop(self):
        """Clean shutdown"""
        self.running = False

    def run(self):
        """Blocking run - for standalone testing"""
        self.start()
        print("[INFO] Speak English into the mic. Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
            print("\n[INFO] Stopped.")

def get_rtsp_url(ip, username, password, path="/cam/realmonitor?channel=1&subtype=0"):
    return f"rtsp://{username}:{urllib.parse.quote(password, safe='')}@{ip}:554{path}"

if __name__ == "__main__":
    ip = "192.168.20.129"
    user = "admin"
    pwd = "admin@123"

    print("="*55)
    print("  AUDIO INTELLIGENCE MODULE - Phase 2")
    print("  English Whisper + Aggression + Music Detection")
    print("="*55)
    print("\nAudio Source:")
    print("1: PC Microphone (for testing)")
    print("2: RTSP Camera Audio")

    choice = input("\nEnter (1 or 2): ").strip()

    if choice == '2':
        rtsp = get_rtsp_url(ip, user, pwd)
        detector = AudioIntelligence(source="rtsp", rtsp_url=rtsp, chunk_duration=1.5)
    else:
        detector = AudioIntelligence(source="mic", chunk_duration=1.5)

    detector.run()
