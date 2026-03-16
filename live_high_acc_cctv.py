"""
Phase 3: Multi-Modal CCTV Fight Detection
Video: R(2+1)D-18 + YOLOv8 Person Tracking
Audio: Whisper Profanity + AST Music/Aggression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
import cv2
import numpy as np
import collections
import time
import threading
import urllib.parse
import sys
try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# Import audio intelligence module
try:
    from audio_intelligence import AudioIntelligence
    AUDIO_AVAILABLE = True
except ImportError:
    print("[WARN] audio_intelligence.py not found. Running video-only mode.")
    AUDIO_AVAILABLE = False
    
try:
    from alert_system import AlertSystem
    ALERT_AVAILABLE = True
except ImportError:
    print("[WARN] alert_system.py not found. Email alerts disabled.")
    ALERT_AVAILABLE = False

# ===========================================================================
#  LAG-FREE VIDEO CAPTURE (CP PLUS OPTIMIZED)
# ===========================================================================
class VideoCaptureThread:
    """Threaded Video Capture to avoid RTSP buffering latency"""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        
        # Extremely important for IP cameras: limit buffer size to 1 so it doesn't queue old frames
        # This keeps the stream instantly live
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                # For IP cameras, using grab() continuously and retrieve() only when needed 
                # discards old frames when the AI takes longer than 1 frame to infer
                ret = self.cap.grab()
                if ret:
                    ret, frame = self.cap.retrieve()
                    with self.lock:
                        self.ret, self.frame = ret, frame
            time.sleep(0.005) # Yield thread slightly

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            # Return a copy to avoid threading conflicts
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

# ===========================================================================
#  R(2+1)D-18 HIGH ACCURACY NETWORK (EXACT COPY FROM TRAIN SCRIPT)
# ===========================================================================
class HighAccFightDetector(nn.Module):
    def __init__(self, model_name='r2plus1d_18', num_classes=2, dropout_rate=0.4):
        super().__init__()
        # Load the base without pre-trained weights since we overwrite them all anyway
        self.backbone = video_models.r2plus1d_18(weights=None)

        in_features = self.backbone.fc.in_features
        # Exact classification head from train_high_accuracy.py
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.75),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ===========================================================================
#  LIVE DETECTION CONTROLLER
# ===========================================================================
class LiveHighAccCCTV:
    def __init__(self, model_path="active_models/high_acc_fight_model_best.pth", 
                 camera_url="", num_frames=32, frame_size=(112, 112),
                 audio_mode="none", audio_rtsp_url="",
                 alert_email="", app_pwd="", dest_email=""):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Multi-Modal CCTV System | Device: {self.device}")
        
        # Load YOLOv8n
        print("Loading YOLOv8 nano for person tracking...")
        self.yolo = YOLO("active_models/yolov8n.pt")
        
        self.camera_url = camera_url
        self.frame_size = frame_size
        self.num_frames = num_frames
        
        # Load fight model
        self.model = HighAccFightDetector().to(self.device)
        print(f"Loading fight model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception:
            checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        try:
            example_input = torch.randn(1, 3, num_frames, 112, 112).to(self.device)
            self.model = torch.jit.trace(self.model, example_input)
            print("Model traced (optimized for speed).")
        except Exception:
            pass
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            
        self.MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1).to(self.device)
        self.STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1).to(self.device)
        
        # Status variables
        self.current_status = "NORMAL"
        self.status_color = (0, 255, 0)
        self.warning_time_accumulated = 0.0
        self.status_since = time.time()
        
        # ---- AUDIO MODULE (Phase 3) ----
        self.audio = None
        self.audio_mode = audio_mode
        if audio_mode != "none" and AUDIO_AVAILABLE:
            print(f"Loading Audio Intelligence (mode: {audio_mode})...")
            if audio_mode == "mic":
                self.audio = AudioIntelligence(source="mic", chunk_duration=1.5)
            elif audio_mode == "rtsp":
                self.audio = AudioIntelligence(source="rtsp", rtsp_url=audio_rtsp_url, chunk_duration=1.5)
            print("Audio Intelligence ready!")
            
        # ---- ALERT SYSTEM (Phase 4) ----
        self.alert_system = None
        if alert_email and app_pwd and dest_email and ALERT_AVAILABLE:
            # Buffer size: roughly 30 FPS * 5 seconds = 150 frames
            self.alert_system = AlertSystem(alert_email, app_pwd, dest_email, fps=30, buffer_seconds=5)
        
    def preprocess_frame(self, frame):
        """Downscale to 112x112 and normalize to 0-1 range"""
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.astype(np.float32) / 255.0

    def update_alarm_status(self, fight_prob, person_count):
        current_time = time.time()
        time_delta = current_time - self.status_since
        self.status_since = current_time
        
        # Read audio signals (thread-safe: these are simple booleans)
        music = self.audio.music_detected if self.audio else False
        profanity = self.audio.profanity_detected if self.audio else False
        aggression = self.audio.aggression_detected if self.audio else False
        
        # ============================================
        # MULTI-MODAL FUSION RULES
        # ============================================
        
        # RULE 1: Music playing -> SUPPRESS fight detection (likely dancing)
        if music and fight_prob >= 0.75 and person_count >= 2:
            # Music is playing, so this is likely dancing not fighting
            # Slow down the timer instead of accumulating
            self.warning_time_accumulated -= (time_delta * 0.5)
            if self.warning_time_accumulated <= 0:
                self.warning_time_accumulated = 0.0
                self.current_status = "NORMAL (Music)"
                self.status_color = (255, 200, 0)  # Cyan-ish - suppressed
            else:
                self.current_status = f"SUPPRESSED ({self.warning_time_accumulated:.1f}s)"
                self.status_color = (255, 200, 0)
            return
        
        # RULE 2: Fight video + profanity/aggression -> ACCELERATE timer (2x)
        if fight_prob >= 0.75 and person_count >= 2:
            speed = 1.0
            if profanity or aggression:
                speed = 2.0  # Audio confirms violence -> escalate faster
            
            self.warning_time_accumulated += (time_delta * speed)
            
            if self.warning_time_accumulated > 1.5:
                # TRIGGERS ALERT!
                if profanity or aggression:
                    self.current_status = "FIGHT + AUDIO CONFIRMED"
                else:
                    self.current_status = "FIGHT DETECTED"
                self.status_color = (0, 0, 255)  # Red
                self.warning_time_accumulated = min(self.warning_time_accumulated, 4.0)
                
                # Fire the email alert system
                if self.alert_system:
                    audio_info = f"Music={music}, Profanity={profanity}, Aggression={aggression}"
                    transcript = self.audio.last_transcript if self.audio else "N/A"
                    self.alert_system.trigger_alert(self.current_status, person_count, audio_info, transcript)
                    
            else:
                self.current_status = f"WARNING ({self.warning_time_accumulated:.1f}s)"
                self.status_color = (0, 165, 255)  # Orange
        
        # RULE 3: No video fight but profanity + aggression -> verbal alert
        elif (profanity or aggression) and person_count >= 2:
            self.current_status = "VERBAL ALERT"
            self.status_color = (0, 200, 255)  # Yellow-orange
            self.warning_time_accumulated = max(self.warning_time_accumulated, 0.3)
        
        # RULE 4: Normal - timer decays
        else:
            self.warning_time_accumulated -= (time_delta * 1.5)
            if self.warning_time_accumulated <= 0:
                self.warning_time_accumulated = 0.0
                self.current_status = "NORMAL"
                self.status_color = (0, 255, 0)  # Green
            elif self.warning_time_accumulated > 1.5:
                # TRIGGERS ALERT!
                self.current_status = "FIGHT DETECTED"
                self.status_color = (0, 0, 255)
                
                # Fire the email alert system
                if self.alert_system:
                    audio_info = f"Music={music}, Profanity={profanity}, Aggression={aggression}"
                    transcript = self.audio.last_transcript if self.audio else "N/A"
                    self.alert_system.trigger_alert(self.current_status, person_count, audio_info, transcript)
                    
            else:
                self.current_status = f"WARNING ({self.warning_time_accumulated:.1f}s)"
                self.status_color = (0, 165, 255)

    def run(self):
        url_display = "PC Webcam" if self.camera_url == 0 else str(self.camera_url).replace('admin%40123', '***')
        print(f"Connecting to stream: {url_display}")
        stream = VideoCaptureThread(self.camera_url)
        
        time.sleep(2)
        if not stream.cap.isOpened():
            print("Failed to open video stream. Check connection.")
            return

        # Start audio module if available
        if self.audio:
            self.audio.start()

        print("Stream connected! Monitoring for fights...")
        print("Press 'q' to quit")
        
        buffer = collections.deque(maxlen=self.num_frames)
        cv2.namedWindow('CCTV Multi-Modal Fight Detection', cv2.WINDOW_NORMAL)
        
        # Shared state between display and inference threads
        self.last_prob = 0.0
        self.last_person_count = 0
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._inference_running = True
        self._yolo_boxes = []
        
        # Background inference thread — keeps display smooth
        def inference_loop():
            last_predict_time = time.time()
            local_buffer = collections.deque(maxlen=self.num_frames)
            yolo_interval = 0  # Counter for running YOLO less often
            
            while self._inference_running:
                # Grab the latest frame
                with self._frame_lock:
                    frame = self._latest_frame
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Preprocess for fight model buffer
                processed = self.preprocess_frame(frame)
                local_buffer.append(processed)
                
                # Run YOLO every ~0.3s (not every frame)
                yolo_interval += 1
                if yolo_interval >= 3:
                    yolo_interval = 0
                    try:
                        results = self.yolo(frame, classes=[0], conf=0.5, verbose=False)
                        boxes = []
                        count = 0
                        if len(results) > 0:
                            count = len(results[0].boxes)
                            for box in results[0].boxes:
                                boxes.append(list(map(int, box.xyxy[0])))
                        self.last_person_count = count
                        self._yolo_boxes = boxes
                    except Exception:
                        pass
                
                # Fight model inference
                if len(local_buffer) == self.num_frames:
                    current_time = time.time()
                    if current_time - last_predict_time >= 0.3:
                        last_predict_time = current_time
                        try:
                            input_tensor = np.stack(list(local_buffer))
                            input_tensor = torch.FloatTensor(input_tensor).permute(3, 0, 1, 2)
                            input_tensor = input_tensor.to(self.device)
                            input_tensor = (input_tensor - self.MEAN) / self.STD
                            input_tensor = input_tensor.unsqueeze(0)
                            
                            with torch.no_grad():
                                output = self.model(input_tensor)
                                probabilities = F.softmax(output, dim=1)[0]
                                fight_prob = probabilities[1].item()
                            
                            self.update_alarm_status(fight_prob, self.last_person_count)
                            self.last_prob = fight_prob
                        except Exception:
                            pass
                
                time.sleep(0.05)  # Small yield
        
        inf_thread = threading.Thread(target=inference_loop, daemon=True)
        inf_thread.start()
        
        try:
            while True:
                ret, frame = stream.read()
                if not ret or frame is None:
                    continue

                # Share frame with inference thread
                with self._frame_lock:
                    self._latest_frame = frame.copy()
                
                # Keep the alert system buffered with the latest 5 seconds of video
                if self.alert_system:
                    self.alert_system.update_frame(frame)
                
                display_frame = frame.copy()
                h, w = display_frame.shape[:2]
                
                # Draw YOLO boxes from last inference
                for box in self._yolo_boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                person_count = self.last_person_count
                
                # ============ DISPLAY OVERLAY ============
                # Top bar background
                cv2.rectangle(display_frame, (0, 0), (w, 120), (0, 0, 0), -1)
                
                # Row 1: Status + Fight Probability
                cv2.putText(display_frame, self.current_status, (20, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.status_color, 3)
                prob_text = f"Fight: {getattr(self, 'last_prob', 0.0)*100:.1f}%"
                cv2.putText(display_frame, prob_text, (w - 250, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Row 2: Persons + Audio Status
                cv2.putText(display_frame, f"Persons: {person_count}", (20, 75), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Audio indicators on the right side
                if self.audio:
                    audio_x = w - 400
                    if self.audio.music_detected:
                        cv2.putText(display_frame, "MUSIC", (audio_x, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                        audio_x += 90
                    if self.audio.profanity_detected:
                        cv2.putText(display_frame, "PROFANITY", (audio_x, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        audio_x += 130
                    if self.audio.aggression_detected:
                        cv2.putText(display_frame, "AGGRO", (audio_x, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                    
                    # Row 3: Last transcript snippet
                    transcript = self.audio.last_transcript[:60] if self.audio.last_transcript else ""
                    if transcript:
                        cv2.putText(display_frame, f"Audio: {transcript}", (20, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                else:
                    cv2.putText(display_frame, "Audio: OFF", (w - 200, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                
                cv2.imshow('CCTV Multi-Modal Fight Detection', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self._inference_running = False
            if self.audio:
                self.audio.stop()
            stream.release()
            cv2.destroyAllWindows()


def get_rtsp_url(ip, username, password, default_path="/cam/realmonitor?channel=1&subtype=0"):
    encoded_password = urllib.parse.quote(password, safe='')
    return f"rtsp://{username}:{encoded_password}@{ip}:554{default_path}"

if __name__ == "__main__":
    ip = "192.168.20.129"
    user = "admin"
    pwd = "admin@123"
    
    print("============================================================")
    print("  MULTI-MODAL CCTV FIGHT DETECTION (Phase 3)")
    print("  Video: R(2+1)D-18 + YOLOv8")
    print("  Audio: Whisper + AST (Profanity/Music/Aggression)")
    print("============================================================")
    
    # Video stream selection
    print("\nVideo Stream:")
    print("1: Main Stream (High Res - Default)")
    print("2: Sub Stream (Low Res - Better FPS)")
    print("3: PC Webcam (for testing without CCTV)")
    v_choice = input("Enter (1, 2, or 3): ").strip()
    
    if v_choice == '2':
        rtsp_url = get_rtsp_url(ip, user, pwd, "/cam/realmonitor?channel=1&subtype=1")
    elif v_choice == '3':
        rtsp_url = 0  # OpenCV index for default laptop webcam
    else:
        rtsp_url = get_rtsp_url(ip, user, pwd, "/cam/realmonitor?channel=1&subtype=0")
    
    # Audio mode selection
    audio_mode = "none"
    if AUDIO_AVAILABLE:
        print("\nAudio Source:")
        print("1: No Audio (video-only mode)")
        print("2: PC Microphone (for testing)")
        print("3: RTSP Camera Audio")
        a_choice = input("Enter (1, 2, or 3): ").strip()
        if a_choice == '2':
            audio_mode = "mic"
        elif a_choice == '3':
            audio_mode = "rtsp"
            
    # Email alert configuration (Hardcoded for convenience)
    sender_em = "janzjozsy019@gmail.com"
    app_pwd = "aspm uhld wlyk omdd"
    # To add more recipients, just separate them with a comma: "email1@gmail.com, email2@college.edu"
    recipient_em = "jin.km24@gmail.com"
    
    if ALERT_AVAILABLE:
        print(f"\nAlert System: ENABLED (Sending to: {recipient_em})")
    
    # Build RTSP audio URL (same camera, different extraction)
    audio_rtsp = get_rtsp_url(ip, user, pwd, "/cam/realmonitor?channel=1&subtype=0")
        
    system = LiveHighAccCCTV(
        model_path="active_models/high_acc_fight_model_best.pth",
        camera_url=rtsp_url,
        num_frames=32,
        frame_size=(112, 112),
        audio_mode=audio_mode,
        audio_rtsp_url=audio_rtsp,
        alert_email=sender_em,
        app_pwd=app_pwd,
        dest_email=recipient_em
    )
    
    system.run()
