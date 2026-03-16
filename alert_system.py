import cv2
import collections
import time
import os
import threading
import smtplib
from email.message import EmailMessage
from datetime import datetime

class AlertSystem:
    def __init__(self, sender_email, app_password, recipient_emails, fps=30, buffer_seconds=5, cooldown_seconds=60):
        self.sender_email = sender_email
        self.app_password = app_password
        
        # Support string (comma separated) or list
        if isinstance(recipient_emails, str):
            self.recipient_emails = [e.strip() for e in recipient_emails.split(',')]
        else:
            self.recipient_emails = recipient_emails
            
        self.fps = fps
        self.buffer_size = fps * buffer_seconds
        
        # Ring buffer to keep the last `buffer_seconds` of frames
        self.frame_buffer = collections.deque(maxlen=self.buffer_size)
        
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0
        self.is_recording_alert = False
        
        # Create alerts directory if it doesn't exist
        self.alerts_dir = "alerts"
        os.makedirs(self.alerts_dir, exist_ok=True)
        
        if self.is_configured():
            print(f"[ALERT SYSTEM] Ready. Sending to {', '.join(self.recipient_emails)}")
        else:
            print("[ALERT SYSTEM] Warning: Creds not fully configured. Email sending disabled.")

    def is_configured(self):
        return bool(self.sender_email and self.app_password and self.recipient_emails)

    def update_frame(self, frame):
        """Called every frame from the main loop to keep the buffer updated."""
        # Only buffer if we are reasonably confident the system might need it soon
        # Actually, best to just keep buffering to always have the lead-up to the event.
        if not self.is_recording_alert:
            self.frame_buffer.append(frame.copy())

    def trigger_alert(self, status, person_count, audio_info, transcript):
        """Called when a fight is confirmed. Saves clip and sends email."""
        current_time = time.time()
        
        if current_time - self.last_alert_time < self.cooldown_seconds:
            # Still in cooldown, ignore
            return
            
        if self.is_recording_alert:
            return
            
        print("\n==================================")
        print("🚨 FIGHT DETECTED! TRIGGERING ALERT 🚨")
        print("==================================\n")
        
        self.last_alert_time = current_time
        self.is_recording_alert = True
        
        # Start the recording and email process in a background thread
        # so we don't block the main CCTV camera feed.
        alert_thread = threading.Thread(
            target=self._process_alert_async,
            args=(list(self.frame_buffer), status, person_count, audio_info, transcript),
            daemon=True
        )
        alert_thread.start()

    def _process_alert_async(self, frames_to_save, status, person_count, audio_info, transcript):
        try:
            if not frames_to_save:
                print("[ALERT SYSTEM] No frames in buffer. Cannot record clip.")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.alerts_dir, f"fight_alert_{timestamp}.mp4")
            
            # 1. Save Video Clip
            print(f"[ALERT SYSTEM] Saving 5-second incident clip: {filename}...")
            height, width, _ = frames_to_save[0].shape
            
            # Using mp4v codec for standard mp4 compatibility
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
            
            for frame in frames_to_save:
                out.write(frame)
            out.release()
            print("[ALERT SYSTEM] Clip saved successfully.")
            
            # 2. Send Email
            if self.is_configured():
                self._send_email(filename, timestamp, status, person_count, audio_info, transcript)
            else:
                print("[ALERT SYSTEM] Skipping email (credentials not configured).")

        except Exception as e:
            print(f"[ALERT SYSTEM ERROR] Failed to process alert: {e}")
        finally:
            self.is_recording_alert = False

    def _send_email(self, video_path, timestamp, status, person_count, audio_info, transcript):
        recipients_str = ', '.join(self.recipient_emails)
        print(f"[ALERT SYSTEM] Sending email alert to {recipients_str}...")
        try:
            msg = EmailMessage()
            msg['Subject'] = f"🚨 URGENT: {status} - Camera 1"
            msg['From'] = self.sender_email
            msg['To'] = recipients_str
            
            # Format the email body
            body = f"""
🚨 SECURITY INCIDENT ALERT 🚨

A fight event has been detected by the CCTV AI system.

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Location: Camera 1 (Main Hall)
Status: {status}
Individuals Detected: {person_count}

--- AUDIO INTELLIGENCE ---
Indicators: {audio_info}
Last Transcript: "{transcript}"

Please find the 5-second incident clip attached.
"""
            msg.set_content(body)
            
            # Attach the video file
            with open(video_path, 'rb') as f:
                video_data = f.read()
                msg.add_attachment(video_data, maintype='video', subtype='mp4', filename=os.path.basename(video_path))
                
            # Connect to Gmail SMTP server
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(self.sender_email, self.app_password)
                smtp.send_message(msg)
                
            print("[ALERT SYSTEM] ✅ Email alert sent successfully!")
            
        except smtplib.SMTPAuthenticationError:
            print("[ALERT SYSTEM ERROR] ❌ SMTP Authentication Failed. Check your Gmail App Password.")
        except Exception as e:
            print(f"[ALERT SYSTEM ERROR] ❌ Failed to send email: {e}")
