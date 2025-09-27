import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray, String
import wave
import os
import time
from datetime import datetime
import threading
import whisper  # pip install openai-whisper
from pydub import AudioSegment, effects
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class WhisperTranscriber(Node):
    def __init__(self):
        super().__init__('whisper_transcriber')

        audio_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.subscription = self.create_subscription(
            ByteMultiArray,
            'audio_raw',
            self.audio_callback,
            audio_qos,
        )

        # NEW: publisher for LlamaProcessor
        self.text_pub = self.create_publisher(String, 'voice_commands', 10)

        # Audio config
        self.channels = 1
        self.rate = 44100
        self.sample_width = 2
        self.chunk_duration = 10  # seconds per file

        # Buffers
        self.audio_buffer = bytearray()
        self.last_flush_time = time.time()

        # Model
        self.model = whisper.load_model("base")  # or "tiny", "small", etc.
        self.get_logger().info("Loaded Whisper model")

        os.makedirs("transcripts", exist_ok=True)

    def audio_callback(self, msg: ByteMultiArray):
        # FIX: ByteMultiArray.data is a sequence of ints -> convert with bytes()
        audio_bytes = b''.join(msg.data)   
        self.audio_buffer.extend(audio_bytes)

        if time.time() - self.last_flush_time >= self.chunk_duration:
            self.flush_and_transcribe()

    def flush_and_transcribe(self):
        if not self.audio_buffer:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wav_path = f"transcripts/audio_{timestamp}.wav"
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.rate)
            wf.writeframes(self.audio_buffer)

        self.get_logger().info(f"Saved {wav_path} - running Whisper...")
        threading.Thread(target=self.run_whisper, args=(wav_path,), daemon=True).start()

        self.audio_buffer.clear()
        self.last_flush_time = time.time()

    def denoise(self, wav_path):
        audio = AudioSegment.from_wav(wav_path)
        filtered = audio.high_pass_filter(300).low_pass_filter(3400)
        amplified = filtered + 10
        normalized = effects.normalize(amplified)
        normalized.export(wav_path, format='wav')

    def run_whisper(self, wav_path):
        self.denoise(wav_path)
        try:
            result = self.model.transcribe(wav_path)
            text = result.get("text", "").strip()
            if text:
                print(f"\n📝 Transcription ({wav_path}):\n{text}\n")
                # NEW: publish so LlamaProcessor can react
                self.text_pub.publish(String(data=text))
            else:
                print(f"\n(No speech detected in {wav_path})\n")
        except Exception as e:
            print(f"❌ Whisper failed on {wav_path}: {e}")

    def destroy_node(self):
        self.flush_and_transcribe()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WhisperTranscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
