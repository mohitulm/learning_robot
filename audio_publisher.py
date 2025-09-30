import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import ByteMultiArray
import pyaudio

class AudioPublisher(Node):
    def __init__(self):
        super().__init__('audio_publisher')

        # Audio stream parameters
        self.chunk = 1024  # Number of frames per buffer
        self.rate = 44100  # Sampling rate
        self.channels = 1  # Mono
        self.format = pyaudio.paInt16


audio_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,   # drop, don’t queue
    history=HistoryPolicy.KEEP_LAST,
    depth=10,                                    # matches your loaned-sample depth
)

        self.publisher_ = self.create_publisher(ByteMultiArray, 'audio_raw', audio_qos)



        # PyAudio stream setup
        self.p = pyaudio.PyAudio()

        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
        except Exception as e:
            self.get_logger().error(f'Could not open audio stream: {e}')
            raise e

        # Timer to read and publish audio chunks
        self.timer = self.create_timer(self.chunk / self.rate, self.publish_audio)

    def publish_audio(self):
        try:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            msg = ByteMultiArray()
            msg.data = tuple(data[i:i+1] for i in range(len(data)))
            self.publisher_.publish(msg)
            self.get_logger().debug('Published audio chunk')
        except Exception as e:
            self.get_logger().warn(f'Audio stream error: {e}')

    def destroy_node(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = AudioPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
