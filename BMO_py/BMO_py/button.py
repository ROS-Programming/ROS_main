import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
import serial
import pynput

class HelloworldPublisher(Node):

    def __init__(self):
        super().__init__('button')
        qos_profile = QoSProfile(depth=10)
        self.publisher = self.create_publisher(String, 'button', qos_profile)
        self.timer = self.create_timer(1, self.publish_msg)
        self.ser = serial.Serial('/dev/ttyACM3', 250000)
        self.ser.reset_input_buffer()
        self.keyboard_button = pynput.keyboard.Controller()

    def key_input(self, data):
        self.keyboard_button.press(data)
        self.keyboard_button.release(data)

    def publish_msg(self):
        if self.ser.readable():
            data = self.ser.readline().decode()
            data = str(data[:len(data)-1]).strip()
            if data == '1':
                self.key_input("w")
                self.get_logger().info("w")
            elif data == '2':
                self.key_input("a")
                self.get_logger().info("a")
            elif data == '3':
                self.key_input("s")
                self.get_logger().info("s")
            elif data == '4':
                self.key_input("d")
                self.get_logger().info("d")
            if data == '5':
                msg = String()
                msg.data = data
                self.publisher.publish(msg)
                self.get_logger().info("change")

def main(args=None):
    rclpy.init(args=args)
    node = HelloworldPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()