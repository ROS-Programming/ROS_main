import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
import serial


class HelloworldPublisher(Node):

    def __init__(self):
        super().__init__('helloworld_publisher')
        qos_profile = QoSProfile(depth=10)
        self.helloworld_publisher = self.create_publisher(String, 'helloworld', qos_profile)
        self.timer = self.create_timer(1, self.publish_helloworld_msg)
        self.ser = serial.Serial('/dev/ttyACM2', 9600, timeout=1)
        self.ser.reset_input_buffer()
        self.input_data_format = ['0', '1', '2', '3', '4', '5']

    def publish_helloworld_msg(self):
        msg = String()
        msg.data = self.ser.readline().decode('utf-8').rstrip()
        if msg.data in self.input_data_format:
            self.helloworld_publisher.publish(msg)
            self.get_logger().info('Published message: {0}'.format(msg.data))

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