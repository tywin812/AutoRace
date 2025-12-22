import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from traffic_light_interfaces.msg import TrafficLightState


class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/color/image',
            self.image_callback,
            10
        )

        self.pub = self.create_publisher(
            TrafficLightState,
            '/traffic_light_state',
            10
        )

        self.roi_x = 34
        self.roi_y = 1
        self.roi_w = 194
        self.roi_h = 291

        self.get_logger().info("Traffic light estimator started")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        roi = frame[
            self.roi_y:self.roi_y + self.roi_h,
            self.roi_x:self.roi_x + self.roi_w
        ]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        red = red1 + red2

        yellow = cv2.inRange(hsv, (20, 70, 50), (35, 255, 255))
        green = cv2.inRange(hsv, (40, 70, 50), (85, 255, 255))

        red_count = cv2.countNonZero(red)
        yellow_count = cv2.countNonZero(yellow)
        green_count = cv2.countNonZero(green)

        if red_count > yellow_count and red_count > green_count:
            state = TrafficLightState.RED
        elif yellow_count > green_count:
            state = TrafficLightState.YELLOW
        else:
            state = TrafficLightState.GREEN
            self.get_logger().info(f"Traffic light is green.")

        msg_out = TrafficLightState()
        msg_out.state = state
        msg_out.stamp = self.get_clock().now().to_msg()

        self.pub.publish(msg_out)

        if state == TrafficLightState.GREEN:
            self.get_logger().info("Traffic light GREEN detected")
            self.destroy_subscription(self.subscription)
            self.create_timer(0.5, self.shutdown)

    def shutdown(self):
        self.get_logger().info("Traffic light estimator finished")
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
    rclpy.spin(node)


if __name__ == '__main__':
    main()