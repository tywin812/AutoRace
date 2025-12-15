import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/color/image',
            self.image_callback,
            10
        )

        self.pub = self.create_publisher(
            Float32,
            '/mission_aruco',
            10
        )

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.published = False

        self.get_logger().info("Aruco detector node started")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, marker_id, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if not self.published and marker_id is not None:
            sqrt_id = math.sqrt(int(marker_id))
            msg_out = Float32()
            msg_out.data = sqrt_id
            self.pub.publish(msg_out)
            self.get_logger().info(f"Detected ArUco ID {int(marker_id)}, sqrt={sqrt_id:.5f}")
            self.published = True


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
