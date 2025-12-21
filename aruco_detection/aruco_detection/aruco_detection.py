import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import math

from aruco_interfaces.msg import ArucoDetection

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

        self.detect_pub = self.create_publisher(
            ArucoDetection,
            '/aruco_detections',
            20
        )


        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.finished = False
        self.min_marker_area = 3500.0

        self.get_logger().info("Aruco detector node started")

    def image_callback(self, msg):
        if self.finished:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, marker_id, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if marker_id is None:
            return

        marker_id = int(marker_id[0][0])
        marker_corners = corners[0][0]

        area = cv2.contourArea(marker_corners)

        det_msg = ArucoDetection()
        det_msg.id = marker_id
        det_msg.area = area
        det_msg.stamp = self.get_clock().now().to_msg()
        self.detect_pub.publish(det_msg)

        if area < self.min_marker_area:
            return
        
        sqrt_id = math.sqrt(int(marker_id))
        msg_out = Float32()
        msg_out.data = sqrt_id
        self.pub.publish(msg_out)

        self.get_logger().info(f"Detected Mission Aruco ID {int(marker_id)}, sqrt={sqrt_id:.5f}")
        
        self.finished = True
        self.destroy_subscription(self.subscription)
        self.create_timer(0.1, self.shutdown)

    def shutdown(self):
        self.get_logger().info("Aruco detector finished, shutting down")
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
