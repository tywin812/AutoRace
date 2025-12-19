#!/usr/bin/env python3
import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, IntegerRange
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageCalibration(Node):

    def __init__(self):
        super().__init__('image_projection')

        descriptor_x = ParameterDescriptor(integer_range=[IntegerRange(from_value=0, to_value=848, step=1)])
        descriptor_y = ParameterDescriptor(integer_range=[IntegerRange(from_value=0, to_value=480, step=1)])

        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera.extrinsic_camera_calibration.top_x', 115, descriptor_x),
                ('camera.extrinsic_camera_calibration.top_y', 240, descriptor_y),
                ('camera.extrinsic_camera_calibration.bottom_x', 410, descriptor_x),
                ('camera.extrinsic_camera_calibration.bottom_y', 480, descriptor_y),
                ('is_calibration_mode', True)
            ]
        )

        self.top_x = self.get_parameter(
            'camera.extrinsic_camera_calibration.top_x').get_parameter_value().integer_value
        self.top_y = self.get_parameter(
            'camera.extrinsic_camera_calibration.top_y').get_parameter_value().integer_value
        self.bottom_x = self.get_parameter(
            'camera.extrinsic_camera_calibration.bottom_x').get_parameter_value().integer_value
        self.bottom_y = self.get_parameter(
            'camera.extrinsic_camera_calibration.bottom_y').get_parameter_value().integer_value

        self.is_calibration_mode = self.get_parameter(
            'is_calibration_mode').get_parameter_value().bool_value

        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.cbGetImageProjectionParam)

        self.sub_image_original = self.create_subscription(
            Image,
            '/color/image',
            self.cbImageProjection,
            1
        )

        self.pub_image_projected = self.create_publisher(Image, '/color/image_projected', 10)

        if self.is_calibration_mode:
            self.pub_image_calib = self.create_publisher(
                Image,
                '/color/image_calib',
                10
            )

        self.cvBridge = CvBridge()

    def cbGetImageProjectionParam(self, parameters):
        for param in parameters:
            if param.name == 'camera.extrinsic_camera_calibration.top_x':
                self.top_x = param.value
            if param.name == 'camera.extrinsic_camera_calibration.top_y':
                self.top_y = param.value
            if param.name == 'camera.extrinsic_camera_calibration.bottom_x':
                self.bottom_x = param.value
            if param.name == 'camera.extrinsic_camera_calibration.bottom_y':
                self.bottom_y = param.value
        self.get_logger().info(f'change: {self.top_x}')
        self.get_logger().info(f'change: {self.top_y}')
        self.get_logger().info(f'change: {self.bottom_x}')
        self.get_logger().info(f'change: {self.bottom_y}')
        return SetParametersResult(successful=True)

    def cbImageProjection(self, msg_img):
        cv_image = self.cvBridge.imgmsg_to_cv2(msg_img, 'bgr8')
        
        height, width, _ = cv_image.shape
        center_x = width // 2 

        pts_src = np.array([
            [center_x - self.top_x, self.top_y],
            [center_x + self.top_x, self.top_y],
            [center_x + self.bottom_x, self.bottom_y],
            [center_x - self.bottom_x, self.bottom_y]
        ], dtype=np.float32)

        pts_dst = np.array([
            [200, 0],   #200 0
            [800, 0],   #800 0
            [800, 600], #800 600
            [200, 600]  #200 600
        ], dtype=np.float32)

        output_width = 1000
        output_height = 600

        if self.is_calibration_mode:
            cv_image_calib = np.copy(cv_image)
            pts_src_int = pts_src.astype(np.int32)
            cv2.polylines(cv_image_calib, [pts_src_int], True, (0, 0, 255), 2)
            for pt in pts_src_int:
                cv2.circle(cv_image_calib, tuple(pt), 5, (0, 255, 0), -1)    
            self.pub_image_calib.publish(self.cvBridge.cv2_to_imgmsg(cv_image_calib, 'bgr8'))

        h, status = cv2.findHomography(pts_src, pts_dst)

        cv_image_homography = cv2.warpPerspective(cv_image, h, (output_width, output_height))

        # H, W = cv_image_homography.shape[:2]

        # mask = np.zeros((H, W), dtype=np.uint8)

        # roi = np.array([[
        #     (200, 0),
        #     (800, 0),
        #     (800, H-1),
        #     (200, H-1),
        # ]], dtype=np.int32)

        # cv2.fillPoly(mask, roi, 255)

        # cv_image_homography = cv2.bitwise_and(cv_image_homography, cv_image_homography, mask=mask)

        self.pub_image_projected.publish(self.cvBridge.cv2_to_imgmsg(cv_image_homography, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = ImageCalibration()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
