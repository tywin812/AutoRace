#!/usr/bin/env python3
import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import UInt8


class DetectLane(Node):

    def __init__(self):
        super().__init__('centroid_detect_lane')

        parameter_descriptor_hue = ParameterDescriptor(
            description='hue parameter range',
            integer_range=[IntegerRange(
                from_value=0,
                to_value=179,
                step=1)]
        )
        parameter_descriptor_saturation_brightness = ParameterDescriptor(
            description='saturation and brightness range',
            integer_range=[IntegerRange(
                from_value=0,
                to_value=255,
                step=1)]
        )
        self.declare_parameters(
            namespace='',
            parameters=[
                ('detect.lane.white.hue_l', 0,
                    parameter_descriptor_hue),
                ('detect.lane.white.hue_h', 179,
                    parameter_descriptor_hue),
                ('detect.lane.white.saturation_l', 0,
                    parameter_descriptor_saturation_brightness),
                ('detect.lane.white.saturation_h', 70,
                    parameter_descriptor_saturation_brightness),
                ('detect.lane.white.brightness_l', 235,
                    parameter_descriptor_saturation_brightness),
                ('detect.lane.white.brightness_h', 255,
                    parameter_descriptor_saturation_brightness),
                ('detect.lane.yellow.hue_l', 10,
                    parameter_descriptor_hue),
                ('detect.lane.yellow.hue_h', 127,
                    parameter_descriptor_hue),
                ('detect.lane.yellow.saturation_l', 70,
                    parameter_descriptor_saturation_brightness),
                ('detect.lane.yellow.saturation_h', 255,
                    parameter_descriptor_saturation_brightness),
                ('detect.lane.yellow.brightness_l', 95,
                    parameter_descriptor_saturation_brightness),
                ('detect.lane.yellow.brightness_h', 255,
                    parameter_descriptor_saturation_brightness),
                ('is_detection_calibration_mode', False)
            ]
        )

        self.hue_white_l = self.get_parameter(
            'detect.lane.white.hue_l').get_parameter_value().integer_value
        self.hue_white_h = self.get_parameter(
            'detect.lane.white.hue_h').get_parameter_value().integer_value
        self.saturation_white_l = self.get_parameter(
            'detect.lane.white.saturation_l').get_parameter_value().integer_value
        self.saturation_white_h = self.get_parameter(
            'detect.lane.white.saturation_h').get_parameter_value().integer_value
        self.brightness_white_l = self.get_parameter(
            'detect.lane.white.brightness_l').get_parameter_value().integer_value
        self.brightness_white_h = self.get_parameter(
            'detect.lane.white.brightness_h').get_parameter_value().integer_value

        self.hue_yellow_l = self.get_parameter(
            'detect.lane.yellow.hue_l').get_parameter_value().integer_value
        self.hue_yellow_h = self.get_parameter(
            'detect.lane.yellow.hue_h').get_parameter_value().integer_value
        self.saturation_yellow_l = self.get_parameter(
            'detect.lane.yellow.saturation_l').get_parameter_value().integer_value
        self.saturation_yellow_h = self.get_parameter(
            'detect.lane.yellow.saturation_h').get_parameter_value().integer_value
        self.brightness_yellow_l = self.get_parameter(
            'detect.lane.yellow.brightness_l').get_parameter_value().integer_value
        self.brightness_yellow_h = self.get_parameter(
            'detect.lane.yellow.brightness_h').get_parameter_value().integer_value

        self.is_calibration_mode = self.get_parameter(
            'is_detection_calibration_mode').get_parameter_value().bool_value
        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.cbGetDetectLaneParam)

        self.sub_image_original = self.create_subscription(
            Image, '/color/image_projected', self.cbFindLane, 1
        )

        self.pub_image_lane = self.create_publisher(
            Image, '/color/lanes_detected', 1
        )

        if self.is_calibration_mode:
            self.pub_image_white_lane = self.create_publisher(
                Image, '/calib/white_lane', 1
                )
            self.pub_image_yellow_lane = self.create_publisher(
                Image, '/calib/yellow_lane', 1
                )

        self.pub_lane_error = self.create_publisher(Float64, '/lane_error', 1)

        self.pub_lane_state = self.create_publisher(UInt8, '/lane_detection_state', 1)

        self.cvBridge = CvBridge()

        self.window_width = 1000
        self.window_height = 600
        
        self.detection_rows = [
            int(0.3 * self.window_height),  
            int(0.5 * self.window_height),  
            int(0.7 * self.window_height),  
        ]
        
        self.row_weights = [0.2, 0.3, 0.5]  
        
        self.centroid_pixels_threshold = 100
        
        self.road_width = 500
        self.road_width_min = 200
        self.road_width_max = 800
        
        self.width_history = []
        self.max_history = 5

    def cbGetDetectLaneParam(self, parameters):
        for param in parameters:
            if param.name == 'detect.lane.white.hue_l':
                self.hue_white_l = param.value
            elif param.name == 'detect.lane.white.hue_h':
                self.hue_white_h = param.value
            elif param.name == 'detect.lane.white.saturation_l':
                self.saturation_white_l = param.value
            elif param.name == 'detect.lane.white.saturation_h':
                self.saturation_white_h = param.value
            elif param.name == 'detect.lane.white.brightness_l':
                self.brightness_white_l = param.value
            elif param.name == 'detect.lane.white.brightness_h':
                self.brightness_white_h = param.value
            elif param.name == 'detect.lane.yellow.hue_l':
                self.hue_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.hue_h':
                self.hue_yellow_h = param.value
            elif param.name == 'detect.lane.yellow.saturation_l':
                self.saturation_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.saturation_h':
                self.saturation_yellow_h = param.value
            elif param.name == 'detect.lane.yellow.brightness_l':
                self.brightness_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.brightness_h':
                self.brightness_yellow_h = param.value
            return SetParametersResult(successful=True)
        
    def cbFindLane(self, image_msg):
        bgr_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')
        h, w, _ = bgr_image.shape
        
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        white_mask = self.maskWhiteLane(hsv_image)
        yellow_mask = self.maskYellowLane(hsv_image)
        
        target_x_list = []
        weights_used = []
        
        for i, row_y in enumerate(self.detection_rows):
            row_margin = 10
            row_start = max(0, row_y - row_margin)
            row_end = min(h, row_y + row_margin)
            
            white_row = white_mask[row_start:row_end, :]
            yellow_row = yellow_mask[row_start:row_end, :]
            
            cx_white = self._getCentroid(white_row)
            cx_yellow = self._getCentroid(yellow_row)
            
            target_x = self._calculate_target_for_row(cx_white, cx_yellow)
            
            if target_x is not None:
                target_x_list.append(target_x)
                weights_used.append(self.row_weights[i])
        
        if len(target_x_list) > 0:
            weights_array = np.array(weights_used)
            weights_array = weights_array / np.sum(weights_array)  
            
            final_target_x = np.sum(np.array(target_x_list) * weights_array)
            
            image_center = w / 2
            error = final_target_x - image_center
            normalized_error = error / image_center
            
            self.pub_lane_error.publish(Float64(data=float(normalized_error)))
            
            vis_image = self._visualize(bgr_image, target_x_list, final_target_x)
            self.pub_image_lane.publish(self.cvBridge.cv2_to_imgmsg(vis_image, 'bgr8'))
        else:
            self.get_logger().warn('No lanes detected on any row!')
    
    def _calculate_target_for_row(self, cx_white, cx_yellow):
        if cx_white is not None and cx_yellow is not None:
            current_width = cx_white - cx_yellow
            
            if self.road_width_min < current_width < self.road_width_max:
                self.width_history.append(current_width)
                if len(self.width_history) > self.max_history:
                    self.width_history.pop(0)
                
                self.road_width = np.median(self.width_history)
                
                return (cx_white + cx_yellow) / 2
            else:
                if cx_yellow is not None:
                    return cx_yellow + self.road_width / 2
                elif cx_white is not None:
                    return cx_white - self.road_width / 2
        elif cx_yellow is not None:
            return cx_yellow + self.road_width / 2
        elif cx_white is not None:
            return cx_white - self.road_width / 2
        return None
    
    def _getCentroid(self, mask):
        M = cv2.moments(mask)
        if M['m00'] > self.centroid_pixels_threshold:
            cx = int(M['m10'] / M['m00'])
            return cx
        return None
    
    def maskWhiteLane(self, hsv):
        lower_white = np.array([self.hue_white_l, self.saturation_white_l, self.brightness_white_l])
        upper_white = np.array([self.hue_white_h, self.saturation_white_h, self.brightness_white_h])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        if self.is_calibration_mode:
            self.pub_image_white_lane.publish(
                self.cvBridge.cv2_to_imgmsg(mask, 'mono8')
            )

        return mask

    def maskYellowLane(self, hsv):
        lower_yellow = np.array([self.hue_yellow_l, self.saturation_yellow_l, self.brightness_yellow_l])
        upper_yellow = np.array([self.hue_yellow_h, self.saturation_yellow_h, self.brightness_yellow_h])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        if self.is_calibration_mode:
            self.pub_image_yellow_lane.publish(
                self.cvBridge.cv2_to_imgmsg(mask, 'mono8')
            )

        return mask
    
    def _visualize(self, image, targets_per_row, final_target_x):
        vis = image.copy()
        
        for i, row_y in enumerate(self.detection_rows):
            
            if i < len(targets_per_row):
                cv2.circle(vis, (int(targets_per_row[i]), row_y), 8, (0, 255, 0), -1)

        h, w, _ = image.shape
        cv2.circle(vis, (int(final_target_x), int(h/2)), 12, (255, 0, 255), -1)

        return vis

def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()