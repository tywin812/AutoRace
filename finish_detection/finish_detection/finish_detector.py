#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.signal import find_peaks


class FinishDetector(Node):

    def __init__(self):
        super().__init__('finish_detector')
        
        self.bridge = CvBridge()
        
        self.sub_image = self.create_subscription(
            Image, '/color/image',
            self.image_callback, 10
        )
        
        self.sub_tunnel_detected = self.create_subscription(
            Bool, '/tunnel_detected',
            self.tunnel_callback, 1
        )
        
        self.pub_finish = self.create_publisher(String, '/robot/finish', 10)
        self.pub_finish_detected = self.create_publisher(Bool, '/finish/detected', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_lane_active = self.create_publisher(Bool, '/lane_control_active', 10)
        
        # Параметры детекции
        self.min_peaks = 8  
        self.max_std_dev = 20.0 
        self.min_area_ratio = 0.2 
        self.detection_threshold = 1 
        
        self.state = "WAITING_FOR_TUNNEL"  # WAITING_FOR_TUNNEL -> SEARCHING -> DETECTED -> FINISHED
        self.detection_count = 0
        self.loss_count = 0
        self.loss_threshold = 5
        
        self.is_active = False
        
    def tunnel_callback(self, msg):
        if msg.data and not self.is_active:
            self.is_active = True
            self.state = "SEARCHING"

    def apply_ipm(self, image):
        h, w = image.shape[:2]
        
        src_points = np.float32([
            [w * 0.10, h * 0.65],
            [w * 0.90, h * 0.65],
            [w * 0.0,  h],
            [w * 1.0,  h]
        ])
        
        dst_points = np.float32([
            [w * 0.2, 0],
            [w * 0.8, 0],
            [w * 0.2, h],
            [w * 0.8, h]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, matrix, (w, h))
        return warped

    def image_callback(self, msg):
        if not self.is_active:
            return
        
        if self.state == "FINISHED":
            self.stop_robot()
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Bridge error: {e}')
            return
        
        h, w = cv_image.shape[:2]
        
        # ROI - берем нижние 2/3
        roi_start = int(h * 1 / 3) 
        roi = cv_image[roi_start:, :]
        
        # Выпрямление перспективы
        ipm_roi = self.apply_ipm(roi)
        
        is_finish = self.detect_checkered_pattern(ipm_roi)
        
        if self.state == "SEARCHING":
            if is_finish:
                self.detection_count += 1
                if self.detection_count >= self.detection_threshold:
                    self.state = "DETECTED"
                    self.loss_count = 0
                    self.get_logger().info('Finish line detected!')
            else:
                self.detection_count = max(0, self.detection_count - 1)
                
        elif self.state == "DETECTED":
            if is_finish:
                self.loss_count = 0
            else:
                self.loss_count += 1
                if self.loss_count >= self.loss_threshold:
                    self.get_logger().info('Finish line crossed - STOPPING!')
                    self.publish_finish()
                    self.state = "FINISHED"

    def detect_checkered_pattern(self, roi):
        h, w = roi.shape[:2]
        
        # Размытие для уменьшения шума
        roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        gray = cv2.cvtColor(roi_blurred, cv2.COLOR_BGR2GRAY)
        
        # Выравнивание гистограммы
        gray = cv2.equalizeHist(gray)
        
        # Определение границ - Auto Canny
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper)
        
        # Горизонтальная проекция
        h_projection = np.sum(edges, axis=0)
        max_val = np.max(h_projection)
        
        if max_val < 255 * 10: 
            return False
        
        # Нормализация
        h_projection = h_projection / max_val
        
        # Поиск пиков
        peaks, _ = find_peaks(h_projection, height=0.2, distance=10)
        
        if len(peaks) < self.min_peaks:
            return False
        
        # Проверка периодичности
        if len(peaks) > 1:
            distances = np.diff(peaks)
            mean_dist = np.mean(distances)
            std_dev = np.std(distances)
            cv = std_dev / mean_dist if mean_dist > 0 else 1.0
            
            if std_dev > self.max_std_dev or cv > 0.5: 
                return False
        
        # Проверка покрытия
        if len(peaks) > 0:
            pattern_width = peaks[-1] - peaks[0]
            width_ratio = pattern_width / w
            if width_ratio < self.min_area_ratio:
                return False
        
        # Проверка наличия черных пикселей
        x_start = peaks[0]
        x_end = peaks[-1]
        roi_gray = gray[:, x_start:x_end]
        black_ratio = np.sum(roi_gray < 50) / roi_gray.size
        
        if black_ratio < 0.05:
            return False
        
        return True

    def stop_robot(self):
        lane_msg = Bool()
        lane_msg.data = False
        self.pub_lane_active.publish(lane_msg)
        
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.pub_cmd_vel.publish(twist)

    def publish_finish(self):
        finish_msg = String()
        finish_msg.data = "EREVAN"
        self.pub_finish.publish(finish_msg)
        
        detected_msg = Bool()
        detected_msg.data = True
        self.pub_finish_detected.publish(detected_msg)
        
        self.stop_robot()


def main(args=None):
    rclpy.init(args=args)
    node = FinishDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
