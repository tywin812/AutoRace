#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.signal import find_peaks


class FinishDetector(Node):
    """Детекция шахматного финиша через анализ градиентов и периодичности"""

    def __init__(self):
        super().__init__('finish_detector')
        
        self.bridge = CvBridge()
        
        # Subscriptions - исправленный топик
        self.sub_image = self.create_subscription(
            Image, '/color/image',
            self.image_callback, 10
        )
        
        # Publishers
        self.pub_finish = self.create_publisher(String, '/robot/finish', 10)
        self.pub_finish_detected = self.create_publisher(Bool, '/finish/detected', 10)
        
        # Debug publishers
        self.pub_debug_edges = self.create_publisher(Image, '/finish/debug/edges', 10)
        self.pub_debug_projection = self.create_publisher(Image, '/finish/debug/projection', 10)
        self.pub_debug_result = self.create_publisher(Image, '/finish/debug/result', 10)
        
        # Параметры
        self.min_peaks = 8
        self.max_std_dev = 15.0
        self.min_area_ratio = 0.3
        self.bw_ratio_threshold = 0.6
        self.detection_threshold = 5
        
        self.detection_count = 0
        self.finish_detected = False
        
        self.get_logger().info('=== Finish Detector Started ===')
        self.get_logger().info(f'Subscribed to: /color/image')
        self.get_logger().info(f'Min peaks: {self.min_peaks}, Max std_dev: {self.max_std_dev}')

    def image_callback(self, msg):
        if self.finish_detected:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Bridge error: {e}')
            return
        
        h, w = cv_image.shape[:2]
        
        # ROI: нижняя половина
        roi_start = h // 3
        roi = cv_image[roi_start:, :]
        
        # Детекция
        is_finish, debug_data = self.detect_checkered_pattern(roi)
        
        if is_finish:
            self.detection_count += 1
            self.get_logger().info(
                f'Финиш обнаружен! Подтверждений: {self.detection_count}/{self.detection_threshold}'
            )
            
            if self.detection_count >= self.detection_threshold:
                self.publish_finish()
        else:
            self.detection_count = max(0, self.detection_count - 1)
        
        self.publish_debug_info(cv_image, roi, roi_start, is_finish, debug_data)

    def detect_checkered_pattern(self, roi):
        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Границы
        edges = cv2.Canny(gray, 50, 150)
        
        # 2. Горизонтальная проекция
        h_projection = np.sum(edges, axis=0)
        
        # 3. Нормализация
        if np.max(h_projection) > 0:
            h_projection = h_projection / np.max(h_projection)
        
        # 4. Поиск пиков
        peaks, _ = find_peaks(
            h_projection, 
            height=0.3,
            distance=5
        )
        
        debug_data = {
            'edges': edges,
            'projection': h_projection,
            'peaks': peaks
        }
        
        if len(peaks) < self.min_peaks:
            return False, debug_data
        
        # 5. Периодичность
        if len(peaks) > 1:
            distances = np.diff(peaks)
            mean_dist = np.mean(distances)
            std_dev = np.std(distances)
            
            cv = std_dev / mean_dist if mean_dist > 0 else 1.0
            
            debug_data['mean_dist'] = mean_dist
            debug_data['std_dev'] = std_dev
            debug_data['cv'] = cv
            
            if std_dev > self.max_std_dev or cv > 0.5:
                return False, debug_data
        
        # 6. Покрытие
        if len(peaks) > 0:
            pattern_width = peaks[-1] - peaks[0]
            width_ratio = pattern_width / w
            
            debug_data['width_ratio'] = width_ratio
            
            if width_ratio < self.min_area_ratio:
                return False, debug_data
        
        # 7. Черно-белый баланс
        black_pixels = np.sum(gray < 100)
        white_pixels = np.sum(gray > 150)
        
        if black_pixels + white_pixels > 0:
            bw_ratio = min(black_pixels, white_pixels) / max(black_pixels, white_pixels)
            debug_data['bw_ratio'] = bw_ratio
            
            if bw_ratio < self.bw_ratio_threshold:
                return False, debug_data
        
        return True, debug_data

    def publish_debug_info(self, full_image, roi, roi_start, is_finish, debug_data):
        h, w = full_image.shape[:2]
        
        # 1. Edges
        if 'edges' in debug_data:
            edges_color = cv2.cvtColor(debug_data['edges'], cv2.COLOR_GRAY2BGR)
            try:
                self.pub_debug_edges.publish(
                    self.bridge.cv2_to_imgmsg(edges_color, "bgr8")
                )
            except:
                pass
        
        # 2. Projection
        if 'projection' in debug_data:
            projection = debug_data['projection']
            peaks = debug_data['peaks']
            
            proj_img = np.zeros((200, len(projection), 3), dtype=np.uint8)
            
            for i in range(len(projection) - 1):
                y1 = int(200 - projection[i] * 180)
                y2 = int(200 - projection[i+1] * 180)
                cv2.line(proj_img, (i, y1), (i+1, y2), (0, 255, 0), 2)
            
            for peak in peaks:
                cv2.line(proj_img, (peak, 0), (peak, 200), (0, 0, 255), 2)
            
            try:
                self.pub_debug_projection.publish(
                    self.bridge.cv2_to_imgmsg(proj_img, "bgr8")
                )
            except:
                pass
        
        # 3. Result
        result_img = full_image.copy()
        
        cv2.rectangle(result_img, (0, roi_start), (w, h), (255, 255, 0), 2)
        
        status_color = (0, 255, 0) if is_finish else (0, 0, 255)
        status_text = "FINISH DETECTED!" if is_finish else "Searching..."
        cv2.putText(
            result_img, status_text,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, status_color, 2
        )
        
        y_offset = 60
        if 'peaks' in debug_data:
            cv2.putText(
                result_img, f"Peaks: {len(debug_data['peaks'])}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )
            y_offset += 25
        
        if 'std_dev' in debug_data:
            cv2.putText(
                result_img, f"Std Dev: {debug_data['std_dev']:.1f}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )
            y_offset += 25
        
        if 'width_ratio' in debug_data:
            cv2.putText(
                result_img, f"Width: {debug_data['width_ratio']:.2f}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )
            y_offset += 25
        
        if 'bw_ratio' in debug_data:
            cv2.putText(
                result_img, f"B/W: {debug_data['bw_ratio']:.2f}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )
        
        cv2.putText(
            result_img, f"Confirm: {self.detection_count}/{self.detection_threshold}",
            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 0), 1
        )
        
        try:
            self.pub_debug_result.publish(
                self.bridge.cv2_to_imgmsg(result_img, "bgr8")
            )
        except:
            pass

    def publish_finish(self):
        if not self.finish_detected:
            self.finish_detected = True
            
            finish_msg = String()
            finish_msg.data = "finish"
            self.pub_finish.publish(finish_msg)
            
            detected_msg = Bool()
            detected_msg.data = True
            self.pub_finish_detected.publish(detected_msg)
            
            self.get_logger().info('🏁 ФИНИШ ЗАФИКСИРОВАН!')


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
