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
    """Детекция шахматного финиша через анализ градиентов и периодичности"""

    def __init__(self):
        super().__init__('finish_detector')
        
        self.bridge = CvBridge()
        
        # Subscriptions
        self.sub_image = self.create_subscription(
            Image, '/color/image',
            self.image_callback, 10
        )
        
        # Publishers
        self.pub_finish = self.create_publisher(String, '/robot/finish', 10)
        self.pub_finish_detected = self.create_publisher(Bool, '/finish/detected', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_lane_active = self.create_publisher(Bool, '/lane_control_active', 10)
        
        # Debug publishers
        self.pub_debug_edges = self.create_publisher(Image, '/finish/debug/edges', 1)
        self.pub_debug_projection = self.create_publisher(Image, '/finish/debug/projection', 1)
        self.pub_debug_result = self.create_publisher(Image, '/finish/debug/result', 1)
        self.pub_debug_ipm = self.create_publisher(Image, '/finish/debug/ipm', 1)
        
        # Параметры
        self.min_peaks = 8  
        self.max_std_dev = 20.0 
        self.min_area_ratio = 0.2 
        self.bw_ratio_threshold = 0.6
        self.detection_threshold = 1 
        
        # State Machine Parameters
        self.state = "SEARCHING" # SEARCHING -> DETECTED -> FINISHED
        self.detection_count = 0
        self.loss_count = 0
        self.loss_threshold = 5 # Frames to confirm we passed the finish line
        
        self.frame_count = 0
        self.log_count = 0
        
        self.get_logger().info('=== Finish Detector Started ===')
        self.get_logger().info(f'Subscribed to: /color/image')
        self.get_logger().info(f'Mode: Stop on Exit + Lane Disable (State: {self.state})')

    def apply_ipm(self, image):
        """Выпрямление перспективы (трапеция -> прямоугольник)"""
        h, w = image.shape[:2]
        
        # Параметры трапеции (подобраны под стандартный наклон камеры робота)
        src_points = np.float32([
            [w * 0.10, h * 0.65],  # Top Left
            [w * 0.90, h * 0.65],  # Top Right
            [w * 0.0,  h],         # Bottom Left
            [w * 1.0,  h]          # Bottom Right
        ])
        
        dst_points = np.float32([
            [w * 0.2, 0],         # Top Left
            [w * 0.8, 0],         # Top Right
            [w * 0.2, h],         # Bottom Left
            [w * 0.8, h]          # Bottom Right
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, matrix, (w, h))
        return warped

    def image_callback(self, msg):
        # Если финиш уже пересечен - постоянно шлем СТОП
        if self.state == "FINISHED":
            self.stop_robot()
            return
        
        self.frame_count += 1
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Bridge error: {e}')
            return
        
        h, w = cv_image.shape[:2]
        
        # 1. ROI - берем нижние 2/3
        roi_start = int(h * 1 / 3) 
        roi = cv_image[roi_start:, :]
        
        # 2. Выпрямление перспективы (IPM)
        ipm_roi = self.apply_ipm(roi)
        
        # Публикуем IPM для отладки
        try:
            self.pub_debug_ipm.publish(self.bridge.cv2_to_imgmsg(ipm_roi, "bgr8"))
        except:
            pass

        # 3. Анализ (ROI stats для логов)
        roi_stats = self.analyze_roi_colors(ipm_roi)
        
        # 4. Детекция
        is_finish, debug_data = self.detect_checkered_pattern(ipm_roi)
        
        # Добавляем ROI stats в debug_data
        debug_data.update(roi_stats)
        
        # --- STATE MACHINE LOGIC ---
        
        if self.state == "SEARCHING":
            if is_finish:
                self.detection_count += 1
                if self.detection_count >= self.detection_threshold:
                    self.state = "DETECTED"
                    self.loss_count = 0
                    self.get_logger().info('🏁 ФИНИШ ЗАМЕЧЕН! (DETECTED) Едем до пересечения...')
            else:
                self.detection_count = max(0, self.detection_count - 1)
                
        elif self.state == "DETECTED":
            if is_finish:
                # Мы все еще видим финиш, сбрасываем счетчик потери
                self.loss_count = 0
            else:
                # Финиш пропал из вида!
                self.loss_count += 1
                self.get_logger().info(f'Финиш теряется... {self.loss_count}/{self.loss_threshold}')
                
                if self.loss_count >= self.loss_threshold:
                    self.publish_finish() # STOP!
                    self.state = "FINISHED"
        
        # --- LOGGING ---
        
        self.log_count += 1
        if self.log_count >= 10:
            self.log_count = 0
            
            # Статистика детекции
            peaks_count = len(debug_data.get('peaks', []))
            
            status_icon = "SEARCH"
            if self.state == "DETECTED": status_icon = "👀 HOLD"
            if self.state == "FINISHED": status_icon = "🛑 DONE"
            
            self.get_logger().info(
                f"State: {status_icon} | Peaks: {peaks_count}/{self.min_peaks} | "
                f"Std: {debug_data.get('std_dev', 0):.1f} | "
                f"CV: {debug_data.get('cv', 0):.2f} | "
                f"Confirm: {self.detection_count} | Loss: {self.loss_count}"
            )
        
        # Always publish debug info
        # Using IPM ROI for visualization
        self.publish_debug_info(ipm_roi, ipm_roi, 0, is_finish, debug_data)

    def analyze_roi_colors(self, roi):
        """Анализ цветов во всей ROI без проверок"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Статистика яркости
        return {
            'very_dark_pct': 0, 
            'dark_pct': 0,
            'medium_pct': 0,
            'light_pct': 0,
            'very_light_pct': 0,
        }

    def detect_checkered_pattern(self, roi):
        h, w = roi.shape[:2]
        
        # 0. Noise Reduction (Gaussian Blur)
        roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        gray = cv2.cvtColor(roi_blurred, cv2.COLOR_BGR2GRAY)
        
        # Equalize
        gray = cv2.equalizeHist(gray)
        
        # 1. Границы - Auto Canny
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper)
        
        # 2. Горизонтальная проекция
        h_projection = np.sum(edges, axis=0)
        max_val = np.max(h_projection)
        
        if max_val < 255 * 10: 
             return False, {'edges': edges, 'projection': h_projection, 'peaks': []}

        # 3. Нормализация
        h_projection = h_projection / max_val
        
        # 4. Поиск пиков
        peaks, _ = find_peaks(h_projection, height=0.2, distance=10)
        
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

        # 7. Check for True Black pixels
        x_start = peaks[0]
        x_end = peaks[-1]
        roi_gray = gray[:, x_start:x_end]
        black_ratio = np.sum(roi_gray < 50) / roi_gray.size
        debug_data['black_ratio'] = black_ratio
        
        if black_ratio < 0.05:
            return False, debug_data

        return True, debug_data

    def publish_debug_info(self, full_image, roi, roi_start, is_finish, debug_data):
        h, w = full_image.shape[:2]
        
        # 1. Result
        result_img = full_image.copy()
        
        # Draw finding rectangle
        if 'peaks' in debug_data and len(debug_data['peaks']) > 0:
            peaks = debug_data['peaks']
            box_color = (0, 255, 0) if is_finish else (0, 0, 255)
            cv2.rectangle(result_img, (peaks[0], 0), (peaks[-1], h), box_color, 3)

        # Draw State Info
        state_color = (255, 255, 0)
        if self.state == "DETECTED": state_color = (0, 255, 255) # Yellow
        if self.state == "FINISHED": state_color = (0, 0, 255)   # Red
        
        cv2.putText(result_img, f"STATE: {self.state}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 2)
                   
        if self.state == "DETECTED":
             cv2.putText(result_img, f"Loss: {self.loss_count}/{self.loss_threshold}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        try:
            self.pub_debug_result.publish(self.bridge.cv2_to_imgmsg(result_img, "bgr8"))
        except:
            pass

    def stop_robot(self):
        """Принудительная остановка"""
        # 1. Disable Lane Follower
        lane_msg = Bool()
        lane_msg.data = False
        self.pub_lane_active.publish(lane_msg)
        
        # 2. Force Stop
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.pub_cmd_vel.publish(twist)

    def publish_finish(self):
        finish_msg = String()
        finish_msg.data = "finish"
        self.pub_finish.publish(finish_msg)
        
        detected_msg = Bool()
        detected_msg.data = True
        self.pub_finish_detected.publish(detected_msg)
        
        self.get_logger().info('🛑 ФИНИШ ПЕРЕСЕЧЕН! ОТПРАВЛЯЮ КОМАНДУ STOP (Disable Lane).')
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
