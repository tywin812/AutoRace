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
        self.pub_debug_edges = self.create_publisher(Image, '/finish/debug/edges', 1)
        self.pub_debug_projection = self.create_publisher(Image, '/finish/debug/projection', 1)
        self.pub_debug_result = self.create_publisher(Image, '/finish/debug/result', 1)
        
        # Параметры
        self.min_peaks = 8  # Increased from 6
        self.max_std_dev = 20.0 # Tightened from 15.0
        self.min_area_ratio = 0.2 # Tightened from 0.2
        self.bw_ratio_threshold = 0.6
        self.detection_threshold = 1 # Increased from 3
        
        self.detection_count = 0
        self.finish_detected = False
        self.frame_count = 0
        self.log_count = 0
        
        self.get_logger().info('=== Finish Detector Started ===')
        self.get_logger().info(f'Subscribed to: /color/image')
        self.get_logger().info(f'Min peaks: {self.min_peaks}, Max std_dev: {self.max_std_dev}')

    def image_callback(self, msg):
        if self.finish_detected:
            return
        
        self.frame_count += 1
        # Process every frame for better detection
        # (removed frame skip optimization)
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Bridge error: {e}')
            return
        
        h, w = cv_image.shape[:2]
        
        # ROI: нижняя половина (Lowered from h // 3 to h // 2)
        roi_start = int(h * 2 / 3)
        roi = cv_image[roi_start:, :]
        
        # ЧИСТАЯ статистика ROI (БЕЗ пиков!)
        roi_stats = self.analyze_roi_colors(roi)
        
        # Детекция
        is_finish, debug_data = self.detect_checkered_pattern(roi)
        
        # Добавляем ROI stats в debug_data
        debug_data.update(roi_stats)
        
        # Logging every 10 frames
        self.log_count += 1
        if self.log_count >= 10:
            self.log_count = 0
            
            # Статистика детекции
            peaks_count = len(debug_data.get('peaks', []))
            std_dev = debug_data.get('std_dev', 0)
            width_ratio = debug_data.get('width_ratio', 0)
            mean_sat = debug_data.get('mean_saturation', 0)
            black_ratio = debug_data.get('black_ratio', 0)
            bw_ratio = debug_data.get('bw_ratio', 0)
            
            status = "✅ FINISH!" if is_finish else "🔍 Searching"
            self.get_logger().info(
                f"{status} | Peaks: {peaks_count}/{self.min_peaks} | "
                f"Std: {std_dev:.1f}/{self.max_std_dev} | "
                f"Width: {width_ratio:.2f}/{self.min_area_ratio} | "
                f"Sat: {mean_sat:.1f}/50 | "
                f"Black: {black_ratio:.2f}/0.05 | "
                f"B/W: {bw_ratio:.2f}/0.4"
            )
            
            # ЧИСТАЯ статистика ROI
            self.get_logger().info(
                f'🎨 [ROI PURE] Brightness | Min: {roi_stats["brightness_min"]:.0f} '
                f'Max: {roi_stats["brightness_max"]:.0f} '
                f'Mean: {roi_stats["brightness_mean"]:.0f} '
                f'Median: {roi_stats["brightness_median"]:.0f}'
            )
            
            self.get_logger().info(
                f'🎨 [ROI PURE] Distribution | '
                f'VDark(<50): {roi_stats["very_dark_pct"]:.1f}% '
                f'Dark(50-100): {roi_stats["dark_pct"]:.1f}% '
                f'Med(100-150): {roi_stats["medium_pct"]:.1f}% '
                f'Light(150-200): {roi_stats["light_pct"]:.1f}% '
                f'VLight(>200): {roi_stats["very_light_pct"]:.1f}%'
            )
        
        if is_finish:
            self.detection_count += 1
            self.get_logger().info(
                f'Финиш обнаружен! Подтверждений: {self.detection_count}/{self.detection_threshold}'
            )
            
            if self.detection_count >= self.detection_threshold:
                self.publish_finish()
        else:
            self.detection_count = max(0, self.detection_count - 1)
        
        # Always publish debug info (removed subscriber check)
        self.publish_debug_info(cv_image, roi, roi_start, is_finish, debug_data)

    def analyze_roi_colors(self, roi):
        """Анализ цветов во всей ROI без проверок"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Статистика яркости
        brightness_min = np.min(gray)
        brightness_max = np.max(gray)
        brightness_mean = np.mean(gray)
        brightness_median = np.median(gray)
        brightness_std = np.std(gray)
        
        # Подсчет пикселей по диапазонам
        very_dark = np.sum(gray < 50)      # Очень темные (черные)
        dark = np.sum((gray >= 50) & (gray < 100))  # Темные
        medium = np.sum((gray >= 100) & (gray < 150))  # Средние (серая дорога)
        light = np.sum((gray >= 150) & (gray < 200))  # Светлые
        very_light = np.sum(gray >= 200)   # Очень светлые (белые)
        
        total_pixels = gray.size
        
        return {
            'brightness_min': brightness_min,
            'brightness_max': brightness_max,
            'brightness_mean': brightness_mean,
            'brightness_median': brightness_median,
            'brightness_std': brightness_std,
            'very_dark_pct': (very_dark / total_pixels) * 100,
            'dark_pct': (dark / total_pixels) * 100,
            'medium_pct': (medium / total_pixels) * 100,
            'light_pct': (light / total_pixels) * 100,
            'very_light_pct': (very_light / total_pixels) * 100,
            'very_dark_count': very_dark,
            'dark_count': dark,
            'medium_count': medium,
            'light_count': light,
            'very_light_count': very_light,
            'total_pixels': total_pixels
        }

    def detect_checkered_pattern(self, roi):
        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Check for overexposure
        mean_brightness = np.mean(gray)
        if mean_brightness > 230:
             self.get_logger().warn(f'Image might be overexposed! Mean brightness: {mean_brightness:.1f}', throttle_duration_sec=2.0)

        # Equalize histogram to improve contrast
        gray = cv2.equalizeHist(gray)
        
        # 1. Границы - Auto Canny
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper)
        
        # 2. Горизонтальная проекция
        h_projection = np.sum(edges, axis=0)
        
        # Check if edges are strong enough (avoid noise)
        max_val = np.max(h_projection)
        if max_val < 255 * 10: # At least 10 pixels of vertical edge
             return False, {
                 'edges': edges, 
                 'projection': h_projection, 
                 'peaks': []
             }

        # 3. Нормализация
        if max_val > 0:
            h_projection = h_projection / max_val
        
        # 4. Поиск пиков
        peaks, _ = find_peaks(
            h_projection, 
            height=0.2, # Increased from 0.3
            distance=3
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
            
            if std_dev > self.max_std_dev or cv > 0.5: # Relaxed CV threshold from 0.3 to 0.5
                return False, debug_data
        
        # 6. Покрытие
        if len(peaks) > 0:
            pattern_width = peaks[-1] - peaks[0]
            width_ratio = pattern_width / w
            
            debug_data['width_ratio'] = width_ratio
            
            if width_ratio < self.min_area_ratio:
                return False, debug_data

        # NEW: Color Saturation Check
        # Finish line is black and white (low saturation).
        # Yellow lines are highly saturated.
        x_start = peaks[0]
        x_end = peaks[-1]
        
        # Extract the strip corresponding to the pattern
        pattern_strip = roi[:, x_start:x_end]
        if pattern_strip.size > 0:
            hsv_strip = cv2.cvtColor(pattern_strip, cv2.COLOR_BGR2HSV)
            saturation = hsv_strip[:, :, 1]
            mean_saturation = np.mean(saturation)
            
            debug_data['mean_saturation'] = mean_saturation
            
            # If saturation is high (> 50), it's likely yellow lines or something else colorful
            if mean_saturation > 50:
                return False, debug_data
        
        # 7. Check for True Black pixels
        # The road is gray (medium brightness), lines are white/yellow (high brightness).
        # The finish flag MUST have black pixels (low brightness).
        roi_gray = gray[:, x_start:x_end]
        # Threshold for "True Black" - adjust based on lighting, but 70 is usually safe for "darker than road"
        black_pixels_count = np.sum(roi_gray < 70) 
        total_pixels = roi_gray.size
        black_ratio = black_pixels_count / total_pixels
        
        debug_data['black_ratio'] = black_ratio
        
        # At least 5% of the area must be truly black
        if black_ratio < 0.05:
            return False, debug_data

        # 8. Черно-белый баланс - Adaptive
        # Use Otsu's thresholding to find optimal threshold
        _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        black_pixels = np.sum(thresh_img == 0)
        white_pixels = np.sum(thresh_img == 255)
        
        if black_pixels + white_pixels > 0:
            bw_ratio = min(black_pixels, white_pixels) / max(black_pixels, white_pixels)
            debug_data['bw_ratio'] = bw_ratio
            
            # Lowered threshold slightly to be more permissive
            if bw_ratio < 0.4: 
                return False, debug_data
        
        return True, debug_data

    def publish_debug_info(self, full_image, roi, roi_start, is_finish, debug_data):
        # Always publish debug images (removed subscriber check optimization)
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
        
        # Draw ROI rectangle
        cv2.rectangle(result_img, (0, roi_start), (w, h), (255, 255, 0), 2)
        
        # Draw detected area if peaks found
        if 'peaks' in debug_data and len(debug_data['peaks']) > 0:
            peaks = debug_data['peaks']
            x_start = peaks[0]
            x_end = peaks[-1]
            
            # Draw bounding box around the pattern
            # Color depends on whether it's considered a valid finish
            box_color = (0, 255, 0) if is_finish else (0, 0, 255)
            cv2.rectangle(result_img, (x_start, roi_start), (x_end, h), box_color, 3)
            
            # Draw vertical lines for each peak
            for peak in peaks:
                cv2.line(result_img, (peak, roi_start), (peak, h), (255, 0, 255), 1)

        status_color = (0, 255, 0) if is_finish else (0, 0, 255)
        status_text = "FINISH DETECTED!" if is_finish else "Searching..."
        cv2.putText(
            result_img, status_text,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, status_color, 2
        )
        
        y_offset = 60
        if 'peaks' in debug_data:
            color = (0, 255, 0) if len(debug_data['peaks']) >= self.min_peaks else (0, 0, 255)
            cv2.putText(
                result_img, f"Peaks: {len(debug_data['peaks'])}/{self.min_peaks}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 1
            )
            y_offset += 25
        
        if 'std_dev' in debug_data:
            std = debug_data['std_dev']
            color = (0, 255, 0) if std <= self.max_std_dev else (0, 0, 255)
            cv2.putText(
                result_img, f"Std Dev: {std:.1f}/{self.max_std_dev}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 1
            )
            y_offset += 25
        
        if 'width_ratio' in debug_data:
            wr = debug_data['width_ratio']
            color = (0, 255, 0) if wr >= self.min_area_ratio else (0, 0, 255)
            cv2.putText(
                result_img, f"Width: {wr:.2f}/{self.min_area_ratio}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 1
            )
            y_offset += 25

        if 'mean_saturation' in debug_data:
            sat = debug_data['mean_saturation']
            color = (0, 255, 0) if sat <= 50 else (0, 0, 255)
            cv2.putText(
                result_img, f"Sat: {sat:.1f}/50",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 1
            )
            y_offset += 25
        
        if 'black_ratio' in debug_data:
            br = debug_data['black_ratio']
            color = (0, 255, 0) if br >= 0.05 else (0, 0, 255)
            cv2.putText(
                result_img, f"True Black: {br:.2f}/0.05",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 1
            )
            y_offset += 25

        if 'bw_ratio' in debug_data:
            bw = debug_data['bw_ratio']
            color = (0, 255, 0) if bw >= 0.4 else (0, 0, 255)
            cv2.putText(
                result_img, f"B/W: {bw:.2f}/0.4",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 1
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
