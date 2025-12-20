# Lane Boundary Filtering Patch

Добавь эти изменения в `construction_and_overtaking/construction_and_overtaking/obstacle_avoidance.py`:

## 1. В `__init__` после строки `self.obstacle_type = "NONE"` добавь:

```python
# Calibration: pixel-to-meter conversion
self.image_center_x = 500.0
self.lane_width_pixels = 600.0
self.lane_width_meters = 0.6
self.pixels_per_meter = self.lane_width_pixels / self.lane_width_meters
self.lane_boundary_margin = 0.05  # meters
```

## 2. После `__init__` добавь 3 новые функции:

```python
def pixels_to_meters(self, pixel_distance):
    """Convert pixel distance to meters."""
    return pixel_distance / self.pixels_per_meter

def get_lane_boundaries_in_meters(self):
    """Get lane boundaries in meters from robot center."""
    if self.left_distance < 999.0:
        left_boundary_m = self.pixels_to_meters(self.left_distance) + self.lane_boundary_margin
    else:
        left_boundary_m = 999.0
        
    if self.right_distance < 999.0:
        right_boundary_m = -(self.pixels_to_meters(self.right_distance) + self.lane_boundary_margin)
    else:
        right_boundary_m = -999.0
        
    return left_boundary_m, right_boundary_m

def is_point_within_lanes(self, y_position, left_boundary, right_boundary):
    """Check if point is within lane boundaries."""
    if left_boundary >= 999.0 and right_boundary <= -999.0:
        return True
        
    if left_boundary < 999.0 and y_position > left_boundary:
        return False
        
    if right_boundary > -999.0 and y_position < right_boundary:
        return False
        
    return True
```

## 3. В `lidar_callback` после строки:
```python
mask = (x > 0) & (x < check_dist) & (np.abs(y) < 1.0)
```

Добавь:
```python
# Get lane boundaries
left_boundary, right_boundary = self.get_lane_boundaries_in_meters()

# FILTER: Only keep points within lane boundaries
if self.lane_state > 0:
    within_lanes = np.array([self.is_point_within_lanes(y[i], left_boundary, right_boundary) 
                             for i in range(len(y))])
    mask = mask & within_lanes
```

## 4. В `log_status` замени строку с логом на:

```python
left_boundary, right_boundary = self.get_lane_boundaries_in_meters()
self.get_logger().info(
    f'State: {current_state_name} | Lanes: L={self.left_distance:.1f}px({left_boundary:.2f}m) '
    f'R={self.right_distance:.1f}px({right_boundary:.2f}m) | LIDAR: F:{self.front_distance:.2f}m'
)
```

## Что это делает:

- Конвертирует расстояния до линий из пикселей в метры
- Фильтрует LiDAR точки — оставляет только те что между линиями
- Объекты за линиями (сбоку от дороги) теперь игнорируются
- В логах показывает границы в метрах

## Калибровка:

Параметры можно подстроить:
- `lane_width_pixels = 600.0` - ширина между линиями в пикселях
- `lane_width_meters = 0.6` - реальная ширина дороги в метрах
- `lane_boundary_margin = 0.05` - запас за линиями (5см)
