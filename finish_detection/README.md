# Finish Detection Package

Пакет для детекции шахматного финиша с камеры через анализ градиентов и периодичности.

## Принцип работы

Алгоритм детектирует шахматный паттерн по следующим признакам:

1. **Градиенты**: Canny edge detection находит границы квадратов
2. **Периодичность**: Горизонтальная проекция показывает пики на одинаковом расстоянии
3. **Черно-белый баланс**: Примерно 50/50 черных и белых пикселей
4. **Покрытие**: Паттерн занимает значительную часть ширины кадра

## Запуск

```bash
# Сборка
cd ~/ros2_ws
colcon build --packages-select finish_detection
source install/setup.bash

# Запуск ноды
ros2 launch finish_detection finish_detection.launch.py

# Или напрямую
ros2 run finish_detection finish_detector
```

## Topics

### Subscriptions
- `/camera/image_raw` (sensor_msgs/Image) - входное изображение с камеры

### Publications
- `/robot/finish` (std_msgs/String) - сигнал финиша для системы судейства
- `/finish/detected` (std_msgs/Bool) - внутренний флаг

### Debug Topics (для RViz)
- `/finish/debug/edges` (sensor_msgs/Image) - изображение границ
- `/finish/debug/projection` (sensor_msgs/Image) - график горизонтальной проекции
- `/finish/debug/result` (sensor_msgs/Image) - результирующее изображение со статистикой

## Отладка в RViz

### 1. Запуск RViz

```bash
rviz2
```

### 2. Добавление визуализаций

Добавьте 4 Image display:

#### a) Оригинальное изображение
- Add -> Image
- Image Topic: `/camera/image_raw`
- Показывает что видит камера

#### b) Детекция границ (Edges)
- Add -> Image
- Image Topic: `/finish/debug/edges`
- Показывает результат Canny edge detection
- **Что смотреть**: должны быть четкие вертикальные линии на границах квадратов

#### c) График проекции
- Add -> Image
- Image Topic: `/finish/debug/projection`
- Показывает горизонтальную проекцию
- **Что смотреть**: 
  - Зеленая линия = интенсивность границ
  - Красные вертикальные линии = найденные пики
  - Пики должны быть на одинаковом расстоянии для шахматки

#### d) Итоговый результат
- Add -> Image
- Image Topic: `/finish/debug/result`
- Показывает исходное изображение + статистику
- **Информация на экране**:
  - `Peaks` - количество найденных пиков (нужно >= 8)
  - `Std Dev` - стандартное отклонение расстояний (нужно < 15)
  - `Width` - доля ширины кадра (нужно > 0.3)
  - `B/W` - соотношение черного/белого (нужно > 0.6)
  - `Confirm` - счетчик подтверждений (5/5 = финиш)
  - Желтый прямоугольник = ROI (область интереса)

## Настройка параметров

Если детекция работает плохо, можно настроить в коде `finish_detector.py`:

```python
self.min_peaks = 8  # Уменьшить если частично виден финиш
self.max_std_dev = 15.0  # Увеличить если неровные квадраты
self.min_area_ratio = 0.3  # Уменьшить для ранней детекции
self.bw_ratio_threshold = 0.6  # Уменьшить при плохом освещении
self.detection_threshold = 5  # Количество подряд кадров для подтверждения
```

## Типичные проблемы

### Проблема: Мало пиков (Peaks < 8)
**Решение**:
- Проверьте `/finish/debug/edges` - видны ли границы?
- Попробуйте уменьшить `self.min_peaks` до 6
- Проверьте освещение - Canny чувствителен к контрасту

### Проблема: Большое Std Dev (> 15)
**Решение**:
- Посмотрите `/finish/debug/projection` - пики должны быть равномерные
- Робот может смотреть под углом - это OK, увеличьте `max_std_dev`

### Проблема: Ложные срабатывания
**Решение**:
- Увеличьте `self.detection_threshold` (5 -> 7)
- Увеличьте `self.min_area_ratio` (0.3 -> 0.5)
- Увеличьте `self.bw_ratio_threshold` (0.6 -> 0.7)

### Проблема: Не видит финиш вообще
**Решение**:
- Проверьте `/camera/image_raw` - работает ли камера?
- Посмотрите `/finish/debug/result` - попадает ли финиш в ROI (желтый прямоугольник)?
- ROI смотрит только на нижнюю половину - можно изменить `roi_start = h // 3`

## Интеграция с race_controller

В вашем главном контроллере подпишитесь на `/finish/detected`:

```python
class RaceController(Node):
    def __init__(self):
        self.sub_finish = self.create_subscription(
            Bool, '/finish/detected',
            self.finish_callback, 10
        )
    
    def finish_callback(self, msg):
        if msg.data:
            self.get_logger().info('🏁 ФИНИШ! Остановка робота...')
            self.stop_robot()
```

## Дополнительно

Алгоритм устойчив к:
- Разному углу обзора
- Частичному попаданию финиша в кадр
- Неидеальному освещению (благодаря Canny)

Не путает с:
- Обычными желтыми/белыми линиями (нет периодичности)
- Другими контрастными паттернами (проверка B/W баланса)
