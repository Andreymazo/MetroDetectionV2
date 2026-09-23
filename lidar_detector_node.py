#!/usr/bin/env python3
"""
🚀 LIVE ROS 2 HUMBLE PRODUCTION NODE
Компонент обработки потока 3D-лидара в реальном времени под требования ТЗ жюри.
"""

import sys
import gc
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

# Стандартный парсер ROS 2 PointCloud2 в NumPy массивы без лишних тяжелых зависимостей
import sensor_msgs_py.point_cloud2 as pc2

# Импортируем ваши проверенные ходовые движки
from cos_processor_v12 import StableLidarOdometryV12
import cos_processor_v12
from generate_submission_v2 import process_point_cloud
from tracker_v2 import LidarObstacleTrackerV2
import config

class SubwayVisionCoreNode(Node):
    def __init__(self):
        super().__init__('subway_vision_core_node')
        
        # 1. ПОДПИСКА НА ТОПИК ЖЮРИ:
        # По ТЗ данные идут из bag-файла. Название топика '/points_raw' является 
        # индустриальным стандартом. Если жюри укажет другой — его можно сменить в config.py.
        self.subscription = self.create_subscription(
            PointCloud2,
            '/points_raw', 
            self.lidar_callback,
            10  # Размер очереди сообщений
        )
        
        # 2. Инициализируем межкадровые движки
        self.odometry_engine = StableLidarOdometryV12()
        self.obstacle_tracker_engine = LidarObstacleTrackerV2()
        self.frame_idx = 0
        self.dt = 0.1  # 10 Гц Hesai 128
        
        self.get_logger().info('🚇 [ИИ-ЯДРО МЕТРО]: ROS 2 Нода успешно запущена и ожидает поток "ros2 bag play"...')

    def lidar_callback(self, msg):
        """Автоматический триггер ROS 2 на каждый прилетающий лазерный кадр"""
        
        # Вытаскиваем координаты X, Y, Z и Интенсивность (колонка 3) из сообщения PointCloud2
        try:
            # Читаем поля структурированного облака жюри
            gen = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
            pts_list = list(gen)
            if not pts_list:
                return
            raw_points = np.array(pts_list, dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Ошибка десериализации PointCloud2: {e}")
            return

        # -----------------------------------------------------------------
        # КОНТУР АКТУАЛЬНОЙ ОДОМЕТРИИ (Физический шаг поезда)
        # -----------------------------------------------------------------
        shift_z_rails, rail_passport = self.odometry_engine.compute_raw_rail_odo_shift(raw_points, self.dt)
        
        # Стены туннеля (Извлекаем макро-облако напрямую из живого NumPy массива)
        # Примечание: Убедитесь, что метод extract_clean_macro_tunnel принимает массив напрямую, 
        # либо адаптируйте его под живой поток вместо чтения .bin файлов.
        shift_z_walls = 0.0
        macro_cloud = self.odometry_engine.extract_clean_macro_tunnel(raw_points)
        if macro_cloud is not None:
            passports = self.odometry_engine.build_passports_via_dbscan(macro_cloud)
            calculate_speed_trigger = bool(self.frame_idx > 0)
            shift_z_walls, _, _ = self.odometry_engine.associate_and_calculate_shift(
                passports, self.dt, calculate_speed=calculate_speed_trigger
            )

        # Подсчет точек в створе для ИИ-шлюза слияния весов
        x_pts, y_pts, z_pts = raw_points[:, 2], raw_points[:, 0], raw_points[:, 1]
        rail_points_mask = (z_pts >= -63.5) & (z_pts <= -3.5) & \
                           (x_pts >= -0.75) & (x_pts <= 0.75) & \
                           (y_pts >= -1.85) & (y_pts <= -1.05)
        current_rail_points_count = int(np.sum(rail_points_mask))

        # Итоговый физический шаг состава вперед от ИИ-шлюза
        shift_z_physical = cos_processor_v12.calculate_adaptive_fusion_shift(
            shift_z_rails, shift_z_walls, self.odometry_engine.prev_velocity_kmh, 
            current_rail_points_count, self.frame_idx, self.dt
        )
        
        calculated_speed_kmh = (shift_z_physical / self.dt) * 3.6
        if calculated_speed_kmh < 0.2:
            calculated_speed_kmh = 0.0
            shift_z_physical = 0.0
            
        self.odometry_engine.prev_velocity_kmh = calculated_speed_kmh

        # -----------------------------------------------------------------
        # КОНТУР БЕЗОПАСНОСТИ (Вызываем детекцию с пробросом шага)
        # -----------------------------------------------------------------
        # Проверяем гео-шлюз разлета стен туннеля
        is_open_space = False
        valid_walls_anchors = [obj for obj in self.odometry_engine.anchor_map if obj.get("status") == "STATUS_VALID_ANCHOR" and obj.get("is_wall", False)]
        if valid_walls_anchors:
            wall_x_coords = [float(w_obj["centroid"]) for w_obj in valid_walls_anchors if "centroid" in w_obj]
            if wall_x_coords and (max(wall_x_coords) > config.V2_OBSTACLE_WALL_OPEN_THRESHOLD or min(wall_x_coords) < -config.V2_OBSTACLE_WALL_OPEN_THRESHOLD):
                is_open_space = True

        # Ваша покадровая детекция (Передаем живой массив точек и шаг одометрии!)
        # Примечание: Убедитесь, что process_point_cloud в Вашем чистовике адаптирован 
        # под прием готового NumPy-массива вместо file_path.
        detected_obstacles = process_point_cloud(raw_points, self.obstacle_tracker_engine, train_step_z=shift_z_physical, is_open_space=is_open_space)

        # -----------------------------------------------------------------
        # КРИТЕРИЙ ТЗ №1 и №2: ЖИВОЙ СИНХРОННЫЙ СТАТУС В КОНСОЛЬ ЖЮРИ
        # -----------------------------------------------------------------
        if len(detected_obstacles) > 0:
            # Находим минимальную дистанцию до ближайшей угрозы (Требование ТЗ)
            min_dist = min([abs(float(obj["center"][2])) for obj in detected_obstacles])
            closest_obs = detected_obstacles[0]
            
            # Warn-логирование подсветит строку ярким желтым/красным цветом в терминале инженера метро
            self.get_logger().warn(
                f"🚨 [КАДР #{self.frame_idx:04d}]: ПРЕПЯТСТВИЕ ОБНАРУЖЕНО! | "
                f"📏 Дистанция до угрозы: {min_dist:.1f} м | "
                f"📦 Форма: {closest_obs.get('shape_text', 'Объект')} | "
                f" Позиция: {closest_obs.get('position_text', 'В колее')} | "
                f"📈 Скорость поезда: {calculated_speed_kmh:.1f} км/ч"
            )
        else:
            # Info-логирование для чистых путей (зеленый/белый цвет)
            self.get_logger().info(
                f"🟢 [КАДР #{self.frame_idx:04d}]: ПУТЬ СВОБОДЕН | "
                f"📏 Дистанция: -1.0 м | "
                f"📈 Скорость поезда: {calculated_speed_kmh:.1f} км/ч"
            )

        self.frame_idx += 1
        gc.collect()

def main(args=None):
    rclpy.init(args=args)
    node = SubwayVisionCoreNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
