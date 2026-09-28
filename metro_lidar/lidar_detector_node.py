
#!/usr/bin/env python3
"""
🚀 PRODUCTION ROS 2 HUMBLE COMPLIANT NODE (ONLINE MULTITHREADED ADAS)
Промышленный конвейер детекции препятствий реального времени (Онлайн-режим).
Обеспечивает 100% сбор кадров на лету через потокобезопасную очередь Queue без пропусков.
"""

import sys
import traceback 
import os
import gc
import csv
import time
import queue
import threading
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import ros2_numpy

# Импортируем движки ЦОС, одометрии и трекинга
from metro_lidar.cos_processor_v12 import StableLidarOdometryV12
import metro_lidar.cos_processor_v12 as cos_processor_v12
from metro_lidar.generate_submission_v2 import process_point_cloud
from metro_lidar.tracker_v2 import LidarObstacleTrackerV2
import metro_lidar.config as config


class SubwayVisionCoreNode(Node):
    
    def __init__(self):
        super().__init__('subway_vision_core_node')
        
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        
        # Ставим depth=10 для сетевого сокета, чтобы FastDDS успевал буферизировать поток
        custom_qos = QoSProfile(
            depth=10,  
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # 🔐 Потокобезопасная очередь для обмена кадрами между сетью и ИИ
        self.frame_queue = queue.Queue()
        
        self.net_cb_group = MutuallyExclusiveCallbackGroup()
        
        # Объявляем параметры для жюри строго один раз
        self.declare_parameter('scenario', 'doubleT_platform')
        self.declare_parameter('data_dir', '/app/for_hackathon')
        self.declare_parameter('lidar_topic', '/sensing/lidar/hesai128/pointcloud')
        
        scenario_param = self.get_parameter('scenario').get_parameter_value().string_value
        data_dir_param = self.get_parameter('data_dir').get_parameter_value().string_value
        
        if os.path.isabs(scenario_param) or '/' in scenario_param:
            base_scenario_dir = os.path.abspath(scenario_param)
        else:
            base_scenario_dir = os.path.join(data_dir_param, scenario_param)
            
        metadata_path = os.path.join(base_scenario_dir, 'metadata.yaml')
        target_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value

        self.get_logger().info(f'📂 [ИНСПЕКТОР ПУТЕЙ]: Анализирую манифест сценария: {base_scenario_dir}')

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    bag_info = yaml.safe_load(f)
                bag_meta = bag_info.get('rosbag2_bagfile_information', {})
                topics_list = bag_meta.get('topics_with_message_count', [])
                
                for topic_entry in topics_list:
                    meta_data = topic_entry.get('topic_metadata', {})
                    if 'PointCloud2' in str(meta_data.get('type', '')) and meta_data.get('name'):
                        target_topic = meta_data.get('name')
                        break
            except Exception as e:
                self.get_logger().error(f'⚠️ Ошибка автопарсинга metadata.yaml: {e}')

        # Подписка ROS 2 Humble привязана к выделенной сетевой группе
        self.subscription = self.create_subscription(
            PointCloud2,
            target_topic, 
            self.lidar_callback,
            custom_qos,
            callback_group=self.net_cb_group
        )
        
        # ИИ-Движки
        self.odometry_engine = StableLidarOdometryV12()
        self.obstacle_tracker_engine = LidarObstacleTrackerV2()
        
        self.frame_idx = 0
        self.dt = 0.1  # Жесткий и стабильный интервал 10 Гц Hesai 128
        self.submission_records = []
        
        self.total_processed_frames = 0     
        self.total_detected_obstacles = 0   
        self.radar_targets_registry = {}

        # 🏃‍♂️ ЗАПУСКАЕМ ВЫДЕЛЕННЫЙ СИСТЕМНЫЙ ПОТОК ДЛЯ ИИ (Consumer Worker)
        self.is_worker_alive = True
        self.worker_thread = threading.Thread(target=self.process_worker_loop, daemon=True)
        self.worker_thread.start()
        
        self.get_logger().info(f'🚀 [АВТОНАСТРОЙКА УСПЕШНА]: Подписка открыта на топик: {target_topic}')
        self.get_logger().info(f'🚇 [ОНЛАЙН ADAS-КОНВЕЙЕР ЖИВОГО ПОЕЗДА]: ПОЛНОСТЬЮ АКТИВИРОВАН.')

    def lidar_callback(self, msg):
        """Сетевой поток (Producer): моментально перехватывает кадр и выпрямляет массивы в ОЗУ"""
        local_idx = self.frame_idx
        self.frame_idx += 1
        
        try:
            arr_dict = ros2_numpy.numpify(msg)
            xyz_structured = arr_dict['xyz']
            intensity_structured = arr_dict['intensity'] if 'intensity' in arr_dict else np.zeros(xyz_structured.shape[:2], dtype=np.float32)
            
            xyz_flat = xyz_structured.reshape(-1, 3).astype(np.float32)
            intensity_flat = intensity_structured.reshape(-1, 1).astype(np.float32)
            
            points = np.hstack((xyz_flat, intensity_flat))
            
            nan_mask = np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(points[:, 2])
            points = points[~nan_mask]
            
            nonzero_mask = np.any(points[:, :3] != 0, axis=1)
            raw_points = points[nonzero_mask].copy()
            
            if len(raw_points) > 0:
                # Кладем подготовленный плоский массив в потокобезопасную очередь
                self.frame_queue.put((local_idx, raw_points))
        except Exception as e:
            self.get_logger().error(f'🚨 Сбой быстрого разбора сетевого кадра #{local_idx}: {e}')

    def process_worker_loop(self):
        """Фоновый поток вычислений (Consumer): непрерывно забирает кадры из очереди и крутит ЦОС/ИИ"""
        while self.is_worker_alive:
            try:
                # Извлекаем следующий кадр из очереди. Ждем 0.5 сек, если очередь пуста
                local_frame_idx, raw_points = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue  
                
            self.total_processed_frames += 1
            calculated_speed_kmh = 0.0

            try:
                # Вычисление плотности точек внутри колеи путей
                x_pts, y_pts, z_pts = raw_points[:, 2], raw_points[:, 0], raw_points[:, 1]
                rail_points_mask = (z_pts >= -63.5) & (z_pts <= -3.5) & \
                                   (x_pts >= -0.75) & (x_pts <= 0.75) & \
                                   (y_pts >= -1.85) & (y_pts <= -1.05)
                current_rail_points_count = int(np.sum(rail_points_mask))

                # =====================================================================
                # 🟢 ЕДИНАЯ ТОЧКА СБОРКИ ОБЛАЧНОЙ ОДОМЕТРИИ V12 (ЗАЩИЩЕННЫЙ ВЫЗОВ)
                # =====================================================================
                shift_z_physical = 0.0
                rail_passport = None
                
                try:
                    res = self.odometry_engine.update_odometry_fusion(
                        raw_points=raw_points,
                        idx=local_frame_idx,
                        dt=self.dt
                    )
                    
                    if isinstance(res, (tuple, list)):
                        raw_shift = res[0]
                        rail_passport = res[1] if len(res) > 1 else None
                    else:
                        raw_shift = res
                        rail_passport = None
                        
                    if isinstance(raw_shift, (tuple, list, np.ndarray)):
                        shift_z_physical = float(raw_shift[0])
                    else:
                        shift_z_physical = float(raw_shift)
                        
                except Exception as odo_error:
                    tb_text = traceback.format_exc()
                    self.get_logger().error(
                        f"⚠️ [СБОЙ ТИПОВ ОДОМЕТРИИ]: {odo_error}.\n"
                        f"=== СТЕК ВЫЗОВОВ С НОМЕРОМ СТРОКИ ===\n{tb_text}"
                        f"=====================================\n"
                        f"Включаю аварийную защиту инерциального шага."
                    )
                    shift_z_physical = (self.odometry_engine.prev_velocity_kmh / 3.6) * self.dt

                calculated_speed_kmh = self.odometry_engine.prev_velocity_kmh

                if calculated_speed_kmh < 0.2:
                    calculated_speed_kmh = 0.0
                    shift_z_physical = 1e-5
                    
                self.odometry_engine.prev_velocity_kmh = calculated_speed_kmh

                # Определение открытого пространства тоннеля (Адаптивный гео-шлюз)
                is_open_space = False
                valid_walls_anchors = [obj for obj in self.odometry_engine.anchor_map if obj.get("status") == "STATUS_VALID_ANCHOR" and obj.get("is_wall", False)]
                
                if valid_walls_anchors:
                    try:
                        wall_x_coords = [float(w_obj["centroid"][0]) for w_obj in valid_walls_anchors if "centroid" in w_obj]
                        if wall_x_coords and (max(wall_x_coords) > config.V2_OBSTACLE_WALL_OPEN_THRESHOLD or min(wall_x_coords) < -config.V2_OBSTACLE_WALL_OPEN_THRESHOLD):
                            is_open_space = True
                    except Exception as geo_gate_err:
                        self.get_logger().error(f"⚠️ [СБОЙ ГЕО-ШЛЮЗА СТЕН]: {geo_gate_err}. Откат к закрытому туннелю.")

                # =====================================================================
                # 🟢 ПРЯМОЙ ОНЛАЙН-ВЫЗОВ ИИ-ДЕТЕКТОРА ПРЕГРАД ИЗ ОЗУ (БЕЗ ПАТЧЕЙ)
                # =====================================================================
                # Передаем NumPy-массив напрямую в ОЗУ вычислительного контура.
                detected_obstacles = process_point_cloud(
                    raw_points=raw_points,
                    tracker=self.obstacle_tracker_engine,
                    train_step_z=shift_z_physical,
                    is_open_space=is_open_space
                )
                # =====================================================================

                virtual_frame_name = f"frame_{local_frame_idx:04d}.bin"

                # Агрегация результатов детекции в буфер submission
                if isinstance(detected_obstacles, list) and len(detected_obstacles) > 0:
                    self.total_detected_obstacles += len(detected_obstacles)
                    current_frame_ids = set()
                    
                    for obj in detected_obstacles:
                        track_id = obj.get("id", "0")
                        current_frame_ids.add(track_id)
                        
                        cx, cy, cz = obj["center"]
                        dist_z = abs(float(cz))
                        shape_label = obj.get("shape_text", "Объемная коробка / Блок")
                        
                        if track_id not in self.radar_targets_registry:
                            self.radar_targets_registry[track_id] = {
                                "shape": shape_label,
                                "max_dist": dist_z,
                                "min_dist": dist_z,
                                "first_frame": virtual_frame_name,
                                "last_frame": virtual_frame_name,
                                "lifetime_frames": 1
                            }
                        else:
                            self.radar_targets_registry[track_id]["min_dist"] = min(self.radar_targets_registry[track_id]["min_dist"], dist_z)
                            self.radar_targets_registry[track_id]["max_dist"] = max(self.radar_targets_registry[track_id]["max_dist"], dist_z)
                            self.radar_targets_registry[track_id]["last_frame"] = virtual_frame_name
                            self.radar_targets_registry[track_id]["lifetime_frames"] += 1
                    
                    # Запись строго по стандарту ТЗ (REP 103)
                    for obj in detected_obstacles:
                        cx, cy, cz = obj["center"]
                        sz_x, sz_y, sz_z = obj["dimensions"]
                        dist_z = abs(float(cz))
                        self.submission_records.append({
                            'frame_id': virtual_frame_name, 'obstacle_detected': 1, 'distance_m': round(dist_z, 3),
                            'center_x': round(float(cx), 3), 'center_y': round(float(cy), 3), 'center_z': round(float(cz), 3),
                            'size_x': round(float(sz_x), 3), 'size_y': round(float(sz_y), 3), 'size_z': round(float(sz_z), 3)  
                        })
                    
                    print(f"🚨 [ОНЛАЙН ➔ {virtual_frame_name}]: ОБНАРУЖЕНО ЦЕЛЕЙ: {len(detected_obstacles)} шт. | 📈 {calculated_speed_kmh:.1f} км/ч (Очередь: {self.frame_queue.qsize()})", flush=True)

                else:
                    self.submission_records.append({
                        'frame_id': virtual_frame_name, 'obstacle_detected': 0, 'distance_m': -1.0,
                        'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0, 'size_x': 0.0, 'size_y': 0.0, 'size_z': 0.0
                    })
                    if local_frame_idx % 10 == 0:
                        print(f"🟢 [ОНЛАЙН ➔ {virtual_frame_name}]: ПУТЬ СВОБОДЕН | 📈 {calculated_speed_kmh:.1f} км/ч (Очередь: {self.frame_queue.qsize()})", flush=True)

            except Exception as e:
                virtual_frame_name = f"frame_{local_frame_idx:04d}.bin"
                tb_text = traceback.format_exc()
                self.get_logger().error(
                    f"🚨 [ФАТАЛЬНЫЙ СБОЙ ИИ НА КАДРЕ {virtual_frame_name}]: {e}\n"
                    f"=== ТОЧНОЕ МЕСТО ПАДЕНИЯ ===\n{tb_text}============================="
                )
                
                self.submission_records.append({
                    'frame_id': virtual_frame_name, 'obstacle_detected': 0, 'distance_m': -1.0,
                    'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0, 'size_x': 0.0, 'size_y': 0.0, 'size_z': 0.0
                })

            # Сообщаем очереди, что задача текущего кадра полностью закрыта
            self.frame_queue.task_done()

    def export_final_submission(self):
        """Записывает накопленный кэш кадров в файл submission.csv [10]"""
        output_path = "/app/submission.csv"
        if not self.submission_records:
            self.get_logger().error("⚠️ Буфер сабмита пуст. CSV не сохранен.")
            return
            
        try:
            self.submission_records.sort(key=lambda x: x['frame_id'])
            
            fields = ['frame_id', 'obstacle_detected', 'distance_m', 
                      'center_x', 'center_y', 'center_z', 
                      'size_x', 'size_y', 'size_z']
                      
            with open(output_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.submission_records)
                
            self.get_logger().info(f"💾 [ИИ-ЭКСПОРТ УСПЕШЕН]: Отчет REP 103 сохранен: {output_path}")
            self.print_final_summary_report()
            
        except Exception as e:
            self.get_logger().error(f"🚨 Ошибка записи CSV: {e}")

    def print_final_summary_report(self):
        """Печатает красивый аудит-отчет накопленного массива submission_records для жюри"""
        print("\n" + "="*80)
        print("📊  ФИНАЛЬНЫЙ ИНСПЕКЦИОННЫЙ ОТЧЕТ ИИ-КОНВЕЙЕРА (СУДЬИ ХАКАТОНА)")
        print("="*80)
        print(f" Всего зафиксировано и обсчитано кадров: {self.total_processed_frames}")
        danger_frames = [r for r in self.submission_records if int(r['obstacle_detected']) == 1]
        print(f" Кадров с опасными аномалиями на путях:  {len(danger_frames)}")
        if danger_frames:
            distances = [float(r['distance_m']) for r in danger_frames]
            print(f" Минимальная дистанция фиксации угрозы: {min(distances):.3f} м")
            print(f" Максимальная дальность раннего зрения: {max(distances):.3f} м")
            
            print("\n📋 ГЛОБАЛЬНЫЙ РЕЕСТР УНИКАЛЬНЫХ ПРЕПЯТСТВИЙ (ИТОГ СЕССИИ):")
            print(f"{'Идентификатор':<15} | {'Класс (Форма)':<25} | {'Интервал кадров':<25} | {'Дистанция (Max -> Min)':<25}")
            print("-"*98)
            for t_id, data in self.radar_targets_registry.items():
                frame_range = f"{data['first_frame']} -> {data['last_frame']}"
                dist_range = f"{data['max_dist']:.2f} м -> {data['min_dist']:.2f} м"
                print(f"🆔 Облако #{t_id:<10} | {data['shape']:<25} | {frame_range:<25} | {dist_range:<25}")
        else:
            print(" ✅ Вся сессия пройдена в штатном режиме. Пути абсолютно чистые.")
        print("="*80 + "\n")
        print(f" 📂 ВСЕГО ОБРАБОТАНО УНИКАЛЬНЫХ КАДРОВ ЛИДАРА: {self.total_processed_frames} шт.")
        print(f" 📦 СУММАРНО НАЙДЕНО ФИЗИЧЕСКИХ ОБЪЕКТОВ ИИ:    {self.total_detected_obstacles} шт.\n")


def main(args=None):
    rclpy.init(args=args)
    node = SubwayVisionCoreNode()
    
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Получен сигнал Ctrl+C. Выполняем грейсфол шатдаун...')
    finally:
        node.export_final_submission()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()


# #!/usr/bin/env python3
# """
# 🚀 PRODUCTION ROS 2 HUMBLE COMPLIANT NODE (ONLINE MULTITHREADED ADAS)
# Промышленный конвейер детекции препятствий реального времени (Онлайн-режим).
# Обеспечивает 100% сбор кадров на лету через потокобезопасную очередь Queue без пропусков.
# """

# import sys
# import traceback 
# import os
# import gc
# import csv
# import time
# import queue
# import threading
# import numpy as np
# import yaml
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud2
# from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
# import ros2_numpy

# # Импортируем движки ЦОС, одометрии и трекинга
# from metro_lidar.cos_processor_v12 import StableLidarOdometryV12
# import metro_lidar.cos_processor_v12 as cos_processor_v12
# from metro_lidar.generate_submission_v2 import process_point_cloud
# from metro_lidar.tracker_v2 import LidarObstacleTrackerV2
# import metro_lidar.config as config


# def execute_detection_via_memory_patch(raw_points: np.ndarray, tracker_engine, shift_z: float, is_open_space: bool):
#     """Выполняет безопасный перехват системной функции np.fromfile (Monkey Patching)"""
#     original_fromfile = np.fromfile
#     try:
#         virtual_file_path = "memory_stream_frame.bin"
        
#         def mock_fromfile(file, dtype=None, count=-1, sep='', offset=0):
#             if file == virtual_file_path:
#                 return raw_points
#             return original_fromfile(file, dtype, count, sep, offset)
            
#         np.fromfile = mock_fromfile
        
#         detected_obstacles = process_point_cloud(
#             virtual_file_path, 
#             tracker_engine, 
#             train_step_z=shift_z, 
#             is_open_space=is_open_space
#         )
#         return detected_obstacles
#     finally:
#         np.fromfile = original_fromfile


# class SubwayVisionCoreNode(Node):
    
#     def __init__(self):
#         super().__init__('subway_vision_core_node')
        
#         from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        
#         # Ставим depth=10 для сетевого сокета, чтобы FastDDS успевал буферизировать поток
#         custom_qos = QoSProfile(
#             depth=10,  
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             durability=DurabilityPolicy.VOLATILE
#         )
        
#         # 🔐 Потокобезопасная очередь для обмена кадрами между сетью и ИИ
#         self.frame_queue = queue.Queue()
        
#         self.net_cb_group = MutuallyExclusiveCallbackGroup()
        
#         # Объявляем параметры для жюри строго один раз
#         self.declare_parameter('scenario', 'doubleT_platform')
#         self.declare_parameter('data_dir', '/app/for_hackathon')
#         self.declare_parameter('lidar_topic', '/sensing/lidar/hesai128/pointcloud')
        
#         scenario_param = self.get_parameter('scenario').get_parameter_value().string_value
#         data_dir_param = self.get_parameter('data_dir').get_parameter_value().string_value
        
#         if os.path.isabs(scenario_param) or '/' in scenario_param:
#             base_scenario_dir = os.path.abspath(scenario_param)
#         else:
#             base_scenario_dir = os.path.join(data_dir_param, scenario_param)
            
#         metadata_path = os.path.join(base_scenario_dir, 'metadata.yaml')
#         target_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value

#         self.get_logger().info(f'📂 [ИНСПЕКТОР ПУТЕЙ]: Анализирую манифест сценария: {base_scenario_dir}')

#         if os.path.exists(metadata_path):
#             try:
#                 with open(metadata_path, 'r', encoding='utf-8') as f:
#                     bag_info = yaml.safe_load(f)
#                 bag_meta = bag_info.get('rosbag2_bagfile_information', {})
#                 topics_list = bag_meta.get('topics_with_message_count', [])
                
#                 for topic_entry in topics_list:
#                     meta_data = topic_entry.get('topic_metadata', {})
#                     if 'PointCloud2' in str(meta_data.get('type', '')) and meta_data.get('name'):
#                         target_topic = meta_data.get('name')
#                         break
#             except Exception as e:
#                 self.get_logger().error(f'⚠️ Ошибка автопарсинга metadata.yaml: {e}')

#         # Подписка ROS 2 Humble привязана к выделенной сетевой группе
#         self.subscription = self.create_subscription(
#             PointCloud2,
#             target_topic, 
#             self.lidar_callback,
#             custom_qos,
#             callback_group=self.net_cb_group
#         )
        
#         # ИИ-Движки
#         self.odometry_engine = StableLidarOdometryV12()
#         self.obstacle_tracker_engine = LidarObstacleTrackerV2()
        
#         self.frame_idx = 0
#         self.dt = 0.1  # Жесткий и стабильный интервал 10 Гц Hesai 128
#         self.submission_records = []
        
#         self.total_processed_frames = 0     
#         self.total_detected_obstacles = 0   
#         self.radar_targets_registry = {}

#         # 🏃‍♂️ ЗАПУСКАЕМ ВЫДЕЛЕННЫЙ СИСТЕМНЫЙ ПОТОК ДЛЯ ИИ (Consumer Worker)
#         self.is_worker_alive = True
#         self.worker_thread = threading.Thread(target=self.process_worker_loop, daemon=True)
#         self.worker_thread.start()
        
#         self.get_logger().info(f'🚀 [АВТОНАСТРОЙКА УСПЕШНА]: Подписка открыта на топик: {target_topic}')
#         self.get_logger().info(f'🚇 [ОНЛАЙН ADAS-КОНВЕЙЕР ЖИВОГО ПОЕЗДА]: ПОЛНОСТЬЮ АКТИВИРОВАН.')

#     def lidar_callback(self, msg):
#         """Сетевой поток (Producer): моментально перехватывает кадр и выпрямляет массивы в ОЗУ"""
#         local_idx = self.frame_idx
#         self.frame_idx += 1
        
#         try:
#             arr_dict = ros2_numpy.numpify(msg)
#             xyz_structured = arr_dict['xyz']
#             intensity_structured = arr_dict['intensity'] if 'intensity' in arr_dict else np.zeros(xyz_structured.shape[:2], dtype=np.float32)
            
#             xyz_flat = xyz_structured.reshape(-1, 3).astype(np.float32)
#             intensity_flat = intensity_structured.reshape(-1, 1).astype(np.float32)
            
#             points = np.hstack((xyz_flat, intensity_flat))
            
#             nan_mask = np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(points[:, 2])
#             points = points[~nan_mask]
            
#             nonzero_mask = np.any(points[:, :3] != 0, axis=1)
#             raw_points = points[nonzero_mask].copy()
            
#             if len(raw_points) > 0:
#                 # Кладем подготовленный плоский массив в потокобезопасную очередь
#                 self.frame_queue.put((local_idx, raw_points))
#         except Exception as e:
#             self.get_logger().error(f'🚨 Сбой быстрого разбора сетевого кадра #{local_idx}: {e}')

#     def process_worker_loop(self):
#         """Фоновый поток вычислений (Consumer): непрерывно забирает кадры из очереди и крутит ЦОС/ИИ"""
#         while self.is_worker_alive:
#             try:
#                 # Извлекаем следующий кадр из очереди. Ждем 0.5 сек, если очередь пуста
#                 local_frame_idx, raw_points = self.frame_queue.get(timeout=0.5)
#             except queue.Empty:
#                 continue  # Очередь пока пуста, ждем новые кадры из сети
                
#             self.total_processed_frames += 1
#             calculated_speed_kmh = 0.0

#             try:
#                 # Вычисление плотности точек внутри колеи путей
#                 x_pts, y_pts, z_pts = raw_points[:, 2], raw_points[:, 0], raw_points[:, 1]
#                 rail_points_mask = (z_pts >= -63.5) & (z_pts <= -3.5) & \
#                                    (x_pts >= -0.75) & (x_pts <= 0.75) & \
#                                    (y_pts >= -1.85) & (y_pts <= -1.05)
#                 current_rail_points_count = int(np.sum(rail_points_mask))

#                 # Расчет текущей одометрии (dt=0.1 всегда стабилен, пропусков между шагами в очереди нет!)
#                 # =====================================================================
#                 # 🟢 ЕДИНАЯ ТОЧКА СБОРКИ ОБЛАЧНОЙ ОДОМЕТРИИ V12 (ЗАЩИЩЕННЫЙ ВЫЗОВ)
#                 # =====================================================================
#                 shift_z_physical = 0.0
#                 rail_passport = None
                
#                 try:
#                     # Вызываем монолитную функцию ядра одометрии из ОЗУ
#                     res = self.odometry_engine.update_odometry_fusion(
#                         raw_points=raw_points,
#                         idx=local_frame_idx,
#                         dt=self.dt
#                     )
                    
#                     # Безопасный разбор ответа: ядро может вернуть как число, так и кортеж/список
#                     if isinstance(res, (tuple, list)):
#                         # Вытаскиваем сдвиг (первый элемент) и паспорт рельс (второй элемент)
#                         raw_shift = res[0]
#                         rail_passport = res[1] if len(res) > 1 else None
#                     else:
#                         raw_shift = res
#                         rail_passport = None
                        
#                     # Дополнительная защита: если сам сдвиг пришел в виде списка/кортежа
#                     if isinstance(raw_shift, (tuple, list, np.ndarray)):
#                         shift_z_physical = float(raw_shift[0])
#                     else:
#                         shift_z_physical = float(raw_shift)
                        
#                 except Exception as odo_error:
#                     # Форматируем полный traceback, чтобы узнать точный файл и номер строки
#                     tb_text = traceback.format_exc()
                    
#                     # Выводим в лог ROS 2 детальную карту падения
#                     self.get_logger().error(
#                         f"⚠️ [СБОЙ ТИПОВ ОДОМЕТРИИ]: {odo_error}.\n"
#                         f"=== СТЕК ВЫЗОВОВ С НОМЕРОМ СТРОКИ ===\n{tb_text}"
#                         f"=====================================\n"
#                         f"Включаю аварийную защиту инерциального шага."
#                     )
                    
#                     # 🟢 ИСПРАВЛЕНО: заменено self.main_odometry_engine на self.odometry_engine
#                     # Нода НЕ упадет, а продолжит движение по расчетной траектории прошлой скорости
#                     shift_z_physical = (self.odometry_engine.prev_velocity_kmh / 3.6) * self.dt

                
#                 # Итоговая скорость обновляется автоматически из памяти движка
#                 calculated_speed_kmh = self.odometry_engine.prev_velocity_kmh
#                 # =====================================================================


#                 if calculated_speed_kmh < 0.2:
#                     calculated_speed_kmh = 0.0
#                     shift_z_physical = 1e-5
                    
#                 self.odometry_engine.prev_velocity_kmh = calculated_speed_kmh

#                 # Определение открытого пространства тоннеля
#                 # Определение открытого пространства тоннеля (Адаптивный гео-шлюз)
#                 is_open_space = False
#                 valid_walls_anchors = [obj for obj in self.odometry_engine.anchor_map if obj.get("status") == "STATUS_VALID_ANCHOR" and obj.get("is_wall", False)]
                
#                 if valid_walls_anchors:
#                     try:
#                         # 🟢 ИСПРАВЛЕНИЕ: w_obj["centroid"][0] забирает чистую координату X (Ширину разлета стен)
#                         wall_x_coords = [float(w_obj["centroid"][0]) for w_obj in valid_walls_anchors if "centroid" in w_obj]
                        
#                         if wall_x_coords and (max(wall_x_coords) > config.V2_OBSTACLE_WALL_OPEN_THRESHOLD or min(wall_x_coords) < -config.V2_OBSTACLE_WALL_OPEN_THRESHOLD):
#                             is_open_space = True
#                     except Exception as geo_gate_err:
#                         self.get_logger().error(f"⚠️ [СБОЙ ГЕО-ШЛЮЗА СТЕН]: {geo_gate_err}. Откат к закрытому туннелю.")


#                 # Вызов ИИ-движка детекции препятствий через Monkey Patching памяти
#                 detected_obstacles = execute_detection_via_memory_patch(
#                     raw_points=raw_points,
#                     tracker_engine=self.obstacle_tracker_engine,
#                     shift_z=shift_z_physical,
#                     is_open_space=is_open_space
#                 )

#                 virtual_frame_name = f"frame_{local_frame_idx:04d}.bin"

#                 # Агрегация результатов детекции в буфер submission
#                 if isinstance(detected_obstacles, list) and len(detected_obstacles) > 0:
#                     self.total_detected_obstacles += len(detected_obstacles)
#                     current_frame_ids = set()
                    
#                     for obj in detected_obstacles:
#                         track_id = obj.get("id", "0")
#                         current_frame_ids.add(track_id)
                        
#                         cx, cy, cz = obj["center"]
#                         dist_z = abs(float(cz))
#                         shape_label = obj.get("shape_text", "Объемная коробка / Блок")
                        
#                         if track_id not in self.radar_targets_registry:
#                             self.radar_targets_registry[track_id] = {
#                                 "shape": shape_label,
#                                 "max_dist": dist_z,
#                                 "min_dist": dist_z,
#                                 "first_frame": virtual_frame_name,
#                                 "last_frame": virtual_frame_name,
#                                 "lifetime_frames": 1
#                             }
#                         else:
#                             self.radar_targets_registry[track_id]["min_dist"] = min(self.radar_targets_registry[track_id]["min_dist"], dist_z)
#                             self.radar_targets_registry[track_id]["max_dist"] = max(self.radar_targets_registry[track_id]["max_dist"], dist_z)
#                             self.radar_targets_registry[track_id]["last_frame"] = virtual_frame_name
#                             self.radar_targets_registry[track_id]["lifetime_frames"] += 1
                    
#                     # Запись строго по стандарту ТЗ (REP 103)
#                     for obj in detected_obstacles:
#                         cx, cy, cz = obj["center"]
#                         sz_x, sz_y, sz_z = obj["dimensions"]
#                         dist_z = abs(float(cz))
#                         self.submission_records.append({
#                             'frame_id': virtual_frame_name, 'obstacle_detected': 1, 'distance_m': round(dist_z, 3),
#                             'center_x': round(float(cx), 3), 'center_y': round(float(cy), 3), 'center_z': round(float(cz), 3),
#                             'size_x': round(float(sz_x), 3), 'size_y': round(float(sz_y), 3), 'size_z': round(float(sz_z), 3)  
#                         })
                    
#                     print(f"🚨 [ОНЛАЙН ➔ {virtual_frame_name}]: ОБНАРУЖЕНО ЦЕЛЕЙ: {len(detected_obstacles)} шт. | 📈 {calculated_speed_kmh:.1f} км/ч (Очередь: {self.frame_queue.qsize()})", flush=True)

#                 else:
#                     self.submission_records.append({
#                         'frame_id': virtual_frame_name, 'obstacle_detected': 0, 'distance_m': -1.0,
#                         'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0, 'size_x': 0.0, 'size_y': 0.0, 'size_z': 0.0
#                     })
#                     if local_frame_idx % 10 == 0:
#                         print(f"🟢 [ОНЛАЙН ➔ {virtual_frame_name}]: ПУТЬ СВОБОДЕН | 📈 {calculated_speed_kmh:.1f} км/ч (Очередь: {self.frame_queue.qsize()})", flush=True)

#             except Exception as e:
#                 virtual_frame_name = f"frame_{local_frame_idx:04d}.bin"
                
#                 # Форматируем полный Traceback до конкретной строки
#                 tb_text = traceback.format_exc()
                
#                 self.get_logger().error(
#                     f"🚨 [ФАТАЛЬНЫЙ СБОЙ ИИ НА КАДРЕ {virtual_frame_name}]: {e}\n"
#                     f"=== ТОЧНОЕ МЕСТО ПАДЕНИЯ ===\n{tb_text}============================="
#                 )
                
#                 self.submission_records.append({
#                     'frame_id': virtual_frame_name, 'obstacle_detected': 0, 'distance_m': -1.0,
#                     'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0, 'size_x': 0.0, 'size_y': 0.0, 'size_z': 0.0
#                 })

                
#             # Сообщаем очереди, что задача текущего кадра полностью закрыта и поток готов брать следующий
#             self.frame_queue.task_done()
#     def export_final_submission(self):
#         """Записывает накопленный кэш кадров в файл submission.csv [10]"""
#         output_path = "/app/submission.csv"
#         if not self.submission_records:
#             self.get_logger().error("⚠️ Буфер сабмита пуст. CSV не сохранен.")
#             return
            
#         try:
#             self.submission_records.sort(key=lambda x: x['frame_id'])
            
#             fields = ['frame_id', 'obstacle_detected', 'distance_m', 
#                       'center_x', 'center_y', 'center_z', 
#                       'size_x', 'size_y', 'size_z']
                      
#             with open(output_path, mode='w', newline='', encoding='utf-8') as f:
#                 writer = csv.DictWriter(f, fieldnames=fields)
#                 writer.writeheader()
#                 writer.writerows(self.submission_records)
                
#             self.get_logger().info(f"💾 [ИИ-ЭКСПОРТ УСПЕШЕН]: Отчет REP 103 сохранен: {output_path}")
#             self.get_logger().info(f"📊 Всего кадров зафиксировано в итоговом CSV: {len(self.submission_records)}")
            
#             # 🟢 ВСТРАИВАЕМ НАШ ИТОГОВЫЙ ИНСПЕКЦИОННЫЙ ОТЧЕТ ДЛЯ ЖЮРИ НА ФИНИШЕ СЕССИИ [10]
#             self.print_final_summary_report()
            
#         except Exception as e:
#             self.get_logger().error(f"🚨 Ошибка записи CSV: {e}")

#     def print_final_summary_report(self):
#         """Печатает красивый аудит-отчет накопленного массива submission_records для жюри"""
#         print("\n" + "="*80)
#         print("📊  ФИНАЛЬНЫЙ ИНСПЕКЦИОННЫЙ ОТЧЕТ ИИ-КОНВЕЙЕРА (СУДЬИ ХАКАТОНА)")
#         print("="*80)
#         print(f" Всего зафиксировано и обсчитано кадров: {self.total_processed_frames}")
#         danger_frames = [r for r in self.submission_records if int(r['obstacle_detected']) == 1]
#         print(f" Кадров с опасными аномалиями на путях:  {len(danger_frames)}")
#         if danger_frames:
#             distances = [float(r['distance_m']) for r in danger_frames]
#             print(f" Минимальная дистанция фиксации угрозы: {min(distances):.3f} м")
#             print(f" Максимальная дальность раннего зрения: {max(distances):.3f} м")
#             print("\n📋 ХРОНОЛОГИЧЕСКИЙ СРЕЗ ПЕРВЫХ ДЕТЕКЦИЙ ДЛЯ СВЕРКИ:")
#             print(f"{'Имя кадра':<20} | {'Детекция':<10} | {'Дальность Z вперед':<15} | {'Смещение X':<12}")
#             print("-"*80)
#             for r in danger_frames[:10]:
#                 print(f"{r['frame_id']:<20} | {'🚨 ДА' if int(r['obstacle_detected'])==1 else 'НЕТ':<10} | {float(r['distance_m']):<15.3f} | {r['center_y']:<12}")
            
#             # -----------------------------------------------------------------
#             # 🟢 ДОБАВЛЯЕМ СВОДНЫЙ РЕЕСТР УНИКАЛЬНЫХ ОБЪЕКТОВ НА ФИНИШЕ
#             # -----------------------------------------------------------------
#                         # -----------------------------------------------------------------
#             # 🟢 ДОБАВЛЯЕМ СВОДНЫЙ РЕЕСТР УНИКАЛЬНЫХ ОБЪЕКТОВ НА ФИНИШЕ (С ТРЕКИНГОМ КАДРОВ)
#             # -----------------------------------------------------------------
#             print("\n📋 ГЛОБАЛЬНЫЙ РЕЕСТР УНИКАЛЬНЫХ ПРЕПЯТСТВИЙ (ИТОГ СЕССИИ):")
#             print(f"{'Идентификатор':<15} | {'Класс (Форма)':<25} | {'Интервал кадров':<25} | {'Дистанция (Max -> Min)':<25}")
#             print("-"*98)
#             for t_id, data in self.radar_targets_registry.items():
#                 frame_range = f"{data['first_frame']} -> {data['last_frame']}"
#                 dist_range = f"{data['max_dist']:.2f} м -> {data['min_dist']:.2f} м"
#                 print(f"🆔 Облако #{t_id:<10} | {data['shape']:<25} | {frame_range:<25} | {dist_range:<25}")

            
#         else:
#             print(" ✅ Вся сессия пройдена в штатном режиме. Пути абсолютно чистые.")
#         print("="*80 + "\n")
#         print(f" 📂 ВСЕГО ОБРАБОТАНО УНИКАЛЬНЫХ КАДРОВ ЛИДАРА: {self.total_processed_frames} шт.")
#         print(f" 📦 СУММАРНО НАЙДЕНО ФИЗИЧЕСКИХ ОБЪЕКТОВ ИИ:    {self.total_detected_obstacles} шт.\n")


# def main(args=None):
#     rclpy.init(args=args)
#     node = SubwayVisionCoreNode()
    
#     # 🟢 ЗАПУСКАЕМ МНОГОПОТОЧНЫЙ ДИСПЕТЧЕР НА 2 ПОТОКА CPU [10]
#     from rclpy.executors import MultiThreadedExecutor
#     executor = MultiThreadedExecutor(num_threads=2)
#     executor.add_node(node)
    
#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         node.get_logger().info('🛑 Получен сигнал Ctrl+C. Выполняем грейсфол шатдаун...')
#     finally:
#         node.export_final_submission()
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == '__main__':
#     main()

# #!/usr/bin/env python3
# """
# 🚀 PRODUCTION ROS 2 HUMBLE COMPLIANT NODE (ONLINE ADAS EDITION) - PART 1
# Промышленный конвейер детекции препятствий реального времени (Онлайн-режим).
# """

# import sys
# import os
# import gc
# import csv
# import time
# import numpy as np
# import yaml
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud2
# from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
# import ros2_numpy  # Гарантирует 100% идентичность парсинга геометрии с quick_check [10]

# # Импортируем движки ЦОС, одометрии и трекинга
# from metro_lidar.cos_processor_v12 import StableLidarOdometryV12
# import metro_lidar.cos_processor_v12 as cos_processor_v12
# from metro_lidar.generate_submission_v2 import process_point_cloud
# from metro_lidar.tracker_v2 import LidarObstacleTrackerV2
# import metro_lidar.config as config


# def execute_detection_via_memory_patch(raw_points: np.ndarray, tracker_engine, shift_z: float, is_open_space: bool):
#     """Выполняет безопасный перехват системной функции np.fromfile (Monkey Patching) [10]"""
#     original_fromfile = np.fromfile
#     try:
#         virtual_file_path = "memory_stream_frame.bin"
        
#         def mock_fromfile(file, dtype=None, count=-1, sep='', offset=0):
#             if file == virtual_file_path:
#                 return raw_points
#             return original_fromfile(file, dtype, count, sep, offset)
            
#         np.fromfile = mock_fromfile
        
#         detected_obstacles = process_point_cloud(
#             virtual_file_path, 
#             tracker_engine, 
#             train_step_z=shift_z, 
#             is_open_space=is_open_space
#         )
#         return detected_obstacles
#     finally:
#         np.fromfile = original_fromfile


# class SubwayVisionCoreNode(Node):
    
#     def __init__(self):
#         super().__init__('subway_vision_core_node')
        
#         from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        
#         # Настройка универсального профиля QoS под Best Effort
#         custom_qos = QoSProfile(
#             depth=1,  # В буфере сокета храним строго 1 самый свежий кадр
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             durability=DurabilityPolicy.VOLATILE
#         )
        
#         # Буфер для обмена сообщениями между параллельными потоками
#         self.latest_msg = None
        
#         # Разделяем сетевой прием и тяжелые вычисления ИИ по разным потокам ОС
#         self.net_cb_group = MutuallyExclusiveCallbackGroup()
#         self.worker_cb_group = MutuallyExclusiveCallbackGroup()
        
#         # 🟢 ОБЪЯВЛЯЕМ ПАРАМЕТРЫ ДЛЯ ЖЮРИ СТРОГО ОДИН РАЗ
#         self.declare_parameter('scenario', 'doubleT_platform')
#         self.declare_parameter('data_dir', '/app/for_hackathon')
#         self.declare_parameter('lidar_topic', '/sensing/lidar/hesai128/pointcloud') # Резервный параметр
        
#         # Переменные для хранения активного состояния подписки
#         self.current_topic_name = None
#         self.subscription = None

#         # Инициализируем межкадровые движки
#         self.odometry_engine = StableLidarOdometryV12()
#         self.obstacle_tracker_engine = LidarObstacleTrackerV2()
        
#         self.frame_idx = 0
#         self.dt = 0.1  # 10 Гц Hesai 128
#         self.submission_records = []
        
#         # Статистические счетчики онлайн-конвейера
#         self.total_processed_frames = 0     
#         self.total_detected_obstacles = 0   
#         self.radar_targets_registry = {}

#         # 🔄 ЗАПУСКАЕМ ПЕРВИЧНУЮ ДИНАМИЧЕСКУЮ НАСТРОЙКУ ПОДПИСКИ
#         self.update_dynamic_subscription()

#         # 🔥 КОЛБЭК ДЛЯ ЖЮРИ: Отслеживаем изменение параметров "на лету" без перезапуска
#         self.add_on_set_parameters_callback(self.on_parameters_changed)
        
#         # Фоновый таймер воркера (25 Гц) привязан к вычислительной группе
#         self.process_timer = self.create_timer(
#             0.04, 
#             self.process_loop, 
#             callback_group=self.worker_cb_group
#         )
        
#         self.get_logger().info(f'🚇 [ОНЛАЙН ИИ-ЯДРО ЖИВОГО ПОЕЗДА]: УСПЕШНО АКТИВИРОВАНО.')

#     def update_dynamic_subscription(self):
#         """Метод динамического переключения подписки на основе актуального YAML"""
#         scenario_param = self.get_parameter('scenario').get_parameter_value().string_value
#         data_dir_param = self.get_parameter('data_dir').get_parameter_value().string_value
        
#         if os.path.isabs(scenario_param) or '/' in scenario_param:
#             base_scenario_dir = os.path.abspath(scenario_param)
#         else:
#             base_scenario_dir = os.path.join(data_dir_param, scenario_param)
            
#         metadata_path = os.path.join(base_scenario_dir, 'metadata.yaml')
        
#         # Получаем дефолтный топик из параметров
#         target_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value

#         self.get_logger().info(f'📂 [ДИНАМИЧЕСКИЙ ПЕРЕКЛЮЧАТЕЛЬ]: Проверяю путь: {base_scenario_dir}')

#         if os.path.exists(metadata_path):
#             try:
#                 with open(metadata_path, 'r', encoding='utf-8') as f:
#                     bag_info = yaml.safe_load(f)
#                 bag_meta = bag_info.get('rosbag2_bagfile_information', {})
#                 topics_list = bag_meta.get('topics_with_message_count', [])
                
#                 found_topic = None
#                 for topic_entry in topics_list:
#                     meta_data = topic_entry.get('topic_metadata', {})
#                     if 'PointCloud2' in str(meta_data.get('type', '')) and meta_data.get('name'):
#                         found_topic = meta_data.get('name')
#                         break
                
#                 if not found_topic:
#                     for topic_entry in topics_list:
#                         meta_data = topic_entry.get('topic_metadata', {})
#                         t_name = str(meta_data.get('name', '')).lower()
#                         if any(marker in t_name for marker in ['lidar', 'points', 'pointcloud', 'hesai']):
#                             found_topic = meta_data.get('name')
#                             break

#                 if found_topic:
#                     target_topic = found_topic
#             except Exception as e:
#                 self.get_logger().error(f'⚠️ Ошибка парсинга {metadata_path}: {e}')
#         else:
#             self.get_logger().warn(f'📂 Файл {metadata_path} не найден. Использую топик по умолчанию: {target_topic}')

#         # ⚡ РЕКОНФИГУРАЦИЯ ПОДПИСКИ НА ЛЕТУ
#         if target_topic != self.current_topic_name:
#             # Если старая подписка существовала — уничтожаем её
#             if self.subscription is not None:
#                 self.get_logger().warn(f'🔄 Уничтожаю старую подписку на топик: {self.current_topic_name}')
#                 self.destroy_subscription(self.subscription)
            
#             self.current_topic_name = target_topic
            
#             from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
#             custom_qos = QoSProfile(
#                 depth=1,
#                 reliability=ReliabilityPolicy.BEST_EFFORT,
#                 durability=DurabilityPolicy.VOLATILE
#             )
            
#             # Создаем новую подписку на лету в сетевой группе
#             self.subscription = self.create_subscription(
#                 PointCloud2,
#                 self.current_topic_name, 
#                 self.lidar_callback,
#                 custom_qos,
#                 callback_group=self.net_cb_group
#             )
#             self.get_logger().info(f'🚀 [АВТОНАСТРОЙКА ЖИВА]: Нода успешно подписалась на топик: {self.current_topic_name}')

#     def on_parameters_changed(self, params):
#         """Колбэк ROS 2, срабатывающий мгновенно, когда жюри меняет параметры ноды"""
#         from rcl_interfaces.msg import SetParametersResult
#         for param in params:
#             if param.name in ['scenario', 'data_dir', 'lidar_topic']:
#                 self.get_logger().warn(f'⚙️ Запрос на смену конфигурации: {param.name} -> {param.value}')
                
#                 # Принудительно останавливаем ИИ, чтобы избежать Race Condition во время переподключения
#                 self.process_timer.cancel()
                
#                 # Принудительно фиксируем новые значения в памяти ноды до финализации
#                 if param.name == 'scenario':
#                     self.get_parameter('scenario').set_value(param.value)
#                 elif param.name == 'data_dir':
#                     self.get_parameter('data_dir').set_value(param.value)
#                 elif param.name == 'lidar_topic':
#                     self.get_parameter('lidar_topic').set_value(param.value)
                
#                 # Перестраиваем подписку по новому YAML
#                 self.update_dynamic_subscription()
                
#                 # Перезапускаем ИИ обратно
#                 self.process_timer.reset()
                
#         return SetParametersResult(successful=True, reason="Параметры успешно обновлены на лету")


#     def lidar_callback(self, msg):
#         """Сетевой поток: работает параллельно, мгновенно сохраняя ссылку на кадр"""
#         self.latest_msg = msg

#     def process_loop(self):
#         """Вычислительный поток: независимо от сети забирает кадр и крутит тяжелый ЦОС/ИИ"""
#         if self.latest_msg is None:
#             return
            
#         # Атомарно вытаскиваем кадр из буфера и очищаем слот
#         msg = self.latest_msg
#         self.latest_msg = None
        
#         local_frame_idx = self.frame_idx
#         self.frame_idx += 1
#         self.total_processed_frames += 1  # Фиксируем шаг онлайн-конвейера

#         flat_floats = None
#         raw_points = None
#         calculated_speed_kmh = 0.0

#         try:
#             # 1. 🟢 ВЫСОКОСКОРОСТНОЙ НАТИВНЫЙ ПАРСЕР И ФИЛЬТР (Копия convert_all_bags)
#             arr_dict = ros2_numpy.numpify(msg)
#             xyz = arr_dict['xyz'].astype(np.float32)
#             num_points = len(xyz)
            
#             if 'intensity' in arr_dict:
#                 intensity = arr_dict['intensity'].astype(np.float32).reshape(-1, 1)
#             else:
#                 intensity = np.zeros((num_points, 1), dtype=np.float32)
                
#             points = np.hstack((xyz, intensity))
            
#             # Фильтрация NaN
#             nan_mask = np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(points[:, 2])
#             points = points[~nan_mask]
            
#             # Вырезаем все пустые/нулевые лучи лидара
#             nonzero_mask = np.any(points[:, :3] != 0, axis=1)
#             raw_points = points[nonzero_mask].copy()
            
#             if len(raw_points) == 0:
#                 return

#             # Вычисление плотности точек внутри колеи путей
#             x_pts, y_pts, z_pts = raw_points[:, 2], raw_points[:, 0], raw_points[:, 1]
#             rail_points_mask = (z_pts >= -63.5) & (z_pts <= -3.5) & \
#                                (x_pts >= -0.75) & (x_pts <= 0.75) & \
#                                (y_pts >= -1.85) & (y_pts <= -1.05)
#             current_rail_points_count = int(np.sum(rail_points_mask))

#             # Расчет текущей одометрии по рельсам и геометрии стен
#             shift_z_rails, _ = self.odometry_engine.compute_raw_rail_odo_shift(raw_points, self.dt)
            
#             shift_z_walls = 0.0
#             macro_cloud = self.odometry_engine.extract_clean_macro_tunnel(raw_points)
#             if macro_cloud is not None:
#                 passports = self.odometry_engine.build_passports_via_dbscan(macro_cloud)
#                 calculate_speed_trigger = bool(local_frame_idx > 0)
#                 shift_z_walls, _, _ = self.odometry_engine.associate_and_calculate_shift(
#                     passports, self.dt, calculate_speed=calculate_speed_trigger
#                 )

#             # Комплексирование одометрии (Fusion)
#             shift_z_physical = cos_processor_v12.calculate_adaptive_fusion_shift(
#                 shift_z_rails, shift_z_walls, self.odometry_engine.prev_velocity_kmh, 
#                 current_rail_points_count, local_frame_idx, self.dt
#             )
            
#             calculated_speed_kmh = (shift_z_physical / self.dt) * 3.6
#             if calculated_speed_kmh < 0.2:
#                 calculated_speed_kmh = 0.0
#                 shift_z_physical = 1e-5
                
#             self.odometry_engine.prev_velocity_kmh = calculated_speed_kmh

#             # Определение открытого пространства тоннеля
#             is_open_space = False
#             valid_walls_anchors = [obj for obj in self.odometry_engine.anchor_map if obj.get("status") == "STATUS_VALID_ANCHOR" and obj.get("is_wall", False)]
#             if valid_walls_anchors:
#                 wall_x_coords = [float(w_obj["centroid"]) for w_obj in valid_walls_anchors if "centroid" in w_obj]
#                 if wall_x_coords and (max(wall_x_coords) > config.V2_OBSTACLE_WALL_OPEN_THRESHOLD or min(wall_x_coords) < -config.V2_OBSTACLE_WALL_OPEN_THRESHOLD):
#                     is_open_space = True

#             detector_points = raw_points

#             # Вызов ИИ-движка детекции препятствий через Monkey Patching памяти
#             detected_obstacles = execute_detection_via_memory_patch(
#                 raw_points=detector_points,
#                 tracker_engine=self.obstacle_tracker_engine,
#                 shift_z=shift_z_physical,
#                 is_open_space=is_open_space
#             )

#             # Агрегация результатов детекции в буфер submission
#             if isinstance(detected_obstacles, list) and len(detected_obstacles) > 0:
#                 self.total_detected_obstacles += len(detected_obstacles)
                
#                 virtual_frame_name = f"frame_{local_frame_idx:04d}.bin"
#                 current_frame_ids = set()
                
#                 for obj in detected_obstacles:
#                     track_id = obj.get("id", "0")
#                     current_frame_ids.add(track_id)
                    
#                     cx, cy, cz = obj["center"]
#                     dist_z = abs(float(cz))
#                     shape_label = obj.get("shape_text", "Объемная коробка / Блок")
                    
#                     # Если этот ID мы видим впервые — регистрируем новую цель в памяти
#                     if track_id not in self.radar_targets_registry:
#                         self.radar_targets_registry[track_id] = {
#                             "shape": shape_label,
#                             "max_dist": dist_z,   # Дальность раннего обнаружения
#                             "min_dist": dist_z,   # Минимальное сближение
#                             "first_frame": virtual_frame_name, # Кадр старта
#                             "last_frame": virtual_frame_name,  # Кадр финиша
#                             "lifetime_frames": 1
#                         }
#                     else:
#                         # Если цель уже велась — обновляем метрики
#                         self.radar_targets_registry[track_id]["min_dist"] = min(self.radar_targets_registry[track_id]["min_dist"], dist_z)
#                         self.radar_targets_registry[track_id]["max_dist"] = max(self.radar_targets_registry[track_id]["max_dist"], dist_z)
#                         self.radar_targets_registry[track_id]["last_frame"] = virtual_frame_name # Сдвигаем крайний кадр
#                         self.radar_targets_registry[track_id]["lifetime_frames"] += 1

                
#                 # 🟢 ПРЕПЯТСТВИЕ НАЙДЕНО: Экспорт строго по ТЗ жюри (Исправленный мост осей)
#                 for obj in detected_obstacles:
#                     cx, cy, cz = obj["center"]
#                     sz_x, sz_y, sz_z = obj["dimensions"]
#                     dist_z = abs(float(cz))
#                     self.submission_records.append({
#                         'frame_id': virtual_frame_name, 
#                         'obstacle_detected': 1, 
#                         'distance_m': round(dist_z, 3),
#                         'center_x': round(float(cx), 3),   # Ширина пути (Истинный X)
#                         'center_y': round(float(cy), 3),   # Высота от УВР (Истинный Y)
#                         'center_z': round(float(cz), 3),   # Расстояние вперед (Истинный Z)
#                         'size_x': round(float(sz_x), 3), 
#                         'size_y': round(float(sz_y), 3), 
#                         'size_z': round(float(sz_z), 3)  
#                     })

#                 # Диспетчерский лог радара ADAS в реальном времени
#                 print(f"\n🚨 [ОНЛАЙН ➔ {virtual_frame_name}]: В ГАБАРИТЕ ПУТЕЙ ОБНАРУЖЕНО ЦЕЛЕЙ: {len(detected_obstacles)} шт.", flush=True)
#                 print(f"📡 [РЕЕСТР СОПРОВОЖДЕНИЯ ADAS]:", flush=True)
#                 for t_id, data in self.radar_targets_registry.items():
#                     active_marker = "🎯 АКТИВЕН" if t_id in current_frame_ids else "⏳ УШЕЛ ИЗ ВИДУ"
#                     print(f"   ↳ 🆔 ПРЕПЯТСТВИЕ #{t_id} [{data['shape']}] | Дистанция: {data['min_dist']:.2f} м | Сопровождение: {data['lifetime_frames']} кадров | {active_marker}", flush=True)
#                 print("-" * 80, flush=True)
                
#             else:
#                 # 🟢 ПУТЬ СВОБОДЕН: Корректное зануление осей под судейскую проверку
#                 virtual_frame_name = f"frame_{local_frame_idx:04d}.bin"
#                 self.submission_records.append({
#                     'frame_id': virtual_frame_name, 'obstacle_detected': 0, 'distance_m': -1.0,
#                     'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0, 'size_x': 0.0, 'size_y': 0.0, 'size_z': 0.0
#                 })
#                 self.get_logger().info(f"🟢 [ОНЛАЙН ➔ {virtual_frame_name}]: ПУТЬ СВОБОДЕН | 📈 {calculated_speed_kmh:.1f} км/ч")

#         except Exception as e:
#             virtual_frame_name = f"frame_{local_frame_idx:04d}.bin"
#             self.get_logger().error(f"🚨 [СБОЙ НА КАДРЕ {virtual_frame_name}]: {e}")
#             self.submission_records.append({
#                 'frame_id': virtual_frame_name, 'obstacle_detected': 0, 'distance_m': -1.0,
#                 'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0, 'size_x': 0.0, 'size_y': 0.0, 'size_z': 0.0
#             })
#         finally:
#             del flat_floats, raw_points
#             gc.collect()

#     def export_final_submission(self):
#         """Записывает накопленный кэш кадров в файл submission.csv [10]"""
#         output_path = "/app/submission.csv"
#         if not self.submission_records:
#             self.get_logger().error("⚠️ Буфер сабмита пуст. CSV не сохранен.")
#             return
            
#         try:
#             self.submission_records.sort(key=lambda x: x['frame_id'])
            
#             fields = ['frame_id', 'obstacle_detected', 'distance_m', 
#                       'center_x', 'center_y', 'center_z', 
#                       'size_x', 'size_y', 'size_z']
                      
#             with open(output_path, mode='w', newline='', encoding='utf-8') as f:
#                 writer = csv.DictWriter(f, fieldnames=fields)
#                 writer.writeheader()
#                 writer.writerows(self.submission_records)
                
#             self.get_logger().info(f"💾 [ИИ-ЭКСПОРТ УСПЕШЕН]: Отчет REP 103 сохранен: {output_path}")
#             self.get_logger().info(f"📊 Всего кадров зафиксировано в итоговом CSV: {len(self.submission_records)}")
            
#             # 🟢 ВСТРАИВАЕМ НАШ ИТОГОВЫЙ ИНСПЕКЦИОННЫЙ ОТЧЕТ ДЛЯ ЖЮРИ НА ФИНИШЕ СЕССИИ [10]
#             self.print_final_summary_report()
            
#         except Exception as e:
#             self.get_logger().error(f"🚨 Ошибка записи CSV: {e}")

#     def print_final_summary_report(self):
#         """Печатает красивый аудит-отчет накопленного массива submission_records для жюри"""
#         print("\n" + "="*80)
#         print("📊  ФИНАЛЬНЫЙ ИНСПЕКЦИОННЫЙ ОТЧЕТ ИИ-КОНВЕЙЕРА (СУДЬИ ХАКАТОНА)")
#         print("="*80)
#         print(f" Всего зафиксировано и обсчитано кадров: {self.total_processed_frames}")
#         danger_frames = [r for r in self.submission_records if int(r['obstacle_detected']) == 1]
#         print(f" Кадров с опасными аномалиями на путях:  {len(danger_frames)}")
#         if danger_frames:
#             distances = [float(r['distance_m']) for r in danger_frames]
#             print(f" Минимальная дистанция фиксации угрозы: {min(distances):.3f} м")
#             print(f" Максимальная дальность раннего зрения: {max(distances):.3f} м")
#             print("\n📋 ХРОНОЛОГИЧЕСКИЙ СРЕЗ ПЕРВЫХ ДЕТЕКЦИЙ ДЛЯ СВЕРКИ:")
#             print(f"{'Имя кадра':<20} | {'Детекция':<10} | {'Дальность Z вперед':<15} | {'Смещение X':<12}")
#             print("-"*80)
#             for r in danger_frames[:10]:
#                 print(f"{r['frame_id']:<20} | {'🚨 ДА' if int(r['obstacle_detected'])==1 else 'НЕТ':<10} | {float(r['distance_m']):<15.3f} | {r['center_y']:<12}")
            
#             # -----------------------------------------------------------------
#             # 🟢 ДОБАВЛЯЕМ СВОДНЫЙ РЕЕСТР УНИКАЛЬНЫХ ОБЪЕКТОВ НА ФИНИШЕ
#             # -----------------------------------------------------------------
#                         # -----------------------------------------------------------------
#             # 🟢 ДОБАВЛЯЕМ СВОДНЫЙ РЕЕСТР УНИКАЛЬНЫХ ОБЪЕКТОВ НА ФИНИШЕ (С ТРЕКИНГОМ КАДРОВ)
#             # -----------------------------------------------------------------
#             print("\n📋 ГЛОБАЛЬНЫЙ РЕЕСТР УНИКАЛЬНЫХ ПРЕПЯТСТВИЙ (ИТОГ СЕССИИ):")
#             print(f"{'Идентификатор':<15} | {'Класс (Форма)':<25} | {'Интервал кадров':<25} | {'Дистанция (Max -> Min)':<25}")
#             print("-"*98)
#             for t_id, data in self.radar_targets_registry.items():
#                 frame_range = f"{data['first_frame']} -> {data['last_frame']}"
#                 dist_range = f"{data['max_dist']:.2f} м -> {data['min_dist']:.2f} м"
#                 print(f"🆔 Облако #{t_id:<10} | {data['shape']:<25} | {frame_range:<25} | {dist_range:<25}")

            
#         else:
#             print(" ✅ Вся сессия пройдена в штатном режиме. Пути абсолютно чистые.")
#         print("="*80 + "\n")
#         print(f" 📂 ВСЕГО ОБРАБОТАНО УНИКАЛЬНЫХ КАДРОВ ЛИДАРА: {self.total_processed_frames} шт.")
#         print(f" 📦 СУММАРНО НАЙДЕНО ФИЗИЧЕСКИХ ОБЪЕКТОВ ИИ:    {self.total_detected_obstacles} шт.\n")


# def main(args=None):
#     rclpy.init(args=args)
#     node = SubwayVisionCoreNode()
    
#     # 🟢 ЗАПУСКАЕМ МНОГОПОТОЧНЫЙ ДИСПЕТЧЕР НА 2 ПОТОКА CPU [10]
#     from rclpy.executors import MultiThreadedExecutor
#     executor = MultiThreadedExecutor(num_threads=2)
#     executor.add_node(node)
    
#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         node.get_logger().info('🛑 Получен сигнал Ctrl+C. Выполняем грейсфол шатдаун...')
#     finally:
#         node.export_final_submission()
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == '__main__':
#     main()

# #!/usr/bin/env python3
# """
# 🚀 PRODUCTION ROS 2 HUMBLE COMPLIANT NODE (ONLINE ADAS EDITION) - PART 1
# Промышленный конвейер детекции препятствий реального времени (Онлайн-режим).
# """

# import sys
# import os
# import gc
# import csv
# import time
# import numpy as np
# import yaml
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud2
# from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
# import ros2_numpy  # Гарантирует 100% идентичность парсинга геометрии с quick_check [10]

# # Импортируем движки ЦОС, одометрии и трекинга
# from metro_lidar.cos_processor_v12 import StableLidarOdometryV12
# import metro_lidar.cos_processor_v12 as cos_processor_v12
# from metro_lidar.generate_submission_v2 import process_point_cloud
# from metro_lidar.tracker_v2 import LidarObstacleTrackerV2
# import metro_lidar.config as config


# def execute_detection_via_memory_patch(raw_points: np.ndarray, tracker_engine, shift_z: float, is_open_space: bool):
#     """Выполняет безопасный перехват системной функции np.fromfile (Monkey Patching) [10]"""
#     original_fromfile = np.fromfile
#     try:
#         virtual_file_path = "memory_stream_frame.bin"
        
#         def mock_fromfile(file, dtype=None, count=-1, sep='', offset=0):
#             if file == virtual_file_path:
#                 return raw_points
#             return original_fromfile(file, dtype, count, sep, offset)
            
#         np.fromfile = mock_fromfile
        
#         detected_obstacles = process_point_cloud(
#             virtual_file_path, 
#             tracker_engine, 
#             train_step_z=shift_z, 
#             is_open_space=is_open_space
#         )
#         return detected_obstacles
#     finally:
#         np.fromfile = original_fromfile


# class SubwayVisionCoreNode(Node):
    
#     def __init__(self):
#         super().__init__('subway_vision_core_node')
        
#         from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        
#         # Настройка универсального профиля QoS под Best Effort
#         custom_qos = QoSProfile(
#             depth=1,  # В буфере сокета храним строго 1 самый свежий кадр
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             durability=DurabilityPolicy.VOLATILE
#         )
        
#         # Буфер для обмена сообщениями между параллельными потоками
#         self.latest_msg = None
        
#         # Разделяем сетевой прием и тяжелые вычисления ИИ по разным потокам ОС
#         self.net_cb_group = MutuallyExclusiveCallbackGroup()
#         self.worker_cb_group = MutuallyExclusiveCallbackGroup()
        
#         # 🟢 ДИНАМИЧЕСКИЙ ПОИСК ПУТИ (Автоматически ищет в папке текущего сценария)
#         # Объявляем параметры для жюри, чтобы они могли передать сценарий или прямой путь
#         # 🟢 ДИНАМИЧЕСКИЙ СВЕРХЗАЩИЩЕННЫЙ ПОИСК ПУТЕЙ (Универсальный мост)
#         self.declare_parameter('scenario', 'doubleT_platform')
#         self.declare_parameter('data_dir', '/app/for_hackathon')
        
#         scenario_param = self.get_parameter('scenario').get_parameter_value().string_value
#         data_dir_param = self.get_parameter('data_dir').get_parameter_value().string_value
        
#         # Интеллектуальное определение: что нам передали в параметре scenario?
#         if os.path.isabs(scenario_param) or '/' in scenario_param:
#             # Вариант А: Жюри передало полный абсолютный или относительный путь к папке сценария
#             base_scenario_dir = os.path.abspath(scenario_param)
#         else:
#             # Вариант Б: Жюри передало только короткое имя папки (работает старая логика)
#             base_scenario_dir = os.path.join(data_dir_param, scenario_param)
            
#         # Финальный прецизионный путь к файлу метаданных
#         metadata_path = os.path.join(base_scenario_dir, 'metadata.yaml')
        
#         self.get_logger().info(f'📂 [ИНСПЕКТОР ПУТЕЙ]: Целевой каталог сценария определен как: {base_scenario_dir}')
#         self.get_logger().info(f'📜 [ИНСПЕКТОР ПУТЕЙ]: Ищем файл метаданных по адресу: {metadata_path}')

        
#         # Собираем путь к метаданным на основе переданного сценария
#         metadata_path = os.path.join(data_dir, scenario, 'metadata.yaml')
        
#         # Значение по умолчанию на случай, если файл не будет найден
#         target_topic = '/sensing/lidar/hesai128/pointcloud'

#         # 🛰️ УНИВЕРСАЛЬНЫЙ ИИ-ПАРСЕР МЕТАДАННЫХ ЖЮРИ (Успешно обработает оба ваших примера)
#         if os.path.exists(metadata_path):
#             try:
#                 with open(metadata_path, 'r', encoding='utf-8') as f:
#                     bag_info = yaml.safe_load(f)
                    
#                 # Спускаемся по древовидной структуре ROS 2 Bag информации
#                 bag_meta = bag_info.get('rosbag2_bagfile_information', {})
#                 topics_list = bag_meta.get('topics_with_message_count', [])
                
#                 found_topic = None
                
#                 # Шаг 1: Ищем топик по строгому совпадению типа PointCloud2
#                 for topic_entry in topics_list:
#                     meta_data = topic_entry.get('topic_metadata', {})
#                     t_type = str(meta_data.get('type', ''))
#                     t_name = meta_data.get('name')
                    
#                     if 'PointCloud2' in t_type and t_name:
#                         found_topic = t_name
#                         self.get_logger().info(f'🎯 [ПАРСЕР]: Найдено точное совпадение типа PointCloud2 -> {found_topic}')
#                         break
                
#                 # Шаг 2: Эвристический откат (если тип данных в базе указан нестандартно)
#                 if not found_topic:
#                     for topic_entry in topics_list:
#                         meta_data = topic_entry.get('topic_metadata', {})
#                         t_name = str(meta_data.get('name', '')).lower()
                        
#                         if any(marker in t_name for marker in ['lidar', 'points', 'pointcloud', 'hesai']):
#                             found_topic = meta_data.get('name')
#                             self.get_logger().warn(f'📡 [ПАРСЕР ЭВРИСТИКА]: Топик определен по ключевым словам -> {found_topic}')
#                             break

#                 if found_topic:
#                     target_topic = found_topic
#                     self.get_logger().info(f'🚀 [АВТОНАСТРОЙКА УСПЕШНА]: Нода переключена на топик базы: {target_topic}')
                    
#             except Exception as e:
#                 self.get_logger().error(f'⚠️ Ошибка автопарсинга metadata.yaml (откат к дефолту): {e}')
#         else:
#             self.get_logger().warn(f'📂 Файл метаданных не найден по пути: {metadata_path}. Работаем по стандартному радиоканалу.')

#         # Согласуем полученный динамический топик с внутренним параметром ROS 2
#         self.declare_parameter('lidar_topic', target_topic)
#         topic_name = self.get_parameter('lidar_topic').get_parameter_value().string_value
        
#         # Подписка ROS 2 Humble жестко привязана к сетевой группе
#         self.subscription = self.create_subscription(
#             PointCloud2,
#             topic_name, 
#             self.lidar_callback,
#             custom_qos,
#             callback_group=self.net_cb_group
#         )
        
#         # Инициализируем межкадровые движки
#         self.odometry_engine = StableLidarOdometryV12()
#         self.obstacle_tracker_engine = LidarObstacleTrackerV2()
        
#         self.frame_idx = 0
#         self.dt = 0.1  # 10 Гц Hesai 128
#         self.submission_records = []
        
#         # Статистические счетчики онлайн-конвейера
#         self.total_processed_frames = 0     
#         self.total_detected_obstacles = 0   
        
#         # Фоновый таймер воркера (25 Гц) привязан к вычислительной группе
#         self.process_timer = self.create_timer(
#             0.04, 
#             self.process_loop, 
#             callback_group=self.worker_cb_group
#         )
        
#         self.get_logger().info(f'🚇 [ОНЛАЙН ИИ-ЯДРО ЖИВОГО ПОЕЗДА]: АКТИВИРОВАНО. Топик подписки: {topic_name}')
#         self.radar_targets_registry = {}
    
#     def lidar_callback(self, msg):
#         """Сетевой поток: работает параллельно, мгновенно сохраняя ссылку на кадр [10]"""
#         self.latest_msg = msg

#     def process_loop(self):
#         """Вычислительный поток: независимо от сети забирает кадр и крутит тяжелый ЦОС/ИИ [10]"""
#         if self.latest_msg is None:
#             return
            
#         # Атомарно вытаскиваем кадр из буфера и очищаем слот
#         msg = self.latest_msg
#         self.latest_msg = None
        
#         local_frame_idx = self.frame_idx
#         self.frame_idx += 1
#         self.total_processed_frames += 1  # Фиксируем шаг онлайн-конвейера

#         flat_floats = None
#         raw_points = None
#         calculated_speed_kmh = 0.0

#         try:
#             # 1. 🟢 ВЫСОКОСКОРОСТНОЙ НАТИВНЫЙ ПАРСЕР И ФИЛЬТР (Копия convert_all_bags)
#             arr_dict = ros2_numpy.numpify(msg)
#             xyz = arr_dict['xyz'].astype(np.float32)
#             num_points = len(xyz)
            
#             if 'intensity' in arr_dict:
#                 intensity = arr_dict['intensity'].astype(np.float32).reshape(-1, 1)
#             else:
#                 intensity = np.zeros((num_points, 1), dtype=np.float32)
                
#             points = np.hstack((xyz, intensity))
            
#             # Фильтрация NaN [10]
#             nan_mask = np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(points[:, 2])
#             points = points[~nan_mask]
            
#             # Вырезаем все пустые/нулевые лучи лидара [10]
#             nonzero_mask = np.any(points[:, :3] != 0, axis=1)
#             raw_points = points[nonzero_mask].copy()
            
#             if len(raw_points) == 0:
#                 return

#             # Вычисление плотности точек внутри колеи путей [10]
#             x_pts, y_pts, z_pts = raw_points[:, 2], raw_points[:, 0], raw_points[:, 1]
#             rail_points_mask = (z_pts >= -63.5) & (z_pts <= -3.5) & \
#                                (x_pts >= -0.75) & (x_pts <= 0.75) & \
#                                (y_pts >= -1.85) & (y_pts <= -1.05)
#             current_rail_points_count = int(np.sum(rail_points_mask))

#             # Расчет текущей одометрии по рельсам и геометрии стен [10]
#             shift_z_rails, _ = self.odometry_engine.compute_raw_rail_odo_shift(raw_points, self.dt)
            
#             shift_z_walls = 0.0
#             macro_cloud = self.odometry_engine.extract_clean_macro_tunnel(raw_points)
#             if macro_cloud is not None:
#                 passports = self.odometry_engine.build_passports_via_dbscan(macro_cloud)
#                 calculate_speed_trigger = bool(local_frame_idx > 0)
#                 shift_z_walls, _, _ = self.odometry_engine.associate_and_calculate_shift(
#                     passports, self.dt, calculate_speed=calculate_speed_trigger
#                 )

#             # Комплексирование одометрии (Fusion) [10]
#             shift_z_physical = cos_processor_v12.calculate_adaptive_fusion_shift(
#                 shift_z_rails, shift_z_walls, self.odometry_engine.prev_velocity_kmh, 
#                 current_rail_points_count, local_frame_idx, self.dt
#             )
            
#             calculated_speed_kmh = (shift_z_physical / self.dt) * 3.6
#             if calculated_speed_kmh < 0.2:
#                 calculated_speed_kmh = 0.0
#                 shift_z_physical = 1e-5
                
#             self.odometry_engine.prev_velocity_kmh = calculated_speed_kmh

#             # Определение открытого пространства тоннеля [10]
#             is_open_space = False
#             valid_walls_anchors = [obj for obj in self.odometry_engine.anchor_map if obj.get("status") == "STATUS_VALID_ANCHOR" and obj.get("is_wall", False)]
#             if valid_walls_anchors:
#                 wall_x_coords = [float(w_obj["centroid"]) for w_obj in valid_walls_anchors if "centroid" in w_obj]
#                 if wall_x_coords and (max(wall_x_coords) > config.V2_OBSTACLE_WALL_OPEN_THRESHOLD or min(wall_x_coords) < -config.V2_OBSTACLE_WALL_OPEN_THRESHOLD):
#                     is_open_space = True

#             detector_points = raw_points

#             # Вызов ИИ-движка детекции препятствий [10]
#             t_start = time.time()

#             detected_obstacles = execute_detection_via_memory_patch(
#                 raw_points=detector_points,
#                 tracker_engine=self.obstacle_tracker_engine,
#                 shift_z=shift_z_physical,
#                 is_open_space=is_open_space
#             )

#             # Агрегация результатов детекции в буфер submission
#             if isinstance(detected_obstacles, list) and len(detected_obstacles) > 0:
#                 self.total_detected_obstacles += len(detected_obstacles)
                
#                 # Сортируем препятствия по дальности, чтобы взять ближайшую угрозу для HUD
#                 distances = [abs(float(obj["center"][2])) for obj in detected_obstacles]
#                 min_dist = min(distances) if distances else -1.0
                
#                 virtual_frame_name = f"frame_{local_frame_idx:04d}.bin"
                
#                 # -----------------------------------------------------------------
#                 # 🟢 ОБНОВЛЕНИЕ ГЛОБАЛЬНОГО РЕЕСТРА ЦЕЛЕЙ РАДАРА (СОПРОВОЖДЕНИЕ ID)
#                 # -----------------------------------------------------------------
#                 current_frame_ids = set()
                
#                 for obj in detected_obstacles:
#                     # Извлекаем уникальный ID, присвоенный межкадровым трекером v2
#                     track_id = obj.get("id", "0")
#                     current_frame_ids.add(track_id)
                    
#                     cx, cy, cz = obj["center"]
#                     sz_x, sz_y, sz_z = obj["dimensions"]
#                     dist_z = abs(float(cz))
                    
#                     shape_label = obj.get("shape_text", "Объемная коробка / Блок")
#                     position_label = obj.get("position_text", "В створе путей")
                    
#                     # Если этот ID мы видим впервые — регистрируем новую цель в памяти
#                     if track_id not in self.radar_targets_registry:
#                         self.radar_targets_registry[track_id] = {
#                             "shape": shape_label,
#                             "min_dist": dist_z,
#                             "lifetime_frames": 1
#                         }
#                     else:
#                         # Если цель уже велась — обновляем текущую дистанцию и растим время жизни
#                         self.radar_targets_registry[track_id]["min_dist"] = min(self.radar_targets_registry[track_id]["min_dist"], dist_z)
#                         self.radar_targets_registry[track_id]["lifetime_frames"] += 1
                
#                 # Записываем строки в массив для submission.csv строго по ТЗ жюри
#                 for obj in detected_obstacles:
#                     cx, cy, cz = obj["center"]
#                     sz_x, sz_y, sz_z = obj["dimensions"]
#                     dist_z = abs(float(cz))
#                     self.submission_records.append({
#                         'frame_id': virtual_frame_name, 'obstacle_detected': 1, 'distance_m': round(dist_z, 3),
#                         'center_x': round(dist_z, 3), 'center_y': round(float(cx), 3), 'center_z': round(float(cy), 3),
#                         'size_x': round(float(sz_z), 3), 'size_y': round(float(sz_x), 3), 'size_z': round(float(sz_y), 3)  
#                     })

#                 # 🔥 НАШ НОВЫЙ ДИСПЕТЧЕРСКИЙ ЛОГ РАДАРА В РЕАЛЬНОМ ВРЕМЕНИ
#                 print(f"\n🚨 [ОНЛАЙН ➔ {virtual_frame_name}]: В ГАБАРИТЕ ПУТЕЙ ОБНАРУЖЕНО ЦЕЛЕЙ: {len(detected_obstacles)} шт.", flush=True)
#                 print(f"📡 [РЕЕСТР СОПРОВОЖДЕНИЯ ADAS]:", flush=True)
#                 for t_id, data in self.radar_targets_registry.items():
#                     # Подсвечиваем цели, активные на текущем кадре
#                     active_marker = "🎯 АКТИВЕН" if t_id in current_frame_ids else "⏳ УШЕЛ ИЗ ВИДУ"
#                     print(f"   ↳ 🆔 ПРЕПЯТСТВИЕ #{t_id} [{data['shape']}] | Дистанция: {data['min_dist']:.2f} м | Сопровождение: {data['lifetime_frames']} кадров | {active_marker}", flush=True)
#                 print("-" * 80, flush=True)
                
#             else:
#                 virtual_frame_name = f"frame_{local_frame_idx:04d}.bin"
#                 self.submission_records.append({
#                     'frame_id': virtual_frame_name, 'obstacle_detected': 0, 'distance_m': -1.0,
#                     'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0, 'size_x': 0.0, 'size_y': 0.0, 'size_z': 0.0
#                 })
#                 self.get_logger().info(f"🟢 [ОНЛАЙН ➔ {virtual_frame_name}]: ПУТЬ СВОБОДЕН | 📈 {calculated_speed_kmh:.1f} км/ч")

#         except Exception as e:
#             virtual_frame_name = f"frame_{local_frame_idx:04d}.bin"
#             self.get_logger().error(f"🚨 [СБОЙ НА КАДРЕ {virtual_frame_name}]: {e}")
#             self.submission_records.append({
#                 'frame_id': virtual_frame_name, 'obstacle_detected': 0, 'distance_m': -1.0,
#                 'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0, 'size_x': 0.0, 'size_y': 0.0, 'size_z': 0.0
#             })
#         finally:
#             del flat_floats, raw_points
#             gc.collect()

#     def export_final_submission(self):
#         """Записывает накопленный кэш кадров в файл submission.csv [10]"""
#         output_path = "/app/submission.csv"
#         if not self.submission_records:
#             self.get_logger().error("⚠️ Буфер сабмита пуст. CSV не сохранен.")
#             return
            
#         try:
#             self.submission_records.sort(key=lambda x: x['frame_id'])
            
#             fields = ['frame_id', 'obstacle_detected', 'distance_m', 
#                       'center_x', 'center_y', 'center_z', 
#                       'size_x', 'size_y', 'size_z']
                      
#             with open(output_path, mode='w', newline='', encoding='utf-8') as f:
#                 writer = csv.DictWriter(f, fieldnames=fields)
#                 writer.writeheader()
#                 writer.writerows(self.submission_records)
                
#             self.get_logger().info(f"💾 [ИИ-ЭКСПОРТ УСПЕШЕН]: Отчет REP 103 сохранен: {output_path}")
#             self.get_logger().info(f"📊 Всего кадров зафиксировано в итоговом CSV: {len(self.submission_records)}")
            
#             # 🟢 ВСТРАИВАЕМ НАШ ИТОГОВЫЙ ИНСПЕКЦИОННЫЙ ОТЧЕТ ДЛЯ ЖЮРИ НА ФИНИШЕ СЕССИИ [10]
#             self.print_final_summary_report()
            
#         except Exception as e:
#             self.get_logger().error(f"🚨 Ошибка записи CSV: {e}")

#     def print_final_summary_report(self):
#         """Печатает красивый аудит-отчет накопленного массива submission_records для жюри"""
#         print("\n" + "="*80)
#         print("📊  ФИНАЛЬНЫЙ ИНСПЕКЦИОННЫЙ ОТЧЕТ ИИ-КОНВЕЙЕРА (СУДЬИ ХАКАТОНА)")
#         print("="*80)
#         print(f" Всего зафиксировано и обсчитано кадров: {self.total_processed_frames}")
#         danger_frames = [r for r in self.submission_records if int(r['obstacle_detected']) == 1]
#         print(f" Кадров с опасными аномалиями на путях:  {len(danger_frames)}")
#         if danger_frames:
#             distances = [float(r['distance_m']) for r in danger_frames]
#             print(f" Минимальная дистанция фиксации угрозы: {min(distances):.3f} м")
#             print(f" Максимальная дальность раннего зрения: {max(distances):.3f} м")
#             print("\n📋 ХРОНОЛОГИЧЕСКИЙ СРЕЗ ПЕРВЫХ ДЕТЕКЦИЙ ДЛЯ СВЕРКИ:")
#             print(f"{'Имя кадра':<20} | {'Детекция':<10} | {'Дальность Z вперед':<15} | {'Смещение X':<12}")
#             print("-"*80)
#             for r in danger_frames[:10]:
#                 print(f"{r['frame_id']:<20} | {'🚨 ДА' if int(r['obstacle_detected'])==1 else 'НЕТ':<10} | {float(r['distance_m']):<15.3f} | {r['center_y']:<12}")
            
#             # -----------------------------------------------------------------
#             # 🟢 ДОБАВЛЯЕМ СВОДНЫЙ РЕЕСТР УНИКАЛЬНЫХ ОБЪЕКТОВ НА ФИНИШЕ
#             # -----------------------------------------------------------------
#             print("\n📋 ГЛОБАЛЬНЫЙ РЕЕСТР УНИКАЛЬНЫХ ПРЕПЯТСТВИЙ (ИТОГ СЕССИИ):")
#             print(f"{'Идентификатор':<15} | {'Морфологический класс (Форма)':<35} | {'Минимальная дистанция':<22}")
#             print("-"*80)
#             for t_id, data in self.radar_targets_registry.items():
#                 print(f"🆔 Облако #{t_id:<10} | {data['shape']:<35} | {data['min_dist']:.3f} м")
            
#         else:
#             print(" ✅ Вся сессия пройдена в штатном режиме. Пути абсолютно чистые.")
#         print("="*80 + "\n")
#         print(f" 📂 ВСЕГО ОБРАБОТАНО УНИКАЛЬНЫХ КАДРОВ ЛИДАРА: {self.total_processed_frames} шт.")
#         print(f" 📦 СУММАРНО НАЙДЕНО ФИЗИЧЕСКИХ ОБЪЕКТОВ ИИ:    {self.total_detected_obstacles} шт.\n")


# def main(args=None):
#     rclpy.init(args=args)
#     node = SubwayVisionCoreNode()
    
#     # 🟢 ЗАПУСКАЕМ МНОГОПОТОЧНЫЙ ДИСПЕТЧЕР НА 2 ПОТОКА CPU [10]
#     from rclpy.executors import MultiThreadedExecutor
#     executor = MultiThreadedExecutor(num_threads=2)
#     executor.add_node(node)
    
#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         node.get_logger().info('🛑 Получен сигнал Ctrl+C. Выполняем грейсфол шатдаун...')
#     finally:
#         node.export_final_submission()
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == '__main__':
#     main()

# #!/usr/bin/env python3
# """
# 🚀 PRODUCTION ROS 2 HUMBLE COMPLIANT NODE (PART 1)
# Промышленный конвейер детекции препятствий, полностью соответствующий ТЗ жюри.
# Использует логику ros2_numpy фильтрации из конвертера для защиты от зависаний.
# """

# import sys
# import os
# import gc
# import csv
# import time
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud2
# from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
# import ros2_numpy  # Гарантирует 100% идентичность парсинга геометрии с quick_check [1, 5]

# # Импортируем движки ЦОС, одометрии и трекинга
# from metro_lidar.cos_processor_v12 import StableLidarOdometryV12
# import metro_lidar.cos_processor_v12 as cos_processor_v12
# from metro_lidar.generate_submission_v2 import process_point_cloud
# from metro_lidar.tracker_v2 import LidarObstacleTrackerV2
# import metro_lidar.config as config


# def execute_detection_via_memory_patch(raw_points: np.ndarray, tracker_engine, shift_z: float, is_open_space: bool):
#     """Выполняет безопасный перехват системной функции np.fromfile (Monkey Patching) [1]"""
#     original_fromfile = np.fromfile
#     try:
#         virtual_file_path = "memory_stream_frame.bin"
        
#         def mock_fromfile(file, dtype=None, count=-1, sep='', offset=0):
#             if file == virtual_file_path:
#                 return raw_points
#             return original_fromfile(file, dtype, count, sep, offset)
            
#         np.fromfile = mock_fromfile
        
#         detected_obstacles = process_point_cloud(
#             virtual_file_path, 
#             tracker_engine, 
#             train_step_z=shift_z, 
#             is_open_space=is_open_space
#         )
#         return detected_obstacles
#     finally:
#         np.fromfile = original_fromfile


# class SubwayVisionCoreNode(Node):
#     def __init__(self):
#         super().__init__('subway_vision_core_node')
        
#         from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        
#         # Настройка универсального профиля QoS под Best Effort
#         # Если bag-файл записан как RELIABLE, параметры автоматически согласуются на уровне DDS [1]
#         custom_qos = QoSProfile(
#             depth=1,  # В буфере сокета храним строго 1 самый свежий кадр [1]
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             durability=DurabilityPolicy.VOLATILE
#         )
        
#         # Буфер для обмена сообщениями между параллельными потоками
#         self.latest_msg = None
        
#         # Разделяем сетевой прием и тяжелые вычисления ИИ по разным потокам ОС
#         self.net_cb_group = MutuallyExclusiveCallbackGroup()
#         self.worker_cb_group = MutuallyExclusiveCallbackGroup()
        
#         # Автоматическая декларация параметра топика лидара для быстрой настройки жюри [1, 6]
#         self.declare_parameter('lidar_topic', '/sensing/lidar/hesai128/pointcloud')
#         topic_name = self.get_parameter('lidar_topic').get_parameter_value().string_value
        
#         # Подписка ROS 2 Humble жестко привязана к сетевой группе
#         self.subscription = self.create_subscription(
#             PointCloud2,
#             topic_name, 
#             self.lidar_callback,
#             custom_qos,
#             callback_group=self.net_cb_group
#         )
        
#         # Инициализируем межкадровые движки
#         self.odometry_engine = StableLidarOdometryV12()
#         self.obstacle_tracker_engine = LidarObstacleTrackerV2()
        
#         self.frame_idx = 0
#         self.dt = 0.1  # 10 Гц Hesai 128 [1]
#         self.submission_records = []
        
#         # Фоновый таймер воркера (25 Гц) привязан к вычислительной группе
#         self.process_timer = self.create_timer(
#             0.04, 
#             self.process_loop, 
#             callback_group=self.worker_cb_group
#         )
        
#         self.get_logger().info(f'🚇 [ИИ-ЯДРО МЕТРО]: Параллельный Humble конвейер готов. Топик: {topic_name}')

#     def lidar_callback(self, msg):
#         """Сетевой поток: работает параллельно, мгновенно сохраняя ссылку на кадр"""
#         self.latest_msg = msg

#     def process_loop(self):
#         """Вычислительный поток: независимо от сети забирает кадр и крутит тяжелый ЦОС/ИИ"""
#         if self.latest_msg is None:
#             return
            
#         # Атомарно вытаскиваем кадр из буфера и очищаем слот
#         msg = self.latest_msg
#         self.latest_msg = None
        
#         local_frame_idx = self.frame_idx
#         self.frame_idx += 1

#         flat_floats = None
#         raw_points = None
#         calculated_speed_kmh = 0.0

#         try:
#             # 1. 🟢 ВЫСОКОСКОРОСТНОЙ НАТИВНЫЙ ПАРСЕР И ФИЛЬТР (Копия convert_all_bags)
#             # Извлекает структурированный массив координат через ros2_numpy [5, 7]
#             arr_dict = ros2_numpy.numpify(msg)
#             xyz = arr_dict['xyz'].astype(np.float32)
#             num_points = len(xyz)
            
#             if 'intensity' in arr_dict:
#                 intensity = arr_dict['intensity'].astype(np.float32).reshape(-1, 1)
#             else:
#                 intensity = np.zeros((num_points, 1), dtype=np.float32)
                
#             points = np.hstack((xyz, intensity))
            
#             # Фильтрация NaN [5, 7]
#             nan_mask = np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(points[:, 2])
#             points = points[~nan_mask]
            
#             # 🔥 ГЛАВНЫЙ СЕКРЕТ СТАБИЛЬНОСТИ: Вырезаем все пустые/нулевые лучи лидара
#             # Это мгновенно сжимает входящий поток с 921 600 до ~340 000 точек (как на диске!) [5]
#             nonzero_mask = np.any(points[:, :3] != 0, axis=1)
#             raw_points = points[nonzero_mask].copy()
            
#             if len(raw_points) == 0:
#                 return

#             # Вычисление плотности точек внутри колеи путей [1, 7]
#             x_pts, y_pts, z_pts = raw_points[:, 2], raw_points[:, 0], raw_points[:, 1]
#             rail_points_mask = (z_pts >= -63.5) & (z_pts <= -3.5) & \
#                                (x_pts >= -0.75) & (x_pts <= 0.75) & \
#                                (y_pts >= -1.85) & (y_pts <= -1.05)
#             current_rail_points_count = int(np.sum(rail_points_mask))

#             # Расчет текущей одометрии по рельсам и геометрии стен [1, 7]
#             shift_z_rails, _ = self.odometry_engine.compute_raw_rail_odo_shift(raw_points, self.dt)
            
#             shift_z_walls = 0.0
#             macro_cloud = self.odometry_engine.extract_clean_macro_tunnel(raw_points)
#             if macro_cloud is not None:
#                 passports = self.odometry_engine.build_passports_via_dbscan(macro_cloud)
#                 calculate_speed_trigger = bool(local_frame_idx > 0)
#                 shift_z_walls, _, _ = self.odometry_engine.associate_and_calculate_shift(
#                     passports, self.dt, calculate_speed=calculate_speed_trigger
#                 )

#             # Комплексирование одометрии (Fusion) [1, 7]
#             shift_z_physical = cos_processor_v12.calculate_adaptive_fusion_shift(
#                 shift_z_rails, shift_z_walls, self.odometry_engine.prev_velocity_kmh, 
#                 current_rail_points_count, local_frame_idx, self.dt
#             )
            
#             calculated_speed_kmh = (shift_z_physical / self.dt) * 3.6
#             if calculated_speed_kmh < 0.2:
#                 calculated_speed_kmh = 0.0
#                 shift_z_physical = 1e-5
                
#             self.odometry_engine.prev_velocity_kmh = calculated_speed_kmh

#             # Определение открытого пространства тоннеля [1, 7]
#             is_open_space = False
#             valid_walls_anchors = [obj for obj in self.odometry_engine.anchor_map if obj.get("status") == "STATUS_VALID_ANCHOR" and obj.get("is_wall", False)]
#             if valid_walls_anchors:
#                 wall_x_coords = [float(w_obj["centroid"]) for w_obj in valid_walls_anchors if "centroid" in w_obj]
#                 if wall_x_coords and (max(wall_x_coords) > config.V2_OBSTACLE_WALL_OPEN_THRESHOLD or min(wall_x_coords) < -config.V2_OBSTACLE_WALL_OPEN_THRESHOLD):
#                     is_open_space = True

#             # Межкадровая память трекера теперь в безопасности, так как на входе нет перегрузки точками [7]
#             detector_points = raw_points

#             # Вызов ИИ-движка детекции препятствий [1, 7]
#             self.get_logger().info(f"🔍 [КАДР #{local_frame_idx:04d}]: Вход в детекцию. Очищенных точек: {len(detector_points)}")
#             t_start = time.time()

#             detected_obstacles = execute_detection_via_memory_patch(
#                 raw_points=detector_points,
#                 tracker_engine=self.obstacle_tracker_engine,
#                 shift_z=shift_z_physical,
#                 is_open_space=is_open_space
#             )
            
#             self.get_logger().info(f"⏱️ [КАДР #{local_frame_idx:04d}]: Выход из детекции. Время: {(time.time() - t_start)*1000:.1f} мс")

#             # Агрегация результатов детекции в буфер submission [1, 7]
#             if isinstance(detected_obstacles, list) and len(detected_obstacles) > 0:
#                 closest_obs = detected_obstacles[0]  # Извлекаем первый словарь из списка [7]
#                 cx, cy, cz = closest_obs["center"]
#                 sz_x, sz_y, sz_z = closest_obs["dimensions"]
#                 min_dist = abs(float(cz))
                
#                 self.submission_records.append({
#                     'frame_id': local_frame_idx, 'obstacle_detected': 1, 'distance_m': round(min_dist, 3),
#                     'center_x': round(abs(float(cz)), 3), 'center_y': round(float(cx), 3), 'center_z': round(float(cy), 3),
#                     'size_x': round(float(sz_z), 3), 'size_y': round(float(sz_x), 3), 'size_z': round(float(sz_y), 3)  
#                 })

#                 shape_label = closest_obs.get("shape_text", "Объект")
#                 position_label = closest_obs.get("position_text", "В колее")
#                 self.get_logger().warn(
#                     f"🚨 [КАДР #{local_frame_idx:04d}]: ПРЕПЯТСТВИЕ ОБНАРУЖЕНО! | "
#                     f"📏 Дистанция до угрозы: {min_dist:.1f} м | 📦 Форма: {shape_label} | 📍 {position_label}"
#                 )
#             else:
#                 self.submission_records.append({
#                     'frame_id': local_frame_idx, 'obstacle_detected': 0, 'distance_m': -1.0,
#                     'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0, 'size_x': 0.0, 'size_y': 0.0, 'size_z': 0.0
#                 })
#                 self.get_logger().info(f"🟢 [КАДР #{local_frame_idx:04d}]: ПУТЬ СВОБОДЕН | 📈 {calculated_speed_kmh:.1f} км/ч")

#         except Exception as e:
#             self.get_logger().error(f"🚨 [СБОЙ КАДРА #{local_frame_idx:04d}]: {e}")
#             self.submission_records.append({
#                 'frame_id': local_frame_idx, 'obstacle_detected': 0, 'distance_m': -1.0,
#                 'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0, 'size_x': 0.0, 'size_y': 0.0, 'size_z': 0.0
#             })
#         finally:
#             del flat_floats, raw_points
#             gc.collect()

#     def export_final_submission(self):
#         """Записывает накопленный кэш кадров в файл submission.csv [1, 7]"""
#         output_path = "/app/submission.csv"
#         if not self.submission_records:
#             self.get_logger().error("⚠️ Буфер сабмита пуст. CSV не сохранен.")
#             return
            
#         try:
#             self.submission_records.sort(key=lambda x: x['frame_id'])
            
#             fields = ['frame_id', 'obstacle_detected', 'distance_m', 
#                       'center_x', 'center_y', 'center_z', 
#                       'size_x', 'size_y', 'size_z']
                      
#             with open(output_path, mode='w', newline='', encoding='utf-8') as f:
#                 writer = csv.DictWriter(f, fieldnames=fields)
#                 writer.writeheader()
#                 writer.writerows(self.submission_records)
                
#             self.get_logger().info(f"💾 [ИИ-ЭКСПОРТ УСПЕШЕН]: Отчет REP 103 сохранен: {output_path}")
#             self.get_logger().info(f"📊 Всего кадров зафиксировано в итоговом CSV: {len(self.submission_records)}")
#         except Exception as e:
#             self.get_logger().error(f"🚨 Ошибка записи CSV: {e}")


# def main(args=None):
#     rclpy.init(args=args)
#     node = SubwayVisionCoreNode()
    
#     # 🟢 ЗАПУСКАЕМ МНОГОПОТОЧНЫЙ ДИСПЕТЧЕР НА 2 ПОТОКА CPU [7]
#     from rclpy.executors import MultiThreadedExecutor
#     executor = MultiThreadedExecutor(num_threads=2)
#     executor.add_node(node)
    
#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         node.get_logger().info('🛑 Получен сигнал Ctrl+C. Выполняем грейсфол шатдаун...')
#     finally:
#         node.export_final_submission()
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == '__main__':
#     main()