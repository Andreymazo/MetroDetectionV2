#!/usr/bin/env python3
"""
🚀 PRODUCTION ROS 2 HUMBLE COMPLIANT NODE (STRICT CHRONO DELAYED BATCH EDITION) - PART 1
Промышленный конвейер детекции препятствий Мосметро строго по ТЗ жюри.
Защищен от прореживания лучей через разделение во времени: Сбор ОЗУ ➔ Автономный ИИ Расчет.
"""

import sys
import os
import gc
import csv
import time
import json
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# Импортируем тяжелые ИИ и ЦОС движки из ядра решения (строго как в quick_check)
from metro_lidar.cos_processor_v12 import StableLidarOdometryV12
from metro_lidar.generate_submission_v2 import process_point_cloud
from metro_lidar.tracker_v2 import LidarObstacleTrackerV2
import metro_lidar.config as config


class SubwayVisionCoreNode(Node):
    def __init__(self):
        super().__init__('subway_vision_core_node')
        
        try:
            os.system("sysctl -w net.core.rmem_max=26214400 > /dev/null 2>&1")
            os.system("sysctl -w net.core.rmem_default=26214400 > /dev/null 2>&1")
        except Exception:
            pass

        # 🟢 НАСТРОЙКА ПРЕЦИЗИОННОГО ВОКЗАЛА В ОЗУ (/dev/shm)
        self.station_dir = "/dev/shm/metro_station_buffer"
        os.makedirs(self.station_dir, exist_ok=True)
        
        # Очищаем перроны вокзала перед стартом новой сессии
        for old_file in os.listdir(self.station_dir):
            try:
                os.remove(os.path.join(self.station_dir, old_file))
            except Exception:
                pass

        self.net_cb_group = MutuallyExclusiveCallbackGroup()

        # 🎯 ИСПРАВЛЕННЫЙ QoS ПОД ВАШ КОРРЕКТНЫЙ ХАКАТОНОВСКИЙ БЭГ
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        custom_qos = QoSProfile(
            depth=100,  # Оставляем большой буфер, чтобы не потерять ни одного кадра
            reliability=ReliabilityPolicy.RELIABLE,  # Гарантия доставки
            durability=DurabilityPolicy.VOLATILE     # Идеально совместим с издателем rosbag
        )
        
        self.declare_parameter('lidar_topic', '/sensing/lidar/hesai128/pointcloud')
        topic_name = self.get_parameter('lidar_topic').get_parameter_value().string_value
        
        # Сетевая подписка ROS 2
        self.subscription = self.create_subscription(
            PointCloud2,
            topic_name, 
            self.lidar_callback,
            custom_qos,
            callback_group=self.net_cb_group
        )
        
        # Нативный ROS 2 издатель телеметрии
        self.telemetry_pub = self.create_publisher(String, '/safetrain/telemetry', 10)
        
        # Инициализируем хронологические движки ИИ-памяти сессии
        self.odometry_engine = StableLidarOdometryV12()
        self.obstacle_tracker_engine = LidarObstacleTrackerV2()
        
        self.net_frame_idx = 0       # Сколько кадров сохранила сеть
        self.compute_frame_idx = 0   # Сколько кадров обсчитал ИИ
        
        self.dt = 0.1                
        self.submission_records = []
        self._csv_exported_flag = False
        self.total_distance_meters = 0.0
        
        # Метка времени для отслеживания таймера тишины сети (безопасные 5.0 сек)
        self.last_activity_time = time.time()
        self.is_running = True
        
        # Флаг-семафор: ИИ-поток спит, пока сеть качает вагоны
        self.network_active_phase = True
        
        # 🔥 АВТОНОМНЫЙ ВЫЧИСЛИТЕЛЬНЫЙ ПОТОК: ИИ-воркер работает независимо от rclpy.spin()
        self.compute_thread = threading.Thread(target=self.detached_compute_worker, daemon=True)
        self.compute_thread.start()
        
        self.get_logger().info('🚇 [БАТЧ-КОНВЕЙЕР АКТИВИРОВАН]: Включен режим 100% сбора лучей колеи!')
        self.get_logger().info(f'📡 Ждем входящий поток PointCloud2 на топике: {topic_name}')
        
        self.total_processed_frames = 0     # Уникальные кадры
        self.total_detected_obstacles = 0   # Всего найденных ИИ объектов

    def lidar_callback(self, msg):
        """ 🚂 СЕТЕВОЙ ПОТОК (PRODUCER): Сбрасывает СЫРЫЕ данные один-в-один как в quick_check """
        import ros2_numpy
        
        if self.net_frame_idx >= 201:
            return
            
        local_net_idx = self.net_frame_idx
        self.net_frame_idx += 1

        try:
            arr_dict = ros2_numpy.numpify(msg)
            xyz = arr_dict['xyz'].astype(np.float32)
            num_points = len(xyz)
            
            if 'intensity' in arr_dict:
                intensity = arr_dict['intensity'].astype(np.float32).reshape(-1, 1)
            else:
                intensity = np.zeros((num_points, 1), dtype=np.float32)
                
            points = np.hstack((xyz, intensity))
            
            nan_mask = np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(points[:, 2])
            points = points[~nan_mask]
            
            nonzero_mask = np.any(points[:, :3] != 0, axis=1)
            pure_points = points[nonzero_mask].copy()
            
            if len(pure_points) == 0:
                return

            # 🎯 ЭТАЛОННОЕ СОХРАНЕНИЕ: Никаких ручных разворотов осей перед записью!
            # Имя формируем по индексам, чтобы гарантировать правильный хронологический порядок
            virtual_file_path = os.path.join(self.station_dir, f"shm_frame_{local_net_idx:04d}.bin")
            pure_points.tofile(virtual_file_path)
                
            # Сдвигаем таймер тишины: сеть активна
            self.last_activity_time = time.time()
                
        except Exception as e:
            self.get_logger().error(f"❌ [СБОЙ СЕТЕВОГО ПРИЕМА НА КАДРЕ #{local_net_idx}]: {e}")

    def detached_compute_worker(self):
        """🏢 АВТОНОМНЫЙ ИИ-ДЕМОН (CONSUMER): Начинает расчет строго после затишья сети """
        while self.is_running:
            # СТАДИЯ А: Ожидание полной остановки бэг-плеера и затишья сети
            if self.network_active_phase:
                if self.net_frame_idx > 0:
                    time_since_last_packet = time.time() - self.last_activity_time
                    
                    # Если сеть молчит дольше 5.0 секунд — плеер гарантированно доиграл бэг
                    if time_since_last_packet > 5.0:
                        print(f"\n⏱️  [ТРИГГЕР ТИШИНЫ]: Сеть молчит {time_since_last_packet:.1f} сек.")
                        print(f"🏁 [ЦУП ПЕРЕХОД]: Сбор завершен. Поймано кадров: {self.net_frame_idx}")
                        print("🧠 [ИИ-ЯДРО]: Включаю автономную разгрузку вокзала ОЗУ на полную мощность...\n")
                        self.network_active_phase = False  # Переходим к этапу вычислений
                
                time.sleep(0.1)
                continue

            # СТАДИЯ Б: Последовательный ИИ-расчет накопленного буфера
            local_compute_idx = self.compute_frame_idx
            total_saved_frames = self.net_frame_idx
            
            # Если мы обсчитали все файлы, сохраненные сетью на вокзале
            if local_compute_idx >= total_saved_frames:
                if not self._csv_exported_flag:
                    print(f"\n🏁 [ЦУП АВТОНОМНЫЙ ФИНИШ]: Вокзал ОЗУ полностью опустел!")
                    print(f"   ↳ Всего успешно обсчитано ИИ-ядром: {local_compute_idx} кадров.")
                    self.export_final_submission()
                    self._csv_exported_flag = True
                    self.print_final_summary_report()
                    print("🟢 [ЦУП ИЗОЛЯЦИЯ]: Сабмит сформирован автоматически. Ожидание Ctrl+C в ноде.\n")
                
                time.sleep(0.1)
                continue

            # Имя виртуального кадра, которое пойдет в судейский CSV
            virtual_frame_name = f"frame_{local_compute_idx:04d}.bin"
            target_bin_path = os.path.join(self.station_dir, f"shm_frame_{local_compute_idx:04d}.bin")
            
            if not os.path.exists(target_bin_path):
                time.sleep(0.01)
                continue

            # Инкрементируем счетчик ИИ-разгрузки вокзала
            self.compute_frame_idx += 1

            try:
                frontend_frame_data = {
                    "danger_alert": False, "train_speed_kmh": 0.0,
                    "total_distance_m": round(self.total_distance_meters, 2),
                    "distance_to_obstacle_m": -1.0, "show_mesh_boxes": config.V2_OBSTACLE_SHOW_3D_MESHBOX,
                    "objects": []
                }

                # =====================================================================
                # 🎯 1. ЭТАЛОННЫЙ ВЫЗОВ ОДОМЕТРИИ КАК В QUICK_CHECK.PY
                # =====================================================================
                shift_z_raw, _ = self.odometry_engine.update_odometry_fusion(
                    target_bin_path, 
                    local_compute_idx, 
                    dt=self.dt
                )
                
                if hasattr(shift_z_raw, "item"):
                    shift_z_physical = float(shift_z_raw.item())
                else:
                    shift_z_physical = float(shift_z_raw)
                
                calculated_speed_kmh = self.odometry_engine.prev_velocity_kmh
                self.total_distance_meters += shift_z_physical

                # Пакуем макро-стены под Three.js топик телеметрии
                valid_walls_anchors = [obj for obj in self.odometry_engine.anchor_map if obj.get("status") == "STATUS_VALID_ANCHOR" and obj.get("is_wall", False)]
                for wall_obj in valid_walls_anchors:
                    pts = wall_obj["raw_points"]
                    if len(pts) > 100: pts = pts[::3]
                    three_pts = [[float(p[0]), float(p[1]), -abs(float(p[2]))] for p in pts]
                    frontend_frame_data["objects"].append({
                        "id": f"WALL_{wall_obj.get('id', '?')}", "class_id": 0, "confidence": 1.0, 
                        "center": [0.0, 0.0, 0.0], "size_3d": [0.0, 0.0, 0.0], "wall_points": three_pts, "obstacle_speed_kmh": 0.0
                    })

                is_open_space = False
                if valid_walls_anchors:
                    wall_x_coords = [float(w_obj["centroid"][0]) for w_obj in valid_walls_anchors if "centroid" in w_obj]
                    if wall_x_coords and (max(wall_x_coords) > config.V2_OBSTACLE_WALL_OPEN_THRESHOLD or min(wall_x_coords) < -config.V2_OBSTACLE_WALL_OPEN_THRESHOLD):
                        is_open_space = False
                safe_shift_z = float(np.abs(shift_z_physical))

                # =====================================================================
                # 🎯 2. ВЫЗОВ ИИ-ДВИЖКОВ ДЕТЕКЦИИ ПРЕПЯТСТВИЙ (v2) КАК В QUICK_CHECK.PY
                # =====================================================================
                detected_obstacles = process_point_cloud(
                    target_bin_path, 
                    tracker=self.obstacle_tracker_engine, 
                    train_step_z=safe_shift_z, 
                    is_open_space=is_open_space
                )
                self.total_processed_frames += 1
                # =====================================================================
                # 🎯 3. ПРЕЦИЗИОННАЯ УПАКОВКА РЕЗУЛЬТАТОВ (СТРОГО ПО ТЗ КОНКУРСА)
                # =====================================================================
                if isinstance(detected_obstacles, list) and len(detected_obstacles) > 0:
                    self.total_detected_obstacles += len(detected_obstacles)
                    frontend_frame_data["danger_alert"] = True
                    
                    distances = [abs(float(obj["center"][2])) for obj in detected_obstacles]
                    min_dist = min(distances) if distances else 0.0
                    frontend_frame_data["distance_to_obstacle_m"] = round(min_dist, 1)
                    
                    conf_val = detected_obstacles[0].get("confidence", 1.0)

                    # 🔥 НАШ КРАСИВЫЙ И НАГЛЯДНЫЙ ПРИНТ ОПАСНОСТИ В РЕАЛЬНОМ ВРЕМЕНИ КАК В QUICK_CHECK
                    print(f"🚨 [{virtual_frame_name}] ➔ ОБНАРУЖЕНО ПРЕПЯТСТВИЙ: {len(detected_obstacles)} шт. | "
                          f"Дистанция угрозы: {min_dist:.2f} м | Уверенность ИИ: {conf_val:.2%}")
                    
                    # Запись строк в массив для формирования итогового submission.csv
                    for obj in detected_obstacles:
                        c = obj["center"]
                        conf_val = obj.get("confidence", 1.0)
                        
                        cx, cy, cz = (float(c[0]), float(c[1]), float(c[2])) if (hasattr(c, "__len__") and len(c) >= 3) else (0.0, 0.0, float(c))
                            
                        self.submission_records.append({
                            "frame_id": virtual_frame_name,
                            "x_center": f"{cx:.4f}", 
                            "y_center": f"{cy:.4f}", 
                            "z_center": f"{cz:.4f}",
                            "class_id": 1, 
                            "confidence": f"{conf_val:.4f}"
                        })

                    # Пробрасываем геометрию преград на фронтенд топика телеметрии
                    for obj_idx, obj in enumerate(detected_obstacles):
                        c = obj["center"]
                        d_vec = obj["dimensions"]
                        
                        ocx, ocy, ocz = (float(c[0]), float(c[1]), float(c[2])) if (hasattr(c, "__len__") and len(c) >= 3) else (0.0, 0.0, float(c))
                        ow, oh, od = (float(d_vec[0]), float(d_vec[1]), float(d_vec[2])) if (hasattr(d_vec, "__len__") and len(d_vec) >= 3) else (0.5, 0.5, 0.5)
                        
                        shape_text, shape_type = "Объемная коробка / Блок", "BOX_BLOCK"
                        if oh > ow and oh > od and oh > 1.0: shape_text, shape_type = "Человек / Вертикальная конструкция", "VERTICAL_SILHOUETTE"
                        elif oh < 0.40 and (ow > 0.6 or od > 0.6): shape_text, shape_type = "Плоский предмет / Настил", "FLAT_OBSTACLE"

                        position_text, position_status = "Строго по центру путей 🚨", "CENTER"
                        if ocx < -0.35: position_text, position_status = "Касается левой кромки габарита ⚠️", "LEFT_EDGE"
                        elif ocx > 0.35: position_text, position_status = "Касается правой кромки габарита ⚠️", "RIGHT_EDGE"

                        three_pts = [[float(pt[0]), float(pt[1]), -abs(float(pt[2]))] for pt in obj.get("raw_points", []) if hasattr(pt, "__len__") and len(pt) >= 3]
                        
                        frontend_frame_data["objects"].append({
                            "id": str(obj.get("id", f"📦_{obj_idx}")), "class_id": 1, "confidence": float(obj.get("confidence", 1.0)),
                            "center": [ocx, ocy, ocz], "size_3d": [ow, oh, od], "obstacle_speed_kmh": float(obj.get("obstacle_speed", 0.0)),
                            "position_status": position_status, "position_text": position_text, "shape_type": shape_type, "shape_text": shape_text, "obstacle_points": three_pts
                        })
                else:
                    # Если препятствий на трассе нет — пишем нулевую строчку по ТЗ хакатона
                    self.submission_records.append({
                        "frame_id": virtual_frame_name,
                        "x_center": "0.0000", "y_center": "0.0000", "z_center": "0.0000",
                        "class_id": 0, "confidence": "0.0000"
                    })
                    
                    # 🟢 Опционально: можно раскомментировать принт ниже, чтобы видеть и чистые кадры
                    # print(f"🟢 [{virtual_frame_name}] ➔ Пути чистые.")


                # Набиваем метаданные поезда для выстрела в веб-панель телеметрии
                frontend_frame_data["train_speed_kmh"] = round(float(calculated_speed_kmh), 1)
                frontend_frame_data["total_distance_m"] = round(float(self.total_distance_meters), 1)
                
                # Заглушка-удержатель для Three.js, если сцена оказалась абсолютно пустой
                if len(frontend_frame_data["objects"]) == 0:
                    frontend_frame_data["objects"].append({
                        "id": "RETAINER", "class_id": -1, "confidence": 1.0, "center": [0.0, -20.0, 0.0], "size_3d": [0.01, 0.01, 0.01], "obstacle_speed_kmh": 0.0
                    })

                # 🟢 ВЫСТРЕЛИВАЕМ СФОРМИРОВАННУЮ СЦЕНУ В НА ТИВНЫЙ ТОПИК ROS 2
                ros_string_msg = String()
                ros_string_msg.data = json.dumps(frontend_frame_data)
                self.telemetry_pub.publish(ros_string_msg)


            except Exception as e:
                import traceback
                print(f"\n❌ [КРИТИЧЕСКИЙ СБОЙ НА КАДРЕ #{local_compute_idx}]: {e}")
                traceback.print_exc(file=sys.stdout)

                self.submission_records.append({
                    "frame_id": virtual_frame_name,
                    "x_center": "0.0000",
                    "y_center": "0.0000",
                    "z_center": "0.0000",
                    "class_id": 0,
                    "confidence": "0.0000"
                })
            finally:
                # Физически удаляем отработавший файл из ОЗУ, освобождая память /dev/shm
                try:
                    if os.path.exists(target_bin_path):
                        os.remove(target_bin_path)
                except Exception:
                    pass
                gc.collect()
    def export_final_submission(self):
        """Записывает накопленный кэш кадров в файл submission.csv строго в судейском формате"""
        output_path = "/app/submission.csv"
        if not self.submission_records:
            self.get_logger().error("⚠️ Буфер сабмита пуст. CSV не сохранен.")
            return
        try:
            self.submission_records.sort(key=lambda x: x['frame_id'])
            fields = ["frame_id", "x_center", "y_center", "z_center", "class_id", "confidence"]
            with open(output_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.submission_records)
            self.get_logger().info(f"💾 [ИИ-ЭКСПОРТ УСПЕШЕН]: Сабмит сохранен в: {output_path}")
        except Exception as e:
            self.get_logger().error(f"🚨 Ошибка записи CSV: {e}")

    def print_final_summary_report(self):
        """Печатает инспекционный отчет для сверки перед отправкой решения"""
        print("\n" + "="*80)
        print("📊  ФИНАЛЬНЫЙ ИНСПЕКЦИОННЫЙ ОТЧЕТ ИИ-КОНВЕЙЕРА (СУДЬИ ХАКАТОНА)")
        print("="*80)
        danger_frames = [r for r in self.submission_records if int(r['class_id']) == 1]
        print(f" Кадров с опасными аномалиями на путях:  {len(danger_frames)}")
        if danger_frames:
            distances = [abs(float(r['z_center'])) for r in danger_frames]
            print(f" Минимальная дистанция фиксации угрозы: {min(distances):.3f} м")
            print(f" Максимальная дальность раннего зрения: {max(distances):.3f} м")
            print("\n📋 ХРОНОЛОГИЧЕСКИЙ СРЕЗ ПЕРВЫХ ДЕТЕКЦИЙ ДЛЯ СВЕРКИ:")
            print(f"{'Имя кадра':<20} | {'Детекция':<10} | {'Дальность Z вперед':<15} | {'Смещение X':<12}")
            print("-"*80)
            for r in danger_frames[:10]:
                print(f"{r['frame_id']:<20} | {'🚨 ДА' if int(r['class_id'])==1 else 'НЕТ':<10} | {abs(float(r['z_center'])):<15.3f} | {r['x_center']:<12}")
        else:
            print(" ✅ Вся сессия пройдена в штатном режиме. Пути абсолютно чистые.")
        print("="*80 + "\n")
    
        print(f" 📂 ВСЕГО ОБРАБОТАНО УНИКАЛЬНЫХ КАДРОВ ЛИДАРА: {self.total_processed_frames} шт.")
        print(f" 📦 СУММАРНО НАЙДЕНО ФИЗИЧЕСКИХ ОБЪЕКТОВ ИИ:    {self.total_detected_obstacles} шт.")


def main(args=None):
    rclpy.init(args=args)
    node = SubwayVisionCoreNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Получен сигнал Ctrl+C. Выполняем шатдаун...')
    finally:
        node.is_running = False
        node.export_final_submission()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()



# #!/usr/bin/env python3
# """
# 🚀 PRODUCTION ROS 2 HUMBLE COMPLIANT NODE (STRICT CHRONO BATOP EDITION) - PART 1
# Промышленный конвейер детекции препятствий Мосметро строго по ТЗ жюри.
# Защищен от вылетов сети через нативный поток Linux и автоматический триггер тишины.
# """

# import sys
# import os
# import gc
# import csv
# import time
# import json
# import threading
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud2
# from std_msgs.msg import String
# from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# # Импортируем тяжелые ИИ и ЦОС движки из ядра решения
# from metro_lidar.cos_processor_v12 import StableLidarOdometryV12
# import metro_lidar.cos_processor_v12 as cos_processor_v12
# from metro_lidar.generate_submission_v2 import process_point_cloud
# from metro_lidar.tracker_v2 import LidarObstacleTrackerV2
# import metro_lidar.config as config


# class SubwayVisionCoreNode(Node):
#     def __init__(self):
#         super().__init__('subway_vision_core_node')
        
#         try:
#             os.system("sysctl -w net.core.rmem_max=26214400 > /dev/null 2>&1")
#             os.system("sysctl -w net.core.rmem_default=26214400 > /dev/null 2>&1")
#         except Exception:
#             pass

#         # 🟢 НАСТРОЙКА ПРЕЦИЗИОННОГО ВОКЗАЛА В ОЗУ (/dev/shm)
#         self.station_dir = "/dev/shm/metro_station_buffer"
#         os.makedirs(self.station_dir, exist_ok=True)
        
#         # Очищаем перроны вокзала перед стартом новой сессии
#         for old_file in os.listdir(self.station_dir):
#             try:
#                 os.remove(os.path.join(self.station_dir, old_file))
#             except Exception:
#                 pass

#         self.net_cb_group = MutuallyExclusiveCallbackGroup()

#         from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
#         custom_qos = QoSProfile(
#             depth=30,  # Огромный буфер, чтобы FastDDS не дропнул кадры при набеге сети
#             reliability=ReliabilityPolicy.RELIABLE,
#             durability=DurabilityPolicy.VOLATILE
#         )
        
#         self.declare_parameter('lidar_topic', '/sensing/lidar/hesai128/pointcloud')
#         topic_name = self.get_parameter('lidar_topic').get_parameter_value().string_value
        
#         # Сетевая подписка ROS 2 (Наш высокоскоростной Накопитель — Producer)
#         self.subscription = self.create_subscription(
#             PointCloud2,
#             topic_name, 
#             self.lidar_callback,
#             custom_qos,
#             callback_group=self.net_cb_group
#         )
        
#         # Нативный ROS 2 издатель телеметрии
#         self.telemetry_pub = self.create_publisher(String, '/safetrain/telemetry', 10)
        
#         # ИИ-память сессии одометрии и трекинга
#         self.odometry_engine = StableLidarOdometryV12()
#         self.obstacle_tracker_engine = LidarObstacleTrackerV2()
        
#         # Раздельные счетчики вокзала:
#         self.net_frame_idx = 0       # Сколько вагонов поймала сеть
#         self.compute_frame_idx = 0   # Сколько вагонов обсчитал ИИ
        
#         self.dt = 0.1                
#         self.submission_records = []
#         self._csv_exported_flag = False
#         self.total_distance_meters = 0.0
        
#         # Метка времени для отслеживания таймера тишины сети
#         self.last_activity_time = time.time()
#         self.is_running = True
        
#         # 🔥 АВТОНОМНЫЙ ЗАПУСК: ИИ-вычислитель работает в нативном независимом потоке Linux
#         self.compute_thread = threading.Thread(target=self.detached_compute_worker, daemon=True)
#         self.compute_thread.start()
        
#         self.get_logger().info(f'🚇 [АВТОНОМНЫЙ ВОКЗАЛ ОЗУ ЗАПУЩЕН]: Ждем поток на топике: {topic_name}')

#     def lidar_callback(self, msg):
#         import ros2_numpy
        
#         if self.net_frame_idx >= 201:
#             return
            
#         local_net_idx = self.net_frame_idx
#         self.net_frame_idx += 1

#         try:
#             arr_dict = ros2_numpy.numpify(msg)
#             xyz = arr_dict['xyz'].astype(np.float32)
#             num_points = len(xyz)
            
#             if 'intensity' in arr_dict:
#                 intensity = arr_dict['intensity'].astype(np.float32).reshape(-1, 1)
#             else:
#                 intensity = np.zeros((num_points, 1), dtype=np.float32)
                
#             points = np.hstack((xyz, intensity))
            
#             nan_mask = np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(points[:, 2])
#             points = points[~nan_mask]
            
#             nonzero_mask = np.any(points[:, :3] != 0, axis=1)
#             pure_points = points[nonzero_mask].copy()
            
#             if len(pure_points) == 0:
#                 return

#             # ТАК ПРАВИЛЬНО: Сохраняем сырой массив без ручного разворота
#             virtual_file_path = os.path.join(self.station_dir, f"shm_frame_{local_net_idx:04d}.bin")
#             pure_points.tofile(virtual_file_path)
                
#             self.last_activity_time = time.time()
                
#         except Exception as e:
#             self.get_logger().error(f"❌ [СБОЙ СЕТЕВОГО ПРИЕМА]: {e}")


#     def detached_compute_worker(self):
#         """🏢 АВТОНОМНЫЙ ИИ-ДЕМОН (CONSUMER): Разгружает вокзал ОЗУ по триггеру тишины"""
#         while self.is_running:
#             local_compute_idx = self.compute_frame_idx
#             target_bin_path = os.path.join(self.station_dir, f"shm_frame_{local_compute_idx:04d}.bin")
            
#             # Проверяем наличие следующего хронологического файла на вокзале ОЗУ
#             if not os.path.exists(target_bin_path):
#                 # 🚨 АВТОМАТИЧЕСКИЙ ТРИГГЕР ТИШИНЫ СЕТИ
#                 # Если мы обработали хотя бы один кадр, но на путях пусто
#                 if local_compute_idx > 0 and not self._csv_exported_flag:
#                     time_since_last_packet = time.time() - self.last_activity_time
                    
#                     # Если сеть молчит дольше 3.0 секунд — плеер гарантированно закрылся!
#                     if time_since_last_packet > 3.0:
#                         print(f"\n⏱️  [ТРИГГЕР ТИШИНЫ]: Сеть молчит {time_since_last_packet:.1f} сек. Автофиниш!")
#                         print(f"🏁 [ЦУП АВТОНОМНЫЙ ФИНИШ]: Всего успешно обсчитано ИИ-ядром: {local_compute_idx} кадров.")
#                         self.export_final_submission()
#                         self._csv_exported_flag = True
#                         self.print_final_summary_report()
#                         print("🟢 [ЦУП ИЗОЛЯЦИЯ]: Сабмит сформирован автоматически. Ожидание Ctrl+C в ноде.\n")
                
#                 # Микропауза 10 мс, чтобы не грузить ядро процессора вхолостую
#                 time.sleep(0.01)
#                 continue

#             # Инкрементируем счетчик ИИ-разгрузки вокзала
#             self.compute_frame_idx += 1

#             try:
#                 raw_points = np.fromfile(target_bin_path, dtype=np.float32).reshape(-1, 4)
                
#                 frontend_frame_data = {
#                     "danger_alert": False, "train_speed_kmh": 0.0,
#                     "total_distance_m": round(self.total_distance_meters, 2),
#                     "distance_to_obstacle_m": -1.0, "show_mesh_boxes": config.V2_OBSTACLE_SHOW_3D_MESHBOX,
#                     "objects": []
#                 }

#                 # Расчет плотности рельс (по уже исправленным оффлайн-осям ИИ)
#                 x_pts, y_pts, z_pts = raw_points[:, 0], raw_points[:, 1], raw_points[:, 2]
#                 rail_points_mask = (z_pts >= -63.5) & (z_pts <= -3.5) & \
#                                    (x_pts >= -0.75) & (x_pts <= 0.75) & \
#                                    (y_pts >= -1.85) & (y_pts <= -1.05)
#                 current_rail_points_count = int(np.sum(rail_points_mask))

#                 # 1. Одометрия рельсов (ОЯР)
#                 shift_z_rails, rail_passport = self.odometry_engine.compute_raw_rail_odo_shift(raw_points, self.dt)
#                 if rail_passport is not None:
#                     rx, ry, rz = rail_passport["centroid"]
#                     frontend_frame_data["objects"].append({
#                         "id": f"RAIL_ANCHOR_{rail_passport.get('id', '888')}", "class_id": 2, "confidence": 1.0,
#                         "center": [float(rx), float(ry), -abs(float(rz))], "size_3d": [0.6, 0.1, 1.2], "obstacle_speed_kmh": 0.0, "ttl": int(rail_passport.get("ttl", 60))
#                     })

#                 # 2. Одометрия стен тоннеля (ЭЯ)
#                 macro_cloud = self.odometry_engine.extract_clean_macro_tunnel(target_bin_path)
#                 if macro_cloud is not None:
#                     passports = self.odometry_engine.build_passports_via_dbscan(macro_cloud)
#                     calculate_speed_trigger = bool(local_compute_idx > 0)
#                     shift_z_walls, _, _ = self.odometry_engine.associate_and_calculate_shift(passports, self.dt, calculate_speed=calculate_speed_trigger)
#                 else:
#                     shift_z_walls = 0.0

#                 # 3. Гибридный ИИ-фьюжн шлюз скоростей
#                 shift_z_physical = cos_processor_v12.calculate_adaptive_fusion_shift(shift_z_rails, shift_z_walls, self.odometry_engine.prev_velocity_kmh, current_rail_points_count, local_compute_idx, self.dt)
#                 calculated_speed_kmh = (shift_z_physical / self.dt) * 3.6
#                 if calculated_speed_kmh < 0.2:
#                     calculated_speed_kmh = 0.0
#                     shift_z_physical = 0.0
                    
#                 self.odometry_engine.prev_velocity_kmh = calculated_speed_kmh
#                 self.total_distance_meters += shift_z_physical

#                 # Пакуем макро-стены ЭЯ под Three.js
#                 valid_walls_anchors = [obj for obj in self.odometry_engine.anchor_map if obj.get("status") == "STATUS_VALID_ANCHOR" and obj.get("is_wall", False)]
#                 for wall_obj in valid_walls_anchors:
#                     pts = wall_obj["raw_points"]
#                     if len(pts) > 100: pts = pts[::3]
#                     three_pts = [[float(p), float(p), -abs(float(p))] for p in pts]
#                     frontend_frame_data["objects"].append({
#                         "id": f"WALL_{wall_obj.get('id', '?')}", "class_id": 0, "confidence": 1.0, "center": [0.0, 0.0, 0.0], "size_3d": [0.0, 0.0, 0.0], "wall_points": three_pts, "obstacle_speed_kmh": 0.0
#                     })

#                 is_open_space = False
#                 if valid_walls_anchors:
#                     wall_x_coords = [float(w_obj["centroid"]) for w_obj in valid_walls_anchors if "centroid" in w_obj]
#                     if wall_x_coords and (max(wall_x_coords) > config.V2_OBSTACLE_WALL_OPEN_THRESHOLD or min(wall_x_coords) < -config.V2_OBSTACLE_WALL_OPEN_THRESHOLD):
#                         is_open_space = True

#                 safe_shift_z = float(np.abs(shift_z_physical))

#                 # 4. ИИ-ДВИЖОК ДЕТЕКЦИИ ПРЕПЯТСТВИЙ (Pipeline v2)
#                 detected_obstacles = process_point_cloud(target_bin_path, self.obstacle_tracker_engine, train_step_z=safe_shift_z, is_open_space=is_open_space)

#                 # 5. ПРЕЦИЗИОННАЯ УПАКОВКА РЕЗУЛЬТАТОВ (БЕЗ ОШИБОК ТИПОВ СПИСКОВ)
#                 if isinstance(detected_obstacles, list) and len(detected_obstacles) > 0:
#                     frontend_frame_data["danger_alert"] = True
                    
#                     # Извлекаем первый подтвержденный объект из списка
#                     closest_obs = detected_obstacles[0]  
#                     center_vector = closest_obs["center"]      # Нативный список [X, Y, Z]
#                     size_vector = closest_obs["dimensions"]    # Нативный список [W, H, D]
                    
#                     # Ювелирно вытаскиваем координаты float по индексам
#                     cx, cy, cz = float(center_vector[0]), float(center_vector[1]), float(center_vector[2])
#                     w, h, d = float(size_vector[0]), float(size_vector[1]), float(size_vector[2])
                    
#                     min_dist = abs(cz)
#                     frontend_frame_data["distance_to_obstacle_m"] = round(min_dist, 1)
                    
#                     # Запись в массив сабмита строго по REP 103 контракту жюри
#                     self.submission_records.append({
#                         'frame_id': f"frame_{local_compute_idx:06d}.bin", 
#                         'obstacle_detected': 1, 'distance_m': round(min_dist, 3),
#                         'center_x': round(abs(cz), 3), 'center_y': round(cx, 3), 'center_z': round(cy, 3),
#                         'size_x': round(d, 3), 'size_y': round(w, 3), 'size_z': round(h, 3)
#                     })

#                     # Пробрасываем геометрию на фронтенд топика телеметрии
#                     for obj_idx, obj in enumerate(detected_obstacles):
#                         ocx, ocy, ocz = obj["center"][0], obj["center"][1], obj["center"][2]
#                         ow, oh, od = obj["dimensions"][0], obj["dimensions"][1], obj["dimensions"][2]
#                         three_x, three_y, three_z = float(ocx), float(ocy), -abs(float(ocz))
                        
#                         shape_text, shape_type = "Объемная коробка / Блок", "BOX_BLOCK"
#                         if oh > ow and oh > od and oh > 1.0: shape_text, shape_type = "Человек / Вертикальная конструкция", "VERTICAL_SILHOUETTE"
#                         elif oh < 0.40 and (ow > 0.6 or od > 0.6): shape_text, shape_type = "Плоский предмет / Настил", "FLAT_OBSTACLE"

#                         position_text, position_status = "Строго по центру путей 🚨", "CENTER"
#                         if three_x < -0.35: position_text, position_status = "Касается левой кромки габарита ⚠️", "LEFT_EDGE"
#                         elif three_x > 0.35: position_text, position_status = "Касается правой кромки габарита ⚠️", "RIGHT_EDGE"

#                         three_pts = [[float(pt[0]), float(pt[1]), -abs(float(pt[2]))] for pt in obj.get("raw_points", [])]
#                         frontend_frame_data["objects"].append({
#                             "id": str(obj.get("id", f"📦_{obj_idx}")), "class_id": 1, "confidence": float(obj.get("confidence", 1.0)),
#                             "center": [three_x, three_y, three_z], "size_3d": [float(ow), float(oh), float(od)], "obstacle_speed_kmh": float(obj.get("obstacle_speed", 0.0)),
#                             "position_status": position_status, "position_text": position_text, "shape_type": shape_type, "shape_text": shape_text, "obstacle_points": three_pts
#                         })
#                 else:
#                     self.submission_records.append({
#                         'frame_id': f"frame_{local_compute_idx:06d}.bin", 
#                         'obstacle_detected': 0, 'distance_m': -1.0,
#                         'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0, 'size_x': 0.0, 'size_y': 0.0, 'size_z': 0.0
#                     })

#                 # Нативные ROS 2 топики продолжают отправку
#                 frontend_frame_data["train_speed_kmh"] = round(float(calculated_speed_kmh), 1)
#                 frontend_frame_data["total_distance_m"] = round(float(self.total_distance_meters), 1)
                
#                 if len(frontend_frame_data["objects"]) == 0:
#                     frontend_frame_data["objects"].append({
#                         "id": "RETAINER", 
#                         "class_id": -1, 
#                         "confidence": 1.0, 
#                         "center": [0.0, -20.0, 0.0], 
#                         "size_3d": [0.01, 0.01, 0.01], 
#                         "obstacle_speed_kmh": 0.0
#                     })

#                 # 🟢 ВЫСТРЕЛИВАЕМ СФОРМИРОВАННУЮ 3D-СЦЕНУ В НА ТИВНЫЙ ТОПИК ROS 2
#                 ros_string_msg = String()
#                 ros_string_msg.data = json.dumps(frontend_frame_data)
#                 self.telemetry_pub.publish(ros_string_msg)

#             except Exception as e:
#                 self.get_logger().error(f"❌ [КРИТИЧЕСКИЙ СБОЙ ИИ-ВЫЧИСЛЕНИЙ НА КАДРЕ #{local_compute_idx}]: {e}")
#                 self.submission_records.append({
#                     'frame_id': f"frame_{local_compute_idx:06d}.bin", 
#                     'obstacle_detected': 0, 'distance_m': -1.0,
#                     'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0, 
#                     'size_x': 0.0, 'size_y': 0.0, 'size_z': 0.0
#                 })
#             finally:
#                 # 🟢 СИГНАЛ НА УТИЛИЗАЦИЮ: Вагон полностью разгружен ИИ-движком. 
#                 # Физически удаляем отработавший файл из ОЗУ-вокзала, чтобы не копить мусор
#                 try:
#                     if os.path.exists(target_bin_path):
#                         os.remove(target_bin_path)
#                 except Exception:
#                     pass
#                 gc.collect()



#     def export_final_submission(self):
#         """Записывает накопленный кэш кадров в файл submission.csv строго перед закрытием"""
#         output_path = "/app/submission.csv"
#         if not self.submission_records:
#             self.get_logger().error("⚠️ Буфер сабмита пуст. CSV не сохранен.")
#             return
#         try:
#             self.submission_records.sort(key=lambda x: x['frame_id'])
#             fields = ["frame_id", "x_center", "y_center", "z_center", "class_id", "confidence"]
#             with open(output_path, mode='w', newline='', encoding='utf-8') as f:
#                 writer = csv.DictWriter(f, fieldnames=fields)
#                 writer.writeheader()
#                 writer.writerows(self.submission_records)
#             self.get_logger().info(f"💾 [ИИ-ЭКСПОРТ УСПЕШЕН]: Сабмит сохранен в: {output_path}")
#             self.get_logger().info(f"📊 Всего лидарных кадров зафиксировано в итоговом CSV: {len(self.submission_records)}")
#         except Exception as e:
#             self.get_logger().error(f"🚨 Ошибка записи CSV: {e}")

#     def print_final_summary_report(self):
#         """Печатает красивый аудит-отчет накопленного массива submission_records для жюри"""
#         print("\n" + "="*80)
#         print("📊  ФИНАЛЬНЫЙ ИНСПЕКЦИОННЫЙ ОТЧЕТ ИИ-КОНВЕЙЕРА (СУДЬИ ХАКАТОНА)")
#         print("="*80)
#         print(f" Всего зафиксировано и обсчитано кадров: {len(self.submission_records)}")
        
#         danger_frames = [r for r in self.submission_records if r['obstacle_detected'] == 1]
#         print(f" Кадров с опасными аномалиями на путях:  {len(danger_frames)}")
        
#         if danger_frames:
#             distances = [r['distance_m'] for r in danger_frames]
#             print(f" Минимальная дистанция фиксации угрозы: {min(distances):.3f} м")
#             print(f" Максимальная дальность раннего зрения: {max(distances):.3f} м")
            
#             print("\n📋 ХРОНОЛОГИЧЕСКИЙ СРЕЗ ПЕРВЫХ ДЕТЕКЦИЙ ДЛЯ СВЕРКИ:")
#             print(f"{'Имя кадра':<20} | {'Детекция':<10} | {'Дистанция':<12} | {'Центр X (Дальность вперед)':<15}")
#             print("-"*80)
#             for r in danger_frames[:10]:
#                 print(f"{r['frame_id']:<20} | {'🚨 ДА' if r['obstacle_detected']==1 else 'НЕТ':<10} | {r['distance_m']:<12.3f} | {r['center_x']:<15.3f}")
#         else:
#             print(" ✅ Вся сессия пройдена в штатном режиме. Пути абсолютно чистые.")
#         print("="*80 + "\n")


# def main(args=None):
#     rclpy.init(args=args)
#     node = SubwayVisionCoreNode()
    
#     try:
#         # Крутим ноду в строгом, надежном одиночном потоке CPU
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('🛑 Получен сигнал Ctrl+C. Выполняем шатдаун...')
#     finally:
#         # Железная страховка: если жюри прервало бэг раньше времени, 
#         # принудительно сбрасываем накопленный кэш в submission.csv
#         node.export_final_submission()
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == '__main__':
#     main()

# #!/usr/bin/env python3
# """
# 🚀 PRODUCTION ROS 2 HUMBLE COMPLIANT NODE (STRICT CHRONO DELAYED BATCH EDITION) - PART 1
# Промышленный конвейер детекции препятствий Мосметро строго по ТЗ жюри.
# Защищен от прореживания лучей через разделение во времени: Сбор ОЗУ ➔ Автономный ИИ Расчет.
# """

# import sys
# import os
# import gc
# import csv
# import time
# import json
# import threading
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud2
# from std_msgs.msg import String
# from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# # Импортируем тяжелые ИИ и ЦОС движки из ядра решения
# from metro_lidar.cos_processor_v12 import StableLidarOdometryV12
# import metro_lidar.cos_processor_v12 as cos_processor_v12
# from metro_lidar.generate_submission_v2 import process_point_cloud
# from metro_lidar.tracker_v2 import LidarObstacleTrackerV2
# import metro_lidar.config as config


# class SubwayVisionCoreNode(Node):
#     def __init__(self):
#         super().__init__('subway_vision_core_node')
        
#         try:
#             os.system("sysctl -w net.core.rmem_max=26214400 > /dev/null 2>&1")
#             os.system("sysctl -w net.core.rmem_default=26214400 > /dev/null 2>&1")
#         except Exception:
#             pass

#         # 🟢 НАСТРОЙКА ПРЕЦИЗИОННОГО ВОКЗАЛА В ОЗУ (/dev/shm)
#         self.station_dir = "/dev/shm/metro_station_buffer"
#         os.makedirs(self.station_dir, exist_ok=True)
        
#         # Очищаем перроны вокзала перед стартом новой сессии
#         for old_file in os.listdir(self.station_dir):
#             try:
#                 os.remove(os.path.join(self.station_dir, old_file))
#             except Exception:
#                 pass

#         self.net_cb_group = MutuallyExclusiveCallbackGroup()

#         from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
#         custom_qos = QoSProfile(
#             depth=100,  # Максимальный карман DDS, чтобы не потерять ни одного луча колеи
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             durability=DurabilityPolicy.VOLATILE
#         )
        
#         self.declare_parameter('lidar_topic', '/sensing/lidar/hesai128/pointcloud')
#         topic_name = self.get_parameter('lidar_topic').get_parameter_value().string_value
        
#         # Сетевая подписка ROS 2 (Наш высокоскоростной Накопитель — Батчер)
#         self.subscription = self.create_subscription(
#             PointCloud2,
#             topic_name, 
#             self.lidar_callback,
#             custom_qos,
#             callback_group=self.net_cb_group
#         )
        
#         # Нативный ROS 2 издатель телеметрии
#         self.telemetry_pub = self.create_publisher(String, '/safetrain/telemetry', 10)
        
#         # Инициализируем хронологические движки ИИ-памяти сессии
#         self.odometry_engine = StableLidarOdometryV12()
#         self.obstacle_tracker_engine = LidarObstacleTrackerV2()
        
#         self.net_frame_idx = 0       # Сколько кадров сохранила сеть
#         self.compute_frame_idx = 0   # Сколько кадров обсчитал ИИ
        
#         self.dt = 0.1                
#         self.submission_records = []
#         self._csv_exported_flag = False
#         self.total_distance_meters = 0.0
        
#         # Метка времени для отслеживания таймера тишины сети (безопасные 15.0 сек)
#         self.last_activity_time = time.time()
#         self.is_running = True
        
#         # Флаг-семафор: ИИ-поток спит, пока сеть качает вагоны
#         self.network_active_phase = True
        
#         # 🔥 АВТОНОМНЫЙ ВЫЧИСЛИТЕЛЬНЫЙ ПОТОК: ИИ-воркер работает независимо от rclpy.spin()
#         self.compute_thread = threading.Thread(target=self.detached_compute_worker, daemon=True)
#         self.compute_thread.start()
        
#         self.get_logger().info(f'🚇 [БАТЧ-КОНВЕЙЕР АКТИВИРОВАН]: Включен режим 100% сбора лучей колеи!')
#         self.get_logger().info(f'📡 Ждем входящий поток PointCloud2 на топике: {topic_name}')

#     def lidar_callback(self, msg):
#         """ 🚂 СЕТЕВОЙ ПОТОК (PRODUCER): Работает со скоростью 200 Гц, только сбрасывает данные в ОЗУ """
#         import ros2_numpy
        
#         if self.net_frame_idx >= 201:
#             return
            
#         local_net_idx = self.net_frame_idx
#         self.net_frame_idx += 1

#         try:
#             # Прецизионный побайтовый оффлайн-формат из convert_all_bags.py
#             arr_dict = ros2_numpy.numpify(msg)
            
#             xyz = arr_dict['xyz'].astype(np.float32)
#             num_points = len(xyz)
            
#             if 'intensity' in arr_dict:
#                 intensity = arr_dict['intensity'].astype(np.float32).reshape(-1, 1)
#             else:
#                 intensity = np.zeros((num_points, 1), dtype=np.float32)
                
#             points = np.hstack((xyz, intensity))
            
#             nan_mask = np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(points[:, 2])
#             points = points[~nan_mask]
            
#             nonzero_mask = np.any(points[:, :3] != 0, axis=1)
#             pure_points = points[nonzero_mask]
            
#             if len(pure_points) == 0:
#                 return

#             # Сбрасываем вагон на вокзал ОЗУ
#             virtual_file_path = os.path.join(self.station_dir, f"shm_frame_{local_net_idx:04d}.bin")
#             pure_points.tofile(virtual_file_path)
                
#             # Постоянно сдвигаем таймер тишины: сеть жива!
#             self.last_activity_time = time.time()
                
#         except Exception as e:
#             self.get_logger().error(f"❌ [СБОЙ СЕТЕВОГО ПРИЕМА НА КАДРЕ #{local_net_idx}]: {e}")
#     def detached_compute_worker(self):
#         """🏢 АВТОНОМНЫЙ ИИ-ДЕМОН (CONSUMER): Начинает расчет только ПОСЛЕ затишья сети"""
#         while self.is_running:
#             # СТАДИЯ А: Ожидание полной остановки бэг-плеера и затишья сети
#             if self.network_active_phase:
#                 if self.net_frame_idx > 0:
#                     time_since_last_packet = time.time() - self.last_activity_time
                    
#                     # Если сеть молчит дольше 15.0 секунд — плеер гарантированно завершил луп!
#                     if time_since_last_packet > 15.0:
#                         print(f"\n⏱️  [ТРИГГЕР ТИШИНЫ]: Сеть молчит {time_since_last_packet:.1f} сек.")
#                         print(f"🏁 [ЦУП ПЕРЕХОД]: Сбор завершен. Поймано вагонов: {self.net_frame_idx}")
#                         print("🧠 [ИИ-ЯДРО]: Включаю автономную разгрузку вокзала ОЗУ на полную мощность...\n")
#                         self.network_active_phase = False  # Переключаем семафор в режим расчетов
                
#                 # Пока сеть качает лучи колеи, ИИ-поток крепко спит и не мешает сокетам
#                 time.sleep(0.1)
#                 continue

#             # СТАДИЯ Б: Последовательный покадровый ИИ-расчет накопленного буфера
#             local_compute_idx = self.compute_frame_idx
#             total_saved_frames = self.net_frame_idx
            
#             # Если мы обсчитали все файлы, сохраненные сетью на вокзале
#             if local_compute_idx >= total_saved_frames:
#                 if not self._csv_exported_flag:
#                     print(f"\n🏁 [ЦУП АВТОНОМНЫЙ ФИНИШ]: Вокзал ОЗУ полностью опустел!")
#                     print(f"   ↳ Всего успешно обсчитано ИИ-ядром: {local_compute_idx} кадров.")
#                     self.export_final_submission()
#                     self._csv_exported_flag = True
#                     self.print_final_summary_report()
#                     print("🟢 [ЦУП ИЗОЛЯЦИЯ]: Сабмит сформирован автоматически. Ожидание Ctrl+C в ноде.\n")
                
#                 time.sleep(0.1)
#                 continue

#             target_bin_path = os.path.join(self.station_dir, f"shm_frame_{local_compute_idx:04d}.bin")
#             if not os.path.exists(target_bin_path):
#                 self.compute_frame_idx += 1
#                 continue

#             # Инкрементируем счетчик ИИ-разгрузки вокзала
#             self.compute_frame_idx += 1

#             try:
#                 frontend_frame_data = {
#                     "danger_alert": False, "train_speed_kmh": 0.0,
#                     "total_distance_m": round(self.total_distance_meters, 2),
#                     "distance_to_obstacle_m": -1.0, "show_mesh_boxes": config.V2_OBSTACLE_SHOW_3D_MESHBOX,
#                     "objects": []
#                 }

#                 # =====================================================================
#                 # 🎯 ЭТАЛОННЫЙ ВЫЗОВ ОДОМЕТРИИ ПУЛЯ-В-ПУЛЮ КАК В QUICK_CHECK.PY
#                 # =====================================================================
#                 # Одометрия нативно зачитает бинарник первой по строковому пути файла
#                 shift_z_raw, _ = self.odometry_engine.update_odometry_fusion(
#                     target_bin_path, 
#                     local_compute_idx, 
#                     dt=self.dt
#                 )
                
#                 # Гарантия скаляра: принудительно вычищаем любые хитрые массивы NumPy
#                 if hasattr(shift_z_raw, "item"):
#                     shift_z_physical = float(shift_z_raw.item())
#                 else:
#                     shift_z_physical = float(shift_z_raw)
                
#                 # Извлекаем скорость и пройденный путь из памяти монолита одометрии
#                 calculated_speed_kmh = self.odometry_engine.prev_velocity_kmh
#                 self.total_distance_meters += shift_z_physical

#                 # Пакуем макро-стены ЭЯ под Three.js топик телеметрии
#                 valid_walls_anchors = [obj for obj in self.odometry_engine.anchor_map if obj.get("status") == "STATUS_VALID_ANCHOR" and obj.get("is_wall", False)]
#                 for wall_obj in valid_walls_anchors:
#                     pts = wall_obj["raw_points"]
#                     if len(pts) > 100: pts = pts[::3]
                    
#                     # 🟢 ИИ-ФИКС: Распаковываем координаты по индексам, исключая ошибки скаляров!
#                     three_pts = [[float(p[0]), float(p[1]), -abs(float(p[2]))] for p in pts]
                    
#                     frontend_frame_data["objects"].append({
#                         "id": f"WALL_{wall_obj.get('id', '?')}", "class_id": 0, "confidence": 1.0, "center": [0.0, 0.0, 0.0], "size_3d": [0.0, 0.0, 0.0], "wall_points": three_pts, "obstacle_speed_kmh": 0.0
#                     })

#                 is_open_space = False
#                 if valid_walls_anchors:
#                     # 🟢 ИИ-ФИКС: Берем координату X (индекс 0) из вектора центроида w_obj["centroid"]
#                     wall_x_coords = [float(w_obj["centroid"][0]) for w_obj in valid_walls_anchors if "centroid" in w_obj]
#                     if wall_x_coords and (max(wall_x_coords) > config.V2_OBSTACLE_WALL_OPEN_THRESHOLD or min(wall_x_coords) < -config.V2_OBSTACLE_WALL_OPEN_THRESHOLD):
#                         is_open_space = True
#                 safe_shift_z = float(np.abs(shift_z_physical))

#                 # 4. ИИ-ДВИЖОК ДЕТЕКЦИИ ПРЕПЯТСТВИЙ (Pipeline v2)
#                 # Передаем честный путь к файлу в ОЗУ, который ждет ваш внутренний .endswith()!
#                 detected_obstacles = process_point_cloud(
#                     target_bin_path, 
#                     self.obstacle_tracker_engine, 
#                     train_step_z=safe_shift_z, 
#                     is_open_space=is_open_space
#                 )

#                 # 5. ПРЕЦИЗИОННАЯ УПАКОВКА РЕЗУЛЬТАТОВ ДЕТЕКЦИИ ДЛЯ ЖЮРИ (БРОНИРОВАННАЯ К ТИПАМ)
#                 if isinstance(detected_obstacles, list) and len(detected_obstacles) > 0:
#                     frontend_frame_data["danger_alert"] = True
                    
#                     # Извлекаем первый объект дляHUD панели
#                     first_obs = detected_obstacles[0]
#                     c_vec = first_obs["center"]
                    
#                     # Безопасно определяем координату дальности Z вперед
#                     if hasattr(c_vec, "__len__") and len(c_vec) >= 3:
#                         cz_val = abs(float(c_vec[2]))
#                     else:
#                         cz_val = abs(float(c_vec))
                        
#                     frontend_frame_data["distance_to_obstacle_m"] = round(cz_val, 1)
                    
#                     # Сохраняем строки для формирования submission.csv в эталонном формате
#                     for obj in detected_obstacles:
#                         c = obj["center"]
#                         conf_val = obj.get("confidence", 1.0)
                        
#                         # Бронированная распаковка центра [X, Y, Z]
#                         if hasattr(c, "__len__") and len(c) >= 3:
#                             cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
#                         else:
#                             cx, cy, cz = 0.0, 0.0, float(c)
                            
#                         self.submission_records.append({
#                             "frame_id": f"frame_{local_compute_idx:04d}.bin",
#                             "x_center": f"{cx:.4f}", "y_center": f"{cy:.4f}", "z_center": f"{cz:.4f}",
#                             "class_id": 1, "confidence": f"{conf_val:.4f}"
#                         })

#                     # Пробрасываем сцену в String топик для веб-панелей фронтенда
#                     for obj_idx, obj in enumerate(detected_obstacles):
#                         c = obj["center"]
#                         d_vec = obj["dimensions"]
                        
#                         if hasattr(c, "__len__") and len(c) >= 3:
#                             ocx, ocy, ocz = float(c[0]), float(c[1]), float(c[2])
#                         else:
#                             ocx, ocy, ocz = 0.0, 0.0, float(c)
                            
#                         if hasattr(d_vec, "__len__") and len(d_vec) >= 3:
#                             ow, oh, od = float(d_vec[0]), float(d_vec[1]), float(d_vec[2])
#                         else:
#                             ow, oh, od = 0.5, 0.5, 0.5
                        
#                         shape_text, shape_type = "Объемная коробка / Блок", "BOX_BLOCK"
#                         if oh > ow and oh > od and oh > 1.0: shape_text, shape_type = "Человек / Вертикальная конструкция", "VERTICAL_SILHOUETTE"
#                         elif oh < 0.40 and (ow > 0.6 or od > 0.6): shape_text, shape_type = "Плоский предмет / Настил", "FLAT_OBSTACLE"

#                         position_text, position_status = "Строго по центру путей 🚨", "CENTER"
#                         if ocx < -0.35: position_text, position_status = "Касается левой кромки габарита ⚠️", "LEFT_EDGE"
#                         elif ocx > 0.35: position_text, position_status = "Касается правой кромки габарита ⚠️", "RIGHT_EDGE"

#                         three_pts = [[float(pt[0]), float(pt[1]), -abs(float(pt[2]))] for pt in obj.get("raw_points", []) if hasattr(pt, "__len__") and len(pt) >= 3]
#                         frontend_frame_data["objects"].append({
#                             "id": str(obj.get("id", f"📦_{obj_idx}")), "class_id": 1, "confidence": float(obj.get("confidence", 1.0)),
#                             "center": [ocx, ocy, ocz], "size_3d": [ow, oh, od], "obstacle_speed_kmh": float(obj.get("obstacle_speed", 0.0)),
#                             "position_status": position_status, "position_text": position_text, "shape_type": shape_type, "shape_text": shape_text, "obstacle_points": three_pts
#                         })
#                 else:
#                     # Если препятствий нет — добавляем чистую строку по регламенту конкурса
#                     self.submission_records.append({
#                         "frame_id": f"frame_{local_compute_idx:04d}.bin",
#                         "x_center": "0.0000", "y_center": "0.0000", "z_center": "0.0000",
#                         "class_id": 0, "confidence": "0.0000"
#                     })


#                 # Нативные ROS 2 топики телеметрии продолжают отправку пакетов
#                 frontend_frame_data["train_speed_kmh"] = round(float(calculated_speed_kmh), 1)
#                 frontend_frame_data["total_distance_m"] = round(float(self.total_distance_meters), 1)
#                 if len(frontend_frame_data["objects"]) == 0:
#                     frontend_frame_data["objects"].append({"id": "RETAINER", "class_id": -1, "confidence": 1.0, "center": [0.0, -20.0, 0.0], "size_3d": [0.01, 0.01, 0.01], "obstacle_speed_kmh": 0.0})

#                 ros_string_msg = String()
#                 ros_string_msg.data = json.dumps(frontend_frame_data)
#                 self.telemetry_pub.publish(ros_string_msg)

#             except Exception as e:
#                 # 🔴 НАША ПРЕЦИЗИОННАЯ СЛУЖБА ДЕБАГА ПРЯМО В КОНСОЛЬ НОДЫ
#                 import traceback
#                 print("\n" + "🚨" * 40)
#                 print(f" КРИТИЧЕСКИЙ СБОЙ НА КАДРЕ #{local_compute_idx}")
#                 print(f" Тип ошибки: {type(e).__name__}")
#                 print(f" Текст ошибки: {e}")
#                 print("📋 ТОЧНЫЙ СТЭК ПАДЕНИЯ ВНУТРИ ИИ-ДВИЖКОВ (СМОТРИ СТРОКУ):")
#                 traceback.print_exc(file=sys.stdout)
#                 print("🚨" * 40 + "\n")

#                 self.submission_records.append({
#                     "frame_id": f"frame_{local_compute_idx:04d}.bin",
#                     "x_center": "0.0000", "y_center": "0.0000", "z_center": "0.0000",
#                     "class_id": 0, "confidence": "0.0000"
#                 })
#             finally:
#                 # Безопасно утилизируем отработавший бинарник из ОЗУ-вокзала
#                 try:
#                     if os.path.exists(target_bin_path):
#                         os.remove(target_bin_path)
#                 except Exception:
#                     pass
#                 gc.collect()

#     def export_final_submission(self):
#         """Записывает накопленный кэш кадров в файл submission.csv в судейском формате"""
#         output_path = "/app/submission.csv"
#         if not self.submission_records:
#             self.get_logger().error("⚠️ Буфер сабмита пуст. CSV не сохранен.")
#             return
#         try:
#             self.submission_records.sort(key=lambda x: x['frame_id'])
#             fields = ["frame_id", "x_center", "y_center", "z_center", "class_id", "confidence"]
#             with open(output_path, mode='w', newline='', encoding='utf-8') as f:
#                 writer = csv.DictWriter(f, fieldnames=fields)
#                 writer.writeheader()
#                 writer.writerows(self.submission_records)
#             self.get_logger().info(f"💾 [ИИ-ЭКСПОРТ УСПЕШЕН]: Сабмит сохранен в: {output_path}")
#         except Exception as e:
#             self.get_logger().error(f"🚨 Ошибка записи CSV: {e}")

#     def print_final_summary_report(self):
#         """Печатает красивый аудит-отчет накопленного массива submission_records для жюри"""
#         print("\n" + "="*80)
#         print("📊  ФИНАЛЬНЫЙ ИНСПЕКЦИОННЫЙ ОТЧЕТ ИИ-КОНВЕЙЕРА (СУДЬИ ХАКАТОНА)")
#         print("="*80)
#         print(f" Всего зафиксировано и обсчитано кадров: {len(self.submission_records)}")
#         danger_frames = [r for r in self.submission_records if int(r['class_id']) == 1]
#         print(f" Кадров с опасными аномалиями на путях:  {len(danger_frames)}")
#         if danger_frames:
#             distances = [abs(float(r['z_center'])) for r in danger_frames]
#             print(f" Минимальная дистанция фиксации угрозы: {min(distances):.3f} м")
#             print(f" Максимальная дальность раннего зрения: {max(distances):.3f} м")
#             print("\n📋 ХРОНОЛОГИЧЕСКИЙ СРЕЗ ПЕРВЫХ ДЕТЕКЦИЙ ДЛЯ СВЕРКИ:")
#             print(f"{'Имя кадра':<20} | {'Детекция':<10} | {'Дальность Z вперед':<15} | {'Смещение X':<12}")
#             print("-"*80)
#             for r in danger_frames[:10]:
#                 print(f"{r['frame_id']:<20} | {'🚨 ДА' if int(r['class_id'])==1 else 'НЕТ':<10} | {abs(float(r['z_center'])):<15.3f} | {r['x_center']:<12}")
#         else:
#             print(" ✅ Вся сессия пройдена в штатном режиме. Пути абсолютно чистые.")
#         print("="*80 + "\n")


# def main(args=None):
#     rclpy.init(args=args)
#     node = SubwayVisionCoreNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('🛑 Получен сигнал Ctrl+C. Выполняем шатдаун...')
#     finally:
#         node.is_running = False
#         node.export_final_submission()
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()

# if __name__ == '__main__':
#     main()
