# # tracker_v2.py
# import numpy as np
# import metro_lidar.config as config
# from unified_object_engine import UnifiedLidarObjectEngine

# class LidarObstacleTrackerV2:
#     """
#     Промышленный межкадровый ИИ-трекер препятствий на базе Единого Паспорта Объекта.
#     Интегрирован с универсальным движком паспортизации и каскадной ассоциации.
#     """
    
#     def __init__(self):
#         """Инициализирует структуры данных, буферы автокалибровки и ИИ-ядро."""
#         self.dynamic_min_z_control = 3.5
#         self.is_cabin_calibrated = False
        
#         # Инициализируем универсальный движок ЕПО-Core для контура препятствий
#         # Задаем параметры воронки под ТЗ детектора вагона
#         self.object_engine = UnifiedLidarObjectEngine(
#             spatial_gate_radius=2.5,          # Радиальные ворота ассоциации
#             feature_match_threshold=0.65,      # Допуск на изменение формы/блеска преграды
#             max_ttl=config.TRACK_MAX_COASTING_AGE
#         )

#     def reset(self):
#         """Полный сброс межкадровой памяти для изоляции WebSocket-сессий FastAPI."""
#         self.is_cabin_calibrated = False
#         self.object_engine.tracked_registry = []
#         self.object_engine.global_id_counter = 0

#     def _calibrate_ego_vehicle_cabin(self, points):
#         """
#         Сканирует пространство перед лидаром и динамически отсекает бампер состава.
#         Синхронизирована с глобальными константами config.py и инвариантна к полярности Z.
#         """
#         if self.is_cabin_calibrated:
#             return self.dynamic_min_z_control

#         cx = points[:, 0]
#         cy = points[:, 1]
#         cz = points[:, 2]

#         # Берем дальность по модулю для защиты от смены полярности осей v12
#         absolute_z = np.abs(cz)

#         cabin_mask = (
#             (cx >= -config.TRAIN_HALF_WIDTH) & (cx <= config.TRAIN_HALF_WIDTH) &
#             (cy >= config.MIN_Y) & (cy <= config.MAX_Y) &
#             (absolute_z >= config.LIDAR_MIN_Z) & (absolute_z <= config.LIDAR_MAX_Z)
#         )
#         cabin_points_z = absolute_z[cabin_mask]

#         if len(cabin_points_z) > config.CABIN_MIN_POINTS_THRESHOLD:
#             max_cabin_edge = float(np.max(cabin_points_z))
#             self.dynamic_min_z_control = max_cabin_edge + 0.40
#             self.is_cabin_calibrated = True
#         else:
#             self.dynamic_min_z_control = config.LIDAR_MIN_Z
#             self.is_cabin_calibrated = True
            
#         return self.dynamic_min_z_control
#     """
#         Выполняет пространственный 3D NMS для склеивания осколков DBSCAN.
#         Принимает: Список массивов точек [N, 4] каждого сырого кластера.
#         Возвращает: Очищенный от дубликатов список укрупненных массивов точек.
#     """
#     def _collapse_duplicates(self, raw_clusters):
#         """
#         Выполняет пространственный 3D NMS для склеивания осколков DBSCAN.
#         Полностью адаптирована под промышленный стандарт словарей-паспортов.
#         """
#         if len(raw_clusters) <= 1:
#             return raw_clusters

#         # 🟢 ВЕКТОРНЫЙ ФИКС: Извлекаем центры из готовых словарей-паспортов без циклов по точкам
#         centers = np.array([c["center"] for c in raw_clusters], dtype=np.float32)
#         collapsed_clusters = []
#         visited = set()

#         for i, obj_a in enumerate(raw_clusters):
#             if i in visited:
#                 continue
                
#             current_macro_group = [obj_a]
#             visited.add(i)
            
#             for j, obj_b in enumerate(raw_clusters):
#                 if j in visited:
#                     continue
                    
#                 # Вычисляем расстояние между центрами 3D-боксов
#                 dist_3d = float(np.linalg.norm(centers[i] - centers[j]))
#                 if dist_3d < 1.5:
#                     current_macro_group.append(obj_b)
#                     visited.add(j)
            
#             # Если объект уникален, сохраняем его как есть
#             if len(current_macro_group) == 1:
#                 collapsed_clusters.append(obj_a)
#             else:
#                 # Если боксы склеились, пересчитываем общие макро-габариты результирующего объекта
#                 merged_obj = current_macro_group[0].copy()
                
#                 group_centers = np.array([c["center"] for c in current_macro_group])
#                 group_sizes = np.array([c["dimensions"] for c in current_macro_group])
                
#                 # Находим новые экстремальные границы объединенного 3D-бокса
#                 max_x = np.max(group_centers[:, 0] + group_sizes[:, 0]/2)
#                 min_x = np.min(group_centers[:, 0] - group_sizes[:, 0]/2)
#                 max_y = np.max(group_centers[:, 1] + group_sizes[:, 1]/2)
#                 min_y = np.min(group_centers[:, 1] - group_sizes[:, 1]/2)
#                 max_z = np.max(group_centers[:, 2] + group_sizes[:, 2]/2)
#                 min_z = np.min(group_centers[:, 2] - group_sizes[:, 2]/2)
                
#                 # Обновляем габариты и выставляем прецизионный центр масс
#                 merged_obj["dimensions"] = [max_x - min_x, max_y - min_y, max_z - min_z]
#                 merged_obj["center"] = [(max_x + min_x)/2.0, (max_y + min_y)/2.0, (max_z + min_z)/2.0]
                
#                 # Суммируем массу точек, если ключи присутствуют
#                 mass_key = "points_count" if "points_count" in merged_obj else "mass"
#                 merged_obj[mass_key] = sum([c.get(mass_key, 1) for c in current_macro_group])
                
#                 # Объединяем внутренние облака точек для WebGL визуализации
#                 if "raw_points" in merged_obj:
#                     combined_pts = []
#                     for c in current_macro_group:
#                         if "raw_points" in c and isinstance(c["raw_points"], list):
#                             combined_pts.extend(c["raw_points"])
#                     merged_obj["raw_points"] = combined_pts
                    
#                 collapsed_clusters.append(merged_obj)
            
#         return collapsed_clusters


#     def track_and_filter_ghosts(self, current_frame_clusters, train_step_z=0.0):
#         """
#         Унифицированный MOT-конвейер межкадровой ассоциации и фильтрации призраков вагона.
#         Принимает: Список массивов точек [N, 4] сырых кластеров из DBSCAN и шаг одометрии.
#         Возвращает: Список подтвержденных ИИ-препятствий для submission.csv и WebGL.
#         """
#         # 1. Запускаем 3D NMS слияние дубликатов на уровне сырых облаков точек
#         clean_clusters = self._collapse_duplicates(current_frame_clusters)
        
#         # 2. Передаем управление универсальному ИИ-ядру ЕПО-Core
#         # Метод сам посчитает паспорта (Массу, Интенсивность, Тензор формы), 
#         # компенсирует шаг train_step_z и проверит хронологию через ИИ-воронку
#         tracked_objects = self.object_engine.match_and_track_frame(
#             current_clusters=clean_clusters, 
#             train_step_z=train_step_z
#         )
        
#         final_safe_objects = []
        
#         # 3. Приведение выходящих паспортов к строгому конкурсному контракту API / Жюри
#         for obj in tracked_objects:
#             cx, cy, cz = obj["center"]
#             w, h, d = obj["dimensions"]
            
#             # На ходу или стоянке рассчитываем аналитику морфологии по сохраненному паспорту формы
#             linearity, planarity, sphericity = obj["geometry"]
            
#             if linearity > planarity and linearity > sphericity:
#                 shape_text = "Человек / Вертикальная конструкция"
#             elif planarity > linearity and planarity > sphericity:
#                 shape_text = "Плоский предмет / Настил"
#             else:
#                 shape_text = "Объемная коробка / Блок"
                
#             # Расчет аналитики отклонения от центра путей (ось X)
#             deviation_x = abs(cx)
#             if deviation_x <= 0.35:
#                 position_text = "Строго по центру путей 🚨"
#             elif cx < -0.35:
#                 position_text = "Касается левой кромки ⚠️"
#             else:
#                 position_text = "Касается правой кромки ⚠️"

#             # Формируем итоговую структуру под контракт generate_submission_v2.py
#             final_safe_objects.append({
#                 "id": obj["id"],
#                 "class_id": 1,
#                 "confidence": 1.0,
#                 "center": [cx, cy, cz],
#                 "dimensions": [max(0.4, w), max(0.4, h), max(0.4, d)],
#                 "points_count": obj["mass"],
#                 "intensity": obj["intensity"],
#                 "shape_text": shape_text,
#                 "position_text": position_text,
#                 "distance_m": round(abs(cz), 1),
#                 "raw_points": obj["raw_points"]
#             })

#         print(f" 🛰️ [ЕПО-MOT ТРЕКЕР препятствий]: Подтверждено стабильных целей: {len(final_safe_objects)} | "
#               f"Всего отслеживается в карте памяти: {len(self.object_engine.tracked_registry)}", flush=True)
              
#         return final_safe_objects

# """
# 🛰️ STATEFUL MOT MODULE v2 (Fixed Axes)
# Компонент межкадровой фильтрации, трекинга скоростей и подавления призраков.
# Работает в фиксированной системе координат одометрии: [0:X_ширина, 1:Y_высота, 2:Z_дальность].
# """

# import numpy as np
# import metro_lidar.config as config

# class LidarObstacleTrackerV2:
#     """Промышленный межкадровый ИИ-трекер на векторных NumPy-операциях."""
    
#     def __init__(self):
#         """Инициализирует структуры данных и буферы автокалибровки."""
#         self.past_tracks = []
#         self.track_id_counter = 0
#         self.dynamic_min_z_control = 3.5
#         self.is_cabin_calibrated = False
#         self.ring_distance_history = []
#         self.dynamic_ring_center_z = None
#         self.dynamic_ring_min_z = 6.5
#         self.dynamic_ring_max_z = 7.0

#     def reset(self):
#         """Полный сброс межкадровой памяти для изоляции WebSocket-сессий FastAPI."""
#         self.past_tracks = []
#         self.track_id_counter = 0
#         self.is_cabin_calibrated = False
#         self.ring_distance_history = []
#         self.dynamic_ring_center_z = None

#     def _calibrate_dynamic_ring_gauge(self, cz):
#         """Вычисляет индивидуальный створ колец инфраструктуры туннеля."""
#         if self.dynamic_ring_center_z is not None:
#             return
#         self.ring_distance_history.append(cz)
#         if len(self.ring_distance_history) >= 15:
#             self.dynamic_ring_center_z = float(np.median(self.ring_distance_history))
#             self.dynamic_ring_min_z = self.dynamic_ring_center_z - 0.25
#             self.dynamic_ring_max_z = self.dynamic_ring_center_z + 0.25

#     def _calibrate_ego_vehicle_cabin(self, points):
#         """
#         Сканирует пространство перед лидаром и динамически отсекает бампер состава.
#         Полностью синхронизирована с глобальными константами config.py без дублирования.
#         Инвариантна к знакам и полярности оси дальности Z.
#         """
#         if self.is_cabin_calibrated:
#             return self.dynamic_min_z_control

#         # Извлекаем физические координаты осей по стандарту монолитного моста
#         cx = points[:, 0]
#         cy = points[:, 1]
#         cz = points[:, 2]

#         # 🟢 ГЛАВНЫЙ ИИ-ФИКС: Берем дальность по модулю abs(), 
#         # благодаря чему маска сработает идеально и при положительном, и при отрицательном Z!
#         absolute_z = np.abs(cz)

#         # Строим ворота фильтрации вагона, используя ТОЛЬКО существующие константы из вашего config.py
#         cabin_mask = (
#             (cx >= -config.TRAIN_HALF_WIDTH) & (cx <= config.TRAIN_HALF_WIDTH) &
#             (cy >= config.MIN_Y) & (cy <= config.MAX_Y) &
#             (absolute_z >= config.LIDAR_MIN_Z) & (absolute_z <= config.LIDAR_MAX_Z)
#         )
#         cabin_points_z = absolute_z[cabin_mask]

#         # Если ИИ нашел плотную структуру бампера вагона прямо перед собой
#         if len(cabin_points_z) > config.CABIN_MIN_POINTS_THRESHOLD:
#             # Находим самую выступающую точку бампера по модулю дальности
#             max_cabin_edge = float(np.max(cabin_points_z))
            
#             # Выставляем живую мертвую зону: край бампера + безопасный отступ (берем из SPIKE_VETO_THRESHOLD_M = 0.05)
#             # или добавляем фиксированный конкурсный зазор 40 см, если бампер длинный
#             self.dynamic_min_z_control = max_cabin_edge + 0.40
#             self.is_cabin_calibrated = True
#         else:
#             # Если точек бампера нет, включаем базовый безопасный створ из конфига
#             self.dynamic_min_z_control = config.LIDAR_MIN_Z
#             self.is_cabin_calibrated = True
            
#         return self.dynamic_min_z_control


#     def _collapse_duplicates(self, objects_list):
#         """Выполняет пространственный 3D NMS для склеивания осколков DBSCAN."""
#         if len(objects_list) <= 1:
#             return objects_list

#         collapsed = []
#         visited = set()

#         for i, obj_a in enumerate(objects_list):
#             if i in visited:
#                 continue
                
#             cx_a, cy_a, cz_a = obj_a["center"]
#             cluster = [obj_a]
#             visited.add(i)
            
#             for j, obj_b in enumerate(objects_list):
#                 if j in visited:
#                     continue
                    
#                 cx_b, cy_b, cz_b = obj_b["center"]
#                 dist_3d = float(np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2 + (cz_a - cz_b)**2))
#                 if dist_3d < 1.5:
#                     cluster.append(obj_b)
#                     visited.add(j)
            
#             if len(cluster) == 1:
#                 collapsed.append(obj_a)
#                 continue
                
#             centers = np.array([c["center"] for c in cluster])
#             sizes = np.array([c["dimensions"] for c in cluster])
#             mean_cx, mean_cy, mean_cz = np.mean(centers, axis=0)
            
#             max_w = float(np.max(centers[:, 0] + sizes[:, 0]/2) - np.min(centers[:, 0] - sizes[:, 0]/2))
#             max_h = float(np.max(centers[:, 1] + sizes[:, 1]/2) - np.min(centers[:, 1] - sizes[:, 1]/2))
#             max_d = float(np.max(centers[:, 2] + sizes[:, 2]/2) - np.min(centers[:, 2] - sizes[:, 2]/2))
            
#             merged_obj = cluster[0].copy()
#             merged_obj["center"] = [mean_cx, mean_cy, mean_cz]
#             merged_obj["dimensions"] = [max(0.4, min(max_w, 3.0)), max(0.4, min(max_h, 3.0)), max(0.4, min(max_d, 3.0))]
#             if "points_count" in cluster[0]:
#                 merged_obj["points_count"] = sum([c.get("points_count", 0) for c in cluster])
                
#             collapsed.append(merged_obj)
            
#         return collapsed
    
#     def _estimate_train_velocity(self, current_objects):
#         """Векторная одометрия по стабильным кластерам стен (ICP-Light на NumPy)."""
#         if not self.past_tracks or not current_objects:
#             return 0.50

#         curr_centers = np.array([obj["center"] for obj in current_objects])
#         past_centers = np.array([tr["center"] for tr in self.past_tracks])

#         # Расстояния по осям X и Y (стабильность боковых стен туннеля)
#         xy_dists = np.linalg.norm(curr_centers[:, np.newaxis, :2] - past_centers[np.newaxis, :, :2], axis=2)
#         best_match_indices = np.argmin(xy_dists, axis=1)
#         min_xy_dists = np.min(xy_dists, axis=1)

#         valid_pairs_mask = min_xy_dists < 0.45

#         if np.sum(valid_pairs_mask) >= 2:
#             # Сдвиг Z — это продольное смещение стен за такт (ход поезда)
#             matched_curr_z = curr_centers[valid_pairs_mask, 2]
#             matched_past_z = past_centers[best_match_indices[valid_pairs_mask], 2]
#             shifts_z = np.abs(matched_curr_z - matched_past_z)
#             valid_shifts = shifts_z[(shifts_z > 0.1) & (shifts_z < 3.0)]
#             if len(valid_shifts) >= 2:
#                 return float(np.mean(valid_shifts))

#         return 0.50

#     def track_and_filter_ghosts(self, current_frame_objects, train_step_z=0.0):
#         """MOT-конвейер покадровой ассоциации со строгим фильтром подтверждения целей."""
#         final_safe_objects = []
#         current_frame_objects = self._collapse_duplicates(current_frame_objects)

#         # Режим ведения вслепую (Coasting) для удержания целей при пропусках кадров
#         if not current_frame_objects:
#             if not self.past_tracks:
#                 return []
#             updated_past_tracks = []
#             for tr in self.past_tracks:
#                 tr["age"] = tr.get("age", 0) + 1
#                 if tr["age"] > config.TRACK_MAX_COASTING_AGE:
#                     continue
                
#                 # 🟢 ИИ-ФИКС ЗНАКА: Сдвигаем упущенные объекты вперед навстречу кабине (+)
#                 tr["center"][2] += float(train_step_z)
#                 updated_past_tracks.append(tr)
                
#                 if tr["hits"] >= 3:
#                     cx, cy, cz = tr["center"]
#                     w, h, d = tr["dimensions"]
#                     final_safe_objects.append({
#                         "id": tr["id"], "class_id": 1, "confidence": 0.8,
#                         "center": [cx, cy, cz], "dimensions": [w, h, d],
#                         "train_speed": 0.0, "obstacle_speed": 0.0
#                     })
#             self.past_tracks = updated_past_tracks
#             return final_safe_objects

#         # Холодный старт цепочки трекинга на первом кадре сессии
#         if not self.past_tracks:
#             new_tracks = []
#             for obj in current_frame_objects:
#                 self.track_id_counter += 1
#                 cx, cy, cz = obj["center"]
#                 w, h, d = obj["dimensions"]
                
#                 track_state = {
#                     "id": self.track_id_counter, "center": [cx, cy, cz], "dimensions": [w, h, d],
#                     "hits": 1, "age": 0
#                 }
#                 new_tracks.append(track_state)
#             self.past_tracks = new_tracks
#             print(f" 🛰️ [Трекер межкадровый]: Инициализация (кадр 0). Накопление истории...", flush=True)
#             return [] 

#         curr_centers = np.array([obj["center"] for obj in current_frame_objects]).reshape(-1, 3)
#         past_centers = np.array([tr["center"] for tr in self.past_tracks]).reshape(-1, 3)
        
#         # 🟢 ПРЕЦИЗИОННЫЙ ИИ-ФИКС ЗНАКА: Прогноз старых треков набегает навстречу поезду (+)
#         past_centers_predicted = past_centers.copy()
#         past_centers_predicted[:, 2] += train_step_z 

#         dists_3d = np.linalg.norm(curr_centers[:, np.newaxis, :] - past_centers_predicted[np.newaxis, :, :], axis=2)
#         best_past_indices = np.argmin(dists_3d, axis=1)
#         min_dists_3d = np.min(dists_3d, axis=1)
        
#         # Ваши родные рабочие ворота ассоциации (2.5 метра — оставляем без изменений)
#         association_gate_mask = min_dists_3d < 2.5 

#         matched_past_indices = set()
#         new_tracks = []

#         for i, obj in enumerate(current_frame_objects):
#             cx, cy, cz = obj["center"]
#             width, height, depth = obj["dimensions"]
#             past_track = None

#             if association_gate_mask[i]:
#                 past_idx = best_past_indices[i]
#                 past_track = self.past_tracks[past_idx]
#                 matched_past_indices.add(past_idx)
#                 hits = past_track["hits"] + 1
#                 track_id = past_track["id"]
#                 age = 0 
#             else:
#                 self.track_id_counter += 1
#                 track_id = self.track_id_counter
#                 hits = 1
#                 age = 0

#             if past_track is not None:
#                 past_w, past_h, past_d = past_track["dimensions"]
#                 render_w = (past_w * 0.7) + (width * 0.3)
#                 render_h = (past_h * 0.7) + (height * 0.3)
#                 render_d = (past_d * 0.7) + (depth * 0.3)
#             else:
#                 render_w, render_h, render_d = width, height, depth

#             track_state = {
#                 "id": track_id, "center": [cx, cy, cz], "dimensions": [render_w, render_h, render_d],
#                 "hits": hits, "age": age
#             }
#             new_tracks.append(track_state)

#             if hits >= 3:
#                 final_safe_objects.append({
#                     "id": track_id, 
#                     "class_id": 1, 
#                     "confidence": 1.0,
#                     "center": [cx, cy, cz], 
#                     "dimensions": [max(0.4, render_w), max(0.4, render_h), max(0.4, render_d)],
#                     "train_speed": 0.0, 
#                     "obstacle_speed": 0.0
#                 })

#         for j, tr in enumerate(self.past_tracks):
#             if j not in matched_past_indices:
#                 tr["age"] = tr.get("age", 0) + 1
#                 if tr["age"] <= config.TRACK_MAX_COASTING_AGE:
#                     # Смещаем Coasting-треки синхронно с общим потоком кадра
#                     tr["center"][2] += float(train_step_z)
#                     new_tracks.append(tr)

#         self.past_tracks = new_tracks
#         print(f" 🛰️ [Трекер межкадровый]: Подтверждено треков (hits>=3): {len(final_safe_objects)} | Всего треков в памяти: {len(self.past_tracks)}", flush=True)
#         return final_safe_objects
#  ++++++++++++++++++++++++++++++++++
# tracker_v2.py
import numpy as np
import metro_lidar.config as config
from metro_lidar.track_vector_module import TrackVectorLock

class LidarObstacleTrackerV2:
    """
    Промышленный межкадровый ИИ-трекер на векторных NumPy-операциях.
    Абсолютно стабильная исходная версия.
    """
    def __init__(self):
        self.past_tracks = []
        self.track_id_counter = 0
        self.dynamic_min_z_control = 3.5
        self.is_cabin_calibrated = False

        self.track_vector_engine = TrackVectorLock()

    def reset(self):
        """Полный сброс памяти для изоляции сессий."""
        self.past_tracks = []
        self.track_id_counter = 0
        self.is_cabin_calibrated = False

    def _calibrate_ego_vehicle_cabin(self, points):
        """Сканирует пространство перед лидаром и динамически отсекает нос состава."""
        if self.is_cabin_calibrated:
            return self.dynamic_min_z_control

        cx, cy, cz = points[:, 0], points[:, 1], points[:, 2]
        absolute_z = np.abs(cz)

        cabin_mask = (
            (cx >= -config.TRAIN_HALF_WIDTH) & (cx <= config.TRAIN_HALF_WIDTH) &
            (cy >= config.MIN_Y) & (cy <= config.MAX_Y) &
            (absolute_z >= config.LIDAR_MIN_Z) & (absolute_z <= config.LIDAR_MAX_Z)
        )
        cabin_points_z = absolute_z[cabin_mask]

        if len(cabin_points_z) > config.CABIN_MIN_POINTS_THRESHOLD:
            max_cabin_edge = float(np.max(cabin_points_z))
            self.dynamic_min_z_control = max_cabin_edge + 0.40
        else:
            self.dynamic_min_z_control = config.LIDAR_MIN_Z
            
        self.is_cabin_calibrated = True
        return self.dynamic_min_z_control

    def _collapse_duplicates(self, objects_list):
        """Выполняет пространственный 3D NMS для склеивания осколков DBSCAN."""
        if len(objects_list) <= 1:
            return objects_list

        collapsed = []
        visited = set()

        for i, obj_a in enumerate(objects_list):
            if i in visited:
                continue
                
            cx_a, cy_a, cz_a = obj_a["center"]
            cluster = [obj_a]
            visited.add(i)
            
            for j, obj_b in enumerate(objects_list):
                if j in visited:
                    continue
                    
                cx_b, cy_b, cz_b = obj_b["center"]
                dist_3d = float(np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2 + (cz_a - cz_b)**2))
                if dist_3d < 1.5:
                    cluster.append(obj_b)
                    visited.add(j)
            
            if len(cluster) == 1:
                collapsed.append(obj_a)
                continue
                
            centers = np.array([c["center"] for c in cluster])
            sizes = np.array([c["dimensions"] for c in cluster])
            mean_cx, mean_cy, mean_cz = np.mean(centers, axis=0)
            
            max_w = float(np.max(centers[:, 0] + sizes[:, 0]/2) - np.min(centers[:, 0] - sizes[:, 0]/2))
            max_h = float(np.max(centers[:, 1] + sizes[:, 1]/2) - np.min(centers[:, 1] - sizes[:, 1]/2))
            max_d = float(np.max(centers[:, 2] + sizes[:, 2]/2) - np.min(centers[:, 2] - sizes[:, 2]/2))
            
            merged_obj = cluster[0].copy()
            merged_obj["center"] = [mean_cx, mean_cy, mean_cz]
            merged_obj["dimensions"] = [max(0.4, min(max_w, 3.0)), max(0.4, min(max_h, 3.0)), max(0.4, min(max_d, 3.0))]
            if "points_count" in cluster[0]:
                merged_obj["points_count"] = sum([c.get("points_count", 0) for c in cluster])
                
            collapsed.append(merged_obj)
            
        return collapsed

    
    def track_and_filter_ghosts(self, current_frame_objects, train_step_z=0.0):
        """
        Промышленный MOT-конвейер межкадровой ассоциации (v13.0).
        Ювелирно ликвидирует дрифт ID (ID Switches) на высоких скоростях и
        вырезает покадровые шумовые вспышки через механизм хронологического карантина.
        """
        final_safe_objects = []
        
        # 1. Схлопываем пространственные дубликаты текущего кадра (3D NMS)
        current_frame_objects = self._collapse_duplicates(current_frame_objects)

        # --- СЦЕНАРИЙ А: КУРСОВОЕ ВЕДЕНИЕ В СЛЕПОТЕ (COASTING) ---
        # Если в текущем кадре DBSCAN пуст — ведем накопленную память треков по инерции
        if not current_frame_objects:
            if not self.past_tracks:
                return []
            updated_past_tracks = []
            for tr in self.past_tracks:
                tr["age"] = tr.get("age", 0) + 1
                # 🛡️ УВЕЛИЧЕННЫЙ ХРОНО-ОКОП: Удерживаем упущенную цель в памяти до 15 тактов (1.5 секунды слепоты)!
                if tr["age"] > 15: 
                    continue
                # Смещаем Coasting-трек по вектору хода поезда
                tr["center"][2] += float(train_step_z)
                updated_past_tracks.append(tr)
                
                # Выводим в финальный submission только жестко подтвержденные временем треки
                if tr["hits"] >= 8:
                    cx, cy, cz = tr["center"]
                    w, h, d = tr["dimensions"]
                    final_safe_objects.append({
                        "id": tr["id"], "class_id": 1, "confidence": 0.75,
                        "center": [cx, cy, cz], "dimensions": [w, h, d],
                        "train_speed": 0.0, "obstacle_speed": 0.0
                    })
            self.past_tracks = updated_past_tracks
            return final_safe_objects

        # --- СЦЕНАРИЙ Б: ХОЛОДНЫЙ СТАРТ СЕССИИ (КАДР 0) ---
        if not self.past_tracks:
            new_tracks = []
            for obj in current_frame_objects:
                self.track_id_counter += 1
                cx, cy, cz = obj["center"]
                w, h, d = obj["dimensions"]
                pts_count = obj.get("points_count", 0)
                
                track_state = {
                    "id": self.track_id_counter, "center": [cx, cy, cz], "dimensions": [w, h, d],
                    "hits": 1, "age": 0, "points_count": pts_count
                }
                new_tracks.append(track_state)
            self.past_tracks = new_tracks
            return [] 

        # --- СЦЕНАРИЙ В: ЖИВАЯ МЕЖКАДРОВАЯ АССОЦИАЦИЯ МАСС ---
        curr_centers = np.array([obj["center"] for obj in current_frame_objects]).reshape(-1, 3)
        past_centers = np.array([tr["center"] for tr in self.past_tracks]).reshape(-1, 3)
        
        # Экстраполируем старые центроиды навстречу поезду с учетом шага одометрии v12
        past_centers_predicted = past_centers.copy()
        past_centers_predicted[:, 2] += train_step_z 

        # Считаем матрицу попарных 3D-расстояний между кадрами
        dists_3d = np.linalg.norm(curr_centers[:, np.newaxis, :] - past_centers_predicted[np.newaxis, :, :], axis=2)
        best_past_indices = np.argmin(dists_3d, axis=1)
        min_dists_3d = np.min(dists_3d, axis=1)
        
        matched_past_indices = set()
        new_tracks = []

        for i, obj in enumerate(current_frame_objects):
            cx, cy, cz = obj["center"]
            width, height, depth = obj["dimensions"]
            pts_count = obj.get("points_count", 0)
            distance = abs(cz)
            past_track = None
            past_idx = best_past_indices[i]

            # =====================================================================
            # 🧠 МАТЕМАТИЧЕСКИЙ АППАРАТ ДЫШАЩИХ СКОРОСТНЫХ ВОРОТ АССОЦИАЦИИ
            # =====================================================================
            # Вместо хардкодных 2.5м, радиус поиска динамически растет вместе с 
            # шагом поезда (train_step_z) и дальностью объекта (abs_z).
            # На 150м при скорости 40 км/ч ворота раскроются до ~6.5 метров, 
            # намертво удерживая сквозной ID и не давая треку разорваться!
            abs_step = abs(train_step_z)
            allowed_gate = 2.5 + (abs_step * 3.5) + (distance * 0.02)
            allowed_gate = np.clip(allowed_gate, 2.5, 7.5) # Верхний срез, чтобы не перепутать два объекта жюри

            # Если объект уложился в скоростные ворота — склеиваем ID с историей
            if min_dists_3d[i] < allowed_gate:
                past_track = self.past_tracks[past_idx]
                matched_past_indices.add(past_idx)
                hits = past_track["hits"] + 1
                track_id = past_track["id"]
                age = 0 
            else:
                # Объект улетел слишком далеко от старых прогнозов — это новый уникальный предмет на трассе
                self.track_id_counter += 1
                track_id = self.track_id_counter
                hits = 1
                age = 0

            # Плавная демпферная фильтрация физических размеров 3D-бокса (сглаживание шума Hesai)
            if past_track is not None:
                past_w, past_h, past_d = past_track["dimensions"]
                render_w = (past_w * 0.8) + (width * 0.2)
                render_h = (past_h * 0.8) + (height * 0.2)
                render_d = (past_d * 0.8) + (depth * 0.2)
            else:
                render_w, render_h, render_d = width, height, depth

            track_state = {
                "id": track_id, "center": [cx, cy, cz], "dimensions": [render_w, render_h, render_d],
                "hits": hits, "age": age, "points_count": pts_count
            }
            new_tracks.append(track_state)

            # =====================================================================
            # 🛡️ ФИЛЬТР ХРОНОЛОГИЧЕСКОГО КАРАНТИНА (ЗАЩИТА submission.csv)
            # =====================================================================
            # Чтобы полностью убрать покадровый мусор станций и пара (вспышки на 1-4 кадра),
            # мы разрешаем вывод преграды в итоговый отчет ТОЛЬКО если объект 
            # стабильно верифицируется трекером на протяжении минимум 8 кадров подряд!
            if hits >= 8:
                final_safe_objects.append({
                    "id": track_id, "class_id": 1, "confidence": 1.0,
                    "center": [cx, cy, cz], 
                    "dimensions": [max(0.4, render_w), max(0.4, render_h), max(0.4, render_d)],
                    "train_speed": 0.0, "obstacle_speed": 0.0
                })

        # Перенос вслепую (Coasting) для треков, которые в этом кадре временно скрылись из видимости
        for j, tr in enumerate(self.past_tracks):
            if j not in matched_past_indices:
                tr["age"] = tr.get("age", 0) + 1
                if tr["age"] <= 15: # Удерживаем инерцию в рамках обновленного TTL
                    tr["center"][2] += float(train_step_z)
                    new_tracks.append(tr)

        self.past_tracks = new_tracks
        
        # Выводим в HUD краткий чистый срез активных целей в памяти
        print(f" 🛰️ [ЕПО-MOT ТРЕКЕР препятствий]: Подтверждено стабильных целей (hits>=8): {len(final_safe_objects)} | "
              f"Всего удерживается в карте памяти: {len(self.past_tracks)}", flush=True)
              
        return final_safe_objects
