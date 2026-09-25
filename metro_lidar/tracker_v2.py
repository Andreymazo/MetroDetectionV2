"""
🛰️ STATEFUL MOT MODULE v2 (Fixed Axes)
Компонент межкадровой фильтрации, трекинга скоростей и подавления призраков.
Работает в фиксированной системе координат одометрии: [0:X_ширина, 1:Y_высота, 2:Z_дальность].
"""

import numpy as np
import metro_lidar.config as config

class LidarObstacleTrackerV2:
    """Промышленный межкадровый ИИ-трекер на векторных NumPy-операциях."""
    
    def __init__(self):
        """Инициализирует структуры данных и буферы автокалибровки."""
        self.past_tracks = []
        self.track_id_counter = 0
        self.dynamic_min_z_control = 3.5
        self.is_cabin_calibrated = False
        self.ring_distance_history = []
        self.dynamic_ring_center_z = None
        self.dynamic_ring_min_z = 6.5
        self.dynamic_ring_max_z = 7.0

    def reset(self):
        """Полный сброс межкадровой памяти для изоляции WebSocket-сессий FastAPI."""
        self.past_tracks = []
        self.track_id_counter = 0
        self.is_cabin_calibrated = False
        self.ring_distance_history = []
        self.dynamic_ring_center_z = None

    def _calibrate_dynamic_ring_gauge(self, cz):
        """Вычисляет индивидуальный створ колец инфраструктуры туннеля."""
        if self.dynamic_ring_center_z is not None:
            return
        self.ring_distance_history.append(cz)
        if len(self.ring_distance_history) >= 15:
            self.dynamic_ring_center_z = float(np.median(self.ring_distance_history))
            self.dynamic_ring_min_z = self.dynamic_ring_center_z - 0.25
            self.dynamic_ring_max_z = self.dynamic_ring_center_z + 0.25

    def _calibrate_ego_vehicle_cabin(self, points):
        """
        Сканирует пространство перед лидаром и динамически отсекает бампер состава.
        Полностью синхронизирована с глобальными константами config.py без дублирования.
        Инвариантна к знакам и полярности оси дальности Z.
        """
        if self.is_cabin_calibrated:
            return self.dynamic_min_z_control

        # Извлекаем физические координаты осей по стандарту монолитного моста
        cx = points[:, 0]
        cy = points[:, 1]
        cz = points[:, 2]

        # 🟢 ГЛАВНЫЙ ИИ-ФИКС: Берем дальность по модулю abs(), 
        # благодаря чему маска сработает идеально и при положительном, и при отрицательном Z!
        absolute_z = np.abs(cz)

        # Строим ворота фильтрации вагона, используя ТОЛЬКО существующие константы из вашего config.py
        cabin_mask = (
            (cx >= -config.TRAIN_HALF_WIDTH) & (cx <= config.TRAIN_HALF_WIDTH) &
            (cy >= config.MIN_Y) & (cy <= config.MAX_Y) &
            (absolute_z >= config.LIDAR_MIN_Z) & (absolute_z <= config.LIDAR_MAX_Z)
        )
        cabin_points_z = absolute_z[cabin_mask]

        # Если ИИ нашел плотную структуру бампера вагона прямо перед собой
        if len(cabin_points_z) > config.CABIN_MIN_POINTS_THRESHOLD:
            # Находим самую выступающую точку бампера по модулю дальности
            max_cabin_edge = float(np.max(cabin_points_z))
            
            # Выставляем живую мертвую зону: край бампера + безопасный отступ (берем из SPIKE_VETO_THRESHOLD_M = 0.05)
            # или добавляем фиксированный конкурсный зазор 40 см, если бампер длинный
            self.dynamic_min_z_control = max_cabin_edge + 0.40
            self.is_cabin_calibrated = True
        else:
            # Если точек бампера нет, включаем базовый безопасный створ из конфига
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
    
    def _estimate_train_velocity(self, current_objects):
        """Векторная одометрия по стабильным кластерам стен (ICP-Light на NumPy)."""
        if not self.past_tracks or not current_objects:
            return 0.50

        curr_centers = np.array([obj["center"] for obj in current_objects])
        past_centers = np.array([tr["center"] for tr in self.past_tracks])

        # Расстояния по осям X и Y (стабильность боковых стен туннеля)
        xy_dists = np.linalg.norm(curr_centers[:, np.newaxis, :2] - past_centers[np.newaxis, :, :2], axis=2)
        best_match_indices = np.argmin(xy_dists, axis=1)
        min_xy_dists = np.min(xy_dists, axis=1)

        valid_pairs_mask = min_xy_dists < 0.45

        if np.sum(valid_pairs_mask) >= 2:
            # Сдвиг Z — это продольное смещение стен за такт (ход поезда)
            matched_curr_z = curr_centers[valid_pairs_mask, 2]
            matched_past_z = past_centers[best_match_indices[valid_pairs_mask], 2]
            shifts_z = np.abs(matched_curr_z - matched_past_z)
            valid_shifts = shifts_z[(shifts_z > 0.1) & (shifts_z < 3.0)]
            if len(valid_shifts) >= 2:
                return float(np.mean(valid_shifts))

        return 0.50

    def track_and_filter_ghosts(self, current_frame_objects, train_step_z=0.0):
        """MOT-конвейер покадровой ассоциации со строгим фильтром подтверждения целей."""
        final_safe_objects = []
        current_frame_objects = self._collapse_duplicates(current_frame_objects)

        # Режим ведения вслепую (Coasting) для удержания целей при пропусках кадров
        if not current_frame_objects:
            if not self.past_tracks:
                return []
            updated_past_tracks = []
            for tr in self.past_tracks:
                tr["age"] = tr.get("age", 0) + 1
                if tr["age"] > config.TRACK_MAX_COASTING_AGE:
                    continue
                
                # 🟢 ИИ-ФИКС ЗНАКА: Сдвигаем упущенные объекты вперед навстречу кабине (+)
                tr["center"][2] += float(train_step_z)
                updated_past_tracks.append(tr)
                
                if tr["hits"] >= 3:
                    cx, cy, cz = tr["center"]
                    w, h, d = tr["dimensions"]
                    final_safe_objects.append({
                        "id": tr["id"], "class_id": 1, "confidence": 0.8,
                        "center": [cx, cy, cz], "dimensions": [w, h, d],
                        "train_speed": 0.0, "obstacle_speed": 0.0
                    })
            self.past_tracks = updated_past_tracks
            return final_safe_objects

        # Холодный старт цепочки трекинга на первом кадре сессии
        if not self.past_tracks:
            new_tracks = []
            for obj in current_frame_objects:
                self.track_id_counter += 1
                cx, cy, cz = obj["center"]
                w, h, d = obj["dimensions"]
                
                track_state = {
                    "id": self.track_id_counter, "center": [cx, cy, cz], "dimensions": [w, h, d],
                    "hits": 1, "age": 0
                }
                new_tracks.append(track_state)
            self.past_tracks = new_tracks
            print(f" 🛰️ [Трекер межкадровый]: Инициализация (кадр 0). Накопление истории...", flush=True)
            return [] 

        curr_centers = np.array([obj["center"] for obj in current_frame_objects]).reshape(-1, 3)
        past_centers = np.array([tr["center"] for tr in self.past_tracks]).reshape(-1, 3)
        
        # 🟢 ПРЕЦИЗИОННЫЙ ИИ-ФИКС ЗНАКА: Прогноз старых треков набегает навстречу поезду (+)
        past_centers_predicted = past_centers.copy()
        past_centers_predicted[:, 2] += train_step_z 

        dists_3d = np.linalg.norm(curr_centers[:, np.newaxis, :] - past_centers_predicted[np.newaxis, :, :], axis=2)
        best_past_indices = np.argmin(dists_3d, axis=1)
        min_dists_3d = np.min(dists_3d, axis=1)
        
        # Ваши родные рабочие ворота ассоциации (2.5 метра — оставляем без изменений)
        association_gate_mask = min_dists_3d < 2.5 

        matched_past_indices = set()
        new_tracks = []

        for i, obj in enumerate(current_frame_objects):
            cx, cy, cz = obj["center"]
            width, height, depth = obj["dimensions"]
            past_track = None

            if association_gate_mask[i]:
                past_idx = best_past_indices[i]
                past_track = self.past_tracks[past_idx]
                matched_past_indices.add(past_idx)
                hits = past_track["hits"] + 1
                track_id = past_track["id"]
                age = 0 
            else:
                self.track_id_counter += 1
                track_id = self.track_id_counter
                hits = 1
                age = 0

            if past_track is not None:
                past_w, past_h, past_d = past_track["dimensions"]
                render_w = (past_w * 0.7) + (width * 0.3)
                render_h = (past_h * 0.7) + (height * 0.3)
                render_d = (past_d * 0.7) + (depth * 0.3)
            else:
                render_w, render_h, render_d = width, height, depth

            track_state = {
                "id": track_id, "center": [cx, cy, cz], "dimensions": [render_w, render_h, render_d],
                "hits": hits, "age": age
            }
            new_tracks.append(track_state)

            if hits >= 3:
                final_safe_objects.append({
                    "id": track_id, 
                    "class_id": 1, 
                    "confidence": 1.0,
                    "center": [cx, cy, cz], 
                    "dimensions": [max(0.4, render_w), max(0.4, render_h), max(0.4, render_d)],
                    "train_speed": 0.0, 
                    "obstacle_speed": 0.0
                })

        for j, tr in enumerate(self.past_tracks):
            if j not in matched_past_indices:
                tr["age"] = tr.get("age", 0) + 1
                if tr["age"] <= config.TRACK_MAX_COASTING_AGE:
                    # Смещаем Coasting-треки синхронно с общим потоком кадра
                    tr["center"][2] += float(train_step_z)
                    new_tracks.append(tr)

        self.past_tracks = new_tracks
        print(f" 🛰️ [Трекер межкадровый]: Подтверждено треков (hits>=3): {len(final_safe_objects)} | Всего треков в памяти: {len(self.past_tracks)}", flush=True)
        return final_safe_objects
 
    # def track_and_filter_ghosts(self, current_frame_objects):
    #     """MOT-конвейер покадровой ассоциации со строгим фильтром по трем кадрам."""
    #     final_safe_objects = []
    #     current_frame_objects = self._collapse_duplicates(current_frame_objects)

    #     # Режим ведения вслепую (Coasting), если на текущем кадре пусто
    #     if not current_frame_objects:
    #         if not self.past_tracks:
    #             return []
    #         updated_past_tracks = []
    #         for tr in self.past_tracks:
    #             tr["age"] = tr.get("age", 0) + 1
    #             if tr["age"] > config.TRACK_MAX_COASTING_AGE:
    #                 continue
    #             updated_past_tracks.append(tr)
                
    #             # Удерживаем на экране только те цели, которые успели подтвердиться
    #             if tr["hits"] >= 3:
    #                 cx, cy, cz = tr["center"]
    #                 w, h, d = tr["dimensions"]
    #                 final_safe_objects.append({
    #                     "id": tr["id"], "class_id": 1, "confidence": 0.8,
    #                     "center": [cx, cy, cz], "dimensions": [w, h, d],
    #                     "train_speed": 0.0, "obstacle_speed": 0.0
    #                 })
    #         self.past_tracks = updated_past_tracks
    #         return final_safe_objects

    #     # Холодный старт цепочки при первом появлении объектов в сессии
    #     if not self.past_tracks:
    #         new_tracks = []
    #         for obj in current_frame_objects:
    #             self.track_id_counter += 1
    #             cx, cy, cz = obj["center"]
    #             w, h, d = obj["dimensions"]
                
    #             track_state = {
    #                 "id": self.track_id_counter, "center": [cx, cy, cz], "dimensions": [w, h, d],
    #                 "hits": 1, "age": 0
    #             }
    #             new_tracks.append(track_state)
    #         self.past_tracks = new_tracks
    #         print(f" 🛰️ [Трекер межкадровый]: Накопление истории... Подтверждено треков: 0", flush=True)
    #         return []

    #     # Извлекаем центры для матричного NumPy-расчета расстояний 3D
    #     curr_centers = np.array([obj["center"] for obj in current_frame_objects]).reshape(-1, 3)
    #     past_centers = np.array([tr["center"] for tr in self.past_tracks]).reshape(-1, 3)

    #     # Вычисляем Евклидову матрицу расстояний между текущим и прошлым кадром
    #     dists_3d = np.linalg.norm(curr_centers[:, np.newaxis, :] - past_centers[np.newaxis, :, :], axis=2)
    #     best_past_indices = np.argmin(dists_3d, axis=1)
    #     min_dists_3d = np.min(dists_3d, axis=1)
    #     association_gate_mask = min_dists_3d < 2.0  # Ворота ассоциации объектов между кадрами

    #     matched_past_indices = set()
    #     new_tracks = []

    #     for i, obj in enumerate(current_frame_objects):
    #         cx, cy, cz = obj["center"]
    #         width, height, depth = obj["dimensions"]
    #         past_track = None

    #         if association_gate_mask[i]:
    #             past_idx = best_past_indices[i]
    #             past_track = self.past_tracks[past_idx]
    #             matched_past_indices.add(past_idx)
    #             hits = past_track["hits"] + 1
    #             track_id = past_track["id"]
    #             age = 0 
    #         else:
    #             self.track_id_counter += 1
    #             track_id = self.track_id_counter
    #             hits = 1
    #             age = 0

    #         # Плавное сглаживание габаритов 3D бокса по экспоненте
    #         if past_track is not None:
    #             past_w, past_h, past_d = past_track["dimensions"]
    #             render_w = (past_w * 0.7) + (width * 0.3)
    #             render_h = (past_h * 0.7) + (height * 0.3)
    #             render_d = (past_d * 0.7) + (depth * 0.3)
    #         else:
    #             render_w, render_h, render_d = width, height, depth

    #         track_state = {
    #             "id": track_id, "center": [cx, cy, cz], "dimensions": [render_w, render_h, render_d],
    #             "hits": hits, "age": age
    #         }
    #         new_tracks.append(track_state)

    #         # 🛑 ЖЕСТКИЙ ФИЛЬТР ПО ТРЕМ КАДРАМ: Пропускаем преграду, только если она стабильна!
    #         if hits >= 3:
    #             final_safe_objects.append({
    #                 "id": track_id, "class_id": 1, "confidence": 1.0,
    #                 "center": [cx, cy, cz], "dimensions": [max(0.4, render_w), max(0.4, render_h), max(0.4, render_d)],
    #                 "train_speed": 0.0, "obstacle_speed": 0.0
    #             })

    #     # Удерживаем в памяти Coasting-треки, потерявшие пару на один такт
    #     for j, tr in enumerate(self.past_tracks):
    #         if j not in matched_past_indices:
    #             tr["age"] = tr.get("age", 0) + 1
    #             if tr["age"] <= config.TRACK_MAX_COASTING_AGE:
    #                 new_tracks.append(tr)

    #     self.past_tracks = new_tracks
    #     print(f" 🛰️ [Трекер межкадровый]: Подтверждено треков (hits>=3): {len(final_safe_objects)} | Всего треков в памяти: {len(self.past_tracks)}", flush=True)
    #     return final_safe_objects

    # def track_and_filter_ghosts(self, current_frame_objects):
    #     """Промышленный MOT-конвейер межкадровой фильтрации и трекинга скоростей."""
    #     final_safe_objects = []
    #     current_frame_objects = self._collapse_duplicates(current_frame_objects)
    #     delta_z = self._estimate_train_velocity(current_frame_objects)
    #     FPS_COEF = 36.0 
    #     train_speed_kmh = delta_z * FPS_COEF

    #     # Режим ведения вслепую (Coasting), если на текущем кадре пусто
    #     if not current_frame_objects:
    #         if not self.past_tracks:
    #             return []
    #         updated_past_tracks = []
    #         for tr in self.past_tracks:
    #             tr["age"] = tr.get("age", 0) + 1
    #             if tr["age"] > config.TRACK_MAX_COASTING_AGE:
    #                 continue
    #             tr["center"][2] -= float(delta_z)
    #             updated_past_tracks.append(tr)
                
    #             if tr["hits"] >= 3:
    #                 cx, cy, cz = tr["center"]
    #                 w, h, d = tr["dimensions"]
    #                 final_safe_objects.append({
    #                     "id": tr["id"], "class_id": 1, "confidence": 0.8,
    #                     "center": [cx, cy, cz], "dimensions": [w, h, d],
    #                     "train_speed": round(train_speed_kmh, 1), "obstacle_speed": 0.0
    #                 })
    #         self.past_tracks = updated_past_tracks
    #         return final_safe_objects

    #     # Холодный старт цепочки при первом появлении объектов
    #     if not self.past_tracks:
    #         new_tracks = []
    #         for obj in current_frame_objects:
    #             self.track_id_counter += 1
    #             cx, cy, cz = obj["center"]
    #             w, h, d = obj["dimensions"]
                
    #             track_state = {
    #                 "id": self.track_id_counter, "center": [cx, cy, cz], "dimensions": [w, h, d],
    #                 "hits": 1, "age": 0, "birthplace_z": cz
    #             }
    #             new_tracks.append(track_state)
                
    #             # Пропускаем в сабмишн со следующего такта для подтверждения скорости
    #             final_safe_objects.append({
    #                 "id": self.track_id_counter, "class_id": 1, "confidence": 1.0,
    #                 "center": [cx, cy, cz], "dimensions": [w, h, d],
    #                 "train_speed": round(train_speed_kmh, 1), "obstacle_speed": 0.0
    #             })
    #         self.past_tracks = new_tracks
    #         return final_safe_objects

    #     # Стандартная векторная ассоциация попарных расстояний 3D
    #     curr_centers = np.array([obj["center"] for obj in current_frame_objects]).reshape(-1, 3)
    #     past_centers = np.array([tr["center"] for tr in self.past_tracks]).reshape(-1, 3)
    #     past_centers_predicted = past_centers.copy()
    #     past_centers_predicted[:, 2] -= delta_z 

    #     dists_3d = np.linalg.norm(curr_centers[:, np.newaxis, :] - past_centers_predicted[np.newaxis, :, :], axis=2)
    #     best_past_indices = np.argmin(dists_3d, axis=1)
    #     min_dists_3d = np.min(dists_3d, axis=1)
    #     association_gate_mask = min_dists_3d < 2.5 

    #     matched_past_indices = set()
    #     new_tracks = []

    #     for i, obj in enumerate(current_frame_objects):
    #         cx, cy, cz = obj["center"]
    #         width, height, depth = obj["dimensions"]
    #         past_track = None

    #         if association_gate_mask[i]:
    #             past_idx = best_past_indices[i]
    #             past_track = self.past_tracks[past_idx]
    #             matched_past_indices.add(past_idx)
    #             hits = past_track["hits"] + 1
    #             track_id = past_track["id"]
    #             birthplace_z = past_track["birthplace_z"]
    #             past_age = past_track.get("age", 0)
    #             age = 0 
    #         else:
    #             self.track_id_counter += 1
    #             track_id = self.track_id_counter
    #             hits = 1
    #             birthplace_z = cz
    #             age = 0
    #             past_age = 0

    #         # Смузинг геометрических размеров
    #         if past_track is not None:
    #             past_w, past_h, past_d = past_track["dimensions"]
    #             render_w = (past_w * 0.7) + (width * 0.3)
    #             render_h = (past_h * 0.7) + (height * 0.3)
    #             render_d = (past_d * 0.7) + (depth * 0.3)
                
    #             if past_age == 0:
    #                 past_cz_val = float(past_track["center"][2])
    #                 distance_convergence = past_cz_val - cz
    #                 absolute_delta_z = distance_convergence - delta_z
    #                 obstacle_speed_kmh = absolute_delta_z * FPS_COEF
    #             else:
    #                 obstacle_speed_kmh = past_track.get("obstacle_speed_filtered", 0.0)
    #         else:
    #             render_w, render_h, render_d = width, height, depth
    #             obstacle_speed_kmh = 0.0

    #         if abs(obstacle_speed_kmh) < 3.5:
    #             obstacle_speed_kmh = 0.0

    #         track_state = {
    #             "id": track_id, "center": [cx, cy, cz], "dimensions": [render_w, render_h, render_d],
    #             "hits": hits, "age": age, "birthplace_z": birthplace_z,
    #             "obstacle_speed_filtered": obstacle_speed_kmh
    #         }
    #         new_tracks.append(track_state)

    #         if hits >= 3:
    #             final_safe_objects.append({
    #                 "id": track_id, "class_id": 1, "confidence": 1.0,
    #                 "center": [cx, cy, cz], "dimensions": [max(0.4, render_w), max(0.4, render_h), max(0.4, render_d)],
    #                 "train_speed": round(train_speed_kmh, 1), "obstacle_speed": round(obstacle_speed_kmh, 1)
    #             })

    #     for j, tr in enumerate(self.past_tracks):
    #         if j not in matched_past_indices:
    #             tr["age"] = tr.get("age", 0) + 1
    #             if tr["age"] <= config.TRACK_MAX_COASTING_AGE:
    #                 tr["center"][2] -= float(delta_z)
    #                 new_tracks.append(tr)

    #     self.past_tracks = new_tracks
    #     print(f" 🛰️ [Трекер межкадровый]: Подтверждено треков (hits>=3): {len(final_safe_objects)} | Всего треков в памяти: {len(self.past_tracks)}", flush=True)
    #     return final_safe_objects


# """
# 🛰️ STATEFUL MOT MODULE v2: Lidar Obstacle & Ghost Tracker (Unified Axes)
# Высокооптимизированный компонент межкадровой фильтрации и трекинга скоростей преград.
# Работает в единой системе координат одометрии: [0:X_ширина, 1:Y_высота, 2:Z_дальность].
# """

# import numpy as np
# import config

# class LidarObstacleTrackerV2:
#     def __init__(self):
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
#         if self.dynamic_ring_center_z is not None:
#             return
#         self.ring_distance_history.append(cz)
#         if len(self.ring_distance_history) >= 15:
#             self.dynamic_ring_center_z = float(np.median(self.ring_distance_history))
#             self.dynamic_ring_min_z = self.dynamic_ring_center_z - 0.25
#             self.dynamic_ring_max_z = self.dynamic_ring_center_z + 0.25
#             print(f"\n🎯 [ИИ-АВТОКАЛИБРОВКА СТВОРА]: Блэклист арки: {self.dynamic_ring_min_z:.2f}м - {self.dynamic_ring_max_z:.2f}м", flush=True)

#     def _calibrate_ego_vehicle_cabin(self, points):
#         """Селф-фильтр бампера: сканирует продольную ось 2 (Z) перед лобовым стеклом."""
#         if self.is_cabin_calibrated:
#             return self.dynamic_min_z_control

#         cx = points[:, 0]
#         cy = points[:, 1]
#         cz = points[:, 2]

#         cabin_mask = (
#             (cx >= -0.85) & (cx <= 0.85) &
#             (cy >= 0.35)  & (cy <= 2.0)  &
#             (cz >= 1.5)   & (cz <= 10.0)
#         )
#         cabin_points_z = cz[cabin_mask]

#         if len(cabin_points_z) > config.CABIN_MIN_POINTS_THRESHOLD:
#             max_cabin_edge = float(np.max(cabin_points_z))
#             self.dynamic_min_z_control = max_cabin_edge + 0.40
#             self.is_cabin_calibrated = True
#             print(f"🚀 [ИИ-СЕЛФ-КАЛИБРОВКА]: Обнаружен бампер вагона. Мертвая зона: {self.dynamic_min_z_control:.2f}м", flush=True)
#         else:
#             self.dynamic_min_z_control = 3.5
#             self.is_cabin_calibrated = True
#             print(f"\n[ИИ-СЕЛФ-КАЛИБРОВКА]: Детали кабины не найдены. Створ: 3.5м\n", flush=True)

#         return self.dynamic_min_z_control

#     def _collapse_duplicates(self, objects_list):
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
#         if not self.past_tracks or not current_objects:
#             return 0.50

#         curr_centers = np.array([obj["center"] for obj in current_objects])
#         past_centers = np.array([tr["center"] for tr in self.past_tracks])

#         # Расстояния по осям X и Y (стабильность боковых стен)
#         xy_dists = np.linalg.norm(curr_centers[:, np.newaxis, :2] - past_centers[np.newaxis, :, :2], axis=2)
#         best_match_indices = np.argmin(xy_dists, axis=1)
#         min_xy_dists = np.min(xy_dists, axis=1)

#         valid_pairs_mask = min_xy_dists < 0.45

#         if np.sum(valid_pairs_mask) >= 2:
#             # Сдвиг Z — это продольное смещение стен за такт
#             matched_curr_z = curr_centers[valid_pairs_mask, 2]
#             matched_past_z = past_centers[best_match_indices[valid_pairs_mask], 2]
#             shifts_z = np.abs(matched_curr_z - matched_past_z)
#             valid_shifts = shifts_z[(shifts_z > 0.1) & (shifts_z < 3.0)]
#             if len(valid_shifts) >= 2:
#                 return float(np.mean(valid_shifts))

#         return 0.50

#     def track_and_filter_ghosts(self, current_frame_objects):
#         final_safe_objects = []
#         current_frame_objects = self._collapse_duplicates(current_frame_objects)
#         delta_z = self._estimate_train_velocity(current_frame_objects)
#         FPS_COEF = 36.0 
#         train_speed_kmh = delta_z * FPS_COEF

#         if not current_frame_objects:
#             if not self.past_tracks:
#                 return []
#             updated_past_tracks = []
#             for tr in self.past_tracks:
#                 tr["age"] = tr.get("age", 0) + 1
#                 if tr["age"] > config.TRACK_MAX_COASTING_AGE:
#                     continue
#                 tr["center"][2] -= float(delta_z)
#                 updated_past_tracks.append(tr)
                
#                 if tr["hits"] >= 3 and not tr.get("is_ghost_infrastructure", False):
#                     cx, cy, cz = tr["center"]
#                     w, h, d = tr["dimensions"]
#                     final_safe_objects.append({
#                         "id": tr["id"], "class_id": 1, "confidence": 0.8,
#                         "center": [cx, cy, cz], "dimensions": [w, h, d],
#                         "train_speed": round(train_speed_kmh, 1), "obstacle_speed": 0.0
#                     })
#             self.past_tracks = updated_past_tracks
#             return final_safe_objects

#         if not self.past_tracks:
#             new_tracks = []
#             for obj in current_frame_objects:
#                 self.track_id_counter += 1
#                 cx, cy, cz = obj["center"]
#                 w, h, d = obj["dimensions"]
                
#                 if (cy >= 0.5) and (w >= 1.3):
#                     self._calibrate_dynamic_ring_gauge(cz)
                
#                 is_ghost = (self.dynamic_ring_min_z <= cz <= self.dynamic_ring_max_z) and (cy >= 0.5) and (w >= 1.3)
#                 track_state = {
#                     "id": self.track_id_counter, "center": [cx, cy, cz], "dimensions": [w, h, d],
#                     "hits": 1, "age": 0, "birthplace_z": cz, "is_ghost_infrastructure": is_ghost
#                 }
#                 new_tracks.append(track_state)
#                 if not is_ghost:
#                     final_safe_objects.append({
#                         "id": self.track_id_counter, "class_id": 1, "confidence": 1.0,
#                         "center": [cx, cy, cz], "dimensions": [w, h, d],
#                         "train_speed": round(train_speed_kmh, 1), "obstacle_speed": 0.0
#                     })
#             self.past_tracks = new_tracks
#             return final_safe_objects

#         curr_centers = np.array([obj["center"] for obj in current_frame_objects]).reshape(-1, 3)
#         past_centers = np.array([tr["center"] for tr in self.past_tracks]).reshape(-1, 3)

#         past_centers_predicted = past_centers.copy()
#         past_centers_predicted[:, 2] -= delta_z 

#         dists_3d = np.linalg.norm(curr_centers[:, np.newaxis, :] - past_centers_predicted[np.newaxis, :, :], axis=2)
#         best_past_indices = np.argmin(dists_3d, axis=1)
#         min_dists_3d = np.min(dists_3d, axis=1)
#         association_gate_mask = min_dists_3d < 2.5 

#         matched_past_indices = set()
#         new_tracks = []

#         for i, obj in enumerate(current_frame_objects):
#             cx, cy, cz = obj["center"]
#             width, height, depth = obj["dimensions"]
#             past_track = None
#             is_ghost = False

#             if association_gate_mask[i]:
#                 past_idx = best_past_indices[i]
#                 past_track = self.past_tracks[past_idx]
#                 matched_past_indices.add(past_idx)
#                 hits = past_track["hits"] + 1
#                 track_id = past_track["id"]
#                 birthplace_z = past_track["birthplace_z"]
#                 past_age = past_track.get("age", 0)
#                 age = 0 
#                 is_ghost = past_track.get("is_ghost_infrastructure", False)
#             else:
#                 self.track_id_counter += 1
#                 track_id = self.track_id_counter
#                 hits = 1
#                 birthplace_z = cz
#                 age = 0
#                 past_age = 0
#             # --- ГЕОМЕТРИЧЕСКИЙ ЩИТ БЕЗОПАСНОСТИ + ОБУЧЕНИЕ КАЛИБРАТОРА (ОСНОВНОЙ ЦИКЛ) ---
#             if (cy >= 0.5) and (width >= 1.3):
#                 self._calibrate_dynamic_ring_gauge(cz)
#             if (self.dynamic_ring_min_z <= cz <= self.dynamic_ring_max_z) and (cy >= 0.5) and (width >= 1.3):
#                 is_ghost = True

#             # Экспоненциальный смузинг размеров бокса для исключения тряски рендеринга
#             if past_track is not None:
#                 past_w, past_h, past_d = past_track["dimensions"]
#                 render_w = (past_w * 0.7) + (width * 0.3)
#                 render_h = (past_h * 0.7) + (height * 0.3)
#                 render_d = (past_d * 0.7) + (depth * 0.3)
                
#                 # Живой расчет скорости (только если объект виделся два кадра подряд)
#                 if past_age == 0 and not is_ghost:
#                     past_cz_val = float(past_track["center"][2])
#                     distance_convergence = past_cz_val - cz
#                     absolute_delta_z = distance_convergence - delta_z
#                     obstacle_speed_kmh = absolute_delta_z * FPS_COEF
#                 else:
#                     obstacle_speed_kmh = past_track.get("obstacle_speed_filtered", 0.0)
#             else:
#                 render_w, render_h, render_d = width, height, depth
#                 obstacle_speed_kmh = 0.0

#             # Стабилизационный нуль-фильтр для исключения покадрового микролюфта DBSCAN
#             if abs(obstacle_speed_kmh) < 3.5:
#                 obstacle_speed_kmh = 0.0

#             # Записываем обновленный паспорт объекта в историю тактов
#             track_state = {
#                 "id": track_id, 
#                 "center": [cx, cy, cz], 
#                 "dimensions": [render_w, render_h, render_d],
#                 "hits": hits, 
#                 "age": age, 
#                 "birthplace_z": birthplace_z, 
#                 "is_ghost_infrastructure": is_ghost,
#                 "obstacle_speed_filtered": obstacle_speed_kmh
#             }
#             new_tracks.append(track_state)

#             if is_ghost:
#                 continue

#             # Нуль-фильтр элементов, жестко зафиксированных на кузове состава (бампер)
#             is_ego_vehicle_part = (obstacle_speed_kmh < -3.5) and (abs(obstacle_speed_kmh + train_speed_kmh) < 2.0)
#             if is_ego_vehicle_part:
#                 continue 

#             if hits >= 3:
#                 final_safe_objects.append({
#                     "id": track_id, 
#                     "class_id": 1, 
#                     "confidence": 1.0,
#                     "center": [cx, cy, cz], 
#                     "dimensions": [max(0.4, render_w), max(0.4, render_h), max(0.4, render_d)],
#                     "train_speed": round(train_speed_kmh, 1), 
#                     "obstacle_speed": round(obstacle_speed_kmh, 1)
#                 })

#         # Сохраняем в историю те старые треки, которые не нашли пару (Coasting сироты)
#         for j, tr in enumerate(self.past_tracks):
#             if j not in matched_past_indices:
#                 tr["age"] = tr.get("age", 0) + 1
#                 if tr["age"] <= config.TRACK_MAX_COASTING_AGE:
#                     tr["center"][2] -= float(delta_z)
#                     new_tracks.append(tr)

#         self.past_tracks = new_tracks
#         print(f" [ИИ ТРЕКЕР]: Активных подтвержденных треков: {len(final_safe_objects)}", flush=True)
#         return final_safe_objects

# +++++++++++++++++++


# """
# 🛰️ STATEFUL MOT MODULE v2 (Fixed Axes & Hit Filter)
# Промышленный компонент межкадровой фильтрации и трекинга скоростей преград.
# Работает в фиксированной системе координат одометрии: [0:X_ширина, 1:Y_высота, 2:Z_дальность].
# """

# import numpy as np
# import config

# class LidarObstacleTrackerV2:
#     """Межкадровый ИИ-трекер на векторных NumPy-операциях с фильтром накопления истории."""
    
#     def __init__(self):
#         """Инициализирует структуры данных памяти тактов."""
#         self.past_tracks = []
#         self.track_id_counter = 0
#         self.dynamic_min_z_control = 3.5
#         self.is_cabin_calibrated = False

#     def reset(self):
#         """Полный сброс межкадровой памяти для изоляции WebSocket-сессий FastAPI."""
#         self.past_tracks = []
#         self.track_id_counter = 0
#         self.is_cabin_calibrated = False

#     def _calibrate_ego_vehicle_cabin(self, points):
#         """Сканирует продольную ось Z перед кабиной и динамически отсекает бампер состава."""
#         if self.is_cabin_calibrated:
#             return self.dynamic_min_z_control

#         cx = points[:, 0]
#         cy = points[:, 1]
#         cz = points[:, 2]

#         cabin_mask = (
#             (cx >= -0.85) & (cx <= 0.85) &
#             (cy >= 0.35)  & (cy <= 2.0)  &
#             (cz >= 1.5)   & (cz <= 10.0)
#         )
#         cabin_points_z = cz[cabin_mask]

#         if len(cabin_points_z) > config.CABIN_MIN_POINTS_THRESHOLD:
#             max_cabin_edge = float(np.max(cabin_points_z))
#             self.dynamic_min_z_control = max_cabin_edge + 0.40
#             self.is_cabin_calibrated = True
#         else:
#             self.dynamic_min_z_control = 3.5
#             self.is_cabin_calibrated = True
#         return self.dynamic_min_z_control

#     def _collapse_duplicates(self, objects_list):
#         """Выполняет пространственный 3D NMS для склеивания близких осколков DBSCAN."""
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
            
#             base_obj = cluster[0]
#             merged_obj = base_obj.copy()
#             merged_obj["center"] = [mean_cx, mean_cy, mean_cz]
#             merged_obj["dimensions"] = [max(0.4, min(max_w, 3.0)), max(0.4, min(max_h, 3.0)), max(0.4, min(max_d, 3.0))]
#             if "points_count" in base_obj:
#                 merged_obj["points_count"] = sum([c.get("points_count", 0) for c in cluster])
                
#             collapsed.append(merged_obj)
            
#         return collapsed

#     def _estimate_train_velocity(self, current_objects):
#         """Векторная одометрия по стабильным кластерам стен (ICP-Light на NumPy)."""
#         if not self.past_tracks or not current_objects:
#             return 0.50

#         curr_centers = np.array([obj["center"] for obj in current_objects])
#         past_centers = np.array([tr["center"] for tr in self.past_tracks])

#         xy_dists = np.linalg.norm(curr_centers[:, np.newaxis, :2] - past_centers[np.newaxis, :, :2], axis=2)
#         best_match_indices = np.argmin(xy_dists, axis=1)
#         min_xy_dists = np.min(xy_dists, axis=1)

#         valid_pairs_mask = min_xy_dists < 0.45

#         if np.sum(valid_pairs_mask) >= 2:
#             matched_curr_z = curr_centers[valid_pairs_mask, 2]
#             matched_past_z = past_centers[best_match_indices[valid_pairs_mask], 2]
#             shifts_z = np.abs(matched_curr_z - matched_past_z)
#             valid_shifts = shifts_z[(shifts_z > 0.1) & (shifts_z < 3.0)]
#             if len(valid_shifts) >= 2:
#                 return float(np.mean(valid_shifts))

#         return 0.50

#     def track_and_filter_ghosts(self, current_frame_objects):
#         """MOT-конвейер покадровой ассоциации со строгим фильтром подтверждения целей."""
#         final_safe_objects = []
#         current_frame_objects = self._collapse_duplicates(current_frame_objects)
#         delta_z = self._estimate_train_velocity(current_frame_objects)
#         FPS_COEF = 36.0 
#         train_speed_kmh = delta_z * FPS_COEF

#         # Режим ведения вслепую (Coasting) для удержания целей при пропусках кадров
#         if not current_frame_objects:
#             if not self.past_tracks:
#                 return []
#             updated_past_tracks = []
#             for tr in self.past_tracks:
#                 tr["age"] = tr.get("age", 0) + 1
#                 if tr["age"] > config.TRACK_MAX_COASTING_AGE:
#                     continue
#                 tr["center"][2] -= float(delta_z)
#                 updated_past_tracks.append(tr)
                
#                 # 🛑 ВЫДАЕМ ТОЛЬКО ПОДТВЕРЖДЕННЫЕ ТРЕКИ
#                 if tr["hits"] >= 3:
#                     cx, cy, cz = tr["center"]
#                     w, h, d = tr["dimensions"]
#                     final_safe_objects.append({
#                         "id": tr["id"], "class_id": 1, "confidence": 0.8,
#                         "center": [cx, cy, cz], "dimensions": [w, h, d],
#                         "train_speed": round(train_speed_kmh, 1), "obstacle_speed": 0.0
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
#                     "hits": 1, "age": 0, "birthplace_z": cz, "obstacle_speed_filtered": 0.0
#                 }
#                 new_tracks.append(track_state)
#             self.past_tracks = new_tracks
#             print(f" 🛰️ [Трекер межкадровый]: Инициализация (кадр 0). Накопление истории...", flush=True)
#             return [] # На 1-м кадре hits=1, жестко отсекаем пачки ложных вспышек

#         curr_centers = np.array([obj["center"] for obj in current_frame_objects]).reshape(-1, 3)
#         past_centers = np.array([tr["center"] for tr in self.past_tracks]).reshape(-1, 3)
#         past_centers_predicted = past_centers.copy()
#         past_centers_predicted[:, 2] -= delta_z 

#         dists_3d = np.linalg.norm(curr_centers[:, np.newaxis, :] - past_centers_predicted[np.newaxis, :, :], axis=2)
#         best_past_indices = np.argmin(dists_3d, axis=1)
#         min_dists_3d = np.min(dists_3d, axis=1)
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
#                 birthplace_z = past_track["birthplace_z"]
#                 past_age = past_track.get("age", 0)
#                 age = 0 
#             else:
#                 self.track_id_counter += 1
#                 track_id = self.track_id_counter
#                 hits = 1
#                 birthplace_z = cz
#                 age = 0
#                 past_age = 0

#             # Экспоненциальное сглаживание габаритов 3D бокса
#             if past_track is not None:
#                 past_w, past_h, past_d = past_track["dimensions"]
#                 render_w = (past_w * 0.7) + (width * 0.3)
#                 render_h = (past_h * 0.7) + (height * 0.3)
#                 render_d = (past_d * 0.7) + (depth * 0.3)
                
#                 if past_age == 0:
#                     past_cz_val = float(past_track["center"][2])
#                     distance_convergence = past_cz_val - cz
#                     absolute_delta_z = distance_convergence - delta_z
#                     obstacle_speed_kmh = absolute_delta_z * FPS_COEF
#                 else:
#                     obstacle_speed_kmh = past_track.get("obstacle_speed_filtered", 0.0)
#             else:
#                 render_w, render_h, render_d = width, height, depth
#                 obstacle_speed_kmh = 0.0

#             if abs(obstacle_speed_kmh) < 3.5:
#                 obstacle_speed_kmh = 0.0

#             track_state = {
#                 "id": track_id, 
#                 "center": [cx, cy, cz], 
#                 "dimensions": [render_w, render_h, render_d],
#                 "hits": hits, 
#                 "age": age, 
#                 "birthplace_z": birthplace_z,
#                 "obstacle_speed_filtered": obstacle_speed_kmh
#             }
#             new_tracks.append(track_state)

#             # 🛑 СТОП-КРАН: Фильтр 3-х кадров в действии. Пропускаем цель только если она стабильна!
#             if hits >= 3:
#                 final_safe_objects.append({
#                     "id": track_id, 
#                     "class_id": 1, 
#                     "confidence": 1.0,
#                     "center": [cx, cy, cz], 
#                     "dimensions": [max(0.4, render_w), max(0.4, render_h), max(0.4, render_d)],
#                     "train_speed": round(train_speed_kmh, 1), 
#                     "obstacle_speed": round(obstacle_speed_kmh, 1)
#                 })

#         for j, tr in enumerate(self.past_tracks):
#             if j not in matched_past_indices:
#                 tr["age"] = tr.get("age", 0) + 1
#                 if tr["age"] <= config.TRACK_MAX_COASTING_AGE:
#                     tr["center"] -= float(delta_z)
#                     new_tracks.append(tr)

#         self.past_tracks = new_tracks
#         print(f" 🛰️ [Трекер межкадровый]: Подтверждено треков (hits>=3): {len(final_safe_objects)} | Всего треков в памяти: {len(self.past_tracks)}", flush=True)
#         return final_safe_objects

# +++++++++++++++++++++++++++++
# """
# 🛰️ STATEFUL MOT MODULE v2 (Clean & Optimized Real-Time Engine)
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

#     def reset(self):
#         """Полный сброс межкадровой памяти для изоляции WebSocket-сессий FastAPI."""
#         self.past_tracks = []
#         self.track_id_counter = 0
#         self.is_cabin_calibrated = False

#     def _calibrate_ego_vehicle_cabin(self, points):
#         """Сканирует продольную ось Z перед кабиной и отсекает бампер состава."""
#         if self.is_cabin_calibrated:
#             return self.dynamic_min_z_control

#         cx = points[:, 0]
#         cy = points[:, 1]
#         cz = points[:, 2]

#         cabin_mask = (
#             (cx >= -0.85) & (cx <= 0.85) &
#             (cy >= 0.35)  & (cy <= 2.0)  &
#             (cz >= 1.5)   & (cz <= 10.0)
#         )
#         cabin_points_z = cz[cabin_mask]

#         if len(cabin_points_z) > config.CABIN_MIN_POINTS_THRESHOLD:
#             max_cabin_edge = float(np.max(cabin_points_z))
#             self.dynamic_min_z_control = max_cabin_edge + 0.40
#             self.is_cabin_calibrated = True
#         else:
#             self.dynamic_min_z_control = 3.5
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

#     def track_and_filter_ghosts(self, current_frame_objects, train_step_z=0.0):
#         """MOT-конвейер покадровой ассоциации со строгим фильтром подтверждения целей."""
#         final_safe_objects = []
#         current_frame_objects = self._collapse_duplicates(current_frame_objects)

#         # Жесткое ограничение максимального размера истории (Защита от утечек оперативной памяти)
#         if len(self.past_tracks) > 30:
#             self.past_tracks = self.past_tracks[-10:]

#         # Режим ведения вслепую (Coasting) для удержания целей при пропусках кадров
#         if not current_frame_objects:
#             if not self.past_tracks:
#                 return []
#             updated_past_tracks = []
#             for tr in self.past_tracks:
#                 tr["age"] = tr.get("age", 0) + 1
#                 if tr["age"] > config.TRACK_MAX_COASTING_AGE:
#                     continue
                
#                 # Безопасно обновляем продольную ось NumPy массива/списка
#                 tr["center"][2] = float(tr["center"][2] + train_step_z)
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
        
#         # Прогноз старых треков сдвигается набегающим потоком одометрии по оси Z
#         past_centers_predicted = past_centers.copy()
#         past_centers_predicted[:, 2] += train_step_z 

#         dists_3d = np.linalg.norm(curr_centers[:, np.newaxis, :] - past_centers_predicted[np.newaxis, :, :], axis=2)
#         best_past_indices = np.argmin(dists_3d, axis=1)
#         min_dists_3d = np.min(dists_3d, axis=1)
        
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
#                     # 🟢 ИИ-ФИКС МАССИВА: Корректно смещаем координату Z внутри списка
#                     tr["center"][2] = float(tr["center"][2] + train_step_z)
#                     new_tracks.append(tr)

#         self.past_tracks = new_tracks
#         print(f" 🛰️ [Трекер межкадровый]: Подтверждено треков (hits>=3): {len(final_safe_objects)} | Всего треков в памяти: {len(self.past_tracks)}", flush=True)
#         return final_safe_objects
