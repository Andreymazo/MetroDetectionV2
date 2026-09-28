# # unified_object_engine.py
# import numpy as np

# class UnifiedLidarObjectEngine:
#     """
#     Промышленный унифицированный движок паспортизации, фильтрации и MOT-сопровождения.
#     Оперирует стандартом ЕПО: [Масса, Интенсивность, Геометрия/Форма].
#     """
#     def __init__(self, spatial_gate_radius=2.5, feature_match_threshold=0.75, max_ttl=10, 
#                  spatial_gate_min=0.05, spatial_gate_max=1.45):
#         # Универсальные параметры ИИ-воронки
#         self.spatial_gate_radius = spatial_gate_radius
#         self.feature_match_threshold = feature_match_threshold
#         self.max_ttl = max_ttl
        
#         # Совместимость с пусковым контуром шпал холодного старта
#         self.spatial_gate_min = spatial_gate_min
#         self.spatial_gate_max = spatial_gate_max
        
#         # Глобальный реестр памяти треков
#         self.tracked_registry = []
#         self.global_id_counter = 0

#     def build_fast_rail_passport(self, pts_in_voxel, voxel_val, cz):
#         """
#         Специализированный легковесный паспорт для путевых сущностей (шпал) холодного старта.
#         """
#         cx = float(np.median(pts_in_voxel[:, 0]))
#         cy = float(np.median(pts_in_voxel[:, 1]))
#         mean_intensity = float(np.mean(pts_in_voxel[:, 3]))
#         mass = len(pts_in_voxel)
        
#         std_x = np.std(pts_in_voxel[:, 0]) + 1e-5
#         std_z = np.std(pts_in_voxel[:, 2]) + 1e-5
#         shape_ratio = float(std_x / std_z)

#         if mean_intensity > 40000.0:
#             material_type = "МЕТАЛЛ (СТРЕЛОЧНЫЙ СТЫК/КРЕПЛЕНИЕ) 💎"
#         else:
#             material_type = "БЕТОН/ДЕРЕВО (МОНОЛИТНАЯ ШПАЛА) 🧱"

#         return {
#             "id": None,
#             "local_voxel": voxel_val,
#             "center": [cx, cy, abs(cz)],
#             "mass": mass,
#             "intensity": mean_intensity,
#             "shape_ratio": shape_ratio,
#             "material": material_type
#         }

#     def compute_feature_distance(self, c_obj, p_obj):
#         """
#         Вспомогательная метрика расстояния для фиксированной путевой гребенки шпал.
#         """
#         mass_diff = abs(c_obj["mass"] - p_obj["mass"]) / float(max(1.0, p_obj["mass"]))
#         intensity_diff = abs(c_obj["intensity"] - p_obj["intensity"]) / float(max(1.0, p_obj["intensity"]))
#         shape_diff = abs(c_obj["shape_ratio"] - p_obj["shape_ratio"]) / float(max(0.001, p_obj["shape_ratio"]))
#         return (mass_diff * 0.1) + (intensity_diff * 0.45) + (shape_diff * 0.45)

#     def build_macro_object_passport(self, cluster_points):
#         """
#         ФУНДАМЕНТАЛЬНЫЙ БЛОК: Расчет Единого Цифрового Паспорта (ЕПО) через тензор ковариации.
#         Принимает: Массив точек кластера DBSCAN [N, 4] -> (X, Y, Z, Intensity)
#         """
#         xyz = cluster_points[:, :3]
#         intensities = cluster_points[:, 3]
        
#         cx, cy, cz = np.median(xyz, axis=0)
#         width = float(np.max(xyz[:, 0]) - np.min(xyz[:, 0]))
#         height = float(np.max(xyz[:, 1]) - np.min(xyz[:, 1]))
#         depth = float(np.max(xyz[:, 2]) - np.min(xyz[:, 2]))
        
#         mass = len(cluster_points)
#         mean_intensity = float(np.mean(intensities))
        
#         # Спектральный анализ формы через собственные числа
#         cov_matrix = np.cov(xyz, rowvar=False)
#         if cov_matrix.ndim == 2 and mass > 3:
#             eigenvalues, _ = np.linalg.eigh(cov_matrix)
#             eigenvalues = np.sort(eigenvalues)[::-1]
#             sum_lam = np.sum(eigenvalues) + 1e-8
#             linearity = float((eigenvalues[0] - eigenvalues[1]) / sum_lam)
#             planarity = float((eigenvalues[1] - eigenvalues[2]) / sum_lam)
#             sphericity = float(eigenvalues[2] / sum_lam)
#         else:
#             linearity, planarity, sphericity = 1.0, 0.0, 0.0

#         return {
#             "center": [cx, cy, cz],
#             "dimensions": [width, height, depth],
#             "mass": mass,
#             "intensity": mean_intensity,
#             "geometry": [linearity, planarity, sphericity]
#         }

#     def match_and_track_frame(self, current_clusters, train_step_z=0.0):
#         """
#         ФУНДАМЕНТАЛЬНЫЙ БЛОК 2: Универсальный каскадный ассоциатор для MOT и ЭЯ.
#         Принимает: Список массивов точек [N, 4] текущего кадра и шаг одометрии поезда.
#         """
#         # Поэлементный расчет паспортов для всех входящих кластеров кадра
#         current_passports = [self.build_macro_object_passport(c) for c in current_clusters]
        
#         # Прокидываем сырые точки в паспорта для прецизионного Open3D ICP одометрии стен
#         for idx, c in enumerate(current_clusters):
#             current_passports[idx]["raw_points"] = c
        
#         # Компенсация хода поезда вперед (прогноз положения старых ориентиров/целей)
#         for track in self.tracked_registry:
#             track["center"][2] += float(train_step_z)
            
#         matched_curr_indices = set()
#         updated_registry = []
#         final_frame_output = []

#         # Каскадный ассоциатор (Оригинальные вложенные питоновские циклы)
#         for track in self.tracked_registry:
#             best_match_idx = -1
#             max_score = 0.0
            
#             for c_idx, curr_obj in enumerate(current_passports):
#                 if c_idx in matched_curr_indices:
#                     continue
                    
#                 # Каскад А: Пространственные ворота
#                 spatial_dist = np.linalg.norm(np.array(track["center"]) - np.array(curr_obj["center"]))
#                 if spatial_dist > self.spatial_gate_radius:
#                     continue
                    
#                 # Каскад Б: Сходство физических параметров (Блеск + Тензор формы)
#                 intensity_sim = 1.0 - abs(track["intensity"] - curr_obj["intensity"]) / max(1.0, track["intensity"])
#                 shape_sim = float(np.exp(-np.linalg.norm(np.array(track["geometry"]) - np.array(curr_obj["geometry"])) * 5.0))
                
#                 total_score = (intensity_sim * 0.45) + (shape_sim * 0.45) + (0.10)
                
#                 if total_score > self.feature_match_threshold and total_score > max_score:
#                     max_score = total_score
#                     best_match_idx = c_idx

#             if best_match_idx != -1:
#                 matched_obj = current_passports[best_match_idx]
#                 track.update(matched_obj)
#                 track["hits"] += 1
#                 track["ttl"] = self.max_ttl
#                 matched_curr_indices.add(best_match_idx)
#                 updated_registry.append(track)
                
#                 if track["hits"] >= 3:
#                     final_frame_output.append(track)
#             else:
#                 track["ttl"] -= 1
#                 if track["ttl"] > 0:
#                     updated_registry.append(track)

#         # Регистрация абсолютно новых объектов
#         for c_idx, curr_obj in enumerate(current_passports):
#             if c_idx not in matched_curr_indices:
#                 self.global_id_counter += 1
#                 curr_obj["id"] = self.global_id_counter
#                 curr_obj["hits"] = 1
#                 curr_obj["ttl"] = self.max_ttl
#                 updated_registry.append(curr_obj)

#         self.tracked_registry = updated_registry
#         return final_frame_output

# unified_object_engine.py
import numpy as np

class UnifiedLidarObjectEngine:
    """
    Промышленный унифицированный движок паспортизации, фильтрации и MOT-сопровождения.
    Оперирует стандартом ЕПО: [Масса, Интенсивность, Геометрия/Форма].
    """
    def __init__(self, spatial_gate_radius=2.5, feature_match_threshold=0.75, max_ttl=10, 
                 spatial_gate_min=0.05, spatial_gate_max=1.45):
        # Универсальные параметры ИИ-воронки
        self.spatial_gate_radius = spatial_gate_radius
        self.feature_match_threshold = feature_match_threshold
        self.max_ttl = max_ttl
        
        # Совместимость с пусковым контуром шпал холодного старта
        self.spatial_gate_min = spatial_gate_min
        self.spatial_gate_max = spatial_gate_max
        
        # Глобальный реестр памяти треков
        self.tracked_registry = []
        self.global_id_counter = 0

    def build_fast_rail_passport(self, pts_in_voxel, voxel_val, cz):
        """
        Специализированный легковесный паспорт для путевых сущностей (шпал) холодного старта.
        """
        cx = float(np.median(pts_in_voxel[:, 0]))
        cy = float(np.median(pts_in_voxel[:, 1]))
        mean_intensity = float(np.mean(pts_in_voxel[:, 3]))
        mass = len(pts_in_voxel)
        
        std_x = np.std(pts_in_voxel[:, 0]) + 1e-5
        std_z = np.std(pts_in_voxel[:, 2]) + 1e-5
        shape_ratio = float(std_x / std_z)

        if mean_intensity > 40000.0:
            material_type = "МЕТАЛЛ (СТРЕЛОЧНЫЙ СТЫК/КРЕПЛЕНИЕ) 💎"
        else:
            material_type = "БЕТОН/ДЕРЕВО (МОНОЛИТНАЯ ШПАЛА) 🧱"

        return {
            "id": None,
            "local_voxel": voxel_val,
            "center": [cx, cy, abs(cz)],
            "mass": mass,
            "intensity": mean_intensity,
            "shape_ratio": shape_ratio,
            "material": material_type
        }

    def compute_feature_distance(self, c_obj, p_obj):
        """
        Вспомогательная метрика расстояния для фиксированной путевой гребенки шпал.
        """
        mass_diff = abs(c_obj["mass"] - p_obj["mass"]) / float(max(1.0, p_obj["mass"]))
        intensity_diff = abs(c_obj["intensity"] - p_obj["intensity"]) / float(max(1.0, p_obj["intensity"]))
        shape_diff = abs(c_obj["shape_ratio"] - p_obj["shape_ratio"]) / float(max(0.001, p_obj["shape_ratio"]))
        return (mass_diff * 0.1) + (intensity_diff * 0.45) + (shape_diff * 0.45)

    def build_macro_object_passport(self, cluster_points):
        """
        ФУНДАМЕНТАЛЬНЫЙ БЛОК: Расчет Единого Цифрового Паспорта (ЕПО) через тензор ковариации.
        Принимает: Массив точек кластера DBSCAN [N, 4] -> (X, Y, Z, Intensity) или [N, 3] -> (X, Y, Z)
        """
        xyz = cluster_points[:, :3]
        mass = len(cluster_points)
        
        # Инициализируем базовое значение интенсивности по умолчанию
        mean_intensity = 1.0
        
        # Проверяем реальное количество колонок в матрице (наличие 4-го канала)
        if cluster_points.shape[1] >= 4:
            intensities = cluster_points[:, 3]
            mean_intensity = float(np.mean(intensities))

        cx, cy, cz = np.median(xyz, axis=0)
        width = float(np.max(xyz[:, 0]) - np.min(xyz[:, 0]))
        height = float(np.max(xyz[:, 1]) - np.min(xyz[:, 1]))
        depth = float(np.max(xyz[:, 2]) - np.min(xyz[:, 2]))
        
        # Спектральный анализ формы через собственные числа
        cov_matrix = np.cov(xyz, rowvar=False)
        if cov_matrix.ndim == 2 and mass > 3:
            eigenvalues, _ = np.linalg.eigh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Сортируем от больших к меньшим [λ1, λ2, λ3]
            
            sum_lam = np.sum(eigenvalues) + 1e-8
            linearity = float((eigenvalues[0] - eigenvalues[1]) / sum_lam)
            planarity = float((eigenvalues[1] - eigenvalues[2]) / sum_lam)
            sphericity = float(eigenvalues[2] / sum_lam)
        else:
            linearity, planarity, sphericity = 1.0, 0.0, 0.0

        return {
            "center": [cx, cy, cz],
            "dimensions": [width, height, depth],
            "mass": mass,
            "intensity": mean_intensity,
            "geometry": [linearity, planarity, sphericity]
        }


    def match_and_track_frame(self, current_clusters, train_step_z=0.0):
        """
        ФУНДАМЕНТАЛЬНЫЙ БЛОК 2: Универсальный каскадный ассоциатор для MOT и ЭЯ.
        Принимает: Список массивов точек [N, 4] текущего кадра и шаг одометрии поезда.
        """
                # Фиксируем базовый тип: проверяем, что прилетело на вход
        is_dict_input = len(current_clusters) > 0 and isinstance(current_clusters[0], dict)

        if is_dict_input:
            # 🟢 ВЕКТОРНЫЙ КОНТУР ДЕТЕКЦИИ (Препятствия): Скорость CPU ➔ 0 мс
            # Извлекаем данные из всех словарей сразу через генерацию списков,
            # но превращаем их в монолитные NumPy-массивы для каскадов скоринга
            current_passports = []
            
            # Быстрая сборка объектов. Физика тензора формы [0.0, 0.0, 1.0] 
            # внедряется мгновенно, минуя тяжелый ковариационный анализ.
            for c in current_clusters:
                current_passports.append({
                    "id": c.get("id", None),
                    "center": c["center"],
                    "dimensions": c["dimensions"],
                    "mass": c.get("mass", c.get("points_count", 1)),
                    "intensity": c.get("intensity", 1.0),
                    "geometry": [0.0, 0.0, 1.0],  # Чистая сферичность
                    "raw_points": c.get("raw_points", [])
                })
        else:
            # 🏢 ВЕКТОРНЫЙ КОНТУР ОДОМЕТРИИ (Стены): Сохраняет исходный ICP-паспорт
            # Здесь по-прежнему вызывается поэлементный расчет ковариации,
            # так как для стен нам физически необходимо извлекать собственные числа.
            current_passports = [self.build_macro_object_passport(c) for c in current_clusters]
            # Прокидываем сырые точки в паспорта стен для прецизионного Open3D ICP
            for idx, c in enumerate(current_clusters):
                current_passports[idx]["raw_points"] = c

        
        # # Компенсация хода поезда вперед (прогноз положения старых ориентиров/целей)
        # for track in self.tracked_registry:
        #     track["center"][2] += float(train_step_z)
            
        # matched_curr_indices = set()
        # updated_registry = []
        # final_frame_output = []

        # for track in self.tracked_registry:
        #     best_match_idx = -1
        #     max_score = 0.0
            
        #     for c_idx, curr_obj in enumerate(current_passports):
        #         if c_idx in matched_curr_indices:
        #             continue
                    
        #         # Каскад А: Пространственные ворота
        #         spatial_dist = np.linalg.norm(np.array(track["center"]) - np.array(curr_obj["center"]))
        #         if spatial_dist > self.spatial_gate_radius:
        #             continue
                    
        #         # Каскад Б: Сходство физических параметров (Блеск + Тензор формы)
        #         intensity_sim = 1.0 - abs(track["intensity"] - curr_obj["intensity"]) / max(1.0, track["intensity"])
        #         shape_sim = float(np.exp(-np.linalg.norm(np.array(track["geometry"]) - np.array(curr_obj["geometry"])) * 5.0))
                
        #         total_score = (intensity_sim * 0.45) + (shape_sim * 0.45) + (0.10)
                
        #         if total_score > self.feature_match_threshold and total_score > max_score:
        #             max_score = total_score
        #             best_match_idx = c_idx

        #     if best_match_idx != -1:
        #         matched_obj = current_passports[best_match_idx]
        #         track.update(matched_obj)
        #         track["hits"] += 1
        #         track["ttl"] = self.max_ttl
        #         matched_curr_indices.add(best_match_idx)
        #         updated_registry.append(track)
                
        #         if track["hits"] >= 3:
        #             final_frame_output.append(track)
        #     else:
        #         track["ttl"] -= 1
        #         if track["ttl"] > 0:
        #             updated_registry.append(track)

        # for c_idx, curr_obj in enumerate(current_passports):
        #     if c_idx not in matched_curr_indices:
        #         self.global_id_counter += 1
        #         curr_obj["id"] = self.global_id_counter
        #         curr_obj["hits"] = 1
        #         curr_obj["ttl"] = self.max_ttl
        #         updated_registry.append(curr_obj)
                # Компенсация хода поезда вперед (прогноз положения старых ориентиров/целей)
        for track in self.tracked_registry:
            track["center"][2] -= float(train_step_z)

        # ⚡ ВЕКТОРНЫЙ РАСЧЕТ МАТРИЦЫ РАССТОЯНИЙ (Broadcasting)
        # Считаем геометрию между всеми старыми треками и новыми паспортами за раз
        if self.tracked_registry and current_passports:
            tracks_centers = np.array([t["center"] for t in self.tracked_registry], dtype=np.float32)  # [M, 3]
            curr_centers = np.array([c["center"] for c in current_passports], dtype=np.float32)      # [N, 3]
            dist_matrix = np.linalg.norm(tracks_centers[:, np.newaxis, :] - curr_centers[np.newaxis, :, :], axis=2)  # [M, N]
        else:
            dist_matrix = np.array([], dtype=np.float32).reshape(0, len(current_passports))
            
        matched_curr_indices = set()
        updated_registry = []
        final_frame_output = []

        # Каскадный ассоциатор (Проход по трекам без вложенных тяжелых вычислений)
        for t_idx, track in enumerate(self.tracked_registry):
            best_match_idx = -1
            max_score = 0.0
            
            if dist_matrix.size > 0:
                for c_idx, curr_obj in enumerate(current_passports):
                    if c_idx in matched_curr_indices:
                        continue
                        
                    # 🟢 МГНОВЕННЫЙ КАСКАД А: Извлекаем готовое расстояние из NumPy-матрицы
                    spatial_dist = float(dist_matrix[t_idx, c_idx])
                    if spatial_dist > self.spatial_gate_radius:
                        continue
                        
                    # Каскад Б: Сходство физических параметров (Блеск + Тензор формы)
                    intensity_sim = 1.0 - abs(track["intensity"] - curr_obj["intensity"]) / max(1.0, track["intensity"])
                    shape_sim = float(np.exp(-np.linalg.norm(np.array(track["geometry"]) - np.array(curr_obj["geometry"])) * 5.0))
                    
                    total_score = (intensity_sim * 0.45) + (shape_sim * 0.45) + (0.10)
                    
                    if total_score > self.feature_match_threshold and total_score > max_score:
                        max_score = total_score
                        best_match_idx = c_idx

            if best_match_idx != -1:
                matched_obj = current_passports[best_match_idx]
                track.update(matched_obj)
                track["hits"] += 1
                track["ttl"] = self.max_ttl
                matched_curr_indices.add(best_match_idx)
                updated_registry.append(track)
                
                if track["hits"] >= 3:
                    final_frame_output.append(track)
            else:
                track["ttl"] -= 1
                if track["ttl"] > 0:
                    updated_registry.append(track)

        # Регистрация новых физических препятствий/ориентиров (Холодный старт трека)
        for c_idx, curr_obj in enumerate(current_passports):
            if c_idx not in matched_curr_indices:
                self.global_id_counter += 1
                curr_obj["id"] = self.global_id_counter
                curr_obj["hits"] = 1
                curr_obj["ttl"] = self.max_ttl
                updated_registry.append(curr_obj)

        self.tracked_registry = updated_registry
        return final_frame_output
