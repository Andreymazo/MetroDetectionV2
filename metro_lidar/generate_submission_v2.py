"""
⚙️ AUTONOMOUS SUBWAY VISION CORE v2 (Fixed Axes)
Компонент покадровой детекции и геометрической фильтрации препятствий.
"""

import os
import csv
import gc
import numpy as np
from sklearn.cluster import DBSCAN
import metro_lidar.config as config

def filter_selective_load_balancing_v2(points):
    """
    Селективный балансировщик нагрузки для отрицательной оси Z (версия v2).
    Использует штатные геометрические параметры габарита поезда для защиты 
    дальних рубежей детекции препятствий (< -V2_OBSTACLE_SAFE_DISTANCE_Z).
    """
    raw_count = len(points)
    
    # Если кадр изначально разреженный, пропускаем его без лишних вычислений
    if raw_count <= config.V2_OBSTACLE_TARGET_POINTS:
        return points
        
    # КРИТИЧЕСКАЯ МАСКА ПРЕПЯТСТВИЙ ДЛЯ V2 (Z идет в минус):
    # Выделяем дальнюю зону строго внутри габаритов поезда, заданных в config.py
    critical_far_mask = (
        (points[:, 0] >= -config.TRAIN_HALF_WIDTH) & 
        (points[:, 0] <= config.TRAIN_HALF_WIDTH) &
        (points[:, 1] >= config.MIN_Y) & 
        (points[:, 1] <= config.MAX_Y) &
        (points[:, 2] < -config.V2_OBSTACLE_SAFE_DISTANCE_Z) # Дальше безопасного рубежа (например, -45м < -40м)
    )
    
    # Все остальные точки (ближняя зона + боковые стены туннеля)
    redundant_mask = ~critical_far_mask
    
    critical_points = points[critical_far_mask]
    redundant_points = points[redundant_mask]
    
    # Вычисляем, сколько места осталось под фоновые точки в рамках лимита детекции препятствий
    allowed_redundant_slots = config.V2_OBSTACLE_TARGET_POINTS - len(critical_points)
    
    if allowed_redundant_slots > 1000:
        skip_step = len(redundant_points) // allowed_redundant_slots
        skip_step = max(1, skip_step)
    else:
        skip_step = 2  # Дефолтный срез, если дальний створ путей забил весь лимит
        
    if skip_step > 1:
        redundant_downsampled = redundant_points[::skip_step]
        # Собираем облако обратно: монолитный дальний габарит + прореженный фон
        points = np.vstack((critical_points, redundant_downsampled))
        print(f"   [СМАРТ-ЩИТ OBSTACLE v2] Вход: {raw_count} точек. Спасено на дальней дистанции: {len(critical_points)}. "
              f"Фон сжат с шагом {skip_step}. Итог: {len(points)} строк.", flush=True)
              
    return points


# def voxel_downsample_adaptive(points):
#     """Сжимает облако точек на входе с помощью адаптивной 3D-сетки вокселей."""
#     if len(points) < 100:
#         return points
#     z_coords = points[:, 2] 
#     dynamic_size = config.VOXEL_SIZE_BASE + (z_coords * config.VOXEL_COEF_Z)
#     indexed_coords = np.floor(points / dynamic_size[:, np.newaxis]).astype(np.int32)
#     _, indices = np.unique(indexed_coords, axis=0, return_index=True)
#     return points[indices]
def voxel_downsample_adaptive(points):
    """
    Сжимает облако точек на входе с помощью адаптивной 3D-сетки.
    Защищена от отрицательных координат одометрии v12.
    """
    if len(points) < 100:
        return points

    # 🟢 КРИТИЧЕСКИЙ ФИКС: Берем дальность по модулю abs(), 
    # чтобы размер вокселя рос конусом вперед, а не схлопывался в ноль в отрицательном Z!
    z_coords = np.abs(points[:, 2])
    
    dynamic_size = config.VOXEL_SIZE_BASE + (z_coords * config.VOXEL_COEF_Z)

    # Квантуем пространство: переводим физические метры в целые 3D-адреса кубиков
    indexed_coords = np.floor(points / dynamic_size[:, np.newaxis]).astype(np.int32)

    # Находим уникальные кубики
    _, indices = np.unique(indexed_coords, axis=0, return_index=True)

    # print(f"   [ЗАЩИТНЫЙ ВОКСЕЛЬНЫЙ ЩИТ] Сжал облако до {len(indices)} точек.", flush=True)
    return points[indices]

# def extract_floor_with_ransac(points):
#     """Вырезает плоскость рельсового полотна вокруг нулевой отметки высоты Y."""
#     if len(points) < 10:
#         mask = points[:, 1] > 0.1
#         return points[mask], points[~mask]
        
#     # Нижняя граница остается жесткой (-2.0), а верхнюю выносим под управление ИИ-детектора
#     floor_mask = (points[:, 1] >= -2.0) & (points[:, 1] <= config.V2_OBSTACLE_FLOOR_MAX_Y)
#     return points[~floor_mask], points[floor_mask]

# def filter_gauge_with_track_bending(all_points, floor_points, live_min_z):
#     """
#     Динамически вычисляет кривизну и уклоны путей в 3D (X и Y) по точкам пола.
#     Защищает систему от слепоты на поворотах и при тряске поезда.
#     """
#     if len(all_points) == 0:
#         return all_points

#     z_all = all_points[:, 2]
#     predicted_x_center = np.zeros_like(z_all)
#     predicted_y_floor = np.zeros_like(z_all) # Массив живого профиля высоты рельс
    
#     STEP_Z = 5.0
#     last_valid_x = 0.0
#     last_valid_y = -1.72  # Дефолтная глубина рельс по паспорту ОЯР
#     has_valid_track = False
    
#     for start_z in np.arange(live_min_z, config.MAX_Z, STEP_Z):
#         end_z = start_z + STEP_Z
#         # Фильтруем сектор с учетом отрицательного Z в v2
#         mask_floor_sector = (
#             (floor_points[:, 2] <= -start_z) & (floor_points[:, 2] > -end_z) &
#             (floor_points[:, 0] >= -2.0) & (floor_points[:, 0] <= 2.0)
#         )
#         floor_sector = floor_points[mask_floor_sector]
        
#         if len(floor_sector) > 15:
#             current_x = np.median(floor_sector[:, 0])
#             current_y = np.median(floor_sector[:, 1]) # Честный замер высоты рельс
#             last_valid_x = current_x
#             last_valid_y = current_y
#             has_valid_track = True
#         else:
#             current_x = last_valid_x
#             current_y = last_valid_y
            
#         sector_all_mask = (z_all <= -start_z) & (z_all > -end_z)
#         predicted_x_center[sector_all_mask] = current_x
#         predicted_y_floor[sector_all_mask] = current_y

#     current_half_width = config.TRAIN_HALF_WIDTH if has_valid_track else 2.40
#     absolute_z = np.abs(all_points[:, 2])

#     # Виртуальный вагон: ищет строго от +15см до +3.0м НАД ЖИВЫМИ РЕЛЬСАМИ в каждом секторе
#     in_curved_gauge_mask = (
#         (all_points[:, 0] >= (predicted_x_center - current_half_width)) &
#         (all_points[:, 0] <= (predicted_x_center + current_half_width)) &
#         (all_points[:, 1] >= (predicted_y_floor + 0.15)) &  
#         (all_points[:, 1] <= (predicted_y_floor + 3.00)) &  
#         (absolute_z >= live_min_z) & (absolute_z <= config.MAX_Z)
#     )
#     return all_points[in_curved_gauge_mask]

def extract_floor_with_ransac(points):
    """
    Динамическая секторная сегментация рельсового полотна (Пункт 1 ТУДУ).
    Вычисляет живой профиль высоты пола в каждом 5-метровом Z-секторе, 
    что исключает слепоту детектора на затяжных спусках, подъемах и при наклоне лидара.
    """
    if len(points) < 10:
        mask = points[:, 1] > 0.1
        return points[mask], points[~mask]
        
    z_coords = points[:, 2]
    # Создаем булеву маску для всего облака, по умолчанию все точки — НЕ пол (False)
    global_floor_mask = np.zeros(len(points), dtype=bool)
    
    STEP_Z = 5.0
    # Итерируемся по дальности. Так как в v2 дальность отрицательная, 
    # start_z идет по модулю от 0 до MAX_Z
    for start_z in np.arange(0.0, config.MAX_Z, STEP_Z):
        end_z = start_z + STEP_Z
        
        # Вырезаем текущий 5-метровый сектор туннеля (Z идет в минус)
        sector_mask = (z_coords <= -start_z) & (z_coords > -end_z)
        
        # Дополнительно сужаем зону поиска пола по ширине X (±1.5м вокруг оси состава),
        # чтобы стены туннеля на поворотах не искажали медиану высоты путей
        search_floor_mask = sector_mask & (points[:, 0] >= -1.5) & (points[:, 0] <= 1.5)
        sector_points = points[search_floor_mask]
        
        if len(sector_points) > 15:
            # Находим честный геометрический уровень рельс (ноль путей) в этом секторе дальности
            sector_floor_y = np.median(sector_points[:, 1])
            
            # Точки в этом секторе признаются полом, если они лежат в строго зажатом зазоре:
            # от 35 см ниже рельс (балласт) до 15 см выше рельс (шпалы/головка рельса)
            floor_y_condition = (points[:, 1] >= (sector_floor_y - 0.35)) & (points[:, 1] <= (sector_floor_y + 0.15))
            
            # Записываем маску пола для текущего сектора в общий массив
            global_floor_mask[sector_mask & floor_y_condition] = True
        else:
            # Если сектор пустой или глубокий горизонт — используем грубый базовый дефолт
            floor_y_condition = (points[:, 1] >= -2.0) & (points[:, 1] <= -1.40)
            global_floor_mask[sector_mask & floor_y_condition] = True
            
    # Разделяем облако точек: инвертированная маска идет в препятствия, прямая — в пол
    return points[~global_floor_mask], points[global_floor_mask]

def filter_gauge_with_track_bending(all_points, floor_points, live_min_z, is_open_space=False):
    """
    Криволинейный створ ворот безопасности 3D (Полностью очищен от хардкода).
    Динамически зажимает область сканирования рельс при прохождении платформ и стрелок.
    """
    if len(all_points) == 0:
        return all_points

    z_all = all_points[:, 2]
    predicted_x_center = np.zeros_like(z_all)
    predicted_y_floor = np.zeros_like(z_all)
    
    STEP_Z = 4.0  
    last_valid_x = 0.0
    last_valid_y = -1.50
    has_valid_track = False
    
    for start_z in np.arange(live_min_z, config.MAX_Z, STEP_Z):
        end_z = start_z + STEP_Z
        
        # 🟢 ПЕРЕВОД НА КОНСТАНТЫ: Адаптивно выбираем ширину створа поиска рельс
        current_search_width = (
            config.V2_OBSTACLE_TRACK_SEARCH_WIDTH_CLOSED if is_open_space 
            else config.V2_OBSTACLE_TRACK_SEARCH_WIDTH_OPEN
        )
        
        mask_floor_sector = (
            (floor_points[:, 2] <= -start_z) & (floor_points[:, 2] > -end_z) &
            (floor_points[:, 0] >= -current_search_width) & (floor_points[:, 0] <= current_search_width)
        )
        floor_sector = floor_points[mask_floor_sector]
        
        if len(floor_sector) > 15:
            current_x = np.median(floor_sector[:, 0])
            current_y = np.median(floor_sector[:, 1])
            last_valid_x = current_x
            last_valid_y = current_y
            has_valid_track = True
        else:
            current_x = last_valid_x
            current_y = last_valid_y
            
        sector_all_mask = (z_all <= -start_z) & (z_all > -end_z)
        predicted_x_center[sector_all_mask] = current_x
        predicted_y_floor[sector_all_mask] = current_y

    # 🟢 ПЕРЕВОД НА КОНСТАНТЫ: Логика раскрытия ворот безопасности по мере дальности Z
    if is_open_space:
        adaptive_half_width = np.full_like(z_all, config.TRAIN_HALF_WIDTH)
    else:
        distance_from_train = np.abs(all_points[:, 2])
        adaptive_half_width = config.TRAIN_HALF_WIDTH + (distance_from_train * config.V2_OBSTACLE_GATE_EXPANSION_COEF)
        adaptive_half_width = np.clip(adaptive_half_width, config.TRAIN_HALF_WIDTH, config.V2_OBSTACLE_GATE_MAX_WIDTH)

    # Вычисляем дальность отдельно, защищая NumPy-конвейер от конфликта типов ufunc
    absolute_z = np.abs(all_points[:, 2])

    in_curved_gauge_mask = (
        (all_points[:, 0] >= (predicted_x_center - adaptive_half_width)) &
        (all_points[:, 0] <= (predicted_x_center + adaptive_half_width)) &
        (all_points[:, 1] >= (predicted_y_floor + 0.15)) &  
        (all_points[:, 1] <= (predicted_y_floor + 3.00)) &  
        (absolute_z >= live_min_z) & (absolute_z <= config.MAX_Z)
    )
    return all_points[in_curved_gauge_mask]
"""
    Схлопывает ложные пространственные дубликаты и "нарезку" монолитных объектов,
    возникающую по оси Z из-за деформации пространства алгоритмом DBSCAN.
    
    МЕТРИКА РАЗДЕЛЕНИЯ ОБЪЕКТОВ (Ответ на физический смысл фильтра):
    Чтобы система гарантированно признала два скопления точек РАЗНЫМИ препятствиями, 
    они должны удовлетворять МИНИМУМ ОДНОМУ из следующих условий:
    
    1. По продольной оси тоннеля (Z): Дистанция между центрами объектов должна быть 
       БОЛЬШЕ ИЛИ РАВНА `distance_gate_z` (по умолчанию 3.0 метра). 
       В бэге жюри объекты расставлены с шагом ~100 метров, поэтому порог в 3.0 метра 
       идеально склеивает "разорванные" куски одной коробки или длинного предмета на рельсах, 
       но гарантированно разделяет идущие друг за другом независимые угрозы.
       
    2. По поперечной оси путей (X): Расстояние между центрами объектов право/лево 
       должно быть БОЛЬШЕ ИЛИ РАВНО `width_gate_x` (по умолчанию 1.2 метра).
       Это исключает склеивание объектов, стоящих на одном расстоянии Z, но в разных 
       габаритах (например, один на левом рельсе, а другой глубоко за пределами путей).

    Параметры:
    ----------
    raw_detected_objects : list
        Список словарей сырых детекций из `find_obstacles_adaptive_density`.
    distance_gate_z : float, default 3.0
        Продольные тиски слияния (в метрах). Защита от дробления длинных предметов вдоль пути.
    width_gate_x : float, default 1.2
        Поперечные тиски слияния (в метрах). Защита от бокового расползания.

    Возвращает:
    -----------
    list
        Очищенный от дубликатов список макро-объектов с пересчитанными 3D-габаритами.
    """
def merge_and_filter_raw_obstacles(raw_detected_objects, distance_gate_z=3.0, width_gate_x=1.2):
    """
    Схлопывает ложные пространственные дубликаты и "нарезку" монолитных объектов,
    возникающую по оси Z из-за деформации пространства алгоритмом DBSCAN.
    
    Учитывает отрицательную полярность оси Z в ИИ-конвейере Metro v12.
    """
    if not raw_detected_objects:
        return []
        
    # Сортируем объекты по дальности Z (от ближних к дальним, учитывая отрицательную ось)
    sorted_objs = sorted(raw_detected_objects, key=lambda x: x["center"][2], reverse=True)
    merged_objects = []
    
    while sorted_objs:
        base_obj = sorted_objs.pop(0)
        bx, by, bz = base_obj["center"]
        
        still_clean_objs = []
        for check_obj in sorted_objs:
            cx, cy, cz = check_obj["center"]
            
            # Проверяем вхождение в ворота дедупликации (используем abs для Z и X)
            if abs(cz - bz) < distance_gate_z and abs(cx - bx) < width_gate_x:
                # Находим крайние точки боксов по всем трем осям для пересчета макро-габаритов
                max_x = max(bx + base_obj["dimensions"][0]/2, cx + check_obj["dimensions"][0]/2)
                min_x = min(bx - base_obj["dimensions"][0]/2, cx - check_obj["dimensions"][0]/2)
                max_y = max(by + base_obj["dimensions"][1]/2, cy + check_obj["dimensions"][1]/2)
                min_y = min(by - base_obj["dimensions"][1]/2, cy - check_obj["dimensions"][1]/2)
                max_z = max(bz + base_obj["dimensions"][2]/2, cz + check_obj["dimensions"][2]/2)
                min_z = min(bz - base_obj["dimensions"][2]/2, cz - check_obj["dimensions"][2]/2)
                
                # Обновляем размеры макро-объекта
                base_obj["dimensions"] = [max_x - min_x, max_y - min_y, max_z - min_z]
                # Смещаем центр масс в прецизионную середину нового общего бокса
                base_obj["center"] = [(max_x + min_x)/2.0, (max_y + min_y)/2.0, (max_z + min_z)/2.0]
                # Суммируем массу точек
                base_obj["points_count"] += check_obj["points_count"]
                
                # Объединяем прореженное облако точек для фронтенда WebGL
                if "raw_points" in base_obj and "raw_points" in check_obj:
                    base_obj["raw_points"].extend(check_obj["raw_points"])
            else:
                still_clean_objs.append(check_obj)
                
        merged_objects.append(base_obj)
        sorted_objs = still_clean_objs
        
    return merged_objects


def find_obstacles_adaptive_density(points_inside_gauge, floor_points):
    """Кластеризует пространственные аномалии внутри габарита сжатием пространства."""
    raw_detected_objects = []
    if len(points_inside_gauge) < 2:
        return raw_detected_objects
        
    dynamic_start_z = 30.0
    if len(floor_points) > 100:
        bins = np.arange(10, int(config.MAX_Z) + 10, 5)
        counts, edges = np.histogram(floor_points[:, 2], bins=bins)
        low_density_indices = np.where(counts < 150)
        if len(low_density_indices) > 0 and len(low_density_indices[0]) > 0:
            dynamic_start_z = float(edges[low_density_indices[0][0]])
        dynamic_start_z = max(15.0, min(dynamic_start_z, 180.0))

    scaled_points = points_inside_gauge.copy()
    z_coords = scaled_points[:, 2]
    scaling_factors = np.where(z_coords > dynamic_start_z, dynamic_start_z / z_coords, 1.0)
    scaled_points[:, 0] *= scaling_factors
    scaled_points[:, 1] *= scaling_factors
    
    db = DBSCAN(eps=config.DBSCAN_OBSTACLE_EPS, min_samples=config.DBSCAN_OBSTACLE_MIN_SAMPLES, algorithm='kd_tree', n_jobs=-1).fit(scaled_points)
    labels = db.labels_
    
    for label in set(labels):
        if label == -1:
            continue
        cluster_indices = np.where(labels == label)
        original_cluster_points = points_inside_gauge[cluster_indices]
        # === НАЧАЛО ПРЕЦИЗИОННОГО РАСЧЕТА ЯДРА ОБЪЕКТА ===
        # Вычисляем честные геометрические габариты бокса вдоль осей
        width = float(np.max(original_cluster_points[:, 0]) - np.min(original_cluster_points[:, 0]))
        height = float(np.max(original_cluster_points[:, 1]) - np.min(original_cluster_points[:, 1]))
        depth = float(np.max(original_cluster_points[:, 2]) - np.min(original_cluster_points[:, 2]))
        
        # Находим прецизионный центр по осям X и Z с помощью np.median
        cx = float(np.median(original_cluster_points[:, 0]))
        cz = float(np.median(original_cluster_points[:, 2]))
        
        # Хирургический фикс оси высоты Y:
        # Находим самую нижнюю точку объекта и поднимаем центр ровно на половину высоты.
        # Это гарантирует, что 3D-модель на фронтенде будет стоять строго НА рельсах, а не тонуть в них.
        min_y_pts = np.min(original_cluster_points[:, 1])
        cy = float(min_y_pts + (height / 2.0))
        
        # Средняя интенсивность лазерного отражения для данного кластера
        cluster_intensities = original_cluster_points[:, 3]
        mean_intensity = float(np.mean(cluster_intensities))
        
        # Собираем СЫРЫЕ объекты, сохраняя ОРИГИНАЛЬНЫЙ интерфейс списка [cx, cy, cz]
                # Собираем СЫРЫЕ объекты, сохраняя интерфейс и добавляя прореженное лазерное облако кластера
        # Берем каждую 5-ю точку кластера, чтобы не перегружать сеть и WebGL рендер (разгрузка в 5 раз!)
        cluster_pts_downsampled = original_cluster_points[::5]
        
        raw_detected_objects.append({
            "class_id": 1,
            "confidence": 1.0,
            "center": [cx, cy, cz],  
            "dimensions": [width, height, depth],
            "points_count": len(original_cluster_points),  
            "intensity": mean_intensity,
            # Сохраняем только координаты X, Y, Z прореженных точек для фронтенда
            "raw_points": cluster_pts_downsampled[:, :3].tolist() 
        })

    return raw_detected_objects

def validate_and_filter_objects(raw_objects, past_tracks=None):
    """
    Кадровая валидация 3D-боксов с молчаливым обогащением признаков морфологии.
    Использует непрерывную динамическую модель («дышащую воронку») для пробития дальности.
    """
    confirmed_objects = []
    raw_clusters_count = len(raw_objects)
    
    # Базовый множитель шума (как в оригинале)
    noise_multiplier = min(2.5, 1.0 + max(0.0, (raw_clusters_count - 20) / 50.0))

    for obj in raw_objects:
        cx, cy, cz = obj["center"]
        width, height, depth = obj["dimensions"]
        points_count = obj["points_count"]
        intensity = obj["intensity"]
        obj_volume = width * height * depth
        distance = abs(cz)
        
        if intensity > config.MAX_TARGET_INTENSITY:
            continue

        # =====================================================================
        # 🧠 МАТЕМАТИЧЕСКИЙ АППАРАТ ДИНАМИЧЕСКОЙ «ДЫШАЩЕЙ ВОРОНКИ»
        # =====================================================================
        # 1. Затухание множителя шума с ростом дистанции.
        # На расстоянии 100+ метров локальный шум у колес поезда больше не ослепляет ИИ.
        local_noise_weight = max(0.0, min(1.0, 1.0 - (distance - 30.0) / 60.0))
        effective_noise_mult = 1.0 + (noise_multiplier - 1.0) * local_noise_weight

        # 2. Непрерывная функция требуемого количества точек (Гиперболическое затухание)
        # Ближний бой (10м) -> ~12 точек; Средний (50м) -> ~5 точек; Дальний (100м+) -> строго 2-3 точки.
        # Формула идеально аппроксимирует физическое расхождение лучей Hesai 128
        dynamic_min_points = 2.0 + (15.0 / (1.0 + 0.05 * distance))
        min_points = max(2, int(dynamic_min_points * effective_noise_mult))

        # 3. Динамический плавный порог объема (Volume Decay)
        # Падает экспоненциально, так как редкие точки на горизонте теряют объемную форму.
        volume_decay = np.exp(-distance / 45.0)
        min_volume = config.NEAR_MIN_VOLUME * effective_noise_mult * max(0.02, volume_decay)

        # 4. Динамический линейный размер (Dimension Decay)
        # Позволяет зацепить коробку 0.3х0.3м на 100 метрах всего по паре лучей (8 см допуска)
        dimension_decay = max(0.3, 1.0 - 0.006 * distance)
        min_dimension = config.NEAR_MIN_DIMENSION * dimension_decay

        # 5. 🛡️ АНТИ-ПРИЗРАК: Динамический высотный предохранитель для сверхдальней зоны
        # Если объект дальше 75 метров подтверждается всего по 2-3 точкам, 
        # он обязан находиться строго в вертикальном габарите пути (не летать под потолком/сводом тоннеля)
        if distance > 75.0 and points_count <= 4:
            # y_center (cy) должен быть в створе рельс и роста препятствия (от -1.5 до 1.5м относительно лидара)
            if cy < -1.5 or cy > 1.5:
                continue # Выбрасываем дальний одиночный шум на стенах/своде

        # =====================================================================
        # Проверка по динамически рассчитанным критериям физики лазера
        # =====================================================================
        if obj_volume >= min_volume and points_count >= min_points and max(width, height, depth) >= min_dimension:
            # Расчет аналитики положения объекта (как в оригинале)
            deviation_x = abs(cx)
            if deviation_x <= 0.35:
                obj["position_status"] = "CENTER"
                obj["position_text"] = "Строго по центру путей 🚨"
            elif cx < -0.35:
                obj["position_status"] = "LEFT_EDGE"
                obj["position_text"] = "Касается левой кромки ⚠️"
            else:
                obj["position_status"] = "RIGHT_EDGE"
                obj["position_text"] = "Касается правой кромки ⚠️"
                
            # Расчет морфологии (формы) объекта (как в оригинале)
            if height > width and height > depth:
                obj["shape_text"] = "Человек / Вертикальная конструкция"
            elif height < 0.40 and (width > 0.8 or depth > 0.8):
                obj["shape_text"] = "Плоский предмет / Настил"
            else:
                obj["shape_text"] = "Объемная коробка / Блок"
                
            obj["distance_m"] = round(distance, 1)
            confirmed_objects.append(obj)
            
    # 🕵️‍♂️ CHECKPOINT 1: ОРИГИНАЛЬНЫЙ ВЫХОД ИЗ ВАЛИДАТОРА ОБЪЕКТОВ
    print(f"\n🔍 [ИИ-ЯДРО ➔ CHECKPOINT 1]: Расчет validate_and_filter_objects\n"
          f"     Всего валидных преград: {len(confirmed_objects)} шт.", flush=True)
    for idx, c_obj in enumerate(confirmed_objects):
        print(f"     -> СУЩНОСТЬ #{idx} | Дистанция distance_m: {c_obj.get('distance_m')} м | "
              f"Центр ИИ [X, Y, Z]: {[round(float(v), 3) for v in c_obj['center']]}", flush=True)
              
    return confirmed_objects

# def validate_and_filter_objects(raw_objects, past_tracks=None):
#     """Кадровая валидация 3D-боксов с молчаливым обогащением признаков морфологии."""
#     confirmed_objects = []
#     raw_clusters_count = len(raw_objects)
#     noise_multiplier = min(2.5, 1.0 + max(0.0, (raw_clusters_count - 20) / 50.0))

#     for obj in raw_objects:
#         cx, cy, cz = obj["center"]
#         width, height, depth = obj["dimensions"]
#         points_count = obj["points_count"]
#         intensity = obj["intensity"]
#         obj_volume = width * height * depth
#         distance = abs(cz)
        
#         if intensity > config.MAX_TARGET_INTENSITY:
#             continue

#         distance_factor = (0.4 + 0.6 * (distance / 15.0)) if distance < 15.0 else max(0.3, 1.0 - 0.003 * (distance - 15.0))
#         size_factor = max(0.3, height / 1.5) if height < 1.5 else 1.0

#         if distance < config.DISTANCE_THRESHOLD_Z:
#             min_volume = config.NEAR_MIN_VOLUME * noise_multiplier
#             min_points = max(config.FAR_MIN_POINTS, int(config.NEAR_MIN_POINTS * noise_multiplier * distance_factor * size_factor))
#             min_dimension = config.NEAR_MIN_DIMENSION
#         else:
#             min_volume = config.FAR_MIN_VOLUME * (noise_multiplier * 1.2)
#             min_points = max(config.FAR_MIN_POINTS, int(config.FAR_MIN_POINTS * noise_multiplier * distance_factor * size_factor))
#             min_dimension = config.FAR_MIN_DIMENSION

#         if obj_volume >= min_volume and points_count >= min_points and max(width, height, depth) >= min_dimension:
#             # Расчет аналитики положения объекта
#             deviation_x = abs(cx)
#             if deviation_x <= 0.35:
#                 obj["position_status"] = "CENTER"
#                 obj["position_text"] = "Строго по центру путей 🚨"
#             elif cx < -0.35:
#                 obj["position_status"] = "LEFT_EDGE"
#                 obj["position_text"] = "Касается левой кромки ⚠️"
#             else:
#                 obj["position_status"] = "RIGHT_EDGE"
#                 obj["position_text"] = "Касается правой кромки ⚠️"
                
#             # Расчет морфологии (формы) объекта
#             if height > width and height > depth:
#                 obj["shape_text"] = "Человек / Вертикальная конструкция"
#             elif height < 0.40 and (width > 0.8 or depth > 0.8):
#                 obj["shape_text"] = "Плоский предмет / Настил"
#             else:
#                 obj["shape_text"] = "Объемная коробка / Блок"
                
#             obj["distance_m"] = round(distance, 1)
#             confirmed_objects.append(obj)
#         # 🕵️‍♂️ CHECKPOINT 1: ВЫХОД ИЗ ВАЛИДАТОРА ОБЪЕКТОВ
#     print(f"\n🔍 [ИИ-ЯДРО ➔ CHECKPOINT 1]: Расчет validate_and_filter_objects\n"
#           f"     Всего валидных преград: {len(confirmed_objects)} шт.", flush=True)
#     for idx, c_obj in enumerate(confirmed_objects):
#         print(f"     -> СУЩНОСТЬ #{idx} | Дистанция distance_m: {c_obj.get('distance_m')} м | "
#               f"Центр ИИ [X, Y, Z]: {[round(float(v), 3) for v in c_obj['center']]}", flush=True)

#     return confirmed_objects

# Было: def process_point_cloud(file_path, tracker, is_open_space=False):
# Стало: передаем чистый физический шаг поезда от одометрии
def process_point_cloud(file_path, tracker, train_step_z=0.0, is_open_space=False):
    """
    Сквозной ИИ-конвейер детекции препятствий (Real-Time Humble Production).
    Защищен от ложного бампера вагона через 'Режим Гашения Стоянки' с такт-инициализацией.
    """
    if not file_path.endswith('.bin'):
        return []
    try:
        # Нативный парсер бинарного облака точек (Инвариантный мост осей)
        raw_points = np.fromfile(file_path, dtype=np.float32).reshape(-1, config.MATRIX_WIDTH_CHANNELS).copy()
    except Exception as e:
        print(f" [CRITICAL]: Ошибка разбора .bin файла: {e}", flush=True)
        return []

    if len(raw_points) == 0:
        return []

    # 🟢 МОНОЛИТНЫЙ СИНХРОННЫЙ МОСТ ОСЕЙ v12:
    # Сопоставляем полярность и индексы одометрии и детектора преград
    points = np.zeros_like(raw_points)
    points[:, 0] = raw_points[:, 0]  # Столбец 0 -> Внутренний X (Ширина путей)
    points[:, 1] = raw_points[:, 2]  # Столбец 2 -> Внутренний Y (Высота над рельсами)
    
    # Принудительно уводим продольный ход в МИНУС, чтобы полностью совпасть с одометрией стен
    points[:, 2] = -np.abs(raw_points[:, 1])  
    points[:, 3] = raw_points[:, 3]  # Интенсивность

    # 1. Вызов селективного балансировщика нагрузки ( Load Balancing )
    points = filter_selective_load_balancing_v2(points)

    # 2. Адаптивное воксельное сжатие 3D-сетки
    points = voxel_downsample_adaptive(points)
    
    # 3. Динамическая селф-калибровка мертвой зоны кабины (уже переведена на abs)
    live_min_z = tracker._calibrate_ego_vehicle_cabin(points)
    
    # 4. Сегментация пола RANSAC и фильтр криволинейной колеи безопасности вагона
    points_above_floor, floor_points = extract_floor_with_ransac(points)
    points_inside_gauge = filter_gauge_with_track_bending(
        points_above_floor, floor_points, live_min_z, is_open_space=is_open_space
    )

    # =====================================================================
    # 🟢 ИИ-ФИКС: ПРОМЫШЛЕННЫЙ РЕЖИМ ГАШЕНИЯ СТОЯНКИ С ТАКТ-ИНИЦИАЛИЗАЦИЕЙ
    # =====================================================================
    # Проверяем, что фаза холодного старта завершена (кабина успешно откалибрована)
    # и шаг одометрии поезда близок к нулю (поезд гарантированно стоит на месте)
    if tracker.is_cabin_calibrated and abs(train_step_z) <= 0.01:
        if len(points_inside_gauge) > 0:
            # Математически жестко стираем бампер вагона и сцепку на расстоянии 2.62 метра.
            # Оставляем строго те точки внутри колеи, которые находятся ДАЛЬШЕ 5.5 метров вперед (-5.5)
            stationary_safe_mask = points_inside_gauge[:, 2] < -5.5
            points_inside_gauge = points_inside_gauge[stationary_safe_mask]
    # =====================================================================
    # =====================================================================
    # 🟢 ЖЕСТКИЙ ФИЛЬТР МЕРТВОЙ ЗОНЫ ВАГОНА (КОНТРАКТ CONFIG.PY)
    # =====================================================================
    # Полностью ликвидирует бампер состава (2.6м) на стоянке и в движении.
    # Так как ось Z в детекторе v2 отрицательная, мы оставляем строго те точки,
    # которые находятся ДАЛЬШЕ лимита config.LIDAR_MIN_Z (3.5м) вперед от лидара.
    if len(points_inside_gauge) > 0:
        # Условие: точки должны быть меньше -3.5 метров (т.е. -4, -10, -50м вперед)
        # Всё, что ближе (-2.60м, -2.73м), отсекается со скоростью NumPy конвейера!
        cabin_clear_mask = points_inside_gauge[:, 2] < -config.LIDAR_MIN_Z
        points_inside_gauge = points_inside_gauge[cabin_clear_mask]
    # =====================================================================

    # Твой родной принт диагностики — теперь тут гарантированно пойдут чистые данные!
    print(f" 📊 [Кадр: {os.path.basename(file_path)}] Точек в колее: {len(points_inside_gauge)} | Точек пола: {len(floor_points)}", flush=True)
    
    # Кластеризация пространственных аномалий внутри очищенной колеи
    raw_detections = find_obstacles_adaptive_density(points_inside_gauge, floor_points)
    # 🛡️ ГЕОМЕТРИЧЕСКИЙ ФИКС: Схлопываем нарезку DBSCAN по оси Z до прохождения валидаторов и трекера
    clean_raw_detections = merge_and_filter_raw_obstacles(raw_detections, distance_gate_z=3.5, width_gate_x=1.2)
    # Пропускаем через валидатор уже чистые макро-объекты
    confirmed_obstacles = validate_and_filter_objects(clean_raw_detections, tracker.past_tracks)
    
    # 🟢 КРИТИЧЕСКИЙ ФИКС: Вызываем трекер СТРОГО ОДИН раз за кадр, передавая шаг одометрии!
    final_safe_objects = tracker.track_and_filter_ghosts(confirmed_obstacles, train_step_z=train_step_z)
    
    # 🟢 ГЕОМЕТРИЧЕСКИЙ ВОССТАНОВИТЕЛЬ ТОЧЕК ДЛЯ ФРОНТЕНДА THREE.JS
    for obj in final_safe_objects:
        cx, cy, cz = obj["center"]
        w, h, d = obj["dimensions"]
        
        lux = 0.10  # Небольшой люфт-запас в 10 см
        in_box_mask = (
            (points_inside_gauge[:, 0] >= (cx - w/2 - lux)) & (points_inside_gauge[:, 0] <= (cx + w/2 + lux)) &
            (points_inside_gauge[:, 1] >= (cy - h/2 - lux)) & (points_inside_gauge[:, 1] <= (cy + h/2 + lux)) &
            (points_inside_gauge[:, 2] >= (cz - d/2 - lux)) & (points_inside_gauge[:, 2] <= (cz + d/2 + lux))
        )
        box_points = points_inside_gauge[in_box_mask]
        
        # Сжимаем плотность в 3 раза, чтобы WebSocket-пакеты летели мгновенно
        downsampled_box_pts = box_points[::3]
        obj["raw_points"] = downsampled_box_pts[:, :3].astype(float).tolist()
        
    return final_safe_objects


# def process_point_cloud(file_path, tracker_engine):
#     """Сквозной ИИ-конвейер с сохранением 100% плотности точек внутри зоны контроля."""
#     if not file_path.endswith('.bin'):
#         return []
#     try:
#         raw_points = np.fromfile(file_path, dtype=np.float32).reshape(-1, config.MATRIX_WIDTH_CHANNELS)
#             # 🔍 ИИ-ДИАГНОСТИКА ХЕШ-СТРУКТУРЫ КАДРА
#         print(f"\n🧠 [КОНКУРСНЫЙ АУДИТ СТРУКТУРЫ]: {os.path.basename(file_path)}", flush=True)
#         print(f"   Сырых строк в файле: {len(raw_points)}", flush=True)
#         if len(raw_points) > 0:
#             print(f"      ↳ Столбец 0: MIN={raw_points[:, 0].min():+.2f}, MAX={raw_points[:, 0].max():+.2f}, MEDIAN={np.median(raw_points[:, 0]):+.2f}, VAR={np.var(raw_points[:, 0]):.2f}", flush=True)
#             print(f"      ↳ Столбец 1: MIN={raw_points[:, 1].min():+.2f}, MAX={raw_points[:, 1].max():+.2f}, MEDIAN={np.median(raw_points[:, 1]):+.2f}, VAR={np.var(raw_points[:, 1]):.2f}", flush=True)
#             print(f"      ↳ Столбец 2: MIN={raw_points[:, 2].min():+.2f}, MAX={raw_points[:, 2].max():+.2f}, MEDIAN={np.median(raw_points[:, 2]):+.2f}, VAR={np.var(raw_points[:, 2]):.2f}", flush=True)

#     except Exception as e:
#         print(f" [CRITICAL]: Ошибка разбора .bin файла: {e}", flush=True)
#         return []

#     if len(raw_points) == 0:
#         return []

#     # Твой эталонный фиксированный мост координат под контракт одометрии
#     points = np.zeros_like(raw_points)
#     points[:, 0] = raw_points[:, 1]  # X (Ширина путей)
#     points[:, 1] = raw_points[:, 2]  # Y (Высота над рельсами)
#     points[:, 2] = np.abs(raw_points[:, 0])  # Z (Дальность вперед)
#     points[:, 3] = raw_points[:, 3]  # Интенсивность

#     # 🟢 УМНЫЙ СКОРОСТНОЙ ЩИТ: Защищаем плотность в колее и нависающем габарите
#     # Выделяем маску потенциальной зоны контроля поезда и проводов
#     gauge_corridor_mask = (
#         (points[:, 0] >= -2.5) & (points[:, 0] <= 2.5) &
#         (points[:, 1] >= config.MIN_Y) & (points[:, 1] < config.MAX_Y)
#     )
    
#     points_inside_corridor = points[gauge_corridor_mask]
#     points_outside_corridor = points[~gauge_corridor_mask]

#     raw_count = len(points_outside_corridor)
#     # Прореживаем только фоновые точки стен туннеля, разгружая CPU
#     if raw_count > config.ADAPTIVE_TARGET_POINTS:
#         skip_step = raw_count // config.ADAPTIVE_TARGET_POINTS
#         points_outside_corridor = points_outside_corridor[::skip_step]

#     # Склеиваем облако обратно: точки в колее заходят со 100% плотностью!
#     points = np.vstack([points_inside_corridor, points_outside_corridor])

#     # Дальнейший стандартный цикл конвейера
#     points = voxel_downsample_adaptive(points)
#     live_min_z = tracker_engine._calibrate_ego_vehicle_cabin(points)
#     points_above_floor, floor_points = extract_floor_with_ransac(points)
#     points_inside_gauge = filter_gauge_with_track_bending(points_above_floor, floor_points, live_min_z)
    
#     print(f" 📊 [Кадр: {os.path.basename(file_path)}] Точек в колее: {len(points_inside_gauge)} | Точек пола: {len(floor_points)} | Мертвая зона Z: {live_min_z:.2f}м", flush=True)

#     raw_detections = find_obstacles_adaptive_density(points_inside_gauge, floor_points)
    
#     print(f"    ⚙️ [АНАЛИЗ DBSCAN]: Входных точек={len(points_inside_gauge)} | Найдено сырых кластеров: {len(raw_detections)}", flush=True)
    
#     confirmed_obstacles = validate_and_filter_objects(raw_detections, tracker_engine.past_tracks)
#     final_safe_objects = tracker_engine.track_and_filter_ghosts(confirmed_obstacles)
#     return final_safe_objects


# def process_point_cloud(file_path, tracker):
#     if not file_path.endswith('.bin'):
#         return []
#     try:
#         raw_points = np.fromfile(file_path, dtype=np.float32).reshape(-1, config.MATRIX_WIDTH_CHANNELS)
#     except Exception as e:
#         print(f" [CRITICAL]: Ошибка разбора .bin файла: {e}", flush=True)
#         return []

#     if len(raw_points) == 0:
#         return []

#     # 🟢 ЭТАЛОННЫЙ МОСТ КООРДИНАТ ИЗ ВЕРСИИ v1 (ЖЕСТКАЯ СТАБИЛИЗАЦИЯ)
#     # Отказываемся от капризного np.var дисперсии, который путает стены туннеля метро!
#     points = np.zeros_like(raw_points)
#     points[:, 0] = raw_points[:, 1]  # Оригинальный Y становится внутренним X (Ширина путей)
#     points[:, 1] = raw_points[:, 2]  # Оригинальный Z становится внутренним Y (Высота над рельсами)
#     points[:, 2] = np.abs(raw_points[:, 0])  # Оригинальный X становится внутренним Z (Дальность вперед)
#     points[:, 3] = raw_points[:, 3]  # Интенсивность

#     raw_count = len(points)
#     if raw_count > config.ADAPTIVE_TARGET_POINTS:
#         points = points[::(raw_count // config.ADAPTIVE_TARGET_POINTS)]

#     points = voxel_downsample_adaptive(points)
#     live_min_z = tracker._calibrate_ego_vehicle_cabin(points)
#     points_above_floor, floor_points = extract_floor_with_ransac(points)
#     points_inside_gauge = filter_gauge_with_track_bending(points_above_floor, floor_points, live_min_z)
    
#     raw_detections = find_obstacles_adaptive_density(points_inside_gauge, floor_points)
#     confirmed_obstacles = validate_and_filter_objects(raw_detections, tracker.past_tracks)
    
#     final_safe_objects = tracker.track_and_filter_ghosts(confirmed_obstacles)
#     return final_safe_objects



# """
# ⚙️ AUTONOMOUS SUBWAY VISION CORE v2 (Unified Detection Pipeline)
# Компонент детекции препятствий с автоматическим определением осей лидара по дисперсии.
# # """

# import os
# import csv
# import gc
# import numpy as np
# from sklearn.cluster import DBSCAN

# import config
# from tracker_v2 import LidarObstacleTrackerV2

# GLOBAL_STATE_TRACKER = LidarObstacleTrackerV2()

# def voxel_downsample_adaptive(points):
#     if len(points) < 100:
#         return points
#     # Столбец 2 — теперь гарантированно продольная дальность туннеля (Z)
#     z_coords = points[:, 2] 
#     dynamic_size = config.VOXEL_SIZE_BASE + (z_coords * config.VOXEL_COEF_Z)
#     indexed_coords = np.floor(points / dynamic_size[:, np.newaxis]).astype(np.int32)
#     _, indices = np.unique(indexed_coords, axis=0, return_index=True)
#     return points[indices]

# def extract_floor_with_ransac(points):
#     """Сегментация рельсового полотна вокруг нулевой отметки высоты (Столбец 1: Y)"""
#     if len(points) < 10:
#         mask = points[:, 1] > 0.1
#         return points[mask], points[~mask]
#     floor_mask = (points[:, 1] >= -0.2) & (points[:, 1] <= 0.15)
#     return points[~floor_mask], points[floor_mask]
# def filter_gauge_with_track_bending(all_points, floor_points, live_min_z):
#     """Криволинейный створ ворот безопасности. Оперирует чистыми осями: 0=X, 1=Y, 2=Z."""
#     if len(all_points) == 0:
#         return all_points

#     z_all = all_points[:, 2]
#     predicted_x_center = np.zeros_like(z_all)
#     STEP_Z = 5.0
    
#     # Посекторный расчет медианы рельсового полотна по оси ширины X (0)
#     for start_z in np.arange(live_min_z, config.MAX_Z, STEP_Z):
#         end_z = start_z + STEP_Z
#         mask_floor_sector = (
#             (floor_points[:, 2] >= start_z) & (floor_points[:, 2] < end_z) &
#             (floor_points[:, 0] >= -2.0) & (floor_points[:, 0] <= 2.0)
#         )
#         floor_sector = floor_points[mask_floor_sector]
#         if len(floor_sector) > 15:
#             predicted_x_center[(z_all >= start_z) & (z_all < end_z)] = np.median(floor_sector[:, 0])

#     in_curved_gauge_mask = (
#         (all_points[:, 0] >= (predicted_x_center - config.TRAIN_HALF_WIDTH)) &
#         (all_points[:, 0] <= (predicted_x_center + config.TRAIN_HALF_WIDTH)) &
#         (all_points[:, 1] >= 0.15) & (all_points[:, 1] < 1.30) &
#         (all_points[:, 2] >= live_min_z) & (all_points[:, 2] <= config.MAX_Z)
#     )
#     return all_points[in_curved_gauge_mask]

# def find_obstacles_adaptive_density(points_inside_gauge, floor_points):
#     raw_detected_objects = []
#     if len(points_inside_gauge) < 2:
#         return raw_detected_objects
        
#     dynamic_start_z = 30.0
#     if len(floor_points) > 100:
#         bins = np.arange(10, int(config.MAX_Z) + 10, 5)
#         counts, edges = np.histogram(floor_points[:, 2], bins=bins)
#         low_density_indices = np.where(counts < 150)
#         if len(low_density_indices) > 0 and len(low_density_indices[0]) > 0:
#             dynamic_start_z = float(edges[low_density_indices[0][0]])
#         dynamic_start_z = max(15.0, min(dynamic_start_z, 180.0))

#     scaled_points = points_inside_gauge.copy()
#     z_coords = scaled_points[:, 2]
#     scaling_factors = np.where(z_coords > dynamic_start_z, dynamic_start_z / z_coords, 1.0)
#     scaled_points[:, 0] *= scaling_factors
#     scaled_points[:, 1] *= scaling_factors
    
#     db = DBSCAN(eps=config.DBSCAN_OBSTACLE_EPS, min_samples=config.DBSCAN_OBSTACLE_MIN_SAMPLES, algorithm='kd_tree', n_jobs=-1).fit(scaled_points)
#     labels = db.labels_
    
#     for label in set(labels):
#         if label == -1:
#             continue
#         cluster_indices = np.where(labels == label)
#         original_cluster_points = points_inside_gauge[cluster_indices]
        
#         center = np.mean(original_cluster_points, axis=0)
#         width = float(np.max(original_cluster_points[:, 0]) - np.min(original_cluster_points[:, 0]))
#         height = float(np.max(original_cluster_points[:, 1]) - np.min(original_cluster_points[:, 1]))
#         depth = float(np.max(original_cluster_points[:, 2]) - np.min(original_cluster_points[:, 2]))
#         mean_intensity = float(np.mean(original_cluster_points[:, 3]))
        
#         # Находим честные геометрические координаты центра кластера
#         cx = float(center[0])
#         cy = float(center[1])
#         cz = float(center[2])
        
#         # === ИСПРАВЛЕННАЯ СБОРКА АНАТОМИИ ПРЕПЯТСТВИЯ ===
#         raw_detected_objects.append({
#             "class_id": 1,
#             "confidence": 1.0,
#             "center": [cx, cy, cz], # 🟢 ТЕПЕРЬ ОСИ РАЗДЕЛЕНЫ ЧЕСТНО И БЕЗ КУБИЗМА
#             "dimensions": [float(width), float(height), float(depth)], # Честные физические размеры [Ширина, Высота, Длина]
#             "points_count": len(original_cluster_points),
#             "intensity": mean_intensity
#         })

#     return raw_detected_objects

# def validate_and_filter_objects(raw_objects, past_tracks=None):
#     confirmed_objects = []
#     raw_clusters_count = len(raw_objects)
#     noise_multiplier = min(2.5, 1.0 + max(0.0, (raw_clusters_count - 20) / 50.0))

#     for obj in raw_objects:
#         cx, cy, cz = obj["center"]
#         width, height, depth = obj["dimensions"]
#         points_count = obj["points_count"]
#         intensity = obj["intensity"]
#         obj_volume = width * height * depth
#         distance = abs(cz)
        
#         if intensity > config.MAX_TARGET_INTENSITY:
#             continue

#         distance_factor = (0.4 + 0.6 * (distance / 15.0)) if distance < 15.0 else max(0.3, 1.0 - 0.003 * (distance - 15.0))
#         size_factor = max(0.3, height / 1.5) if height < 1.5 else 1.0

#         if distance < config.DISTANCE_THRESHOLD_Z:
#             min_volume = config.NEAR_MIN_VOLUME * noise_multiplier
#             min_points = max(config.FAR_MIN_POINTS, int(config.NEAR_MIN_POINTS * noise_multiplier * distance_factor * size_factor))
#             min_dimension = config.NEAR_MIN_DIMENSION
#         else:
#             min_volume = config.FAR_MIN_VOLUME * (noise_multiplier * 1.2)
#             min_points = max(config.FAR_MIN_POINTS, int(config.FAR_MIN_POINTS * noise_multiplier * distance_factor * size_factor))
#             min_dimension = config.FAR_MIN_DIMENSION

#         if obj_volume >= min_volume and points_count >= min_points and max(width, height, depth) >= min_dimension:
#             confirmed_objects.append(obj)
            
#     return confirmed_objects

# def process_point_cloud(file_path):
#     """Сквозной ИИ-конвейер детекции с инвариантным дисперсионным разворотом осей."""
#     if not file_path.endswith('.bin'):
#         return []
#     try:
#         raw_points = np.fromfile(file_path, dtype=np.float32).reshape(-1, config.MATRIX_WIDTH_CHANNELS)
#     except Exception as e:
#         print(f" [CRITICAL]: Ошибка разбора .bin файла: {e}", flush=True)
#         return []

#     if len(raw_points) == 0:
#         return []

#     spatial_points = raw_points[:, :3]
#     intensity = raw_points[:, 3]

#     # 🔥 [ИНВАРИАНТНЫЙ ИИ-ДЕТЕКТОР ОСЕЙ ПО ДИСПЕРСИИ]:
#     variances = np.var(spatial_points, axis=0)
    
#     # 1. Максимальный разброс точек — продольный ход состава (Z)
#     forward_axis_idx = np.argmax(variances)
#     # 2. Минимальный разброс — лоток и свод тоннеля (Высота: Y)
#     height_axis_idx = np.argmin(variances)
#     # 3. Средний разброс — ширина колеи и стен (Ширина: X)
#     width_axis_idx = 3 - (forward_axis_idx + height_axis_idx)

#     # Пересобираем облако в жесткий стандарт одометрии cos_processor_v11
#     points = np.zeros_like(raw_points)
#     points[:, 0] = spatial_points[:, width_axis_idx]
#     points[:, 1] = spatial_points[:, height_axis_idx]
#     points[:, 2] = np.abs(spatial_points[:, forward_axis_idx]) 
#     points[:, 3] = intensity

#     # Балансировщик нагрузки (Load Balancing)
#     raw_count = len(points)
#     if raw_count > config.ADAPTIVE_TARGET_POINTS:
#         points = points[::(raw_count // config.ADAPTIVE_TARGET_POINTS)]

#     points = voxel_downsample_adaptive(points)
#     live_min_z = GLOBAL_STATE_TRACKER._calibrate_ego_vehicle_cabin(points)
#     points_above_floor, floor_points = extract_floor_with_ransac(points)
#     points_inside_gauge = filter_gauge_with_track_bending(points_above_floor, floor_points, live_min_z)
    
#     raw_detections = find_obstacles_adaptive_density(points_inside_gauge, floor_points)
#     confirmed_obstacles = validate_and_filter_objects(raw_detections, GLOBAL_STATE_TRACKER.past_tracks)
    
#     # Межкадровый трекер убирает ложные кольца тюбингов
#     final_safe_objects = GLOBAL_STATE_TRACKER.track_and_filter_ghosts(confirmed_obstacles)
#     return final_safe_objects

# def main():
#     test_data_dir = "./test_lidar_frames/roundT_doubleT"
#     output_csv_path = "submission.csv"
#     csv_header = ["frame_id", "class_id", "confidence", "x_center", "y_center", "z_center", "width", "height", "depth"]
    
#     if not os.path.exists(test_data_dir):
#         return
        
#     frame_files = sorted([f for f in os.listdir(test_data_dir) if f.endswith('.bin')])
    
#     with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(csv_header)
        
#         for file_name in frame_files:
#             file_path = os.path.join(test_data_dir, file_name)
#             detections = process_point_cloud(file_path)
            
#             for obj in detections:
#                 cx, cy, cz = obj["center"]
#                 w, h, d = obj["dimensions"]
                
#                 # ОБРАТНЫЙ МОСТ ПОД ТЗ ЖЮРИ (REP 103) ДЛЯ SUBMISSION.CSV:
#                 # В ТЗ: X=вперед, Y=влево, Z=вверх
#                 # У нас: 0:X=ширина, 1:Y=высота, 2:Z=дальность
#                 writer.writerow([
#                     file_name, obj["class_id"], round(obj["confidence"], 4),
#                     round(cz, 2), round(cx, 2), round(cy, 2),
#                     round(d, 2),  round(w, 2),  round(h, 2)
#                 ])
#             del detections
#             gc.collect()

# if __name__ == "__main__":
#     main()
