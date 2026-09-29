"""
⚙️ AUTONOMOUS SUBWAY ODOMETRY CORE v12 (Fixed Axes)
Высокоточный компонент облачной одометрии стен туннеля на базе Open3D ICP.
Синхронизирован с монолитным стандартом осей: [0:X_ширина, 1:Y_высота, 2:Z_дальность].
"""

import os
import sys
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import time
from metro_lidar.cold_starter import calculate_initial_velocity_voxels
import metro_lidar.config as config

def calculate_adaptive_fusion_shift(shift_rails, shift_walls, prev_velocity_kmh, rail_points_count, idx, dt=0.1):
    """
    Умный динамический шлюз: если один источник врет или его плотность точек упала 
    ниже критического порога config.RAIL_MIN_POINTS_TRUST — доверие к нему падает до нуля.
    На стартовых кадрах (idx <= 2) отключает проверку по ускорению, чтобы поймать начальную скорость.
    """
    # Вычисляем ожидаемый ход поезда по прошлому такту (инерциальный прогноз)
    expected_shift_z = (prev_velocity_kmh / 3.6) * dt
    
    # 🟢 ЛОГИКА СТАРТА: На первых 3 тактах (0, 1, 2) отключаем капризный фильтр дельты ускорения,
    # чтобы система могла выполнить "холодный зацеп" за реальную стартовую скорость датасета.
    is_starting_phase = bool(idx <= 2)
    
    is_rail_lying = False
    
    # 1. Прямой контроль достоверности рельс по количеству точек (массе)
    if rail_points_count is not None and rail_points_count < config.RAIL_MIN_POINTS_TRUST:
        is_rail_lying = True
        print(f"   📉 [ИИ-ДОВЕРИЕ]: Плотность рельс упала до {rail_points_count} точек (Порог: {config.RAIL_MIN_POINTS_TRUST})! Источник заблокирован.", flush=True)
    
    # 2. Если по точкам всё ок, проверяем рельсы на физическую адекватность сдвига
    elif shift_rails is not None:
        rail_delta = abs(abs(shift_rails) - expected_shift_z)
        
        if is_starting_phase:
            # На старте защищаем только от экстремальных спайков соударения (например, > 1.2м)
            if abs(shift_rails) > config.SPIKE_ABSOLUTE_LIMIT_Z:
                is_rail_lying = True
        else:
            # В стабильном движении включаем жесткий барьер по паспортному ускорению
            if rail_delta > config.MAX_PHYSICAL_ACCEL_Z or abs(shift_rails) > config.SPIKE_ABSOLUTE_LIMIT_Z:
                is_rail_lying = True

    is_wall_lying = False
    has_walls = (shift_walls is not None)
    
    # 3. Проверяем одометрию стен на физическую адекватность
    if has_walls:
        wall_delta = abs(abs(shift_walls) - expected_shift_z)
        if is_starting_phase:
            if abs(shift_walls) > config.SPIKE_ABSOLUTE_LIMIT_Z:
                is_wall_lying = True
        else:
            if wall_delta > config.MAX_PHYSICAL_ACCEL_Z or abs(shift_walls) > config.SPIKE_ABSOLUTE_LIMIT_Z:
                is_wall_lying = True

    # --- ЖЕСТКОЕ РАСПРЕДЕЛЕНИЕ ТРАСТА МЕЖДУ СТЕНАМИ И РЕЛЬСАМИ ---
    
    # Сценарий А: Рельсы врут/ослепли, а стены валидны и стабильны -> 100% контроля СТЕНАМ
    if is_rail_lying and has_walls and not is_wall_lying:
        if abs(shift_walls) <= config.WALL_SIGNIFICANCE_THRESHOLD_Z:
            print(f"   ⚠️ [ИИ-ШЛЮЗ СТАБИЛИЗАЦИЯ]: Рельсы дисквалифицированы, но стены подтверждают СТОЯНКУ (0.0 см).", flush=True)
            return 0.0
        else:
            print(f"   ⚠️ [ИИ-ШЛЮЗ]: Рельсы дисквалифицированы! Скорость удерживают СТЕНЫ ({abs(shift_walls)*100:.1f} см).", flush=True)
            return shift_walls
        
    # Сценарий Б: Стены врут, а рельсы адекватны и плотные -> 100% контроля РЕЛЬСАМ
    if is_wall_lying and shift_rails is not None and not is_rail_lying:
        print(f"   ⚠️ [ИИ-ШЛЮЗ]: Спайк стен ({abs(shift_walls)*100:.1f} см)! Скорость удерживают РЕЛЬСЫ.", flush=True)
        return shift_rails

    # Сценарий В: Кризис (оба источника выдали дичь или рельсы ослепли, а стен нет) -> Спасает инерция
    if is_rail_lying and (not has_walls or is_wall_lying):
        print(f"   🚨 [ИИ-ШЛЮЗ КРИЗИС ЗАЩИТА]: Сбой всех датчиков кадра! Включаю аварийную ИНЕРЦИЮ.", flush=True)
        return expected_shift_z

    # Сценарий Г: Штатный режим движения -> Смешиваем строго по пропорциям из config.py
    if shift_rails is not None and has_walls:
        # Если оба датчика около нуля — фиксируем чистую стоянку
        if abs(shift_walls) <= config.WALL_SIGNIFICANCE_THRESHOLD_Z and abs(shift_rails) <= config.MAX_PHYSICAL_ACCEL_Z:
            return 0.0
        return (shift_rails * config.WEIGHT_RAIL_ODOMETRY) + (shift_walls * config.WEIGHT_WALL_ODOMETRY)
    
    if shift_rails is not None:
        return shift_rails
    return shift_walls if has_walls else 0.0


def compute_adaptive_geometry_gates(anchor_obj, config_mod, expected_train_shift, blind_expansion):
    """Рассчитывает динамические пространственные ворота захвата по осям XY и продольной оси Z."""
    mass = anchor_obj.get("mass", 0)
    is_wall = anchor_obj.get("is_wall", False)
    abs_shift = float(np.abs(expected_train_shift))
    
    if bool(np.greater(mass, config_mod.MASS_CLASS_GIANT)) or is_wall:
        max_xy_gate = config_mod.WALL_XY_GATE_M
        z_min = -config_mod.GATE_Y_MAX_LIMIT_M
        z_max = config_mod.GATE_Y_MAX_LIMIT_M
    elif bool(np.greater(mass, config_mod.MASS_CLASS_MEDIUM)):
        max_xy_gate = 0.35
        z_min = -abs_shift - 0.35 - blind_expansion
        z_max = abs_shift + 0.35 + blind_expansion
    else:
        max_xy_gate = 0.20
        if bool(abs_shift == 0.0):
            z_min = -1.50 - blind_expansion
            z_max = +1.50 + blind_expansion
        else:
            z_min = -abs_shift - config_mod.GATE_Y_STATIONARY_BASE - blind_expansion
            z_max = abs_shift + config_mod.GATE_Y_MOVING_MAX_BASE + blind_expansion
        
    return max_xy_gate, z_min, z_max

def compute_weighted_median(shifts, weights):
    """Универсальная взвешенная медиана для стабильного голосования масс."""
    if not shifts or not weights or len(shifts) == 0:
        return 0.0
        
    shifts_arr = np.array(shifts, dtype=np.float32)
    weights_arr = np.array(weights, dtype=np.float32)
    
    sort_idx = np.argsort(shifts_arr)
    shifts_sorted = shifts_arr[sort_idx]
    weights_sorted = weights_arr[sort_idx]
    
    sum_w = np.sum(weights_sorted)
    if sum_w == 0:
        return float(np.median(shifts_arr))
        
    cum_weights = np.cumsum(weights_sorted)
    cutoff = sum_w / 2.0
    idx = np.searchsorted(cum_weights, cutoff)
    
    if idx < len(shifts_sorted) - 1 and np.isclose(cum_weights[idx], cutoff):
        return float((shifts_sorted[idx] + shifts_sorted[idx + 1]) / 2.0)
        
    return float(shifts_sorted[idx])


class StableLidarOdometryV12:
    """Промышленный движок трекинга опорных элементов ячеек (ЭЯ) стен туннеля."""
    
    def __init__(self, is_stabilizer=False):
        # Телеметрия и память ЦОС стен
        self.prev_velocity_kmh = 0.0
        self.prev_acceleration = 0.0
        self.blind_frames_counter = 0
        self.stationary_accumulator = 0.0  
        self.anchor_map = []
        self.global_id_counter = 0
        
        # 🟢 ДОБАВЛЯЕМ СЮДА ПЕРЕМЕННЫЕ ХРАНИЛИЩА СЫРОГО ОЯР:
        self.rail_raw_anchor = None       # Хранилище сырого опорного PCD путей
        self.rail_ttl = 0                 # Таймер удержания (6 кадров)
        self.rail_accum_z_predicted = 0.0 # Инерциальный накопитель сдвига внутри петли
        # 🟢 АРХИТЕКТУРНЫЙ МАРКЕР ПАРАЛЛЕЛЬНОГО ДВОЙНИКА
        self.is_stabilizer = is_stabilizer
        if not is_stabilizer:
            self.stabilizer_engine = None  # Инициализируется на такте сброса
            self.stabilizer_age = 0

    # def update_odometry_fusion(self, file_path, idx, dt=0.1):
    #     """
    #     Единая точка сборки одометрии v12. 
    #     Инкапсулирует весь конвейер [Рельсы + Стены + ИИ-Шлюз] внутри ядра.
    #     Возвращает: (итоговый_сдвиг_z, rail_passport)
    #     """
    #     # 🔍 ВЫВОДИМ ВХОДНЫЕ ПАРАМЕТРЫ ТАКТА В API:
    #     if idx <= 2:
    #         print(f"🔍 [ВХОД В ЯДРО]: Кадр idx={idx} | Текущая prev_velocity_kmh в памяти: {self.prev_velocity_kmh} км/ч", flush=True)
            
    #     if not os.path.exists(file_path):
    #         return 0.0, None
    #     # =====================================================================
    #     # 🚀 [ИИ-ПЕРЕХВАТ ХОЛОДНОГО СТАРТА V12] 🚀
    #     # =====================================================================
    #     # Если это самый первый кадр сессии, мы находимся в полной слепоте.
    #     # Вызываем внешнюю боевую функцию, передавая ей путь к папке сценария.
    #             # === ВНУТРИ update_odometry_fusion (idx == 0) ===
    #     if idx == 0 and self.prev_velocity_kmh == 0.0:
    #         try:
    #             scene_dir = os.path.dirname(file_path)
    #             start_speed_kmh = calculate_initial_velocity_voxels(scene_dir)
                
    #             # 🔍 ДОБАВЛЯЕМ ЭТОТ ПРИНТ ДЛЯ ПРОВЕРКИ ВЫХОДА ИЗ ФУНКЦИИ:
    #             print(f"\n🔍 [СТАРТЕР ПРОВЕРКА 1]: Функция calculate_initial_velocity_voxels вернула: {start_speed_kmh} км/ч\n", flush=True)
                
    #             if start_speed_kmh > 1.0:
    #                 self.prev_velocity_kmh = start_speed_kmh
    #         except Exception as e:
    #             print(f"   ⚠️ Ошибка холодного старта: {e}", flush=True)


    #     # Читаем сырые точки один раз за такт
    #     raw_points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        
    #     # 1. Расчет количества точек в колее рельс для ИИ-шлюза доверия
    #     x_pts, y_pts, z_pts = raw_points[:, 2], raw_points[:, 0], raw_points[:, 1]
    #     rail_points_mask = (z_pts >= -63.5) & (z_pts <= -3.5) & \
    #                        (x_pts >= -0.75) & (x_pts <= 0.75) & \
    #                        (y_pts >= -1.85) & (y_pts <= -1.05)
    #     current_rail_points_count = int(np.sum(rail_points_mask))

    #     # 2. Вызываем рельсовый одометр (ОЯР)
    #     shift_z_rails, rail_passport = self.compute_raw_rail_odo_shift(raw_points, dt)

    #     # 3. Вызываем одометр стен туннеля (ЭЯ)
    #     macro_cloud = self.extract_clean_macro_tunnel(file_path)
    #     if macro_cloud is not None:
    #         passports = self.build_passports_via_dbscan(macro_cloud)
    #         calculate_speed_trigger = bool(idx > 0)
    #         shift_z_walls, matches_count, _ = self.associate_and_calculate_shift(passports, dt, calculate_speed=calculate_speed_trigger, idx=idx)

    #     else:
    #         shift_z_walls, matches_count = 0.0, 0

    #             # =====================================================================
    #     # 4. МАТЕМАТИЧЕСКОЕ СЛИЯНИЕ СКОРОСТЕЙ И ФИЛЬТРАЦИЯ СПАЙКОВ
    #     # =====================================================================
    #     if idx > 0:
    #         # Жестко берем модули сдвигов датчиков, чтобы убрать конфликт знаков!
    #         safe_rails = np.abs(shift_z_rails) if shift_z_rails is not None else 0.0
    #         safe_walls = np.abs(shift_z_walls) if shift_z_walls is not None else 0.0
            
    #         final_shift_z = calculate_adaptive_fusion_shift(
    #             safe_rails, safe_walls, self.prev_velocity_kmh, current_rail_points_count, idx, dt
    #         )
            
    #         calculated_speed_kmh = (float(final_shift_z) / dt) * 3.6
    #         if calculated_speed_kmh < 0.2:
    #             calculated_speed_kmh = 0.0
                
    #         self.prev_velocity_kmh = calculated_speed_kmh
    #     else:
    #         # На кадре 0 ИИ-пускач уже прописал скорость, ход равен 0
    #         final_shift_z = 0.0

    #     # 🟢 ВЕРНЫЙ ВЫХОД: Строка ретерна стоит НА ОДНОМ УРОВНЕ с if/else!
    #     # Функция ГАРАНТИРОВАННО вернет кортеж (0.0, rail_passport) на кадре 0!
    #     return float(final_shift_z), rail_passport
    
    def update_odometry_fusion(self, raw_points, idx, dt=0.1):
        """
        Единая точка сборки одометрии v12. 
        Инкапсулирует двухконтурный параллельный контур (Текущий + Стабилизирующий)
        с прецизионной распаковкой кортежей ICP-вычислений.
        """
        # --- ФАЗА ВЫЧИСЛЕНИЙ ДЛЯ ИЗОЛИРОВАННОГО ПЕСОЧНОГО СТАБИЛИЗАТОРА (КОНТУР Б) ---
        if self.is_stabilizer:
            x_pts, y_pts, z_pts = raw_points[:, 2], raw_points[:, 0], raw_points[:, 1]
            rail_points_mask = (z_pts >= -63.5) & (z_pts <= -3.5) & \
                               (x_pts >= -0.75) & (x_pts <= 0.75) & \
                               (y_pts >= -1.85) & (y_pts <= -1.05)
            current_rail_points_count = int(np.sum(rail_points_mask))

            shift_z_rails, rail_passport = self.compute_raw_rail_odo_shift(raw_points, dt)
            
            macro_cloud = self.extract_clean_macro_tunnel_from_memory(raw_points)
            if macro_cloud is not None:
                passports = self.build_passports_via_dbscan(macro_cloud)
                calculate_speed_trigger = bool(idx > 0)
                res_walls = self.associate_and_calculate_shift(passports, dt, calculate_speed=calculate_speed_trigger, idx=idx)
                
                # 🟢 ХИРУРГИЧЕСКИЙ ФИКС 1: Вытаскиваем строго индекс 0 из кортежа одометрии стен
                if isinstance(res_walls, (tuple, list)) and len(res_walls) > 0:
                    shift_z_walls = float(res_walls[0])
                else:
                    shift_z_walls = float(res_walls) if res_walls is not None else 0.0
            else:
                shift_z_walls = 0.0

            if idx > 0:
                final_shift_z = calculate_adaptive_fusion_shift(
                    np.abs(shift_z_rails) if shift_z_rails is not None else 0.0, 
                    np.abs(shift_z_walls), self.prev_velocity_kmh, current_rail_points_count, idx, dt
                )
            else:
                final_shift_z = 0.0
                
            if isinstance(final_shift_z, (tuple, list, np.ndarray)):
                final_shift_z = float(final_shift_z[0]) if len(final_shift_z) > 0 else 0.0
            else:
                final_shift_z = float(final_shift_z)

            # 🟢 ИСПРАВЛЕНО: Расчёт и сохранение собственной скорости Стабилизатора
            if idx > 0:
                # Переводим локальный сдвиг такта в км/ч
                self.prev_velocity_kmh = (final_shift_z / dt) * 3.6
                if self.prev_velocity_kmh < 0.2:
                    self.prev_velocity_kmh = 0.0
            else:
                # На кадре 0 сдвига ещё нет, скорость остаётся дефолтной или с пускового вокселя
                pass

            # 🟢 ДИАГНОСТИЧЕСКИЙ ПРИНТ ДЛЯ АУДИТА СКОРОСТИ СТАБИЛИЗАТОРА
            print(f"   📊 [КОНТУР Б ПАМЯТЬ]: Шаг {final_shift_z*100:.2f} см ➔ Скорость Стабилизатора успешно обновлена: {self.prev_velocity_kmh:.2f} км/ч", flush=True)
                
            return float(final_shift_z), rail_passport


        # === ⚔️ ФАЗА АРБИТРАЖА ДЛЯ ТЕКУЩЕГО (ОСНОВНОГО) КОНТУРА А ===
        self.stabilizer_age += 1
        
        if self.stabilizer_age % 20 == 0 or self.stabilizer_engine is None:
            self.stabilizer_engine = StableLidarOdometryV12(is_stabilizer=True)
            local_stab_idx = 0
        else:
            local_stab_idx = self.stabilizer_age % 20

        # Расчет плотности рельсового полотна для Текущего контура
        x_pts, y_pts, z_pts = raw_points[:, 2], raw_points[:, 0], raw_points[:, 1]
        rail_points_mask = (z_pts >= -63.5) & (z_pts <= -3.5) & \
                           (x_pts >= -0.75) & (x_pts <= 0.75) & \
                           (y_pts >= -1.85) & (y_pts <= -1.05)
        current_rail_points_count = int(np.sum(rail_points_mask))

        # --- РАСЧЕТ КОНТУРА А (ТЕКУЩИЙ) ---
        shift_z_rails_main, rail_passport_main = self.compute_raw_rail_odo_shift(raw_points, dt)
        macro_cloud_main = self.extract_clean_macro_tunnel_from_memory(raw_points)
        if macro_cloud_main is not None:
            passports_main = self.build_passports_via_dbscan(macro_cloud_main)
            calculate_speed_trigger_main = bool(idx > 0)
            res_walls_main = self.associate_and_calculate_shift(
                passports_main, dt, calculate_speed=calculate_speed_trigger_main, idx=idx
            )
            
            # 🟢 ХИРУРГИЧЕСКИЙ ФИКС 2: Вытаскиваем строго индекс 0 из основного кортежа стен
            if isinstance(res_walls_main, (tuple, list)) and len(res_walls_main) > 0:
                shift_z_walls_main = float(res_walls_main[0])
            else:
                shift_z_walls_main = float(res_walls_main) if res_walls_main is not None else 0.0
        else:
            shift_z_walls_main = 0.0

        # --- БЕЗОПАСНАЯ КИНЕМАТИКА КОНТУРА А (ОСНОВНОЙ) ---
        shift_z_physical_main = calculate_adaptive_fusion_shift(
            np.abs(shift_z_rails_main) if shift_z_rails_main is not None else 0.0, 
            np.abs(shift_z_walls_main), self.prev_velocity_kmh, current_rail_points_count, idx, dt
        )
        
        if isinstance(shift_z_physical_main, (tuple, list, np.ndarray)):
            shift_z_physical_main = float(shift_z_physical_main[0]) if len(shift_z_physical_main) > 0 else 0.0
        else:
            shift_z_physical_main = float(shift_z_physical_main)

        # 🟢 ИСПРАВЛЕНО: берем модуль от шага, чтобы инверсия осей стен не превращала скорость в отрицательную!
        calculated_speed_kmh_main = (np.abs(shift_z_physical_main) / dt) * 3.6

        if calculated_speed_kmh_main < 0.2:
            calculated_speed_kmh_main = 0.0
            shift_z_physical_main = 1e-5
        self.prev_velocity_kmh = calculated_speed_kmh_main

        # --- БЕЗОПАСНАЯ КИНЕМАТИКА КОНТУРА Б (СТАБИЛИЗАТОР) ---
        shift_z_physical_stab = 0.0
        try:
            res_stab = self.stabilizer_engine.update_odometry_fusion(raw_points, local_stab_idx, dt)
            if isinstance(res_stab, (tuple, list)) and len(res_stab) > 0:
                shift_z_physical_stab = float(res_stab[0])
            else:
                shift_z_physical_stab = float(res_stab) if res_stab is not None else 0.0
        except Exception:
            shift_z_physical_stab = 0.0

        stab_speed = self.stabilizer_engine.prev_velocity_kmh

        # --- МОДЕРНИЗИРОВАННЫЙ ОРГАН АРБИТРАЖА НА ТАКТЕ 3 (УПРАВЛЕНИЕ ИЗ CONFIG) ---
        if local_stab_idx == 3:
            expected_stab_step = (stab_speed / 3.6) * dt
            stab_step_delta = abs(abs(shift_z_physical_stab) - abs(expected_stab_step))

            # Считаем относительное различие между контурами (защита от деления на ноль +0.1)
            velocity_ratio = (stab_speed + 0.1) / (self.prev_velocity_kmh + 0.1)
            
            # Условие А: Вывод основного контура из комы (уснул около нуля, а Б едет)
            cond_coma = bool(
                self.prev_velocity_kmh <= config.ARBITER_COMA_THRESHOLD_KMH and 
                stab_speed > config.ARBITER_WAKEUP_SPEED_KMH
            )
            
            # Условие Б: Критический разрыв пропорций в движении (дрифт знака или ложные стены)
            cond_ratio_divergence = bool(
                (velocity_ratio >= config.ARBITER_VELOCITY_RATIO_MAX or 
                 velocity_ratio <= config.ARBITER_VELOCITY_RATIO_MIN) and 
                stab_speed > config.ARBITER_MIN_STAB_SPEED_KMH
            )

            # Если шаг Стабилизатора физически адекватен и сработал один из триггеров недоверия
            if (cond_coma or cond_ratio_divergence) and stab_step_delta <= config.MAX_PHYSICAL_ACCEL_Z:
                
                print(f"\n⚡ [ЯДРО ОДОМЕТРИИ АРБИТРАЖ ТРИГГЕР]: Зафиксировано критическое расхождение контуров! "
                      f"Текущая А: {self.prev_velocity_kmh:.2f} км/ч | Стабилизатор Б: {stab_speed:.2f} км/ч. "
                      f"Пропорция разрыва: {velocity_ratio:.1f}х (Порог: {config.ARBITER_VELOCITY_RATIO_MAX}х). "
                      f"Принудительно инжектирую скорость и карту независимой песочницы Б!\n", flush=True)
                
                # Полная сквозная синхронизация Хроно-Карты и памяти скоростей из Контура Б в Контур А
                self.prev_velocity_kmh = stab_speed
                self.anchor_map = self.stabilizer_engine.anchor_map
                self.rail_raw_anchor = self.stabilizer_engine.rail_raw_anchor
                self.rail_accum_z_predicted = self.stabilizer_engine.rail_accum_z_predicted
                
                shift_z_physical_main = shift_z_physical_stab


        return float(shift_z_physical_main), rail_passport_main


    def extract_clean_macro_tunnel_from_memory(self, raw_points):
        """СТАДИЯ 1.0 & 1.5 (ОНЛАЙН В ОЗУ): Адаптированная версия без чтения диска."""
        # Монолитный фиксированный мост осей под контракт одометрии стен
        points = np.zeros((len(raw_points), 4))
        points[:, 0] = raw_points[:, 0]  # X_ширина
        points[:, 1] = raw_points[:, 2]  # Y_высота
        points[:, 2] = raw_points[:, 1] * config.AXIS_POLARITY  # Z_хода поезда
        points[:, 3] = raw_points[:, 3]  # Интенсивность
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        tunnel_mask = (y > config.TUNNEL_Y_MIN) & (y < config.TUNNEL_Y_MAX) & \
                      (z > config.LIDAR_MIN_Z) & (z < config.LIDAR_MAX_Z) & \
                      (y > config.MIN_Y) & (y < config.MAX_Y)
                      
        outside_tracks_mask = (x < config.TRACK_GAUGE_LEFT) | (x > config.TRACK_GAUGE_RIGHT)
        
        clean_mask = tunnel_mask & outside_tracks_mask
        tunnel_pts = points[clean_mask]
        
        if len(tunnel_pts) < 100:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tunnel_pts[:, :3])
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=config.VOXEL_SIZE_BASE)
        
        downsampled_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=config.VOXEL_SIZE_BASE * 2.0, max_nn=30)
        )
        return downsampled_pcd



    def compute_raw_rail_odo_shift(self, raw_points, dt=0.1):
        """
        Высокочувствительный параллельный контур ОЯР по геометрическому принципу удержания.
        Сопровождает облако рельс на всем протяжении продольного коридора видимости.
        """
        # Базис осей строго по замерам из консоли Docker
        points = np.zeros_like(raw_points)
        points[:, 0] = raw_points[:, 2]  # Истинная Ширина X -> Лево/Право
        points[:, 1] = raw_points[:, 0]  # Истинная Высота Y -> Вверх/Вниз
        points[:, 2] = raw_points[:, 1]  # Истинная Дальность Z -> Продольный ход поезда
        points[:, 3] = raw_points[:, 3]
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Настраиваем маски на однозначные математические пазы (60-метровый створ)
        mask_z = np.logical_and(np.greater_equal(z, -63.5), np.less_equal(z, -3.5))
        mask_x = np.logical_and(np.greater_equal(x, -0.75), np.less_equal(x, 0.75))
        mask_y = np.logical_and(np.greater_equal(y, -1.85), np.less_equal(y, -1.05))
        
        rail_mask = np.logical_and(np.logical_and(mask_z, mask_x), mask_y)
        curr_rail_pts = points[rail_mask]
        
        print(f"   [ОЯР ГЕОМЕТРИЯ ПУТИ] В створе шпал найдено сырых точек: {len(curr_rail_pts)}", flush=True)
        
        # Если точки внезапно пропали (глухая слепота лидара)
        if len(curr_rail_pts) < 80:
            # Не сбрасываем якорь сразу, даем ему шанс дождаться восстановления данных
            return None, None

        curr_pcd = o3d.geometry.PointCloud()
        curr_pcd.points = o3d.utility.Vector3dVector(curr_rail_pts[:, :3])
        
        cx = float(np.median(curr_rail_pts[:, 0]))
        cy = float(np.median(curr_rail_pts[:, 1]))
        cz = float(np.median(curr_rail_pts[:, 2]))
        
        # === ФАЗА 1: ЗАХВАТ ИЛИ ПЕРЕФИКСАЦИЯ ЯКОРЯ ===
        # Якорь создается при первом старте или когда предыдущий вылетел из маски дальности
        if self.rail_raw_anchor is None:
            self.rail_raw_anchor = curr_pcd
            self.rail_accum_z_predicted = 0.0  # Сброс одометрического хода для нового ориентира
            
            rail_passport = {
                "id": 888, 
                "centroid": [cx, cy, cz],
                "status": "STATUS_RAIL_FIX", 
                "ttl": 999  # Условный бесконечный TTL для совместимости с логированием
            }
            v_ms = (self.prev_velocity_kmh / 3.6)
            return v_ms * dt, rail_passport

        # === ФАЗА 2: ГЕОМЕТРИЧЕСКОЕ СОПРОВОЖДЕНИЕ (ICP МАТРИЦА СКОЛЬЖЕНИЯ) ===
        v_ms = (self.prev_velocity_kmh / 3.6)
        expected_shift_z = v_ms * dt
        
        # Копим пройденное расстояние с момента фиксации текущего якоря путей
        self.rail_accum_z_predicted += expected_shift_z

        # Восстанавливаем прецизионную матрицу трансформации Open3D 4х4
        init_trans = np.identity(4, dtype=np.float64)
        # Прогнозируем сдвиг: в нативных осях Hesai сближение идет вдоль смещения
        init_trans[0, 3] = -float(self.rail_accum_z_predicted)

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=60
        )
        
        # Сопоставляем текущий кадр с удерживаемым в памяти эталоном рельс
        reg_result = o3d.pipelines.registration.registration_icp(
            curr_pcd, self.rail_raw_anchor, 
            max_correspondence_distance=0.60,  # Немного расширяем радиус поиска для быстрых тактов
            init=init_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=criteria
        )
        
        # Извлекаем истинный сдвиг по оси трансформации ICP
        total_icp_z = -float(reg_result.transformation[0, 3])
        diff_shift_z = total_icp_z - (self.rail_accum_z_predicted - expected_shift_z)
        
        # Рассчитываем положение центроида для отрисовки в Three.js
        cz_rendered = cz - total_icp_z
        
        rail_passport = {
            "id": 888, 
            "centroid": [cx, cy, cz_rendered],
            "status": "STATUS_RAIL_TRACK", 
            "ttl": int(60.0 - self.rail_accum_z_predicted)  # Показываем примерный остаток дистанции в метрах
        }

        # Фильтр аномальных скачков скорости (Spike Veto)
        if bool(np.greater(np.abs(diff_shift_z), 1.2)):
            diff_shift_z = expected_shift_z

        # === ФАЗА 3: УТИЛИЗАЦИЯ И ОТПУСКАНИЕ ПО ГЕОМЕТРИЧЕСКОЙ МАСКЕ ===
        # Защитный рубеж: длина коридора вырезки 60 метров. 
        # Если поезд проехал больше 45-50 метров относительно точки захвата, 
        # значит старый якорь ушел под бампер (вышел из маски). Отпускаем его!
        MAX_TRACKING_DISTANCE_METERS = 48.0
        
        if bool(np.greater_equal(np.abs(self.rail_accum_z_predicted), MAX_TRACKING_DISTANCE_METERS)):
            print(f"   🛤️ [ОЯР ВЫСВОБОЖДЕНИЕ]: Рельсовый якорь прошел {self.rail_accum_z_predicted:.2f}м и отпущен из маски. Захват следующего сектора...", flush=True)
            self.rail_raw_anchor = None  # На следующем кадре сработает ФАЗА 1 и захватит новый чистый кусок пути

        return float(diff_shift_z), rail_passport


    def extract_clean_macro_tunnel(self, filepath):
        """СТАДИЯ 1.0 & 1.5: Сжатие облака вокселями и жесткая вырезка створа стен туннеля."""
        if not os.path.exists(filepath):
            return None
            
        raw_points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        
        # 🟢 ОЧИЩЕННЫЙ ФИКСИРОВАННЫЙ МОСТ ОСЕЙ ПО КОНТРАКТУ ОДОМЕТРИИ (Индекс 1 — это ход Z):
        points = np.zeros((len(raw_points), 4))
        points[:, 0] = raw_points[:, 0]  # RAW_WIDTH_AXIS = 0   -> Твой Столбец 0 (X_ширина)
        points[:, 1] = raw_points[:, 2]  # RAW_HEIGHT_AXIS = 2  -> Твой Столбец 1 (Y_высота)
        points[:, 2] = raw_points[:, 1] * config.AXIS_POLARITY  # RAW_FORWARD_AXIS = 1 -> Твой Столбец 2 (Z_хода)
        points[:, 3] = raw_points[:, 3]  # Интенсивность лазера
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Геометрическая маска туннеля и рельс на основе фиксированных осей
        tunnel_mask = (y > config.TUNNEL_Y_MIN) & (y < config.TUNNEL_Y_MAX) & \
                      (z > config.LIDAR_MIN_Z) & (z < config.LIDAR_MAX_Z) & \
                      (y > config.MIN_Y) & (y < config.MAX_Y)
                      
        outside_tracks_mask = (x < config.TRACK_GAUGE_LEFT) | (x > config.TRACK_GAUGE_RIGHT)
        
        clean_mask = tunnel_mask & outside_tracks_mask
        tunnel_pts = points[clean_mask]
        
        if len(tunnel_pts) < 100:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tunnel_pts[:, :3])
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=config.VOXEL_SIZE_BASE)
        
        downsampled_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=config.VOXEL_SIZE_BASE * 2.0, max_nn=30)
        )
        return downsampled_pcd
    def build_passports_via_dbscan(self, macro_pcd):
        """ЭТАП 2: Сплошной DBSCAN по скелету стен для вычисления паспортов геометрии."""
        if macro_pcd is None:
            return []
            
        xyz = np.asarray(macro_pcd.points)
        raw_normals = np.asarray(macro_pcd.normals) if macro_pcd.has_normals() else None
        
        if len(xyz) < config.DBSCAN_MIN_SAMPLES:
            return []
             
        db = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES, n_jobs=-1).fit(xyz)
        labels = db.labels_
        
        anchors_list = []
        for label in set(labels):
            if label == -1: 
                continue  
                
            cluster_mask = (labels == label)
            pts_xyz = xyz[cluster_mask]
            pts_normals = raw_normals[cluster_mask] if raw_normals is not None else None
            
            if len(pts_xyz) < 4:
                continue
                
            is_wall_structure = bool(len(pts_xyz) >= 200) 
            mass = len(pts_xyz)

            cx = float(np.median(pts_xyz[:, 0]))
            cy = float(np.median(pts_xyz[:, 1]))
            cz = float(np.median(pts_xyz[:, 2]))
            
            cov_matrix = np.cov(pts_xyz, rowvar=False)
            if cov_matrix.ndim == 2:
                eigenvalues, _ = np.linalg.eigh(cov_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]  
                sum_lam = eigenvalues[0] + eigenvalues[1] + eigenvalues[2] + 1e-8
                
                linearity = (eigenvalues[0] - eigenvalues[1]) / sum_lam
                planarity = (eigenvalues[1] - eigenvalues[2]) / sum_lam
                sphericity = eigenvalues[2] / sum_lam
            else:
                linearity, planarity, sphericity = 1.0, 0.0, 0.0
            
            ki_percentages = [0.1, 0.7, 0.2, 0.0, 0.0] 
            mass_weight_factor = 50.0 if is_wall_structure else float(mass)

            anchors_list.append({
                "dbscan_label": int(label),
                "centroid": [cx, cy, cz],
                "mass": mass,
                "mass_weight": mass_weight_factor, 
                "geometry": [linearity, planarity, sphericity],
                "ki": ki_percentages,
                "raw_points": pts_xyz,
                "raw_normals": pts_normals, 
                "is_wall": is_wall_structure
            })
            
        anchors_list.sort(key=lambda x: x["centroid"][2])
        return anchors_list

    def associate_and_calculate_shift(self, curr_pso, dt=0.1, calculate_speed=False, idx=0):

        """ЭТАП 3: Каскадное ICP сопоставление кадра с Единой Хроно-Картой Якорей."""
        active_map = self.anchor_map
        
        if not active_map:
            initialized_map = []
            for c_obj in curr_pso:
                self.global_id_counter += 1
                c_obj["id"] = self.global_id_counter
                c_obj["ttl"] = config.MAX_TTL
                c_obj["total_matches"] = 1
                c_obj["status"] = "STATUS_CANDIDATE"
                c_obj["history_frames"] = 1
                initialized_map.append(c_obj)
            self.anchor_map = initialized_map
            return 0.0, 0, []
            
        shifts, weights = [], []
        matched_count = 0
        comparison_logs = []
        stats = {"gate_z": 0, "gate_xy": 0, "ii_drop": 0}
        
        blind_expansion = float(self.blind_frames_counter * config.GATE_Y_BLIND_STEP_M)
        expected_train_shift = -(self.prev_velocity_kmh / 3.6) * dt  
        
        matched_map_indices = set()
        matched_curr_indices = set()
        
        for m_idx, m_obj in enumerate(active_map):
            best_match_idx, max_total_score = -1, 0.0
            best_ki_sim, best_shape_sim, rough_shift_z = 0.0, 0.0, 0.0
            p_center = m_obj.get("centroid")
            
            for c_idx, c_obj in enumerate(curr_pso):
                if c_idx in matched_curr_indices or m_obj.get("is_wall") != c_obj.get("is_wall"):
                    continue
                    
                c_center = c_obj.get("centroid")
                shift_z = p_center[2] - c_center[2]
                xy_dist = np.linalg.norm(np.array(p_center[:2]) - np.array(c_center[:2]))
                
                max_xy_gate, z_gate_min, z_gate_max = compute_adaptive_geometry_gates(
                    m_obj, config, expected_train_shift, blind_expansion
                )

                if bool(np.less(shift_z, z_gate_min) or np.greater(shift_z, z_gate_max)):
                    stats["gate_z"] += 1
                    continue
                if bool(np.greater(xy_dist, max_xy_gate)):
                    stats["gate_xy"] += 1
                    continue
                    
                ki_sim = float(np.sum(np.minimum(m_obj.get("ki"), c_obj.get("ki"))))
                geo_dist = np.linalg.norm(np.array(m_obj.get("geometry")) - np.array(c_obj.get("geometry")))
                shape_sim = float(np.exp(-geo_dist * 5.0))
                total_score = ki_sim * 0.5 + shape_sim * 0.5
                
                if bool(np.greater(m_obj.get("mass", 0), config.MASS_CLASS_GIANT)):
                    dynamic_match_thresh = config.THRESH_MATCH_GIANT
                elif bool(np.greater(m_obj.get("mass", 0), config.MASS_CLASS_MEDIUM)):
                    dynamic_match_thresh = config.THRESH_MATCH_MEDIUM
                else:
                    dynamic_match_thresh = config.THRESH_MATCH_SHUM

                if bool(np.less(total_score, dynamic_match_thresh)):
                    stats["ii_drop"] += 1
                    continue

                if bool(np.greater(total_score, max_total_score)):
                    max_total_score = total_score
                    best_match_idx = c_idx
                    best_ki_sim, best_shape_sim = ki_sim, shape_sim
                    rough_shift_z = shift_z

            final_thresh = config.THRESH_MATCH_GIANT if m_obj.get("is_wall") else config.THRESH_MATCH_MEDIUM
            if best_match_idx != -1 and bool(np.greater(max_total_score, final_thresh)):
                c_obj = curr_pso[best_match_idx]
                v_meters_per_tact = float((np.abs(self.prev_velocity_kmh) / 3.6) * dt)
                icp_search_dist = max(0.40, (config.ICP_SEARCH_DIST_WALL if m_obj.get("is_wall") else config.ICP_SEARCH_DIST_BASE) + v_meters_per_tact)
                
                p_pcd, c_pcd = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
                p_pcd.points = o3d.utility.Vector3dVector(m_obj["raw_points"])
                c_pcd.points = o3d.utility.Vector3dVector(c_obj["raw_points"])
                
                if m_obj.get("raw_normals") is not None:
                    p_pcd.normals = o3d.utility.Vector3dVector(m_obj["raw_normals"])
                if c_obj.get("raw_normals") is not None:
                    c_pcd.normals = o3d.utility.Vector3dVector(c_obj["raw_normals"])
                
                init_trans = np.identity(4, dtype=np.float64)
                init_trans[2, 3] = float(expected_train_shift) if m_obj.get("is_wall") else -float(rough_shift_z)
                effective_rough_shift = expected_train_shift if m_obj.get("is_wall") else rough_shift_z
                
                algo_estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane() if p_pcd.has_normals() else o3d.pipelines.registration.TransformationEstimationPointToPoint()
                criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=100)

                reg_result = o3d.pipelines.registration.registration_icp(c_pcd, p_pcd, icp_search_dist, init_trans, algo_estimation, criteria)
                final_icp_shift_z = float(reg_result.transformation[2, 3])
                
                if bool(np.greater(np.abs(final_icp_shift_z - effective_rough_shift), config.SPIKE_VETO_THRESH_WALL if m_obj.get("is_wall") else config.SPIKE_VETO_THRESH_CABLE)):
                    continue  
                
                m_obj["history_frames"] += 1
                if bool(np.greater_equal(m_obj["history_frames"], 2)):
                    m_obj["status"] = "STATUS_VALID_ANCHOR"
                
                if m_obj.get("is_wall") == False and bool(np.greater_equal(total_score, 0.95)):
                    anchor_weight = float(m_obj.get("mass", 1)) * total_score * 500.0
                elif m_obj.get("is_wall") == True:
                    anchor_weight = 10.0 * total_score
                else:
                    anchor_weight = float(m_obj.get("mass_weight", 1)) * total_score

                shifts.append(final_icp_shift_z)
                weights.append(anchor_weight)
                matched_count += 1

                m_obj["ttl"] = config.MAX_TTL  
                m_obj["total_matches"] += 1   
                
                m_obj["centroid"] = (np.array(m_obj["centroid"]) * 0.9 + np.array(c_obj["centroid"]) * 0.1).tolist()
                m_obj["raw_points"] = c_obj["raw_points"]
                if c_obj.get("raw_normals") is not None:
                    m_obj["raw_normals"] = c_obj["raw_normals"]
                    
                m_obj["geometry"] = c_obj["geometry"]
                m_obj["ki"] = c_obj["ki"]
                m_obj["mass"] = c_obj["mass"]
                
                matched_map_indices.add(m_idx)
                matched_curr_indices.add(best_match_idx)
                
                cx_map, cy_map, cz_map = m_obj["centroid"]
                comparison_logs.append({
                    "past_id": m_obj["id"], 
                    "curr_id": c_obj["dbscan_label"], 
                    "ki_pct": best_ki_sim * 100.0, 
                    "shape_pct": best_shape_sim * 100.0,
                    "shift_cm": final_icp_shift_z * 100.0, 
                    "status": m_obj["status"]
                })

                # Вывод паспорта центроидов в фиксированных осях одометрии
                wall_side = "⬅️ ЛЕВОЕ КРЫЛО" if cx_map < 0 else "➡️ ПРАВОЕ КРЫЛО"
                print(f"      📍 [ГЕОМЕТРИЯ ЭЯ]: Якорь #{m_obj['id']:03d} | Физический центр: X={cx_map:+.2f}м, Y={cy_map:+.2f}м, Z={cz_map:5.2f}м | Локация: {wall_side}", flush=True)

        # =====================================================================
        # --- ЧАСТЬ 3: ИСПРАВЛЕННЫЙ ФИЗИЧЕСКИЙ ДЕМПФЕР СТОЯНКИ V12.0 (ИНЕРЦИОННЫЙ ЗАМОК) ---
        # =====================================================================
        calculated_shift_z = compute_weighted_median(shifts, weights)
        
        # Разворачиваем триггеры для прецизионной аналитики
        cond_deadband = bool(np.less_equal(np.abs(calculated_shift_z), config.DEADBAND_STATIONARY_M))
        cond_no_matches = bool(matched_count == 0)
        
        # Паспортный порог: если поезд ехал быстрее 3.0 км/ч, потеря стен — это СЛЕПОТА, а не стоянка
        is_moving_prior = bool(self.prev_velocity_kmh > 3.0)

        # 🧠 ЖЕСТКИЙ ИНЕРЦИОННЫЙ ЗАМОК (СТРАТЕГИЯ УДЕРЖАНИЯ):
        # Если ориентиры полностью потерялись на ходу, мы запрещаем демпферу занулять ход.
        # Подставляем прецизионный шаг, рассчитанный строго на основе последней известной скорости.
        if cond_no_matches and is_moving_prior:
            # Сдвиг = (Скорость_км_ч / 3.6) * dt. В осях стен сдвиг идет со знаком минус
            calculated_shift_z = -float((self.prev_velocity_kmh / 3.6) * dt)
            is_train_standing = False
            print(f"   🚨 [ИИ-ЯДРО V12 ИНЕРЦИЯ]: Ориентиры стен потеряны на ходу! Жестко удерживаю скорость: {self.prev_velocity_kmh:.2f} км/ч (Шаг: {calculated_shift_z*100:.1f} см)", flush=True)
        else:
            # В штатном режиме или около нуля — демпфер работает по мертвой зоне лазера
            is_train_standing = cond_deadband or cond_no_matches

        # Аналитический принт для сквозного контроля (idx проброшен в аргументы)
        MIN_VELOCITY_STAND_THRESHOLD = getattr(config, "MIN_VELOCITY_STAND_THRESHOLD", 0.1)
        cond_low_speed = bool(self.prev_velocity_kmh < MIN_VELOCITY_STAND_THRESHOLD)
        
                # --- ДИНАМИЧЕСКИЙ ПРЕФИКС ДЛЯ РАЗДЕЛЕНИЯ ЛОГОВ КОНТУРОВ ---
        if self.is_stabilizer:
            prefix = "⏳ [КОНТУР Б: СТАБИЛИЗАТОР]"
        else:
            prefix = "🎯 [КОНТУР А: ТЕКУЩАЯ ОДОМЕТРИЯ]"

        if idx <= 250:
            print(f"🔍 {prefix} Кадр #{idx:03d} | Шаг такта Z: {calculated_shift_z*100:.2f} см | "
                  f"Прошлая V: {self.prev_velocity_kmh:.2f} км/ч | "
                  f"Триггеры -> Мертвая зона: {cond_deadband}, Нет стен: {cond_no_matches}, Инерция активна: {cond_no_matches and is_moving_prior}", flush=True)

        # Обработка демпфера стоянки (срабатывает только при истинной остановке)
        if is_train_standing:
            self.stationary_accumulator += np.abs(calculated_shift_z)
            if bool(self.stationary_accumulator >= config.START_MOTION_THRESHOLD_M):
                calculated_shift_z = self.stationary_accumulator
                self.stationary_accumulator = 0.0
                print(f"   🚀 [ИСТИННЫЙ СТАРТ V12] Поезд преодолел мертвую зону платформы! Высвобождено: {calculated_shift_z*100:+.2f} см", flush=True)
            else:
                calculated_shift_z = 0.0
        else:
            self.stationary_accumulator = 0.0

        if bool(matched_count == 0 and calculate_speed == True):
            self.blind_frames_counter += 1
        else:
            self.blind_frames_counter = 0

        # =====================================================================
        # 🟢 ХРОНО-УПРАВЛЕНИЕ КАРТОЙ ПАМЯТИ ЯКОРЕЙ (СДВИГАЕМ СТРОГО ИНДЕКС 2)
        # =====================================================================
        updated_map = []
        for m_idx, m_obj in enumerate(active_map):
            if m_idx in matched_map_indices:
                m_obj["ttl"] = config.MAX_TTL
                updated_map.append(m_obj)
            else:
                m_obj["ttl"] -= 1
                if bool(np.greater(m_obj["ttl"], 0)):
                    # Внимание: calculated_shift_z отрицательный в движении, 
                    # вычитание (- calculated_shift_z) корректно смещает центроиды вперед
                    m_obj["centroid"][2] = float(m_obj["centroid"][2] - calculated_shift_z)
                    
                    m_macro_pts = np.array(m_obj["raw_points"])
                    m_macro_pts[:, 2] = m_macro_pts[:, 2] - calculated_shift_z
                    m_obj["raw_points"] = m_macro_pts
                    updated_map.append(m_obj)


        # --- ЧАСТЬ 3.5: ПРОСТРАНСТВЕННОЕ ВЕТО НА КЛОНОВ (ВЫДЕЛЕНИЕ НОВЫХ ЯКОРЕЙ) ---
        if bool(calculate_speed == True or len(active_map) < 100):
            for c_idx, c_obj in enumerate(curr_pso):
                if c_idx not in matched_curr_indices:
                    if bool(c_obj.get("mass", 0) >= config.MASS_CLASS_MEDIUM or c_obj.get("is_wall") == True):
                        c_center = np.array(c_obj["centroid"])
                        is_duplicate_clon = False
                        
                        for existing_obj in updated_map:
                            e_center = np.array(existing_obj["centroid"])
                            spatial_dist = np.linalg.norm(e_center - c_center)
                            if bool(spatial_dist < 1.50):  
                                is_duplicate_clon = True
                                break
                        
                        if not is_duplicate_clon:
                            self.global_id_counter += 1
                            c_obj["id"] = self.global_id_counter
                            c_obj["ttl"] = config.MAX_TTL
                            c_obj["total_matches"] = 1
                            c_obj["status"] = "STATUS_CANDIDATE"
                            c_obj["history_frames"] = 1
                            updated_map.append(c_obj)
                        
        self.anchor_map = updated_map
        
        if bool(len(curr_pso) > 0):
            print(f"   📊 [ВОРОНКА ФИЛЬТРАЦИИ V10]: Из {len(active_map)} Якорей единой карты проверено комбинаций:", flush=True)
            print(f"      -> Срезано по продольному коридору Z: {stats['gate_z']}", flush=True)
            print(f"      -> Срезано по боковому люфту XY:      {stats['gate_xy']}", flush=True)
            print(f"      -> Отклонено по динамическому ИИ-порогу: {stats['ii_drop']}", flush=True)

        return calculated_shift_z, matched_count, comparison_logs

def print_match_matrix_v10(match_logs, step_name):
    print(f"\n📊 [МАТРИЦА СРАВНЕНИЯ ОУиР V10 | {step_name}]")
    print("   Глобальный ID Карты ЕЯ ➔ Кадр (DBSCAN Label) | Сходство КИ | Сходство Тензора | Статус ОУиР | Рассчитанный ΔZ")
    print("   " + "-" * 115)
    for log in match_logs:
        status_flag = "🟢 ЭТАЛОН" if log.get("status") == "STATUS_VALID_ANCHOR" else "⏳ КАРАНТИН"
        print(f"    Якорь #{log.get('past_id'):03d}        ➔ Кластер {log.get('curr_id'):02d} |    {log.get('ki_pct'):.1f}%    |      {log.get('shape_pct'):.1f}%      |  {status_flag}  |   {log['shift_cm']:+.4f} см")

def main():
    # Выбираем целевой сценарий
    scenario = "doubleT_platform"
    scene_dir = f"./test_lidar_frames/{scenario}"
    
    if not os.path.exists(scene_dir):
        print(f"❌ Ошибка: Директория {scene_dir} не найдена!")
        return

    files = sorted([f for f in os.listdir(scene_dir) if f.endswith('.bin')])
    if len(files) < 10:
        print("❌ Ошибка: Мало кадров в папке.")
        return
        
    dt = 0.1  # 10 Гц
    TIME_SLEEP_SECONDS = 0.1  
    
    log_filepath = "./v12_run_dump.txt"
    log_file = open(log_filepath, "w", encoding="utf-8")
    
    def dual_print(text):
        print(text)
        log_file.write(text + "\n")
        log_file.flush()

    dual_print("=" * 115)
    dual_print(f"🚀 [ИНИЦИАЛИЗАЦИЯ ПАРАЛЛЕЛЬНОГО КОНВЕЙЕРА ОДОМЕТРИИ V12.0 | СЦЕНАРИЙ: {scenario}]")
    dual_print("=" * 115)
    
    engine = StableLidarOdometryV12()
    total_distance_m = 0.0
    
    for idx, filename in enumerate(files):
        log_file.write(f"\n--- СТЕК КАДРА {idx:03d} ({filename}) ---\n")
        file_path = os.path.join(scene_dir, filename)
        
        # -----------------------------------------------------------------
        # ВЕТКА 1: ПАРАЛЛЕЛЬНЫЙ СЫРОЙ КОНТУР ОЯР (60 МЕТРОВ ШПАЛ)
        # -----------------------------------------------------------------
         # Читаем сырые точки один раз из файла
        raw_points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        
        # Вычисляем количество точек локально для передачи в ИИ-шлюз
        x_pts, y_pts, z_pts = raw_points[:, 2], raw_points[:, 0], raw_points[:, 1]
        rail_points_mask = (z_pts >= -63.5) & (z_pts <= -3.5) & \
                           (x_pts >= -0.75) & (x_pts <= 0.75) & \
                           (y_pts >= -1.85) & (y_pts <= -1.05)
        current_rail_points_count = int(np.sum(rail_points_mask))

        # Вызываем ОЯР
        shift_z_rails, rail_passport = engine.compute_raw_rail_odo_shift(raw_points, dt)

        
        if rail_passport is not None:
            rx, ry, rz = rail_passport["centroid"]
            log_file.write(f"   🛤️  [РЕЛЬСОВЫЙ ЯКОРЬ ОЯР]: ID #{rail_passport['id']} | X={rx:+.2f}м, Y={ry:+.2f}м, Z={rz:5.2f}м | TTL: {rail_passport['ttl']}\n")

        # -----------------------------------------------------------------
        # ВЕТКА 2: ПАРАЛЛЕЛЬНЫЙ СЖАТЫЙ КОНТУР СТЕН ТУННЕЛЯ
        # -----------------------------------------------------------------
        macro_cloud = engine.extract_clean_macro_tunnel(file_path)
        
        if macro_cloud is not None:
            passports = engine.build_passports_via_dbscan(macro_cloud)
            calculate_speed_trigger = bool(idx > 0)
            shift_z_walls, matches_count, logs = engine.associate_and_calculate_shift(
                passports, dt, calculate_speed=calculate_speed_trigger
            )
        else:
            shift_z_walls, matches_count, logs = 0.0, 0, []

                # -----------------------------------------------------------------
        # МАТЕМАТИЧЕСКОЕ СЛИЯНИЕ СКОРОСТЕЙ (УМНЫЙ ИИ-ШЛЮЗ ЧЕРЕЗ CONFIG)
        # -----------------------------------------------------------------
        if idx > 0:
            # 🟢 ИСПРАВЛЕНО: Передаем ровно 6 параметров в соответствии с сигнатурой функции!
            final_shift_z = calculate_adaptive_fusion_shift(
                shift_z_rails, 
                shift_z_walls, 
                engine.prev_velocity_kmh, 
                current_rail_points_count, 
                idx, 
                dt
            )
        else:
            # Защита самого первого кадра поездки
            if shift_z_rails is not None:
                final_shift_z = shift_z_rails
            else:
                final_shift_z = shift_z_walls if matches_count > 0 else 0.0



        # Переводим результирующий сдвиг в физический ход поезда вперед
        if matches_count > 0 or shift_z_rails is not None:
            shift_z_physical = float(final_shift_z)
            raw_speed_kmh = (shift_z_physical / dt) * 3.6
        else:
            shift_z_physical = 0.0
            raw_speed_kmh = 0.0

        # Ограничитель перегрузок (Motion Prior)
        MAX_PHYSICAL_DV_KMH = 1.5
        if idx > 0 and engine.prev_velocity_kmh > 0.5:
            speed_delta = raw_speed_kmh - engine.prev_velocity_kmh
            if abs(speed_delta) > MAX_PHYSICAL_DV_KMH:
                clamped_delta = np.clip(
                    speed_delta, 
                    config.MAX_DECELERATION_MSS * 3.6 * dt, 
                    config.MAX_ACCELERATION_MSS * 3.6 * dt
                )
                calculated_speed_kmh = engine.prev_velocity_kmh + clamped_delta
                shift_z_physical = (calculated_speed_kmh / 3.6) * dt
            else:
                calculated_speed_kmh = raw_speed_kmh
        else:
            calculated_speed_kmh = raw_speed_kmh

        if calculated_speed_kmh < 0.1:
            calculated_speed_kmh = 0.0
            shift_z_physical = 0.0

        # Сохранение параметров для следующего такта
        engine.prev_velocity_kmh = calculated_speed_kmh
        total_distance_m += shift_z_physical
        
        valid_anchors = sum(1 for x in engine.anchor_map if x["status"] == "STATUS_VALID_ANCHOR")
        rail_status = f" | 🛤️  ОЯР: {rail_passport['ttl']}t" if rail_passport else " | 🛤️  ОЯР: Поиск"
        
        dual_print(f"[КАДР {idx:03d}] ({filename}) | Опора ЭЯ: {valid_anchors}{rail_status} | Шаг ΔZ: {shift_z_physical*100:+.2f} см | Скорость: {calculated_speed_kmh:.2f} км/ч | Путь: {total_distance_m:.3f} м")

        time.sleep(TIME_SLEEP_SECONDS)

    dual_print("\n" + "=" * 115)
    dual_print(f"🏁 [ПАРАЛЛЕЛЬНЫЙ КОНВЕЙЕР УСПЕШНО ЗАВЕРШЕН]:")
    dual_print(f"    👉 Итоговый путь состава: {total_distance_m:.3f} метров")
    dual_print(f"    👉 Итоговая скорость состава: {engine.prev_velocity_kmh:.2f} км/ч")
    dual_print("=" * 115)
    log_file.close()

if __name__ == '__main__':
    main()
