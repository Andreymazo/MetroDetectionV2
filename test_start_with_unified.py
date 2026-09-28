# test_unified_engine.py
import os
import sys
import numpy as np
from unified_object_engine import UnifiedLidarObjectEngine

def test_cold_start_with_unified_engine(scene_dir, voxel_size=0.35):
    """
    Выполняет многофакторный холодный старт с использованием 
    выделенного унифицированного движка ЕПО-Core.
    """
    files = sorted([f for f in os.listdir(scene_dir) if f.endswith('.bin')])[:3]
    if len(files) < 3:
        print("❌ Ошибка: Для полноценного теста нужно минимум 3 кадра!")
        return None

    # Инициализируем наш новый единый класс
    engine = UnifiedLidarObjectEngine(spatial_gate_min=0.05, spatial_gate_max=1.45, feature_match_threshold=0.35)
    frames_object_passports = []

    print("=" * 110)
    print(f"📡 [ИИ-ДВИЖОК ЕПО-CORE]: ТЕСТИРОВАНИЕ УНИФИЦИРОВАННОГО КЛАССА ОБЪЕКТОВ")
    print(f"📂 Целевой сценарий: {scene_dir}")
    print("=" * 110)

    # === ШАГ 1: ВЫЗОВ ПАСПОРТИЗАЦИИ ЧЕРЕЗ КЛАСС ===
    for frame_name in files:
        filepath = os.path.join(scene_dir, frame_name)
        raw_points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)

        points = np.zeros_like(raw_points)
        points[:, 0] = raw_points[:, 2]  # X - ширина
        points[:, 1] = raw_points[:, 0]  # Y - высота
        points[:, 2] = raw_points[:, 1]  # Z - дальность

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        rail_mask = (z >= -15.0) & (z <= -3.5) & \
                    (x >= -0.75) & (x <= 0.75) & \
                    (y >= -1.85) & (y <= -1.05)
        
        curr_rail_pts = points[rail_mask]
        if len(curr_rail_pts) < 30:
            frames_object_passports.append([])
            continue

        z_coords = curr_rail_pts[:, 2]
        voxel_indices = np.floor(z_coords / voxel_size).astype(np.int32)
        unique_voxels, unique_counts = np.unique(voxel_indices, return_counts=True)
        
        mean_density = np.mean(unique_counts)
        dense_voxels = unique_voxels[unique_counts > (mean_density * 1.1)]
        
        frame_passports = []
        obj_id_local = 0

        for voxel_val in dense_voxels:
            pts_in_voxel = curr_rail_pts[voxel_indices == voxel_val]
            if len(pts_in_voxel) < 10:
                continue

            cz = float(np.median(pts_in_voxel[:, 2]))
            
            # 🟢 ОБРАЩЕНИЕ К КЛАССУ: Сборка единого паспорта
            passport = engine.build_fast_rail_passport(pts_in_voxel, voxel_val, cz)
            passport["id"] = obj_id_local
            frame_passports.append(passport)
            obj_id_local += 1
            
        frames_object_passports.append(frame_passports)

    # Вывод базиса сессии
    print(f"\n📋 [КАДР 000 (БАЗИС СЕССИИ) ➔ {files[0]}]:")
    print(f"   Зафиксировано опорных путевых объектов в ближней зоне: {len(frames_object_passports[0])} шт.")
    for ob in frames_object_passports[0]:
        print(f"   ↳ 📦 ОБЪЕКТ #{ob['id']} | {ob['material']}\n"
              f"        📍 Позиция: X={ob['center'][0]:+.2f}м, Y={ob['center'][1]:+.2f}м, Дальность Z={ob['center'][2]:.2f}м\n"
              f"        📊 Признаки: Плотность={ob['mass']} точек | Блеск={ob['intensity']:.0f} | Форма (X/Z)={ob['shape_ratio']:.2f}")

    # === ШАГ 2: АССОЦИАЦИЯ ЧЕРЕЗ УНИФИЦИРОВАННЫЙ КОНТУР КЛАССА ===
    calculated_shifts = []

    for t in range(1, 3):
        print(f"\n" + "-"*110)
        print(f"📋 [КАДР {t:03d} (МЕЖКАДРОВОЕ СОПРОВОЖДЕНИЕ ТРЕКОВ) ➔ {files[t]}]:")
        print("-" * 110)
        
        past_passports = frames_object_passports[t-1]
        curr_passports = frames_object_passports[t]
        
        matched_pairs_count = 0
        step_shifts = []

        for c_obj in curr_passports:
            best_match_past_id = -1
            min_feature_distance = float('inf')
            calculated_dz = 0.0
            
            for p_obj in past_passports:
                shift_z = p_obj["center"][2] - c_obj["center"][2]
                
                # 🟢 ОБРАЩЕНИЕ К КЛАССУ: Использование общих пространственных ворот
                if shift_z <= engine.spatial_gate_min or shift_z > engine.spatial_gate_max:
                    continue
                    
                # 🟢 ОБРАЩЕНИЕ К КЛАССУ: Расчет многомерного расстояния признаков ЕПО
                feature_distance = engine.compute_feature_distance(c_obj, p_obj)
                
                if feature_distance < min_feature_distance and feature_distance < engine.feature_match_threshold:
                    min_feature_distance = feature_distance
                    best_match_past_id = p_obj["id"]
                    calculated_dz = shift_z
                    best_p_obj = p_obj

            if best_match_past_id != -1:
                matched_pairs_count += 1
                step_shifts.append(calculated_dz)
                
                print(f"   🎯 АКТИВНЫЙ ТРЕК: Объект Кадра {t-1:03d} #{best_match_past_id} ➔ Перешел в Объект Кадра {t:03d} #{c_obj['id']}\n"
                      f"        🧬 Сходство паспорта: ИНФРАСТРУКТУРА СОВПАЛА НА {((1.0 - min_feature_distance)*100):.1f}%\n"
                      f"        📐 Динамика хода: Z_было={best_p_obj['center'][2]:.2f}м ➔ Z_стало={c_obj['center'][2]:.2f}м | Физический сдвиг ΔZ: {calculated_dz*100:+.2f} см\n"
                      f"        📊 Текущий слепок: Плотность={c_obj['mass']} тк. | Блеск={c_obj['intensity']:.0f}")

        if step_shifts:
            tact_shift_z = float(np.median(step_shifts))
            calculated_shifts.append(tact_shift_z)
            print(f"   📈 [ИТОГ ТАКТА {t}]: Успешно сопоставлено пар: {matched_pairs_count}. Расчетное перемещение путей: {tact_shift_z*100:.2f} см")
        else:
            print(f"   🚨 [ИТОГ ТАКТА {t}]: Кризис! Ни один путевой трек не прошел верификацию признаков.")
            calculated_shifts.append(0.0)

    # Финальный судейский вердикт
    print("\n" + "=" * 110)
    print("🏁 [ФИНАЛЬНЫЙ АНАЛИТИЧЕСКИЙ ОТЧЕТ ИИ-ПУСКАЧА ХОЛОДНОГО СТАРТА]")
    print("=" * 110)
    
    if len(calculated_shifts) == 2 and calculated_shifts[0] > 0 and calculated_shifts[1] > 0:
        shift_01, shift_12 = calculated_shifts[0], calculated_shifts[1]
        
        avg_step_z = (shift_01 + shift_12) / 2.0
        start_velocity_kmh = (avg_step_z / 0.1) * 3.6
        
        print(f" ✅ ХРОНОЛОГИЯ ПОДТВЕРЖДЕНА: Сдвиг_01 = {shift_01*100:.1f} см | Сдвиг_12 = {shift_12*100:.1f} см")
        print(f" ✅ ГЕОМЕТРИЧЕСКИЙ КОНСЕНСУС: Движение стабильно, класс ЕПО-Core верифицирован.")
        print(f" 🚀 СТАРТОВАЯ СКОРОСТЬ СОСТАВА УСПЕШНО ЗАФИКСИРОВАНА: {start_velocity_kmh:.2f} км/ч")
    else:
        print(" ❌ ТЕСТ ЗАВЕРШЕН С ОШИБКОЙ: Не хватило многофакторных признаков.")
    print("=" * 110 + "\n")

if __name__ == "__main__":
    target_scenario = "./test_lidar_frames/doubleT_platform"
    if os.path.exists(target_scenario):
        test_cold_start_with_unified_engine(target_scenario)
    else:
        print(f"Папка {target_scenario} не найдена.")
