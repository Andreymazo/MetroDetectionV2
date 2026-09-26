import os
import numpy as np
import metro_lidar.config as config

def calculate_initial_velocity_voxels(scene_dir, voxel_size=0.35):
    """
    Боевой 'молчаливый' модуль холодного старта.
    Обсчитывает первые 3 кадра, извлекает многофакторный паспорт шпал 
    и возвращает стартовую скорость для разогрева основного ядра одометрии.
    """
    try:
        # Находим первые 3 кадра в папке
        files = sorted([f for f in os.listdir(scene_dir) if f.endswith('.bin')])[:3]
        if len(files) < 3:
            return 0.0

        frames_voxels_z = []

        for filepath in [os.path.join(scene_dir, f) for f in files]:
            raw_points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)

            # Монолитный мост осей v12
            points = np.zeros_like(raw_points)
            points[:, 0] = raw_points[:, 2]  # X - ширина путей
            points[:, 1] = raw_points[:, 0]  # Y - высота над рельсами
            points[:, 2] = raw_points[:, 1]  # Z - продольный ход

            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            # Извлекаем жесткий ближний створ путей по константам config.py
            rail_mask = (z >= -15.0) & (z <= -3.5) & \
                        (x >= -0.75) & (x <= 0.75) & \
                        (y >= -1.85) & (y <= -1.05)
            
            curr_rail_pts = points[rail_mask]
            if len(curr_rail_pts) < 30:
                frames_voxels_z.append(np.array([]))
                continue

            # Плотностное квантование оси Z
            z_coords = curr_rail_pts[:, 2]
            voxel_indices = np.floor(z_coords / voxel_size).astype(np.int32)
            unique_voxels, unique_counts = np.unique(voxel_indices, return_counts=True)
            
            mean_density = np.mean(unique_counts)
            dense_voxels = unique_voxels[unique_counts > (mean_density * 1.1)]
            
            frame_passports = []
            for voxel_val in dense_voxels:
                pts_in_voxel = curr_rail_pts[voxel_indices == voxel_val]
                if len(pts_in_voxel) < 10:
                    continue

                cz = float(np.median(pts_in_voxel[:, 2]))
                mean_intensity = float(np.mean(pts_in_voxel[:, 3]))
                
                std_x = np.std(pts_in_voxel[:, 0]) + 1e-5
                std_z = np.std(pts_in_voxel[:, 2]) + 1e-5
                shape_ratio = float(std_x / std_z)

                frame_passports.append({
                    "z": abs(cz),
                    "mass": len(pts_in_voxel),
                    "intensity": mean_intensity,
                    "shape": shape_ratio
                })
            frames_voxels_z.append(frame_passports)

        # Многофакторная межкадровая ассоциация треков гребенки
        calculated_shifts = []
        for t in range(1, 3):
            past_passports = frames_voxels_z[t-1]
            curr_passports = frames_voxels_z[t]
            step_shifts = []

            for c_obj in curr_passports:
                best_match_dz = None
                min_feature_dist = float('inf')
                
                for p_obj in past_passports:
                    shift_z = p_obj["z"] - c_obj["z"]
                    if shift_z <= 0.05 or shift_z > 1.45:
                        continue
                        
                    # Множители отклонения признаков с защитой от деления на ноль
                    mass_diff = abs(c_obj["mass"] - p_obj["mass"]) / float(max(1.0, p_obj["mass"]))
                    intensity_diff = abs(c_obj["intensity"] - p_obj["intensity"]) / float(max(1.0, p_obj["intensity"]))
                    shape_diff = abs(c_obj["shape"] - p_obj["shape"]) / float(max(0.001, p_obj["shape"]))
                    
                    # Форма и блеск материала в приоритете
                    feature_dist = (mass_diff * 0.1) + (intensity_diff * 0.45) + (shape_diff * 0.45)
                    
                    if feature_dist < min_feature_dist and feature_dist < 0.35:
                        min_feature_dist = feature_dist
                        best_match_dz = shift_z

                if best_match_dz is not None:
                    step_shifts.append(best_match_dz)

            if step_shifts:
                calculated_shifts.append(float(np.median(step_shifts)))

        # Финальный расчет и верификация консенсуса
        if len(calculated_shifts) == 2:
            shift_01, shift_12 = calculated_shifts[0], calculated_shifts[1]
            if abs(shift_01 - shift_12) < 0.20:
                avg_step_z = (shift_01 + shift_12) / 2.0
                return float((avg_step_z / 0.1) * 3.6)
                
        return 0.0
    except Exception as e:
        import traceback
        print("\n💥 [КРИТИЧЕСКИЙ СБОЙ ИИ-СТАРТЕРА]: Ошибка внутри cold_starter.py!")
        print(f"   ↳ Описание ошибки: {e}")
        print("   ↳ Точный стек вызовов (Traceback):")
        traceback.print_exc()
        print("=" * 110 + "\n")
        return 0.0

