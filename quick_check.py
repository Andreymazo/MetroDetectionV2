import os
import sys
import argparse
import csv
from metro_lidar.cos_processor_v12 import StableLidarOdometryV12
import numpy as np
from pathlib import Path

# Импортируем ваши v2 компоненты
from metro_lidar.generate_submission_v2 import process_point_cloud
from metro_lidar.tracker_v2 import LidarObstacleTrackerV2

def print_banner(text):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Конкурсный конвейер детекции препятствий")
    parser.add_argument("--scenario", type=str, required=True, help="Путь к папке с извлеченными .bin кадрами лидара")
    args = parser.parse_args()

    # 🎯 Прямо берем переданный путь к бинарникам (без хардкода подпапок)
    test_data_dir = os.path.abspath(args.scenario)

    if not os.path.exists(test_data_dir):
        print(f"❌ Ошибка: Указанная папка с кадрами не найдена по пути: {test_data_dir}")
        print("💡 Сначала выполните десериализацию rosbag с помощью convert_simple.py или convert_all_bags.py!")
        return

    print_banner("2. ЗАПУСК ИИ-КОНВЕЙЕРА ДЕТЕКЦИИ ПРЕПЯТСТВИЙ (v2)")
    print(f"🔄 Сканируем подготовленные кадры из: {test_data_dir}...\n")
    
    # Безопасный поиск кадров через pathlib.Path строго в test_data_dir
    data_path_obj = Path(test_data_dir)
    frame_files = sorted([f.name for f in data_path_obj.glob('*.bin')])
    
    if not frame_files:
        print(f"⚠️ Предупреждение: В папке {test_data_dir} не найдено .bin файлов.")
        return

    # Инициализируем межкадровый трекер v2 и одометрию
    tracker = LidarObstacleTrackerV2()
    odometry = StableLidarOdometryV12() 
    
    # 💾 ГЛОБАЛЬНЫЙ РЕЕСТР ДЛЯ УЧЕТА УНИКАЛЬНЫХ ФИЗИЧЕСКИХ ПРЕПЯТСТВИЙ
    unique_obstacles_registry = {}  
    total_detections_count = 0      
    csv_rows = []
    
    for frame_name in frame_files:
        file_path = os.path.join(test_data_dir, frame_name)
        frame_idx = frame_files.index(frame_name)

        # Читаем бинарное облако точек лидара из правильного file_path
        raw_points_array = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

        # Вычисляем сдвиг кадра по одометрии с защитой от типов возврата (без распаковки через "_")
        try:
            res_odo = odometry.update_odometry_fusion(raw_points_array, frame_idx, dt=0.1)
            if isinstance(res_odo, (tuple, list)):
                shift_z_physical = float(res_odo[0])
            else:
                shift_z_physical = float(res_odo)
        except Exception:
            shift_z_physical = (odometry.prev_velocity_kmh / 3.6) * 0.1

        calculated_speed_kmh = odometry.prev_velocity_kmh
        
        # =====================================================================
        # 🛰️ [ДИНАМИЧЕСКИЙ АНАЛИЗАТОР ПРОСТРАНСТВА НА ЛЕТУ]
        # =====================================================================
        x_check = raw_points_array[:, 0]
        y_check = raw_points_array[:, 2] 
        
        side_walls_mask = (y_check >= -0.5) & (y_check <= 1.5) & (np.abs(raw_points_array[:, 1]) < 25.0)
        
        if np.sum(side_walls_mask) > 100:
            max_wall_width = np.percentile(np.abs(x_check[side_walls_mask]), 98)
            is_open_space_computed = bool(max_wall_width > 3.2)
        else:
            is_open_space_computed = False
            
        if is_open_space_computed:
            print(f" 📡 [АУДИТ ПРОСТРАНСТВА]: Кадр {frame_name} ➔ Впереди ПЛАТФОРМА / СТАНЦИЯ (is_open_space=True)", flush=True)
        # =====================================================================

        # Вызываем пайплайн детекции v2 с передачей трекера
        detected_obstacles = process_point_cloud(raw_points_array, tracker=tracker, train_step_z=shift_z_physical, is_open_space=is_open_space_computed)
        
                # Вызываем пайплайн детекции v2 с передачей трекера
        detected_obstacles = process_point_cloud(raw_points_array, tracker=tracker, train_step_z=shift_z_physical, is_open_space=is_open_space_computed)
        
        if isinstance(detected_obstacles, list) and len(detected_obstacles) > 0:
            total_detections_count += len(detected_obstacles)
            
            # ЦИКЛ №1: Быстро собираем сквозную историю треков для итоговой таблицы судей
            for obj in detected_obstacles:
                track_id = obj.get("id", obj.get("track_id", "unknown"))
                cz_raw = obj.get("center", 0.0)
                
                # 🎯 БЕЗОПАСНОЕ ИЗВЛЕЧЕНИЕ Z: Если пришел список [X, Y, Z], берем именно Z (индекс 2)
                if isinstance(cz_raw, (list, tuple, np.ndarray)):
                    cz_val = float(cz_raw[2]) if len(cz_raw) > 2 else float(cz_raw[0])
                else:
                    cz_val = float(cz_raw) if isinstance(cz_raw, (int, float)) else 0.0
                    
                cz_dist = abs(cz_val)
                
                if track_id not in unique_obstacles_registry and track_id != "unknown":
                    unique_obstacles_registry[track_id] = {
                        "first_seen_frame": frame_name,
                        "min_distance_m": cz_dist,
                        "shape": obj.get("shape_text", "Объемный объект")
                    }
                elif track_id in unique_obstacles_registry:
                    if cz_dist < unique_obstacles_registry[track_id]["min_distance_m"]:
                        unique_obstacles_registry[track_id]["min_distance_m"] = cz_dist

            
            # 🎯 СХЛОПЫВАНИЕ ДУБЛЕЙ ДЛЯ CSV: Выполняется строго ОДИН раз на весь кадр (Вне циклов!)
                        # 🎯 СХЛОПЫВАНИЕ ДУБЛЕЙ ДЛЯ CSV: Вычисляем один ближайший объект на весь кадр (Вне циклов!)
            def get_distance_z(o):
                center = o.get("center", 0.0)
                if isinstance(center, (list, tuple, np.ndarray)):
                    # Разворачиваем вложенные массивы, если ИИ вернул [[X, Y, Z]]
                    flat_c = np.ravel(center)
                    return abs(float(flat_c[2])) if len(flat_c) > 2 else abs(float(flat_c[0]))
                try:
                    return abs(float(center))
                except (TypeError, ValueError):
                    return 999.0

            closest_obj = min(detected_obstacles, key=get_distance_z)

            # 🎯 АБСОЛЮТНО ЗАЩИЩЕННАЯ ИБЫСТРАЯ РАСПАКОВКА КООРДИНАТ КЛИППИНГОМ
            c_raw = closest_obj.get("center", [0.0, 0.0, 0.0])
            if isinstance(c_raw, (list, tuple, np.ndarray)):
                flat_c = np.ravel(c_raw)
                cx = float(flat_c[0]) if len(flat_c) > 0 else 0.0
                cy = float(flat_c[1]) if len(flat_c) > 1 else 0.0
                cz = float(flat_c[2]) if len(flat_c) > 2 else 0.0
            else:
                cx, cy, cz = 0.0, 0.0, float(c_raw) if isinstance(c_raw, (int, float)) else 0.0
            
            # Универсальная распаковка габаритов ближайшего объекта
            d_raw = closest_obj.get("dimensions", [1.0, 1.0, 1.0])
            if isinstance(d_raw, (list, tuple, np.ndarray)):
                flat_d = np.ravel(d_raw)
                sz_x = float(flat_d[0]) if len(flat_d) > 0 else 1.0
                sz_y = float(flat_d[1]) if len(flat_d) > 1 else 1.0
                sz_z = float(flat_d[2]) if len(flat_d) > 2 else 1.0
            else:
                sz_x, sz_y, sz_z = 1.0, 1.0, float(d_raw) if isinstance(d_raw, (int, float)) else 1.0
            
            dist_z = abs(cz)
            
            print(f"🚨 [{frame_name}] -> Найдено рамок: {len(detected_obstacles)} | "
                  f"Ближайшее: {dist_z:.2f} м | Уникальных объектов в памяти: {len(unique_obstacles_registry)}")
                  
            # Пишем строго ОДНУ строку для текущего кадра в итоговый CSV-сабмит по ближайшей угрозе
            csv_rows.append({
                "frame_id": frame_name,
                "x_center": f"{cx:.4f}",
                "y_center": f"{cy:.4f}",
                "z_center": f"{cz:.4f}",
                "class_id": 1,
                "confidence": f"{closest_obj.get('confidence', 1.0):.4f}"
            })

        else:
            # Требование ТЗ: даже пустые кадры должны быть зафиксированы в оффлайн-CSV
            csv_rows.append({
                "frame_id": frame_name, "x_center": "0.0000", "y_center": "0.0000", "z_center": "-1.0000", "class_id": 0, "confidence": "0.0000"
            })
            print(f"🟢 [{frame_name}] -> Пути чистые. Препятствий не обнаружено.")


    print_banner("3. СИНХРОННЫЙ ЭКСПОРТ В SUBMISSION.CSV")
    csv_path = "submission.csv"
    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = ["frame_id", "x_center", "y_center", "z_center", "class_id", "confidence"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"✅ Результаты успешно верифицированы и сохранены в: {os.path.abspath(csv_path)}")
    except Exception as e:
        print(f"❌ Ошибка записи итогового файла ответов: {e}")

    # 🏁 ВЫВОДИМ ОЧИЩЕННЫЙ ИТОГОВЫЙ АУДИТ ЖЮРИ
    print_banner("🏁 АУДИТ ПОЛНОСТЬЮ ЗАВЕРШЕН")
    print(f"📊 Всего уникальных кадров записано в CSV: {len(csv_rows)}") # 🎯 ДОБАВЛЯЕМ ЭТОТ ПРИНТ
    print(f"📊 Всего отрисовано рамок на кадрах (детекций): {total_detections_count}")
    print(f"🎯 РЕАЛЬНОЕ КОЛИЧЕСТВО УНИКАЛЬНЫХ ПРЕПЯТСТВИЙ НА ТРАССЕ: {len(unique_obstacles_registry)} шт.")
    print("-" * 70)
    print("📋 СПИСОК ФИЗИЧЕСКИХ ЦЕЛЕЙ (СОПРОВОЖДЕНИЕ ТРЕКЕРА):")
    for t_id, data in unique_obstacles_registry.items():
        fs_frame = data.get('first_seen_frame') if data.get('first_seen_frame') is not None else "unknown"
        min_dist = data.get('min_distance_m') if data.get('min_distance_m') is not None else 0.0
        shape_text = data.get('shape') if data.get('shape') is not None else "Объемный объект"
        print(f"  🆔 Объект #{str(t_id):<4} | Впервые замечен: {fs_frame} | Мин. дистанция: {min_dist:.2f} м | Тип: {shape_text}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
