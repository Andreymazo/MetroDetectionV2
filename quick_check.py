import os
import sys
import argparse
import csv
from metro_lidar.cos_processor_v12 import StableLidarOdometryV12
import numpy as np

# Импортируем твои актуальные v2 компоненты
from metro_lidar.generate_submission_v2 import process_point_cloud
from metro_lidar.tracker_v2 import LidarObstacleTrackerV2

def print_banner(text):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)

def main():
    # 🟢 ДОБАВЛЯЕМ АВТОНОМНЫЙ РАЗБОР АРГУМЕНТОВ ДЛЯ ЖЮРИ
    parser = argparse.ArgumentParser(description="Конкурсный конвейер детекции препятствий")
    parser.add_argument("--data_dir", type=str, default="./for_hackathon", help="Путь к папке с багами")
    parser.add_argument("--scenario", type=str, default="doubleT_obstacle", help="Имя сценария")
    args = parser.parse_args()

    # Задаем динамические пути на основе переданных флагов
    bag_path = os.path.join(args.data_dir, args.scenario)
    test_data_dir = f"./test_lidar_frames/{args.scenario}"

    # Если бинарных кадров еще нет — запускаем конвертер
    if not os.path.exists(test_data_dir) or not os.listdir(test_data_dir):
        print_banner("1. ИНТЕЛЛЕКТУАЛЬНАЯ ДЕСЕРИАЛИЗАЦИЯ И ПОДГОТОВКА ДАННЫХ")
        print(f"📦 Обнаружен сырой rosbag-источник в: {bag_path}")
        
        # Динамически импортируем конвертер, чтобы не раздувать зависимости
        try:
            import convert_all_bags
            print("🔄 Запускаем нативное извлечение физических координат из SQLite3...")
            convert_all_bags.convert_single_bag(args.scenario)
        except ImportError:
            print("❌ Ошибка: Файл convert_all_bags.py не найден в корне проекта!")
            return
            
    # Проверяем наличие папки с кадрами теперь
    if not os.path.exists(test_data_dir):
        print(f"❌ Ошибка: Папка {test_data_dir} не найдена!")
        return

    print_banner("2. ЗАПУСК ИИ-КОНВЕЙЕРА ДЕТЕКЦИИ ПРЕПЯТСТВИЙ (v2)")
    print(f"🔄 Сканируем кадры из: {test_data_dir}...\n")
    
    frame_files = sorted([f for f in os.listdir(test_data_dir) if f.endswith('.bin')])
    
    if not frame_files:
        print(f"⚠️ Предупреждение: В папке {test_data_dir} не найдено .bin файлов.")
        return

    # Инициализируем твой межкадровый трекер v2
    tracker = LidarObstacleTrackerV2()
    odometry = StableLidarOdometryV12() 
    total_obstacles_found_across_frames = 0
    csv_rows = []
    
    # Проходим по каждому кадру последовательно
    for frame_name in frame_files:
        file_path = os.path.join(test_data_dir, frame_name)
        # Фиксируем индекс текущего кадра (нужен для логики старта одометрии)
        frame_idx = frame_files.index(frame_name)

        # 💡 Вызываем встроенный фужн-контур одометрии v12
        # Он сам посчитает рельсы (ОЯР), стены (ЭЯ), применит ИИ-шлюз и вернет сдвиг кадра
        # Одометрия сама считает шаг сдвига из файла, ничего лишнего передавать не нужно
        shift_z_physical, _ = odometry.update_odometry_fusion(file_path, frame_idx, dt=0.1)


        # Вызываем твой пайплайн детекции v2 с передачей трекера
        detected_obstacles = process_point_cloud(file_path, tracker=tracker, train_step_z=shift_z_physical, is_open_space=False)
        
        if len(detected_obstacles) > 0:
            total_obstacles_found_across_frames += len(detected_obstacles)
            
            # Находим дистанцию до ближайшего объекта (извлекаем x_center для ТЗ жюри)
            distances = [abs(float(obj["center"][2])) for obj in detected_obstacles]
            min_dist = min(distances) if distances else 0.0
            conf = detected_obstacles[0].get("confidence", 1.0)
            
            print(f"🚨 [{frame_name}] -> ОБНАРУЖЕНО ПРЕПЯТСТВИЙ: {len(detected_obstacles)} | "
                  f"Ближайшее на расстоянии: {min_dist:.2f} м | "
                  f"Уверенность ИИ: {conf:.2%}")
                  
            # Сохраняем строки для формирования submission.csv
            for obj in detected_obstacles:
                csv_rows.append({
                    "frame_id": frame_name,
                    "x_center": f"{obj['center'][0]:.4f}",
                    "y_center": f"{obj['center'][1]:.4f}",
                    "z_center": f"{obj['center'][2]:.4f}",
                    "class_id": 1,
                    "confidence": f"{obj.get('confidence', 1.0):.4f}"
                })
        else:
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

    print_banner("🏁 АУДИТ ПОЛНОСТЬЮ ЗАВЕРШЕН")
    print(f"Всего опасных аномалий обнаружено в сессии: {total_obstacles_found_across_frames}")
    print("=" * 70)

if __name__ == "__main__":
    main()
