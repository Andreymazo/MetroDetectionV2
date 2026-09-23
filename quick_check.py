import os
import sys
import argparse
import csv
import numpy as np

# Импортируем твои актуальные v2 компоненты
from generate_submission_v2 import process_point_cloud
from tracker_v2 import LidarObstacleTrackerV2

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
    total_obstacles_found_across_frames = 0
    csv_rows = []
    
    # Проходим по каждому кадру последовательно
    for frame_name in frame_files:
        file_path = os.path.join(test_data_dir, frame_name)
        
        # Вызываем твой пайплайн детекции v2 с передачей трекера
        detected_obstacles = process_point_cloud(file_path, tracker)
        
        if len(detected_obstacles) > 0:
            total_obstacles_found_across_frames += len(detected_obstacles)
            
            # Находим дистанцию до ближайшего объекта (извлекаем x_center для ТЗ жюри)
            distances = [abs(float(obj["center"][0])) for obj in detected_obstacles]
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

# import os
# import numpy as np

# # Импортируем компоненты из новой версии ИИ-конвейера
# from generate_submission_v2 import process_point_cloud
# from tracker_v2 import LidarObstacleTrackerV2

# # Настраиваем пути к датасету
# # TEST_DATA_DIR = "./test_lidar_frames/roundT_doubleT"
# # TEST_DATA_DIR = "./test_lidar_frames/doubleT_obstacle"
# # TEST_DATA_DIR = "./test_lidar_frames/doubleT_platform"

# # TEST_DATA_DIR = "./test_lidar_frames/roundT_doubleT"
# # TEST_DATA_DIR = "./test_lidar_frames/roundT_squareT_pressureGate_squareT"
# # TEST_DATA_DIR = "./test_lidar_frames/squareT_platform_squareT_switch"

# def print_banner(text):
#     print("\n" + "=" * 70)
#     print(f" {text}")
#     print("=" * 70)

# def main():
#     # Проверяем наличие папки с кадрами
#     if not os.path.exists(TEST_DATA_DIR):
#         print(f"❌ Ошибка: Папка {TEST_DATA_DIR} не найдена!")
#         return

#     print_banner("1. ЗАПУСК ИИ-КОНВЕЙЕРА ДЕТЕКЦИИ ПРЕПЯТСТВИЙ (v2)")
#     print(f"🔄 Сканируем кадры из: {TEST_DATA_DIR}...\n")
    
#     # Получаем и сортируем список файлов лидара
#     frame_files = sorted([f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.bin')])
    
#     if not frame_files:
#         print(f"⚠️ Предупреждение: В папке {TEST_DATA_DIR} не найдено .bin файлов.")
#         return

#     # Инициализируем межкадровый трекер для симуляции непрерывного движения поезда
#     tracker = LidarObstacleTrackerV2()
    
#     total_obstacles_found_across_frames = 0
    
#     # Проходим по каждому кадру последовательно
#     for frame_name in frame_files:
#         file_path = os.path.join(TEST_DATA_DIR, frame_name)
        
#         # Вызываем пайплайн детекции v2 с передачей трекера
#         # detected_obstacles = process_point_cloud(file_path, tracker)
#         detected_obstacles = process_point_cloud(file_path)
#         if len(detected_obstacles) > 0:
#             total_obstacles_found_across_frames += len(detected_obstacles)
            
#             # Находим дистанцию до ближайшего объекта (координата Z в v2 отвечает за дальность вперед)
#             min_dist = min([abs(float(obj["center"][2])) for obj in detected_obstacles])
#             # Берём confidence первого препятствия
#             conf = detected_obstacles[0].get("confidence", 1.0)
            
#             print(f"🚨 [{frame_name}] -> ОБНАРУЖЕНО ПРЕПЯТСТВИЙ: {len(detected_obstacles)} | "
#                   f"Ближайшее на расстоянии: {min_dist:.2f} м | "
#                   f"Уверенность ИИ: {conf:.2%}")
#         else:
#             print(f"🟢 [{frame_name}] -> Пути чистые. Препятствий не обнаружено.")

#     print_banner("🏁 СИНХРОННЫЙ АУДИТ ЗАВЕРШЕН")
#     print(f"Всего опасных аномалий обнаружено в сессии: {total_obstacles_found_across_frames}")
#     print(f"Активных треков осталось в памяти трекера: {len(tracker.past_tracks)}")
#     print("=" * 70)

# if __name__ == "__main__":
#     main()

# import os
# import csv
# from generate_submission import main as run_submission_pipeline

# # Настраиваем пути
# # TEST_DATA_DIR = "./test_lidar_frames/roundT_doubleT"
# TEST_DATA_DIR = "./test_lidar_frames/doubleT_obstacle"
# # TEST_DATA_DIR = "./test_lidar_frames/roundT_doubleT"
# # TEST_DATA_DIR = "./test_lidar_frames/roundT_doubleT"
# # TEST_DATA_DIR = "./test_lidar_frames/roundT_doubleT"

# CSV_PATH = "submission.csv"

# def print_banner(text):
#     print("\n" + "=" * 70)
#     print(f" {text}")
#     print("=" * 70)

# def main():
#     # Проверяем наличие папки с кадрами
#     if not os.path.exists(TEST_DATA_DIR):
#         print(f"❌ Ошибка: Папка {TEST_DATA_DIR} не найдена!")
#         return

#     # 1. ЗАПУСК ДЕТЕКЦИИ ПРЕПЯТСТВИЙ
#     print_banner("1. ЗАПУСК ИИ-КОНВЕЙЕРА ДЕТЕКЦИИ ПРЕПЯТСТВИЙ")
#     print(f"🔄 Сканируем кадры из: {TEST_DATA_DIR}...")
    
#     # Временно подменяем рабочую папку в кодовой базе, чтобы generate_submission читал нужный датасет
#     # (так как в его коде test_data_dir по умолчанию жестко прописан)
#     import generate_submission
#     generate_submission.test_data_dir = TEST_DATA_DIR
    
#     # Запускаем оригинальный процесс генерации submission.csv
#     run_submission_pipeline()

#     # 2. АНАЛИЗ СФОРМИРОВАННОГО CSV-ФАЙЛА
#     print_banner("2. РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ ИЗ SUBMISSION.CSV")
    
#     if not os.path.exists(CSV_PATH):
#         print(f"❌ Ошибка: Файл {CSV_PATH} не был создан!")
#         return

#     # Группируем обнаруженные объекты по кадрам
#     frames_report = {}
    
#     with open(CSV_PATH, mode="r", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             frame_id = row["frame_id"]
#             # Нам нужна истинная дальность (координата X_center в ТЗ жюри / REP 103)
#             distance = float(row["x_center"])
#             class_id = int(row["class_id"])
#             confidence = float(row["confidence"])
            
#             if frame_id not in frames_report:
#                 frames_report[frame_id] = []
                
#             frames_report[frame_id].append({
#                 "dist": distance,
#                 "class": class_id,
#                 "conf": confidence
#             })

#     # Выводим компактную статистику по каждому кадру
#     total_obstacles_found = 0
    
#     for frame_name in sorted(frames_report.keys()):
#         objects = frames_report[frame_name]
#         obstacles = [obj for obj in objects if obj["class"] == 1] # Фильтруем именно препятствия
        
#         if len(obstacles) > 0:
#             total_obstacles_found += len(obstacles)
#             # Находим дистанцию до ближайшего объекта на путях в этом кадре
#             min_dist = min([obj["dist"] for obj in obstacles])
            
#             print(f"🚨 [{frame_name}] -> ОБНАРУЖЕНО ПРЕПЯТСТВИЙ: {len(obstacles)} | "
#                   f"Ближайшее на расстоянии: {min_dist:.2f} м | "
#                   f"Уверенность ИИ: {obstacles[0]['conf']:.2%}")
#         else:
#             print(f"🟢 [{frame_name}] -> Пути чистые. Препятствий не обнаружено.")

#     print_banner("🏁 ИТОГОВЫЙ АУДИТ ЗАВЕРШЕН")
#     print(f"Всего опасных аномалий в файле ответов: {total_obstacles_found}")
#     print(f"Результаты успешно сохранены и проверены в: {os.path.abspath(CSV_PATH)}")
#     print("=" * 70)

# if __name__ == "__main__":
#     main()
