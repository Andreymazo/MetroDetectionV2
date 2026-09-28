import os
import sys
import numpy as np
from sklearn.cluster import DBSCAN
from unified_object_engine import UnifiedLidarObjectEngine

def test_real_tracking_loop(scene_dir, voxel_size=0.35):
    """
    ФУНДАМЕНТАЛЬНЫЙ СКВОЗНОЙ СТРЕСС-ТЕСТ КОНТУРА MOT (MULTIPLE OBJECT TRACKING).
    
    НАЗНАЧЕНИЕ ТЕСТА:
    Валидация сквозной межкадровой ассоциации макро-объектов туннеля (ЭЯ) в условиях 
    реального высокочастотного потока лидарных данных (.bin) беспилотного метро.
    
    ЧТО И ЗАЧЕМ МЫ ПРОВЕРЯЕМ:
    1. Интеграцию метода `match_and_track_frame`: Проверяется скрытая логика, которая 
       была полностью исключена из тестов холодного старта (где вызывалась только 
       изолированная паспортизация отдельных вокселей рельс).
    2. Корректность знака компенсации одометрии поезда: Метод `match_and_track_frame` 
       самостоятельно сдвигает старые треки в памяти на величину `train_step_z`. 
       Тест верифицирует, совпадает ли направление и знак этого сдвига с реальным 
       набеганием облака точек на лидар (уменьшение локальной координаты Z).
    3. Выносливость ИИ-воронки (Spatial Gate + Feature Match Threshold): Проверяется, 
       удерживает ли многомерный дескриптор формы и блеска реальные физические объекты 
       туннеля (кабели, кронштейны, светофоры) при изменении ракурса сканирования, 
       или система начнет "терять" треки и плодить клоны (новые ID).
    4. Стабильность RAM (Утилизация по TTL): Верификация корректного уменьшения 
       и сброса `ttl` для объектов, вышедших из зоны видимости датчика под бампер поезда.
       
    ВХОДНЫЕ ПАРАМЕТРЫ ДЛЯ МЕТОДА match_and_track_frame:
    - current_clusters (list of np.array [N, 4]): Список реальных пространственных 
      кластеров инфраструктуры туннеля, выделенных на текущем кадре с помощью DBSCAN.
    - train_step_z (float): Динамический физический шаг продвижения состава вперед 
      (в метрах), рассчитанный верифицированным контуром одометрии путей.
    """
    print("=" * 115)
    print("🧪 СУПЕР-ТЕСТ ЕПО-CORE: СКВОЗНАЯ ВАЛИДАЦИЯ КОНТУРА МЕЖКАДРОВОГО ТРЕКИНГА НА РЕАЛЬНЫХ ДАННЫХ")
    print(f"📂 Целевой сценарий: {scene_dir}")
    print("=" * 115)

    # Берем первые 4 последовательных кадра для отслеживания динамики накопленияHits
    files = sorted([f for f in os.listdir(scene_dir) if f.endswith('.bin')])[:4]
    if len(files) < 3:
        print("❌ Ошибка: В папке недостаточно кадров для глубокого трекинга.")
        return

    # Инициализируем ЕПО-Core с вашими промышленными параметрами фильтрации
    engine = UnifiedLidarObjectEngine(
        spatial_gate_radius=2.5, 
        feature_match_threshold=0.75, 
        max_ttl=10
    )

    # Физические сдвиги поезда вперед, зафиксированные пусковым контуром
    # Шаг измеряется в метрах. Положительное значение означает движение состава вперед.
    real_train_shifts = [0.0, 1.0930, 1.2553, 1.1500]

    # Основной цикл симуляции движения поезда по перегону
    for idx, frame_name in enumerate(files):
        filepath = os.path.join(scene_dir, frame_name)
        raw_points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)

        # Монолитный мост осей v12: [0:X_ширина, 1:Y_высота, 2:Z_дальность]
        points = np.zeros((len(raw_points), 4))
        points[:, 0] = raw_points[:, 2]  # X - ширина туннеля (лево/право)
        points[:, 1] = raw_points[:, 0]  # Y - высота туннеля (пол/потолок)
        points[:, 2] = raw_points[:, 1]  # Z - продольная дальность (глубина)
        points[:, 3] = raw_points[:, 3]  # Интенсивность (блеск материала)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Стадия фильтрации пространства: Вырезаем габарит туннеля
        tunnel_mask = (y > -1.85) & (y < 2.0) & \
                      (z >= -45.0) & (z <= -3.5)
        
        # Защитный барьер безопасности: Исключаем рельсовое полотно и колею поезда,
        # чтобы изолировать стационарные объекты стен туннеля (ЭЯ)
        outside_tracks_mask = (x < -0.8) | (x > 0.8)
        
        clean_mask = tunnel_mask & outside_tracks_mask
        macro_pts = points[clean_mask]

        if len(macro_pts) < 10:
            print(f"\n🎬 Кадр #{idx:03d} ({frame_name}) ➔ Туннель пуст. Недостаточно точек для кластеризации.")
            continue

        # Пространственная сегментация: Выделяем физические контуры макро-объектов туннеля
        db = DBSCAN(eps=0.5, min_samples=10, n_jobs=-1).fit(macro_pts[:, :3])
        labels = db.labels_
        
        # Формируем список кластеров текущего кадра для передачи в ЕПО-Core
        current_frame_clusters = []
        for label in set(labels):
            if label == -1:
                continue
            current_frame_clusters.append(macro_pts[labels == label])

        print(f"\n🎬 Кадр #{idx:03d} ({frame_name})")
        print(f"   📥 Выделено реальных пространственных кластеров лидара: {len(current_frame_clusters)} шт.")
        print(f"   🚊 Физический шаг поезда на этом такте: {real_train_shifts[idx]*100:.2f} см")

        # 🔥 ВЫЗЫВАЕМ БОЕВОЙ МЕТОД СОПРОВОЖДЕНИЯ ТРЕКОВ
        # Конвейер передает сформированные кластеры кадра и дельту хода поезда из одометрии
        confirmed_output = engine.match_and_track_frame(
            current_clusters=current_frame_clusters, 
            train_step_z=real_train_shifts[idx]
        )

        # Выводим глубокий аудит состояния памяти реестра треков
        print(f"   📊 Текущий размер tracked_registry: {len(engine.tracked_registry)} треков в памяти")
        
        # Сортируем треки по степени стабильности для детального анализа тренда
        active_tracks = [t for t in engine.tracked_registry if t.get("hits", 0) > 1]
        new_tracks = [t for t in engine.tracked_registry if t.get("hits", 0) == 1]
        
        print(f"      ↳ 🔄 Стабильно сопровождаются (Hits > 1): {len(active_tracks)} шт.")
        print(f"      ↳ ⏳ Новые кандидаты на проверке (Hits == 1): {len(new_tracks)} шт.")
        print(f"   🎯 Выдано подтвержденных целей (Hits >= 3): {len(confirmed_output)} шт.")
        
        # Если есть стабильные треки, выводим динамику изменения их локальных координат Z
                # Если есть стабильные треки, покажем динамику их движения в СК поезда
        if active_tracks:
            print("   📋 Срез активных треков в памяти:")
            # Перебираем первые 3 трека из списка активных
            for track in active_tracks[:3]:
                # Безопасно извлекаем координаты центра масс
                center_coords = track.get("center", [0.0, 0.0, 0.0])
                cx = center_coords[0]
                cy = center_coords[1]
                cz = center_coords[2]
                
                print(f"      🆔 Трек ID: #{track.get('id')} | "
                      f"Засечек (Hits): {track.get('hits')} | "
                      f"TTL: {track.get('ttl')} | "
                      f"Центр: X={cx:+.2f}м, Y={cy:+.2f}м, Z={cz:.2f}м")


    print("\n" + "=" * 115)
    print("🏁 ТЕСТ ЗАВЕРШЕН. СМОТРИМ НА ТРЕНД ИЗМЕНЕНИЯ КОЛИЧЕСТВА ТРЕКОВ")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    # Указываем путь к папке с бинарными кадрами туннеля (сценарий платформы метро)
    target_scenario = "./test_lidar_frames/doubleT_platform"
    if os.path.exists(target_scenario):
        test_real_tracking_loop(target_scenario)
    else:
        print(f"❌ Ошибка: Папка {target_scenario} не найдена. Проверьте пути к датасету хакатона.")
