import os
import gc
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Импортируем движок одометрии стен тоннеля
# from cos_processor_v11 import StableLidarOdometryV10
from cos_processor_v12 import StableLidarOdometryV12
import cos_processor_v12
# Импортируем изолированный пайплайн поиска препятствий
# from generate_submission import process_point_cloud
from generate_submission_v2 import process_point_cloud
from tracker_v2 import LidarObstacleTrackerV2
import config

app = FastAPI(title="Autonomous Subway Lidar API & Telemetry Core", version="8.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Жёстко фиксируем путь к вашему датасету
# TEST_DATA_DIR = "./test_lidar_frames/roundT_doubleT"
TEST_DATA_DIR = "./test_lidar_frames/doubleT_obstacle"
# TEST_DATA_DIR = "./test_lidar_frames/roundT_doubleT"
# TEST_DATA_DIR = "./test_lidar_frames/roundT_squareT_pressureGate_squareT"
# TEST_DATA_DIR = "./test_lidar_frames/squareT_platform_squareT_switch"


@app.websocket("/ws/monitoring")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] 3D Cockpit успешно подключился к ИИ-конвейеру одометрии!", flush=True)
    
    session_journal = {
        "current_frame_index": 0,
        "total_distance_meters": 0.0
    }
    
    # Изолируем экземпляры движков внутри сессии для предотвращения OOM и конфликтов вкладок
    odometry_engine = StableLidarOdometryV12()
    obstacle_tracker_engine = LidarObstacleTrackerV2()
    
    frame_files = []
    if os.path.exists(TEST_DATA_DIR):
        frame_files = sorted([f for f in os.listdir(TEST_DATA_DIR) if f.endswith(('.pcd', '.bin'))])
        
    dt = 0.1  # 10 Гц Hesai 128
    
    try:
        async for message in websocket.iter_text():
            if message == "READY_FOR_NEXT_FRAME":
                frame_data = {
                    "danger_alert": False,
                    "train_speed_kmh": 0.0,
                    "total_distance_m": round(session_journal["total_distance_meters"], 2),
                    "distance_to_obstacle_m": -1.0,
                    # 🟢 ПРОБРАСЫВАЕМ УПРАВЛЕНИЕ КОРОБКАМИ НА ФРОНТЕНД ИЗ CONFIG.PY
                    "show_mesh_boxes": config.V2_OBSTACLE_SHOW_3D_MESHBOX,
                    "objects": []
                }
                
                if len(frame_files) > 0:
                    idx = session_journal["current_frame_index"]
                    if idx >= len(frame_files):
                        # Мягкий перезапуск круга симуляции при окончании кадров датасета
                        idx = 0
                        session_journal["current_frame_index"] = 0
                        session_journal["total_distance_meters"] = 0.0
                        odometry_engine = StableLidarOdometryV12()
                        obstacle_tracker_engine.reset()
                        
                    target_file = frame_files[idx]
                    file_path = os.path.join(TEST_DATA_DIR, target_file)
                    
                    # -----------------------------------------------------------------
                    # ВЕТКА А: ИЗОЛИРОВАННАЯ ДЕТЕКЦИЯ ПРЕПЯТСТВИЙ (v2)
                    # -----------------------------------------------------------------
                    # -----------------------------------------------------------------
                    # ВЕТКА А: ДЕТЕКЦИЯ ПРЕПЯТСТВИЙ С УЧЕТОМ ГЕОМЕТРИИ ТУННЕЛЯ
                    # -----------------------------------------------------------------
                    valid_walls_anchors = [
                        obj for obj in odometry_engine.anchor_map 
                        if obj.get("status") == "STATUS_VALID_ANCHOR" and obj.get("is_wall", False) is True
                    ]
                    
                    is_open_space = False
                    if len(valid_walls_anchors) > 0:
                        wall_x_coords = [float(wall_obj["center"]) for wall_obj in valid_walls_anchors if "center" in wall_obj]
                        if wall_x_coords:
                            max_wall_x = max(wall_x_coords)
                            min_wall_x = min(wall_x_coords)
                            
                            # Чистый вызов через константу: проверяем разлет стен туннеля
                            if max_wall_x > config.V2_OBSTACLE_WALL_OPEN_THRESHOLD or min_wall_x < -config.V2_OBSTACLE_WALL_OPEN_THRESHOLD:
                                is_open_space = True
                                
                    if is_open_space:
                        print(f" 🛬 [ГЕО-ШЛЮЗ]: Зафиксировано расширение туннеля (Разлет стен: {min_wall_x:.1f}м ... {max_wall_x:.1f}м). Смена режима следования колеи.", flush=True)

                    detected_obstacles = process_point_cloud(file_path, obstacle_tracker_engine, is_open_space=is_open_space)

                    
                    # -----------------------------------------------------------------
                    # ВЕТКА Б: ЧЕСТНАЯ ОДОМЕТРИЯ СТЕН И РЕЛЬСОВ (ВЫЗОВ ИЗ ЯДРА v12)
                    # -----------------------------------------------------------------
                    # Читаем бинарный файл один раз за такт сокета
                    raw_points = np.fromfile(file_path, dtype=np.float32).reshape(-1, config.MATRIX_WIDTH_CHANNELS)

                    # 1. Запускаем параллельный контур рельсового одометра (ОЯР)
                    shift_z_rails, rail_passport = odometry_engine.compute_raw_rail_odo_shift(raw_points, dt)

                    # Вывод оригинального лога диагностики ОЯР путей
                    if rail_passport is not None:
                        print(f"   [API ВЫХОД ОЯР] Такт кадра #{idx:03d} | rail_passport: dict (Есть: True)", flush=True)
                        rx, ry, rz = rail_passport["centroid"]
                        print(f"      ↳ Паспорт ОЯР -> ID: {rail_passport['id']}, Центр: [{rx}, {ry}, {rz}], Статус: {rail_passport['status']}", flush=True)

                    # 2. Запускаем контур стен туннеля (ЭЯ)
                    macro_cloud = odometry_engine.extract_clean_macro_tunnel(file_path)
                    
                    if macro_cloud is not None:
                        passports = odometry_engine.build_passports_via_dbscan(macro_cloud)
                        calculate_speed_trigger = bool(idx > 0)
                        
                        # Сопоставляем стены (якоря напечатаются в консоль внутри метода)
                        shift_z_walls, matches_count, logs = odometry_engine.associate_and_calculate_shift(
                            passports, dt, calculate_speed=calculate_speed_trigger
                        )
                    else:
                        shift_z_walls, matches_count = 0.0, 0

                    # 3. МАТЕМАТИЧЕСКИЙ ИИ-ФЬЮЖН ШЛЮЗ (БЕЗ ХАРДКОДА И ПОВТОРОВ КОДА)
                    # Подсчитываем локальное количество точек в створе для верификации датчиков
                    x_pts, y_pts, z_pts = raw_points[:, 2], raw_points[:, 0], raw_points[:, 1]
                    rail_points_mask = (z_pts >= -63.5) & (z_pts <= -3.5) & \
                                       (x_pts >= -0.75) & (x_pts <= 0.75) & \
                                       (y_pts >= -1.85) & (y_pts <= -1.05)
                    current_rail_points_count = int(np.sum(rail_points_mask))

                    # Передаем ровно 6 параметров в соответствии с вашим новым ядром v12
                    # Функция сама отключит капризные фильтры на старте и отсечет прыжки скорости!
                    shift_z_physical = cos_processor_v12.calculate_adaptive_fusion_shift(
                        shift_z_rails, 
                        shift_z_walls, 
                        odometry_engine.prev_velocity_kmh, 
                        current_rail_points_count, 
                        idx, 
                        dt
                    )

                    # Переводим результирующий чистый физический сдвиг шлюза в скорость км/ч
                    calculated_speed_kmh = (shift_z_physical / dt) * 3.6

                    # Стабилизационный нуль-фильтр остановки у перрона
                    if calculated_speed_kmh < 0.2:
                        calculated_speed_kmh = 0.0
                        shift_z_physical = 0.0

                    # Обновляем хронологическую память сессионного журнала
                    odometry_engine.prev_velocity_kmh = calculated_speed_kmh
                    session_journal["total_distance_meters"] += shift_z_physical
                    
                    # Упаковываем чистую телеметрию одометрии в корень JSON-пакета
                    frame_data["train_speed_kmh"] = round(float(calculated_speed_kmh), 1)
                    frame_data["total_distance_m"] = round(float(session_journal["total_distance_meters"]), 1)

                    # -----------------------------------------------------------------
                    # УПАКОВКА ОБЪЕКТОВ СИНХРОННО ПОД ТРЕБОВАНИЯ THREE.JS
                    # -----------------------------------------------------------------
                    frame_data["danger_alert"] = len(detected_obstacles) > 0
                    
                    # 1. Пакуем реальные опасные препятствия (class_id = 1)
                    if len(detected_obstacles) > 0:
                        min_dist = min([abs(float(obj["center"][2])) for obj in detected_obstacles])
                        frame_data["distance_to_obstacle_m"] = round(min_dist, 1)
                        
                        for obj_idx, obj in enumerate(detected_obstacles):
                            cx, cy, cz = obj["center"]
                            w, h, d = obj["dimensions"]
                            
                            three_x = float(cx)
                            three_y = float(cy)
                            three_z = -abs(float(cz))
                            
                            # === ЖЕЛЕЗОБЕТОННЫЙ МАТЕМАТИЧЕСКИЙ КЛАССИФИКАТОР МОРФОЛОГИИ ===
                            # Если высота существенно больше ширины и глубины — это человек/столб
                            if h > w and h > d and h > 1.0:
                                shape_text = "Человек / Вертикальная конструкция"
                                shape_type = "VERTICAL_SILHOUETTE"
                            # Если объект очень плоский по высоте — это настил/препятствие на рельсах
                            elif h < 0.40 and (w > 0.6 or d > 0.6):
                                shape_text = "Плоский предмет / Настил"
                                shape_type = "FLAT_OBSTACLE"
                            # Во всех остальных случаях — объемный блок или коробка
                            else:
                                shape_text = "Объемная коробка / Блок"
                                shape_type = "BOX_BLOCK"

                            # Пересчитываем позицию и зоны риска по прецизионной медиане three_x
                            deviation_x = abs(three_x)
                            if deviation_x <= 0.35:
                                position_status = "CENTER"
                                position_text = "Строго по центру путей 🚨"
                            elif three_x < -0.35:
                                position_status = "LEFT_EDGE"
                                position_text = "Касается левой кромки габарита ⚠️"
                            else:
                                position_status = "RIGHT_EDGE"
                                position_text = "Касается правой кромки габарита ⚠️"
                            # =============================================================

                            # Извлекаем массив прореженных сырых точек кластера из DBSCAN
                            raw_pts = obj.get("raw_points", [])
                            three_pts = []
                            for pt in raw_pts:
                                # Перетасовываем оси точек препятствия под Three.js (Z уводим в минус)
                                three_pts.append([float(pt[0]), float(pt[1]), -abs(float(pt[2]))])

                            frame_data["objects"].append({
                                "id": str(obj.get("id", f"📦_{obj_idx}")),
                                "class_id": 1,
                                "confidence": float(obj.get("confidence", 1.0)),
                                "center": [three_x, three_y, three_z],
                                "size_3d": [float(w), float(h), float(d)],
                                "obstacle_speed_kmh": float(obj.get("obstacle_speed", 0.0)),
                                # Пробрасываем честные живые данные на фронтенд
                                "position_status": position_status,
                                "position_text": position_text,
                                "shape_type": shape_type,
                                "shape_text": shape_text,
                                # Наше живое облегченное лазерное облако препятствия
                                "obstacle_points": three_pts 
                            })


                    else:
                        frame_data["distance_to_obstacle_m"] = -1.0

                    # 2. Пакуем активные ЭЯ СТЕН тоннеля (class_id = 0)
                    valid_walls_anchors = [
                        obj for obj in odometry_engine.anchor_map 
                        if obj.get("status") == "STATUS_VALID_ANCHOR" and obj.get("is_wall", False) is True
                    ]
                    
                    for wall_obj in valid_walls_anchors:
                        try:
                            pts = wall_obj["raw_points"]
                            if len(pts) > 100:
                                pts = pts[::3]  # Прореживаем шагом для разгрузки WebGL рендеринга
                            
                            three_pts = []
                            for pt in pts:
                                # Внутренний ИИ контракт: pt[0]=X_ширина, pt[1]=Y_высота, pt[2]=Z_дальность
                                # Уводим продольную ось в минус для Three.js канона
                                three_pts.append([float(pt[0]), float(pt[1]), -abs(float(pt[2]))])
                            
                            frame_data["objects"].append({
                                "id": f"WALL_{wall_obj.get('id', '?')}",
                                "class_id": 0,
                                "confidence": 1.0,
                                "center": [0.0, 0.0, 0.0],
                                "size_3d": [0.0, 0.0, 0.0],
                                "wall_points": three_pts,
                                "obstacle_speed_kmh": 0.0
                            })
                        except Exception as e:
                            continue
                    # 3. Пакуем активный РЕЛЬСОВЫЙ ЯКОРЬ ОЯР под кабину (class_id = 2)
                    if rail_passport is not None:
                        rx, ry, rz = rail_passport["centroid"]
                        
                        frame_data["objects"].append({
                            "id": f"RAIL_ANCHOR_{rail_passport.get('id', '888')}",
                            "class_id": 2,
                            "confidence": 1.0,
                            "center": [float(rx), float(ry), -abs(float(rz))],
                            "size_3d": [0.6, 0.1, 1.2],  # Габарит подсветки шпалы
                            "obstacle_speed_kmh": 0.0,
                            "ttl": int(rail_passport.get("ttl", 60))
                        })

                    # Защитный сокетный щит для удержания фронтенд-петли, если сцена пуста
                    if len(frame_data["objects"]) == 0:
                        frame_data["objects"].append({
                            "id": "RETAINER", 
                            "class_id": -1, 
                            "confidence": 1.0,
                            "center": [0.0, -20.0, 0.0], 
                            "size_3d": [0.01, 0.01, 0.01],
                            "obstacle_speed_kmh": 0.0
                        })
                    # ТВОЙ ОРИГИНАЛЬНЫЙ КОНЦЕВОЙ ПРИНТ СТАТИСТИКИ ТАКТА
                    # =====================================================================
                    # 📊 СИНХРОНИЗИРОВАННЫЙ ИИ-ЛОГ ТАКТА (БЕРЕМ ДАННЫЕ НАПРЯМУЮ ИЗ ШЛЮЗА API)
                    # =====================================================================
                    ttl_val = rail_passport['ttl'] if rail_passport else "Поиск"
                    
                    # Извлекаем из готового JSON-пакета только объекты препятствий (class_id == 1)
                    api_obstacles = [obj for obj in frame_data["objects"] if obj.get("class_id") == 1]
                    
                    if len(api_obstacles) > 0:
                        # Берем первое препятствие такта
                        closest_api_obs = api_obstacles[0]
                        obs_status_str = f"🚨 {len(api_obstacles)} ПРЕДМЕТА ({closest_api_obs.get('position_text')}, {closest_api_obs.get('shape_text')})"
                    else:
                        obs_status_str = "🟢 ЧИСТО"

                    print(f"[TAКТ {idx:03d}] Стены: {len(valid_walls_anchors)} ЭЯ | "
                          f"🛤️  ОЯР: {ttl_val}t | "
                          f"V поезда: {calculated_speed_kmh:.1f} км/ч | "
                          f"Путь: {session_journal['total_distance_meters']:.1f} м | "
                          f"Преграды: {obs_status_str}", flush=True)
                    # =====================================================================

                    # Инкрементируем индекс текущей сессии для следующего кадра
                    session_journal["current_frame_index"] = idx + 1
                    
                    # Отправляем сформированную 3D-сцену и телеметрию на фронтенд Three.js
                    await websocket.send_json(frame_data)
                    
                    # Глубокая очистка памяти от тяжелых локальных массивов текущего такта
                    del macro_cloud, passports, detected_obstacles, valid_walls_anchors
                    gc.collect()
                else:
                    await websocket.send_json(frame_data)
                    await asyncio.sleep(0.1)


    except WebSocketDisconnect:
        print("[WS] 3D Cockpit отключился.", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

