import os
import numpy as np
from pathlib import Path
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import ros2_numpy

def convert_single_bag(bag_name):
    bag_path = f"./for_hackathon/{bag_name}"
    output_dir = f"./test_lidar_frames/{bag_name}"
    
    if not os.path.exists(bag_path):
        print(f"[ПРОПУСК] Папка {bag_path} не найдена.")
        return
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n[ПАКЕТНЫЙ КОНВЕРТЕР] Обработка сценария: {bag_name}")
    
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr', output_serialization_format='cdr'
    )
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"  ❌ Ошибка открытия bag-файла: {e}")
        return

    # --- ИНТЕЛЛЕКТУАЛЬНЫЙ АВТОПОИСК ТОПИКА ЛИДАРА ---
    topic_types = reader.get_all_topics_and_types()
    target_topic = None
    msg_type_str = "sensor_msgs/msg/PointCloud2"
    
    for topic in topic_types:
        if topic.type == msg_type_str:
            target_topic = topic.name
            break
            
    if not target_topic:
        print(f"  ❌ Внутри записи не найдено ни одного топика с типом {msg_type_str}!")
        return
        
    print(f"  🔍 Нативныи автопоиск определил канал данных: {target_topic}")
    
    msg_type = get_message(msg_type_str)
    frame_counter = 0
    
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        if topic_name != target_topic:
            continue
            
        try:
            msg = deserialize_message(data, msg_type)
            arr_dict = ros2_numpy.numpify(msg)
            
            xyz = arr_dict['xyz'].astype(np.float32)
            num_points = len(xyz)
            
            if 'intensity' in arr_dict:
                intensity = arr_dict['intensity'].astype(np.float32).reshape(-1, 1)
            else:
                intensity = np.zeros((num_points, 1), dtype=np.float32)
                
            points = np.hstack((xyz, intensity))
            
            nan_mask = np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(points[:, 2])
            points = points[~nan_mask]
            
            nonzero_mask = np.any(points[:, :3] != 0, axis=1)
            points = points[nonzero_mask]
            
            if len(points) == 0:
                continue
                
            filename = f"frame_{frame_counter:06d}.bin"
            filepath = os.path.join(output_dir, filename)
            points.tofile(filepath)
            frame_counter += 1
            
        except Exception as e:
            continue
            
    print(f"  ✅ УСПЕХ! Извлечено кадров: {frame_counter} -> сохранены в {output_dir}")

def main():
    # Список всех папок сценариев жюри
    scenarios = [
        "doubleT_obstacle",
        "doubleT_platform",
        "roundT_doubleT",
        "roundT_pressureGate_roundT",
        "roundT_squareT_pressureGate_squareT",
        "squareT_platform_squareT_switch"
    ]
    
    print("="*70)
    print("[ПАКЕТНЫЙ КОНВЕРТЕР] Запуск адаптивнои десериализации всех данных жюри...")
    print("="*70)
    
    for scenario in scenarios:
        convert_single_bag(scenario)
        
    print("\n" + "="*70)
    print("[УСПЕХ] Все доступные сценарии жюри изолированно разложены по папкам!")
    print("="*70)

if __name__ == "__main__":
    main()
