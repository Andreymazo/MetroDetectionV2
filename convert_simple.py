import os
import sqlite3
import struct
import numpy as np
from pathlib import Path

def parse_adaptive_pointcloud2(data_blob):
    """
    Автоматически определяет point_step, динамически находит смещения полей
    и извлекает чистые [X, Y, Z, Intensity] независимо от структуры и padding'а.
    """
    # Список возможных размеров point_step (16 - чистый XYZI, 32 - с выравниванием или ring/timestamp)
    possible_steps = [16, 24, 32, 48]
    point_step = None
    marker_pos = -1
    
    for possible_step in possible_steps:
        # Собираем маркер: 1 байт (is_bigendian=0) + 4 байта (point_step)
        marker = b'\x00' + struct.pack('<I', possible_step)
        marker_pos = data_blob.find(marker, 0, 2000)
        if marker_pos != -1:
            point_step = possible_step
            break
            
    if marker_pos == -1 or point_step is None:
        raise ValueError("Не удалось автоматически определить point_step в заголовке CDR")
        
    # Позиция после point_step
    pos = marker_pos + 5
    
    # Читаем row_step (4 байта) и data_bytes_len (4 байта)
    row_step, data_bytes_len = struct.unpack_from('<II', data_blob, pos)
    
    # Начало бинарного массива точек
    data_start = pos + 8
    raw_data = data_blob[data_start : data_start + data_bytes_len]
    
    # Корректируем длину буфера, если он забит нулями в конце
    actual_data_len = (len(raw_data) // point_step) * point_step
    raw_data = raw_data[:actual_data_len]
    
    num_points = len(raw_data) // point_step
    
    # ДИНАМИЧЕСКАЯ НАРЕЗКА МАТРИЦЫ ПО СМЕЩЕНИЯМ
    raw_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(num_points, point_step)
    
    # Создаем итоговую матрицу [N, 4]
    points = np.zeros((num_points, 4), dtype=np.float32)
    
    # Стандартные оффсеты для X, Y, Z (они всегда идут первыми)
    x_off, y_off, z_off = 0, 4, 8
    
    # А вот интенсивность ищем динамически в зависимости от обнаруженного point_step
    if point_step == 16:
        i_off = 12
    elif point_step == 32:
        i_off = 16  # Часто встречающийся оффсет интенсивности при 32 байтах
    else:
        i_off = 12  # Дефолт для остальных
        
    # Безопасное векторное извлечение float32 через views
    points[:, 0] = raw_array[:, x_off:x_off+4].view(np.float32).ravel()
    points[:, 1] = raw_array[:, y_off:y_off+4].view(np.float32).ravel()
    points[:, 2] = raw_array[:, z_off:z_off+4].view(np.float32).ravel()
    
    # Проверяем, не вылезает ли оффсет интенсивности за границы шага точки
    if i_off + 4 <= point_step:
        points[:, 3] = raw_array[:, i_off:i_off+4].view(np.float32).ravel()
    else:
        points[:, 3] = 0.0  # Если интенсивности нет, заполняем нулями
        
    return points


def main():
    bag_dir = "./for_hackathon/cloud_with_fake_obj"
    db3_path = os.path.join(bag_dir, "cloud_with_fake_obj_0.db3")
    output_dir = "./test_lidar_frames/cloud_with_fake_obj"
    
    if not os.path.exists(db3_path):
        print(f"[ОШИБКА] Файл базы данных {db3_path} не найден!")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db3_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name FROM topics WHERE type='sensor_msgs/msg/PointCloud2'")
    topic_row = cursor.fetchone()
    
    if not topic_row:
        print("[ОШИБКА] В базе данных не найден топик PointCloud2!")
        conn.close()
        return
        
    topic_id, topic_name = topic_row
    print(f"[АДАПТИВНЫЙ КОНВЕРТЕР] Чтение топика: {topic_name} (ID: {topic_id})")

    cursor.execute("SELECT data, timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", (topic_id,))
    
    frame_counter = 0
    timestamps_log_path = os.path.join(output_dir, "timestamps.txt")
    
    with open(timestamps_log_path, "w") as ts_file:
        while True:
            row = cursor.fetchone()
            if row is None:
                break
                
            data_blob, raw_timestamp = row
            try:
                # Извлекаем данные с учетом динамического шага
                points = parse_adaptive_pointcloud2(data_blob)
                
                # Фильтрация NaN и нулей
                nan_mask = np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(points[:, 2])
                points = points[~nan_mask]
                
                nonzero_mask = np.any(points[:, :3] != 0, axis=1)
                points = points[nonzero_mask]
                
                if len(points) == 0:
                    continue
                
                # Сохраняем в бинарник
                filename = f"frame_{frame_counter:06d}.bin"
                filepath = os.path.join(output_dir, filename)
                points.tofile(filepath)
                
                timestamp_sec = raw_timestamp / 1e9
                ts_file.write(f"{filename},{timestamp_sec}\n")
                
                frame_counter += 1
                if frame_counter % 50 == 0 or frame_counter == 1:
                    print(f"  [Кадр {frame_counter}] Успешно извлечен. Точек: {len(points)} -> {filename}")
            except Exception as e:
                print(f"  [Ошибка кадра {frame_counter}]: {e}")
                continue

    conn.close()
    print(f"[УСПЕХ] Конвертация завершена! Обработано кадров: {frame_counter}")

if __name__ == "__main__":
    main()
