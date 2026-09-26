import os
import sqlite3
import struct

def inspect_pointcloud2_fields(db3_path):
    if not os.path.exists(db3_path):
        print(f"[ОШИБКА] Файл {db3_path} не найден!")
        return

    conn = sqlite3.connect(db3_path)
    cursor = conn.cursor()
    
    # 1. Находим нужный топик
    cursor.execute("SELECT id, name FROM topics WHERE type='sensor_msgs/msg/PointCloud2'")
    topic_row = cursor.fetchone()
    if not topic_row:
        print("[ОШИБКА] Топик PointCloud2 не найден в базе!")
        conn.close()
        return
        
    topic_id, topic_name = topic_row
    print(f"=== АНАЛИЗ ТОПИКА: {topic_name} ===")
    
    # 2. Берем ровно одно сообщение для теста
    cursor.execute("SELECT data FROM messages WHERE topic_id = ? LIMIT 1", (topic_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print("[ОШИБКА] В топике нет сообщений!")
        return
        
    data = row[0]
    
    # 3. Начинаем разбор заголовка CDR
    pos = 4  # Пропускаем ROS 2 CDR заголовок (4 байта)
    
    # Пропускаем std_msgs/Header (stamp: 8 байт)
    pos += 8
    
    # Пропускаем frame_id string
    frame_id_len = struct.unpack_from('<I', data, pos)[0]
    pos += 4 + frame_id_len
    pos = (pos + 3) & ~3  # Выравнивание по 4 байтам
    
    # Читаем геометрию кадра
    height, width = struct.unpack_from('<II', data, pos)
    pos += 8
    print(f"Разрешение кадра: Width={width}, Height={height} (Всего точек: {width * height})")
    
    # Читаем массив описания полей (fields)
    fields_len = struct.unpack_from('<I', data, pos)[0]
    pos += 4
    
    print("\n[СТРУКТУРА ПОЛЕЙ В СООБЩЕНИИ]:")
    print(f"{'Имя поля':<15} | {'Смещение (Offset)':<18} | {'Тип данных':<12} | {'Количество':<10}")
    print("-" * 65)
    
    # Карта типов данных ROS 2 PointCloud2 для наглядности
    datatype_map = {1: 'INT8', 2: 'UINT8', 3: 'INT16', 4: 'UINT16', 5: 'INT32', 6: 'UINT32', 7: 'FLOAT32', 8: 'FLOAT64'}

    for i in range(fields_len):
        # Длина строки имени
        name_len = struct.unpack_from('<I', data, pos)[0]
        # Само имя (исключая нулевой символ в конце, если есть)
        name = data[pos+4 : pos+4+name_len-1].decode('utf-8').strip('\x00')
        pos += 4 + name_len
        pos = (pos + 3) & ~3  # Выравнивание
        
        # offset (uint32), datatype (uint8), count (uint32)
        offset, datatype, count = struct.unpack_from('<IBI', data, pos)
        pos += 9
        pos = (pos + 3) & ~3  # Выравнивание
        
        type_str = datatype_map.get(datatype, f"UNKNOWN ({datatype})")
        print(f"{name:<15} | {offset:<18} | {type_str:<12} | {count:<10}")

    # Финальные параметры упаковки
    is_bigendian, point_step, row_step = struct.unpack_from('<?II', data, pos)
    print("-" * 65)
    print(f"Размер одной точки в байтах (point_step): {point_step}")
    print(f"Размер одной строки в байтах (row_step): {row_step}")
    print(f"Big Endian: {is_bigendian}")

if __name__ == "__main__":
    # Укажите путь к вашему файлу
    db3_file = "./for_hackathon/cloud_with_fake_obj/cloud_with_fake_obj_0.db3"
    inspect_pointcloud2_fields(db3_file)
