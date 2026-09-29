from pathlib import Path
from rosbags.rosbag2 import Reader

# Путь к папке с багом (убедитесь, что внутри лежат metadata.yaml и .db3)
bag_path = Path('for_hackathon/cloud_with_fake_obj')

if not bag_path.exists():
    print(f"❌ Ошибка: Путь {bag_path.resolve()} не существует!")
    print("Проверьте, правильно ли указано имя папки/путь.")
    exit(1)

try:
    with Reader(bag_path) as reader:
        print("==================================================")
        print("            МЕТАДАННЫЕ ТЕСТОВОГО БАГА             ")
        print("==================================================")
        print(f"⏱️ Продолжительность записи: {reader.duration / 1e9:.2f} сек")
        print(f"📦 Всего кадров (сообщений): {reader.message_count}")
        
        print("\n📝 ДОСТУПНЫЕ ТОПИКИ (КАНАЛЫ ДАННЫХ):")
        print("--------------------------------------------------")
        
        has_odom = False
        has_gt = False
        
        for connection in reader.connections:
            topic = connection.topic
            msgtype = connection.msgtype
            print(f"🔹 Топик: {topic:<35} | Тип: {msgtype}")
            
            # Проверяем ключевые слова на скорость и разметку
            t_lower = topic.lower()
            if any(w in t_lower for w in ['odom', 'twist', 'velocity', 'speed', 'state']):
                has_odom = True
            if any(w in t_lower for w in ['label', 'truth', 'marker', 'object', 'obstacle', 'detection']):
                if 'cloud_with_fake_obj' not in t_lower:  # исключаем имя самого баг-файла
                    has_gt = True

        print("--------------------------------------------------")
        print("\n🔍 АНАЛИЗ ВОЗМОЖНОСТЕЙ НАБОРА ДАННЫХ:")
        
        if has_odom:
            print("✅ НАЙДЕН ТОПИК СКОРОСТИ/ОДОМЕТРИИ! Данные о движении поезда есть внутри бага.")
        else:
            print("❌ ТОПИК СКОРОСТИ ОТСУТСТВУЕТ. Скорость поезда в явном виде не пишется.")
            
        if has_gt:
            print("✅ НАЙДЕН ТОПИК РАЗМЕТКИ/ПРЕПЯТСТВИЙ! В баге есть готовые 3D-боксы или метки целей.")
        else:
            print("❌ ТОПИК РАЗМЕТКИ ОТСУТСТВУЕТ. Готовых ответов 'есть ли препятствие' внутри бага нет.")

        print("\n📊 ТЕСТ ЧТЕНИЯ ПЕРВЫХ КАДРОВ:")
        print("--------------------------------------------------")
        count = 0
        for connection, timestamp, rawdata in reader.messages():
            print(f"Сообщение {count+1}: Время {timestamp} нс | Топик: {connection.topic}")
            count += 1
            if count >= 5:
                break
        print("==================================================")

except Exception as e:
    print(f"❌ Произошла ошибка при чтении баг-файла: {e}")
    print("Убедитесь, что в указанной папке лежат файлы metadata.yaml и .db3 одновременно.")
