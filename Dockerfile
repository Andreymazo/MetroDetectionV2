FROM ros:humble-perception

# Обновляем репозитории и ставим pip для докачки тяжелых Python ИИ-библиотек
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Доставляем специфический ИИ-стек и ros2-numpy
# Доставляем специфический ИИ-стек, ros2-numpy и полноценный uvicorn с веб-сокетами
RUN pip3 install --no-cache-dir \
    open3d \
    scikit-learn \
    fastapi \
    "uvicorn[standard]" \
    websockets \
    ros2-numpy


# Настраиваем рабочую директорию решения
WORKDIR /app

# 🟢 ЭТАЛОННЫЙ КАНOН FASTБDS: Генерируем прецизионный XML-профиль Shared Memory
RUN echo '<?xml version="1.0" encoding="UTF-8" ?>\n\
<dds xmlns="http://eprosima.com">\n\
    <profiles>\n\
        <!-- Сначала явно регистрируем конфигурацию самого SHM-транспорта -->\n\
        <transport_descriptors>\n\
            <transport_descriptor>\n\
                <transport_id>shm_transport</transport_id>\n\
                <type>SHM</type>\n\
            </transport_descriptor>\n\
        </transport_descriptors>\n\
\n\
        <!-- Привязываем созданный транспорт к дефолтному профилю участника -->\n\
        <participant profile_name="shm_only_participant_profile" is_default_profile="true">\n\
            <rtps>\n\
                <useBuiltinTransports>false</useBuiltinTransports>\n\
                <userTransports>\n\
                    <transport_id>shm_transport</transport_id>\n\
                </userTransports>\n\
            </rtps>\n\
        </participant>\n\
    </profiles>\n\
</dds>' > /app/fastdds_shm.xml

ENV FASTRTPS_DEFAULT_PROFILES_FILE=/app/fastdds_shm.xml
ENV PYTHONPATH=/app

# Копируем всю кодовую базу
COPY . /app

# Делаем питоновский скрипт Humble-ноды исполняемым
RUN chmod +x /app/metro_lidar/lidar_detector_node.py

# Автоматический запуск ИИ-ноды в режиме Wall Time (use_sim_time=false)
CMD ["python3", "/app/metro_lidar/lidar_detector_node.py", "--ros-args", "-p", "use_sim_time:=false"]
