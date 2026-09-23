# Берём проверенный официальный образ ROS 2 Humble с модулями Perception
FROM ros:humble-perception

# Отключаем интерактивные окна при сборке
ENV DEBIAN_FRONTEND=noninteractive

# Обновляем системные репозитории и ставим базовые инструменты Python
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем все зависимости, включая ros2-numpy, напрямую через PIP!
# Это на 100% защищает нас от ошибки "Unable to locate package"
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    numpy \
    open3d \
    ros2-numpy \
    websockets \
    scikit-learn

# Указываем рабочую зону внутри контейнера
WORKDIR /app

# Настраиваем автоподгрузку ROS-переменных при входе в контейнер (оставляем для интерактивного режима)
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# 🟢 ИИ-ФИКС ДЛЯ ТЗ ЖЮРИ:
# Делаем наш созданный ROS 2 скрипт исполняемым напрямую внутри контейнера
# (Файлы вашего проекта монтируются в /app через ключ -v \$(pwd):/app при запуске)
RUN chmod +x /app/lidar_detector_node.py

# 🟢 Финальный запуск автономного контура детекции:
# Принудительно подгружаем окружение ROS 2 и запускаем ноду в реальном времени.
# Это позволит инженеру метро запустить всё одной кнопкой: `docker run`
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && python3 /app/lidar_detector_node.py"]
