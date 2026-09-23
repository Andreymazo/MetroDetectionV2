# Берём проверенный официальный образ ROS 2 Humble с модулями Perception
FROM ros:humble-perception

# Отключаем интерактивные окна при сборке
ENV DEBIAN_FRONTEND=noninteractive

# Обновляем системные репозитории и ставим базовые инструменты Python
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем все зависимости, включаяros2-numpy, напрямую через PIP!
# Это на 100% защищает нас от ошибки "Unable to locate package"
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    numpy \
    open3d \
    ros2-numpy \
    websockets\
    scikit-learn


# Указываем рабочую зону внутри контейнера
WORKDIR /app

# Настраиваем автоподгрузку ROS-переменных при входе в контейнер
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Запускаем командную строку по умолчанию
CMD ["/bin/bash"]
