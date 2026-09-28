import numpy as np
import metro_lidar.config as config

class TrackVectorLock:
    """
    Промышленный модуль векторной фиксации пути (Track Locking).
    Хранит, обновляет и сдвигает межкадровую цифровую трассу рельсового лотка.
    Генерирует прецизионную 3D-маску коридора безопасности для DBSCAN.
    """
    def __init__(self):
        # Вектор центроидов пути: массив [M, 3], где каждая строка — [X, Y, Z]
        # Заполняется дискретными 5-метровыми блоками от 0 до config.MAX_Z
        self.track_vector = None
        self.step_z = 5.0
        self.z_nodes = np.arange(0.0, config.MAX_Z + self.step_z, self.step_z)
        self.reset_to_passport_default()

    def reset_to_passport_default(self):
        """Аварийный откат или инициализация вектора трассы по жестким паспортным константам."""
        M = len(self.z_nodes)
        self.track_vector = np.zeros((M, 3), dtype=np.float32)
        self.track_vector[:, 0] = 0.0                      # Идеальная прямая по центру X
        self.track_vector[:, 1] = -1.52                     # Паспортная глубина лотка метро по Y
        self.track_vector[:, 2] = -self.z_nodes             # Направление продольного хода (в минус под v12)

    def update_track_geometry(self, floor_points, train_step_z):
        """
        ФАЗА 1 и 2: Межкадровый сдвиг одометрии и обновление "головы" вектора из ближнего боя.
        Принимает:
            floor_points: массив [K, 4] — точки пола, найденные RANSAC в текущем кадре.
            train_step_z: float — честный шаг поезда из ядра одометрии v12.
        """
        # 1. 🟢 ВЕКТОРНЫЙ СДВИГ ТРАССЫ ИЗ ПРОШЛОГО (Инерциальный ход)
        # Так как поезд едет вперед, старые блоки пути набегают НА поезд.
        # Переносим (интерполируем) накопленные X и Y профили с учетом пройденного расстояния.
        if abs(train_step_z) > 0.001:
            old_z_shifted = self.track_vector[:, 2] + abs(train_step_z) # Точки сместились ближе к бамперу
            # Интерполируем старые смещения X и Y на новые жесткие Z-ноды
            self.track_vector[:, 0] = np.interp(self.track_vector[:, 2], old_z_shifted, self.track_vector[:, 0])
            self.track_vector[:, 1] = np.interp(self.track_vector[:, 2], old_z_shifted, self.track_vector[:, 1])

        # 2. 🟢 ОБНОВЛЕНИЕ БЛИЖНЕЙ ЗОНЫ ЖИВЫМИ ДАННЫМИ ЛАЗЕРА (до 30 метров)
        if floor_points is not None and len(floor_points) > 20:
            z_floor = floor_points[:, 2]
            # Берем только надежную ближнюю зону, где нет препятствий и плотность лучей 100%
            near_mask = (z_floor >= -30.0) & (z_floor <= -3.5)
            near_floor = floor_points[near_mask]

            if len(near_floor) > 15:
                # Квантуем ближний лоток по нашим 5-метровым секторам
                voxel_indices = np.floor(np.abs(near_floor[:, 2]) / self.step_z).astype(np.int32)
                
                for idx_node, z_val in enumerate(self.z_nodes):
                    if z_val > 30.0:
                        break # Дальнюю зону не трогаем, она живет за счет памяти изшлюзованного прошлого!
                        
                    sector_mask = (voxel_indices == int(z_val / self.step_z))
                    sector_pts = near_floor[sector_mask]
                    
                    if len(sector_pts) > 10:
                        # Прецизионно обновляем только X и Y координаты ближних векторов
                        self.track_vector[idx_node, 0] = np.median(sector_pts[:, 0])
                        self.track_vector[idx_node, 1] = np.median(sector_pts[:, 1])

        # 3. 🛡️ МАТЕМАТИЧЕСКИЙ ПРЕДОХРАНИТЕЛЬ: Зажимаем профиль высоты Y в жесткие тиски лотка
        # Это исключает взрыв полинома в космос на пустых участках и станциях
        self.track_vector[:, 1] = np.clip(self.track_vector[:, 1], -1.80, -1.35)

    def generate_adaptive_gauge_mask(self, points, is_open_space=False):
        """
        ФАЗА 3: Сверхбыстрая генерация булевой маски коридора безопасности.
        Принимает: points массив [N, 4] сырого кадра ОЗУ.
        Возвращает: булев массив [N], где True — точки внутри Векторного Пути-Объекта.
        """
        if len(points) == 0:
            return np.array([], dtype=bool)

        x_pts = points[:, 0]
        y_pts = points[:, 1]
        z_pts = points[:, 2]
        abs_z = np.abs(z_pts)

        # Шаг 1. Через интерполяцию мгновенно находим живую ось X и Y пути для КАЖДОЙ из N точек облака
        grid_x_center = np.interp(z_pts, self.track_vector[:, 2], self.track_vector[:, 0])
        grid_y_floor  = np.interp(z_pts, self.track_vector[:, 2], self.track_vector[:, 1])

        # Шаг 2. Расчет адаптивных динамических створов ворот на основе конфига
        if is_open_space:
            # 🚉 НА СТАНЦИИ / СТРЕЛКЕ: Раскрываем боковой габарит конусом, чтобы заглянуть в кривую путей
            adaptive_half_width = np.clip(config.TRAIN_HALF_WIDTH + (abs_z * config.V2_OBSTACLE_GATE_EXPANSION_COEF), 
                                          config.TRAIN_HALF_WIDTH, config.V2_OBSTACLE_TRACK_SEARCH_WIDTH_OPEN)
            # Открываем потолок на максимум над рельсами
            dynamic_max_y = np.full_like(z_pts, 3.80)
        else:
            # 🚇 В ТУННЕЛЕ: Жестко зажимаем ширину в габарит состава, полностью срезая тюбинги стен!
            adaptive_half_width = np.clip(config.TRAIN_HALF_WIDTH + (abs_z * 0.001), 
                                          config.TRAIN_HALF_WIDTH, config.V2_OBSTACLE_TRACK_SEARCH_WIDTH_CLOSED)
            # Безопасный конус потолка: плавно ведем от 3.3м до 3.0 метров вдалеке, исключая ложный верхний свод
            dynamic_max_y = np.clip(3.30 - (abs_z * 0.002), 3.00, 3.30)

        # Шаг 3. Финальный векторный отбор точек лазера, зажатых внутри Векторного Пути-Объекта
        in_gauge_mask = (
            # Прецизионный боковой обхват X относительно живой оси поворота путей
            (x_pts >= (grid_x_center - adaptive_half_width)) &
            (x_pts <= (grid_x_center + adaptive_half_width)) &
            
            # Прецизионный высотный обхват Y строго НАД уровнем рельс и под потолком маски
            (y_pts >= (grid_y_floor + config.MIN_Y)) &  
            (y_pts <= (grid_y_floor + dynamic_max_y)) &  
            
            # Дальностный срез и мертвая зона бампера вагона
            (abs_z >= config.MIN_Z_CONTROL) & (abs_z <= config.MAX_Z)
        )

        return in_gauge_mask
