import os  # Импортируем модуль os для работы с файловой системой

# Определение корневого пути проекта
ROOT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Определение пути к папке ресурсов
RESOURCES_PATH = os.path.normpath(os.path.join(ROOT_PATH, 'resources'))

# Определение пути к файлу весов модели
WEIGHTS_PATH = os.path.normpath(os.path.join(RESOURCES_PATH, 'last_weights.weights.h5'))
# Определение длины набора данных
DATASET_LENGTH = 90
# Определение частоты дискретизации
SAMPLING_RATE = 500
# Определение формы данных
DATA_SHAPE = (14, 1)

