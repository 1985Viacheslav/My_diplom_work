from typing import List, Tuple
import numpy as np
import tensorflow.keras as k
from joblib import load
import logging
import os

from src.model import Descriptor
from src.utils import const

# === Настройка логирования (совместимо с Python 3.8) ===
os.makedirs('logs', exist_ok=True)

log_path = 'logs/classifier.log'
log_format = '%(asctime)s | %(levelname)s | %(message)s'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter(log_format))

if not logger.handlers:
    logger.addHandler(file_handler)


class Classifier:
    def __init__(self):
        # Загружаем модель
        try:
            self._model = k.models.load_model(const.MODEL_PATH)
            logger.info(f"Модель успешно загружена из: {const.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

        # Загружаем StandardScaler
        try:
            self._scaler = load(const.SCALER_PATH)
            logger.info(f"Scaler успешно загружен из: {const.SCALER_PATH}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке scaler'а: {e}")
            raise

    def classify(self, descriptors: List[Descriptor]) -> Tuple[int, float]:
        """
        Классифицирует список дескрипторов ЭКГ-сегментов.
        Возвращает наиболее вероятный класс и уровень уверенности.
        """
        if not descriptors:
            logger.warning("Получен пустой список дескрипторов")
            raise ValueError("Список дескрипторов пуст")

        # Преобразуем каждый дескриптор в массив
        data = np.array([d.as_array() for d in descriptors], dtype=float)
        logger.info(f"Получено {len(data)} дескрипторов для классификации")

        # Масштабируем
        data = self._scaler.transform(data)
        logger.info("Данные масштабированы с использованием StandardScaler")

        # Предсказания
        predictions = self._model.predict(data, verbose=0)
        y_pred = np.argmax(predictions, axis=-1)

        pred_class = np.bincount(y_pred).argmax()
        confidence = np.sum(y_pred == pred_class) / len(y_pred)

        logger.info(f"Предсказанные классы сегментов: {y_pred}")
        logger.info(f"Выбранный класс: {pred_class} | Уверенность: {confidence:.2%}")
        logger.debug(f"Предсказанные вероятности (первые 2 сегмента):\n{predictions[:2]}")

        return pred_class, confidence



