from typing import List, Tuple

import numpy as np
import tensorflow.keras as k

from src.model import Descriptor
from src.utils import const


class Classifier:
    def __init__(self):
        # Загружаем модель из файла .keras
        self._model = k.models.load_model(const.MODEL_PATH)
        print("✅ Модель успешно загружена")

    def classify(self, descriptors: List[Descriptor]) -> Tuple[int, float]:
        """
        Классифицирует список дескрипторов ЭКГ.
        Возвращает наиболее вероятный класс и уровень уверенности.
        """
        # Преобразуем дескрипторы в массив для предсказания
        data = np.array([descriptor.as_array() for descriptor in descriptors], dtype=float)

        # Предсказываем вероятности классов
        predictions = self._model.predict(data)
        y_pred = np.argmax(predictions, axis=-1)

        # Определяем наиболее частый класс и уровень уверенности
        pred_class = np.bincount(y_pred).argmax()
        confidence = np.sum(np.where(y_pred == pred_class, 1, 0)) / len(y_pred)

        return pred_class, confidence
