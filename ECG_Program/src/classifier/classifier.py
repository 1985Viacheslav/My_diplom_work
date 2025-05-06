from typing import (
    List,  # Импорт типа List из модуля typing
    Tuple,  # Импорт типа Tuple из модуля typing
)

import numpy as np  # Импорт библиотеки numpy для работы с массивами
import keras as k  # Импорт библиотеки Keras из TensorFlow для создания нейронных сетей

from src.model import Descriptor  # Импорт класса Descriptor из модуля src.model
from src.utils import const  # Импорт констант из модуля src.utils

class Classifier:
    def __init__(self, class_num: int = const.DATASET_LENGTH):
        # Инициализация объекта класса Classifier с заданным количеством классов
        self._model = self.make_model(class_num=class_num)

    @staticmethod
    def make_model(class_num: int):
        # Метод для создания и возвращения модели нейронной сети
        model = k.Sequential(name='ECGPersonIdentifier')  # Создание последовательной модели Keras

        model.add(k.Input(const.DATA_SHAPE))  # Добавление входного слоя с формой данных из константы
        model.add(k.layers.Flatten(name='flatten'))  # Преобразование многомерного ввода в одномерный массив

        # Добавление плотных (Dense) слоев
        for i, neurons in enumerate([335, 104, 214, 57]):  # Проходим по списку с количеством нейронов - лучшие параметры взяты из результатов автоматического подбора параметров model_training
            model.add(k.layers.Dense(units=neurons, activation='relu', name=f'dense{i}'))  # Добавляем плотный слой с функцией активации ReLU
                        
        # Добавление выходного слоя с функцией активации Softmax для классификации
        model.add(k.layers.Dense(class_num, activation='softmax', name='classifier'))

        # Загрузка весов модели из файла, указанного в константе
        model.load_weights(const.WEIGHTS_PATH)

        return model

    def classify(self, descriptors: List[Descriptor]) -> Tuple[int, float]:
        # Метод для классификации дескрипторов
        data = np.array([descriptor.as_array() for descriptor in descriptors], dtype=float)  # Преобразование дескрипторов в массив чисел
        y_pred = np.argmax(self._model.predict(data), axis=-1)  # Прогнозирование классов для каждого дескриптора

        pred_class = np.bincount(y_pred).argmax()  # Определение наиболее часто встречающегося класса
        confidence = np.sum(np.where(y_pred == pred_class, 1, 0)) / len(y_pred)  # Подсчет уровня уверенности в классификации
        return pred_class, confidence  # Возвращение класса и уровня уверенности
