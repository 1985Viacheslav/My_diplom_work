import antropy  # Импортируем библиотеку antropy для вычисления энтропии и нулевых пересечений
import numpy as np  # Импортируем библиотеку numpy для работы с массивами

from src.features.mfcc import mfcc  # Импортируем функцию mfcc из модуля src.features.mfcc

class Segment:
    def __init__(self, signal: np.ndarray):
        # Инициализация объекта Segment с сигналом, представленным в виде массива numpy
        self._signal = signal

    def cepstral_coeffs(self) -> np.ndarray:
        # Вычисление кепстральных коэффициентов для сегмента
        ceps, _, _ = mfcc(self._signal, nceps=12)  # Вызываем функцию mfcc для получения 12 кепстральных коэффициентов

        return ceps[0]  # Возвращаем первый набор кепстральных коэффициентов

    def zcr(self) -> int:
        # Вычисление нулевого пересечения для сегмента
        return antropy.num_zerocross(self._signal.flatten())  # Используем библиотеку antropy для вычисления нулевых пересечений

    def entropy(self) -> float:
        # Вычисление энтропии для сегмента
        return antropy.perm_entropy(self._signal.flatten())  # Используем библиотеку antropy для вычисления энтропии
