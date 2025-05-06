from typing import (
    List,  # Импортируем тип List из модуля typing
    Optional,  # Импортируем тип Optional из модуля typing
)

import biosppy  # Импортируем библиотеку biosppy для работы с сигналами
import numpy as np  # Импортируем библиотеку numpy для работы с массивами
import wfdb  # Импортируем библиотеку wfdb для работы с сигналами ЭКГ

from src.features.mfcc import NWIN  # Импортируем константу NWIN из модуля src.features.mfcc
from .segment import Segment  # Импортируем класс Segment из модуля src.model.segment
from src.utils import const  # Импортируем константы из модуля src.utils

class Record:
    def __init__(self, path: str):
        # Инициализация объекта Record с путем к файлу
        signals, _ = wfdb.rdsamp(path, channels=[1])  # Чтение сигнала ЭКГ из файла
        self._signals: np.ndarray = signals  # Сохранение сигнала как массива numpy

    def peaks(self) -> np.ndarray:
        # Выделение R-пиков сигнала ЭКГ
        return biosppy.signals.ecg.engzee_segmenter(
            signal=self._signals,  # Входной сигнал
            sampling_rate=const.SAMPLING_RATE,  # Частота дискретизации
        ).as_dict()['rpeaks']  # Возвращение массива R-пиков

    @staticmethod
    def normalize(signals: np.ndarray) -> np.ndarray:
        # Нормализация сигнала
        scale: float = max(np.abs(signals.max()), np.abs(signals.min()))  # Определение максимального значения амплитуды
        return signals / scale  # Возвращение нормализованного сигнала

    def max_segment_len(self) -> int:
        # Определение максимальной длины сегмента
        peaks = self.peaks()  # Выделение R-пиков
        segment_lengths: List[int] = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 2)]  # Вычисление длин сегментов между пиками
        return max(segment_lengths)  # Возвращение максимальной длины сегмента

    def segments(self, length: Optional[int] = None, normalize: bool = True) -> List[Segment]:
        # Разделение сигнала на сегменты
        res = []  # Инициализация списка для хранения сегментов

        peaks = self.peaks()  # Выделение R-пиков
        for i in range(len(peaks) - 2):
            if length:
                signals = self._signals[peaks[i]:peaks[i] + length]  # Разделение сигнала на сегмент заданной длины
            elif peaks[i + 1] - peaks[i] >= NWIN:
                signals = self._signals[peaks[i]:peaks[i + 1]]  # Разделение сигнала на сегмент между пиками
            else:
                continue

            if normalize:
                signals = self.normalize(signals)  # Нормализация сегмента
            res.append(Segment(signals))  # Добавление сегмента в список

        return res  # Возвращение списка сегментов
