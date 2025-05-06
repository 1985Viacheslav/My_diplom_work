from dataclasses import dataclass  # Импортируем декоратор dataclass для создания класса данных

from humanfriendly.tables import format_pretty_table  # Импортируем функцию format_pretty_table для форматирования таблиц
import numpy as np  # Импортируем библиотеку numpy для работы с массивами

@dataclass
class Descriptor:
    # Описание класса Descriptor с полями для кепстральных коэффициентов, нулевого пересечения и энтропии
    cepstral_coefficients: np.ndarray  # Массив кепстральных коэффициентов
    zcr: int  # Нулевое пересечение (Zero Crossing Rate)
    entropy: float  # Энтропия

    def as_array(self) -> np.ndarray:
        # Метод для преобразования дескриптора в массив numpy
        res = [coeff for coeff in self.cepstral_coefficients]  # Добавляем кепстральные коэффициенты в список
        res.append(float(self.zcr))  # Добавляем нулевое пересечение в список
        res.append(self.entropy)  # Добавляем энтропию в список

        return np.array(res, dtype=float)  # Преобразуем список в массив numpy

    def __repr__(self):
        # Метод для представления дескриптора в виде строки
        return ', '.join([f'{value:.03f}' for value in self.as_array()])  # Форматируем значения с 3 знаками после запятой

    def __str__(self):
        # Метод для представления дескриптора в виде таблицы
        column_names = ['Cepstral coefficients', 'Zero crossing rate', 'Entropy']  # Названия колонок таблицы

        cepstral_formatted = ', '.join([f'{coeff:.03f}' for coeff in self.cepstral_coefficients])  # Форматируем кепстральные коэффициенты
        values = [[cepstral_formatted, str(self.zcr), f'{self.entropy:.03f}']]  # Форматируем значения дескриптора

        return format_pretty_table(values, column_names)  # Возвращаем таблицу с форматированными значениями
