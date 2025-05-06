from typing import List  # Импортируем тип List из модуля typing

from humanfriendly.terminal.spinners import Spinner  # Импортируем класс Spinner для отображения спиннера

from ..model import (  # Импортируем необходимые классы из модуля src.model
    Descriptor,
    Record,
)

def get_descriptors(record_path: str) -> List[Descriptor]:
    # Создаем объект записи с указанным путем
    record = Record(record_path)
    # Получаем сегменты записи без нормализации
    segments = record.segments(normalize=False)

    # Создаем спиннер для отображения прогресса вычисления дескрипторов
    with Spinner(label='Calculating descriptors from record segments', total=len(segments)) as spinner:
        res = []
        # Проходим по каждому сегменту записи
        for i, segment in enumerate(segments, start=1):
            # Добавляем дескриптор для каждого сегмента в список
            res.append(
                Descriptor(
                    cepstral_coefficients=segment.cepstral_coeffs(),  # Вычисляем кепстральные коэффициенты
                    zcr=segment.zcr(),  # Вычисляем нулевое пересечение
                    entropy=segment.entropy(),  # Вычисляем энтропию
                )
            )
            # Обновляем шаг спиннера
            spinner.step(i)

    # Возвращаем список дескрипторов
    return res

