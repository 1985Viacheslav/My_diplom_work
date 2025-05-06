import warnings  # Импортируем модуль warnings для работы с предупреждениями

import numpy as np  # Импортируем библиотеку numpy для работы с массивами

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    # Функция для сегментации массива вдоль указанной оси

    if axis is None:
        a = np.ravel(a)  # Преобразуем массив в одномерный
        axis = 0

    l = a.shape[axis]  # Длина вдоль указанной оси

    if overlap >= length:
        raise ValueError("frames cannot overlap by more than 100%")  # Ошибка, если перекрытие больше или равно длине окна
    if overlap < 0 or length <= 0:
        raise ValueError("overlap must be nonnegative and length must be positive")  # Ошибка, если перекрытие отрицательное или длина не положительная

    if l < length or (l - length) % (length - overlap):
        if l > length:
            roundup = length + (1 + (l - length) // (length - overlap)) * (length - overlap)
            rounddown = length + ((l - length) // (length - overlap)) * (length - overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length - overlap) \
               or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1, axis)  # Меняем оси местами

        if end == 'cut':
            a = a[..., :rounddown]  # Обрезаем массив
        elif end in ['pad', 'wrap']:
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s, dtype=a.dtype)  # Создаем новый массив
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue  # Дополняем массив значением endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup - l]  # Оборачиваем массив
            a = b

        a = a.swapaxes(-1, axis)  # Возвращаем оси на место

    l = a.shape[axis]
    if l == 0:
        raise ValueError("Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'")  # Ошибка, если недостаточно данных
    assert l >= length
    assert (l - length) % (length - overlap) == 0
    n = 1 + (l - length) // (length - overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n, length) + a.shape[axis + 1:]  # Новая форма массива
    newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]  # Новые шаги

    try:
        return np.ndarray.__new__(
            np.ndarray, strides=newstrides,
            shape=newshape, buffer=a, dtype=a.dtype
        )
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")  # Предупреждение о проблеме с созданием ndarray
        a = a.copy()  # Копируем массив
  
        newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]  # Новые шаги
        return np.ndarray.__new__(
            np.ndarray, strides=newstrides,
            shape=newshape, buffer=a, dtype=a.dtype
        )

