import numpy as np  # Импортируем библиотеку numpy для работы с массивами
from scipy.fftpack import fft  # Импортируем функцию fft для вычисления дискретного преобразования Фурье
from scipy.fftpack.realtransforms import dct  # Импортируем функцию dct для вычисления дискретного косинусного преобразования
from scipy.signal import lfilter  # Импортируем функцию lfilter для фильтрации сигналов

from .segment_axis import segment_axis  # Импортируем функцию segment_axis из локального модуля

NWIN = 256  # Определяем константу для размера окна

def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
    # Создание фильтробанка для анализа частотных составляющих

    nfilt = nlinfilt + nlogfilt  # Общее количество фильтров

    # Инициализация массива частотных границ
    freqs = np.zeros(nfilt + 2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc  # Линейная шкала частот
    freqs[nlinfilt:] = freqs[nlinfilt - 1] * logsc ** np.arange(1, nlogfilt + 3)  # Логарифмическая шкала частот
    heights = 2. / (freqs[2:] - freqs[0:-2])  # Высоты треугольных фильтров

    # Инициализация фильтробанка
    fbank = np.zeros((nfilt, nfft))
    nfreqs = np.arange(nfft) / (1. * nfft) * fs  # Частотная шкала
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i + 1]
        hi = freqs[i + 2]

        # Индексы для левой и правой части треугольных фильтров
        lid = np.arange(
            np.floor(low * nfft / fs) + 1,
            np.floor(cen * nfft / fs) + 1, dtype=int
        )
        lslope = heights[i] / (cen - low)  # Наклон левой части треугольного фильтра
        rid = np.arange(
            np.floor(cen * nfft / fs) + 1,
            np.floor(hi * nfft / fs) + 1, dtype=int
        )
        rslope = heights[i] / (hi - cen)  # Наклон правой части треугольного фильтра
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank, freqs  # Возвращаем фильтробанк и границы частот

def mfcc(signal, nwin=NWIN, nfft=512, fs=16000, nceps=13):
    # Функция для вычисления кепстральных коэффициентов мел-частот (MFCC)

    over = nwin - 160  # Перекрытие окон
    prefac = 0.97  # Коэффициент предыскажения
    lowfreq = 133.33  # Нижняя граница частот
    linsc = 200 / 3.  # Линейная шкала частот
    logsc = 1.0711703  # Логарифмическая шкала частот

    nlinfil = 13  # Количество фильтров на линейной шкале
    nlogfil = 27  # Количество фильтров на логарифмической шкале
    nfil = nlinfil + nlogfil  # Общее количество фильтров

    w = np.hamming(nwin)  # Создание окна Хэмминга

    fbank = trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)[0]  # Создание фильтробанка

    extract = preemp(signal, prefac)  # Применение предыскажения к сигналу
    framed = segment_axis(extract, nwin, over) * w  # Разделение сигнала на сегменты и применение окна

    spec = np.abs(fft(framed, nfft))  # Вычисление спектра
    mspec = np.log10(np.dot(spec, fbank.T))  # Вычисление мел-спектра
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, :nceps]  # Вычисление MFCC
    return ceps, mspec, spec  # Возвращение MFCC, мел-спектра и спектра

def preemp(signal, p):
    # Функция для предыскажения сигнала
    return lfilter([1., -p], 1, signal)
