from .descriptor import Descriptor  # Импортируем класс Descriptor из локального модуля descriptor
from .record import Record  # Импортируем класс Record из локального модуля record
from .segment import Segment  # Импортируем класс Segment из локального модуля segment

__all__ = (
    Descriptor,  # Включаем класс Descriptor в список экспортируемых символов
    Record,  # Включаем класс Record в список экспортируемых символов
    Segment,  # Включаем класс Segment в список экспортируемых символов
)
