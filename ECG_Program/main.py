import os

# Импортируем класс AutomaticSpinner из библиотеки humanfriendly
from humanfriendly.terminal.spinners import AutomaticSpinner

# Устанавливаем уровень логирования для TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    try:
        # Создаем спиннер для валидации аргументов
        with AutomaticSpinner('Валидация аргументов'):
            from src.cli.args import get_args
            args = get_args()  # Получаем аргументы командной строки

        # Создаем спиннер для подготовки экстрактора дескрипторов
        with AutomaticSpinner('Подготовка экстрактора дескрипторов'):
            from src.cli.descriptors import get_descriptors

        descriptors = get_descriptors(args.record)  # Получаем дескрипторы записи

        # Создаем спиннер для подготовки классификатора
        with AutomaticSpinner('Подготовка классификатора'):
            from src.classifier.classifier import Classifier
            clf = Classifier()  # Инициализируем классификатор

        # Создаем спиннер для классификации записи
        with AutomaticSpinner('Классификация записи'):
            person_id, confidence = clf.classify(descriptors)  # Классифицируем запись

        # Выводим результат классификации
        print(f'Указанная запись соответствует Person_{person_id + 1} с уверенностью {confidence:.2%}')

    except FileNotFoundError as e:
        # Обрабатываем исключение, если файл не найден
        print(e)

if __name__ == '__main__':
    main()  # Запускаем основную функцию
