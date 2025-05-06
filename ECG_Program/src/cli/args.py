import argparse
import os

def get_args() -> argparse.Namespace:
    # Создание парсера аргументов командной строки
    parser = argparse.ArgumentParser(description='Классификатор человека по сигналу ЭКГ', )

    # Добавление аргумента -r/--record для указания пути к записи ЭКГ
    parser.add_argument(
        '-r',
        '--record',
        required=True,
        action='store',
        type=str,
        help='Путь к записи ЭКГ. Указывать расширение файла не обязательно.'
    )

    # Валидация аргументов командной строки
    return validate_args(parser.parse_args())

def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    # Валидация аргумента record
    args.record = validate_record(args.record)
    return args

def validate_record(record: str) -> str:
    # Преобразование пути записи в абсолютный путь
    record = os.path.abspath(record)

    # Получение директории записи
    record_dir = os.path.dirname(record)
    # Проверка существования директории
    if not os.path.exists(record_dir):
        raise FileNotFoundError(f'Папка "{record_dir}" не существует')
    elif not os.path.isdir(record_dir):
        raise FileNotFoundError(f'"{record_dir}" не является папкой')

    # Получение имени файла записи без расширения
    record_name = os.path.basename(record)
    record_name = record_name.rsplit('.', maxsplit=1)[0]  # Удаление расширения файла

    # Поиск файлов, соответствующих записи
    record_entities = list(
        filter(
            lambda file_name: file_name.startswith(record_name + '.') and os.path.isfile(
                os.path.join(record_dir, file_name)
            ), os.listdir(record_dir)
        )
    )

    # Проверка наличия необходимых файлов записи
    if len(record_entities) == 0:
        raise FileNotFoundError(f'Запись с именем "{record_name}" не существует по пути "{record_dir}"')
    elif (record_data_file := f'{record_name}.dat') not in record_entities:
        raise FileNotFoundError(f'Запись с именем "{record_name}" не имеет файла данных "{record_data_file}"')
    elif (record_header_file := f'{record_name}.hea') not in record_entities:
        raise FileNotFoundError(f'Запись с именем "{record_name}" не имеет файла заголовка "{record_header_file}"')

    # Возвращение абсолютного пути к записи без расширения
    record = os.path.abspath(os.path.join(record_dir, record_name))

    return record

