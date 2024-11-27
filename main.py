import os
from concurrent.futures import ThreadPoolExecutor

from clstr import light_green, light_cyan, green
from data_processing import get_coords_from_projection, merge_data_sets, combine_arrays, \
    interpolate_missing_values
from excel import parse_camera_settings, write_results_to_excel
from video_processing import VideoProcessor


def process_video(video_processor: VideoProcessor) -> list[dict]:
    print(f'\tОбработка видео "{video_processor.video_path}"...')
    coords = video_processor.get_all_coords()
    print(f'\tОбработка видео "{video_processor.video_path} завершена"')
    return coords


def main() -> None:
    videoset = int(input(green("Введите номер видеосета (1-8): ")))
    read_path = os.path.abspath(f'tests\\step1\\videoset{videoset}\\Seq{videoset}_settings.xlsx')

    print(light_cyan(f'\nОбработка файла "{read_path}"...'))

    cs = parse_camera_settings(read_path, videoset)
    camera_info = list(cs.values())

    print(f'\nКамеры:\n\t{'\n\t'.join([f'{i+1}: {camera}' for i, camera in enumerate(camera_info)])}')

    print(light_cyan(f'\nСбор видео...\n'))

    video_processors = []
    video_processors_t = []

    for i in range(3):
        path = os.path.abspath(f'tests\\step1\\videoset{videoset}\\Seq{videoset}_camera{i + 1}.mov')
        path_t = os.path.abspath(f'tests\\step1\\videoset{videoset}\\Seq{videoset}_camera{i + 1}T.mov')
        video_processors.append(VideoProcessor(path, camera_info[i], 'rgb'))
        video_processors_t.append(VideoProcessor(path_t, camera_info[i], 'ir'))

    print(f'RGB-видео:\n\t{'\n\t'.join([f'{i+1}: {vp.video_path}' for i, vp in enumerate(video_processors)])}')
    print(f'IR-видео:\n\t{'\n\t'.join([f'{i+1}: {vp.video_path}' for i, vp in enumerate(video_processors_t)])}')

    input(green('\nНажмите Enter для начала обработки видео'))

    results_rgb = []
    results_ir = []

    print(light_cyan('Обработка RGB видео...'))
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_video, vp) for i, vp in enumerate(video_processors)]
        for future in futures:
            results_rgb.append(future.result())

    print(light_cyan('Обработка IR видео...'))
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_video, vp) for i, vp in enumerate(video_processors_t)]
        for future in futures:
            results_ir.append(future.result())

    print(light_green('Обработка видео завершена'))
    print(light_cyan('Обработка данных...'))

    results = [merge_data_sets(rgb, ir) for rgb, ir in zip(results_rgb, results_ir)]
    results_ = []
    for i, result in enumerate(results):
        # print(f"Камера {i + 1}:")
        results_.append([])
        for j, data in enumerate(result):
            data_ = get_coords_from_projection((data['x'], data['y']), video_processors[i].frame_size, data['diameter'], camera_info[i]) if data else None
            # print(f"\n\tКадр {j}:\n\t\tПроекция: {data}\n\t\tКоординаты 1: {data_}")
            results_[i].append(data_)

    result = combine_arrays(*results_)
    result = interpolate_missing_values(result)

    print(light_cyan('Сохранение результатов...'))

    write_path = os.path.abspath(f'output{videoset}.xlsx')
    write_results_to_excel(result, write_path)

    print(light_green(f'Результаты сохранены в файле "{write_path}"'))


if __name__ == "__main__":
    main()