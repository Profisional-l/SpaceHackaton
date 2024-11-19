import os
from concurrent.futures import ThreadPoolExecutor
from camera_data import CameraData
from data_processing import get_coords_from_projection, merge_data_sets, calculate_sphere_coordinates, combine_arrays, \
    interpolate_missing_values
from excel import parse_camera_settings, write_results_to_excel
from video_processing import VideoProcessor


def process_video(video_processor, camera_info):
    coords = video_processor.get_all_coords()
    results = []
    for i, data in enumerate(coords):
        if data is not None:
            projected_coords = get_coords_from_projection((data['x'], data['y']), video_processor.frame_size, data['diameter'], camera_info)
            results.append((i, data, projected_coords))
        else:
            results.append((i, data, None))
    return coords


def main():
    videoset = int(input("Введите номер видеосета (1-8): "))

    cs = parse_camera_settings(os.path.abspath(f'tests\\step1\\videoset{videoset}\\Seq{videoset}_settings.xlsx'), videoset)
    camera_info = list(cs.values())
    print(camera_info)
    input('Нажмите Enter для продолжения...')
    video_processors = []
    video_processors_t = []

    for i in range(3):
        path = os.path.abspath(f'tests\\step1\\videoset{videoset}\\Seq{videoset}_camera{i + 1}.mov')
        path_t = os.path.abspath(f'tests\\step1\\videoset{videoset}\\Seq{videoset}_camera{i + 1}T.mov')
        video_processors.append(VideoProcessor(path, camera_info[i], 'rgb'))
        video_processors_t.append(VideoProcessor(path_t, camera_info[i], 'ir'))

    results_rgb = []
    results_ir = []
    # print(video_processors[0].frame_size)

    print('Обработка RGB видео...')
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_video, vp, camera_info[i]) for i, vp in enumerate(video_processors)]
        for future in futures:
            results_rgb.append(future.result())

    print('Обработка IR видео...')
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_video, vp, camera_info[i]) for i, vp in enumerate(video_processors_t)]
        for future in futures:
            results_ir.append(future.result())

    print('Объединение данных...')
    results = [merge_data_sets(rgb, ir) for rgb, ir in zip(results_rgb, results_ir)]
    results_ = []
    for i, result in enumerate(results):
        print(f"Камера {i + 1}:")
        results_.append([])
        for j, data in enumerate(result):
            # не работает
            data_ = get_coords_from_projection((data['x'], data['y']), video_processors[i].frame_size, data['diameter'], camera_info[i]) if data else None
            # data2 = calculate_sphere_coordinates(camera_info[i], video_processors[i].frame_size, (data['x'], data['y'], data['diameter'])) if data else None
            print(f"Кадр {j}:\n\tПроекция: {data}\n\tКоординаты 1: {data_}")
            results_[i].append(data_)

    result = combine_arrays(*results_)
    # print(result)
    result = interpolate_missing_values(result)

    print(result)
    write_results_to_excel(result, f'output{videoset}.xlsx')



if __name__ == "__main__":
    main()