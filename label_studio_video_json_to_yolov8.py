import cv2
import json
import random
from pathlib import Path, PurePosixPath
import os
import yaml
import configparser

from icecream import ic
LIST_BOX_COLOR = colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 165, 0),
    (255, 255, 0),
    (128, 0, 128),
    (255, 192, 203),
    (165, 42, 42),
    (128, 128, 128),
    (0, 0, 0),
    (255, 255, 255)
]
BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White
SECURITY_MARGIN = 0.5  # half second
TRAIN_DIRECTORY = r"train"
VALID_DIRECTORY = r"valid"
TEST_DIRECTORY = r"test"
IMAGES_DIRECTORY = r"images"
LABELS_DIRECTORY = r"labels"
IMAGE_EXTENSION = r".jpg"
LABEL_EXTENSION = r".txt"


def convert(path_video_source_raw, path_json_raw, path_destination_raw, path_google_collab_raw):
    # A partir del json que devuelve el label-studio, lo convertimos en yolo8 format.
    # 1- Descomponemos el video en frames (iamgen) individuales
    # 2- A cada frame le asignamos un fichero txt (yolo) con las etiquetas correspondientes
    # 3- Creamos el fichero de configuración general de yolo
    if config.getboolean('ic', 'disable'):
        ic.disable()
    json = get_json(path_json_raw)
    category_id_to_name = {}
    videos_info = []
    for json_video_metadata in json:
        video_bboxes, video_category_ids, category_id_to_name, video_path = \
            convert_json(json_video_metadata, path_video_source_raw, category_id_to_name)
        ic(video_bboxes)
        ic(len(video_bboxes))
        distribution_frames = \
            distribute_frames_by_type(len(video_bboxes[0]), 
                                      config.getint('yolov8', 'perc_train'), 
                                      config.getint('yolov8', 'perc_valid'), 
                                      config.getint('yolov8', 'perc_test'))
        video_info = {'video_name': Path(video_path).stem,
                      'video_path': video_path,
                      'video_bboxes': video_bboxes,
                      'video_category_ids': video_category_ids,
                      'distribution_frames': distribution_frames}
        videos_info.append(video_info)

    dst_images_path, dst_labels_path, path_data_yaml = \
        init_destination_directories(path_destination_raw)

    write_data_yaml_yolov8(path_data_yaml, path_google_collab_raw, category_id_to_name)
    for video_info in videos_info:
        video_name = video_info['video_name']
        video_path = video_info['video_path']
        video_bboxes = video_info['video_bboxes']
        video_category_ids = video_info['video_category_ids']
        distribution_frames = video_info['distribution_frames']
        num_frame = 0
        cap = cv2.VideoCapture(str(video_path))
        for bboxes in zip(*video_bboxes):
            success, img = cap.read()
            if success:
                if has_bboxes(bboxes):
                    if config.getboolean('results', 'write_frame_video'):
                        write_image_file_yolov8(img, video_name, num_frame, dst_images_path, distribution_frames)
                    if config.getboolean('results', 'write_annotation'):
                        write_label_file_yolov8(video_category_ids, bboxes, video_name, num_frame, dst_labels_path, distribution_frames)
                visualize(img, bboxes, video_category_ids, category_id_to_name)
            else:
                break
            num_frame += 1


def get_json(path_json_raw):
    # Open the orders.json file
    with open(path_json_raw) as file:
        # Load its content and make a new dictionary
        data = json.load(file)
    return data


def get_json_video_name(anotations, path_video_source_raw):
    path_json_video_name_raw = anotations["data"]['video']
    path_json_video_name = Path(path_json_video_name_raw)
    json_video_name = path_json_video_name.name
    video_name = json_video_name.split("-", 1)[1]
    return video_name


def convert_json(json_video_metadata, path_video_source_raw, category_id_to_name_pre):
    #duration, frame_count, json_path
    video_name = get_json_video_name(json_video_metadata, path_video_source_raw)
    ic(video_name)
    video_path = Path(path_video_source_raw) / video_name
    ic(video_path)
    duration, frame_count, frame_times = get_video_information(video_path)
    # Open the orders.json file
    # with open(json_path) as file:
    #     # Load its content and make a new dictionary
    #     data = json.load(file)
    category_name_to_id = dict((v, k) for k, v in category_id_to_name_pre.items())  # Inverse dictionary
    video_bboxes = []
    video_category_ids = []
    # if len(data) == 0:
    #     raise ValueError(f"JSon {json_path} vacio");
    for result in json_video_metadata['annotations'][0]['result']:
        # ic(result['value']['labels'])
        if 'labels' in result['value'] and len(result['value']['sequence']) > 0:
            all_coordinates, label = get_anotations_label(duration, frame_count, frame_times, result, video_path)
            video_bboxes.append(all_coordinates)
            if label in category_name_to_id:
                video_category_ids.append(category_name_to_id[label])
            else:
                id_label = len(category_name_to_id)
                category_name_to_id[label] = id_label
                video_category_ids.append(id_label)
        category_id_to_name = dict((v, k) for k, v in category_name_to_id.items())  # Inverse dictionary
    return video_bboxes, video_category_ids, category_id_to_name, video_path


def get_video_information(path_video_source):
    ic(path_video_source)
    frame_times = []
    # Open the video file
    cap = cv2.VideoCapture(str(path_video_source))
    ic("Loading all time frames.....")
    # Loop through each frame and read it
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame_times.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        else:
            break
        # Do something with each frame here, like display it or process it

    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    # Release the video capture object and close any windows
    cap.release()
    cv2.destroyAllWindows()
    return duration, frame_count, frame_times


def get_anotations_label(duration, frame_count, frame_times, result, path_video_source):
    label = result['value']['labels'][0]
    ic(label)
    cap_calc = initialize_calcul_video(path_video_source)
    ic(result['value']['sequence'][0]['frame'])
    # ic(result['value']['sequence'][0])
    duration_labels = result['value']['duration']
    ic(duration, duration_labels)
    if abs(duration - duration_labels) > SECURITY_MARGIN:
        raise ValueError(
            f"No coinciden los tiempos de los videos, con el tiempo de las etiquetas. Duración video: {duration}, Duración anotaciones {duration_labels}")
    sequences = result['value']['sequence']
    # https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    # frame_step = duration / frame_count
    # frame_step_adjustment = 2
    frame_step_json = 0 # 0.04
    all_coordinates = []
    current_time = frame_times[0]
    first_loop = True
    for sequence in sequences:
        if len(all_coordinates) >= frame_count:
            break
        next_x = sequence['x']
        next_y = sequence['y']
        next_w = sequence['width']
        next_h = sequence['height']
        next_t = sequence['time'] - frame_step_json
        next_e = sequence['enabled']
        next_yolo = get_yolo_value(next_x, next_y, next_w, next_h)
        ic(next_t, current_time)
        if first_loop:
            first_loop = False
            num_steps_start = 0
            next_current_time = frame_times[0]
            next_next_current_time = frame_times[1]
            first_frame_annotation = True if abs(next_t - next_current_time) < 0.05 else False;
            ic(num_steps_start, next_t, next_current_time, next_next_current_time, abs(next_t - next_current_time), abs(next_t - next_next_current_time))
            while next_current_time < next_t and abs(next_t - next_current_time) > abs(next_t - next_next_current_time):
                num_steps_start += 1
                if first_frame_annotation:
                    all_coordinates.append(next_yolo)
                else:
                    all_coordinates.append(None)
                next_current_time = next_next_current_time
                next_next_current_time = frame_times[len(all_coordinates) + (1 if (len(all_coordinates) + 1) <= frame_count else 0)]
                ic(num_steps_start, next_t, next_current_time, next_next_current_time, abs(next_t - next_current_time), abs(next_t - next_next_current_time))
                visualize_calcul_video(cap_calc, label, next_yolo, None, all_coordinates[-1])
            all_coordinates.append(next_yolo)
            current_time = frame_times[len(all_coordinates) - 1]
            ic(current_time)
            visualize_calcul_video(cap_calc, label, next_yolo, None, all_coordinates[-1])
        else:
            # añadidmos las etiquetas a los frames intermedios de cada anotación
            num_steps = 0
            next_current_time = frame_times[len(all_coordinates) + (0 if (len(all_coordinates) + 1) <= frame_count else -1)]
            ic(num_steps, next_t, current_time, next_current_time, abs(next_t - current_time), abs(next_t - next_current_time))
            while current_time < next_t and abs(next_t - current_time) > abs(next_t - next_current_time):
                num_steps += 1
                if len(all_coordinates) + num_steps > frame_count:
                    break
                current_time = frame_times[len(all_coordinates) - 1 + num_steps]
                next_current_time = frame_times[len(all_coordinates) + (num_steps if (len(all_coordinates) + num_steps + 1) <= frame_count else num_steps - 1)]
                ic(num_steps, next_t, current_time, next_current_time, abs(next_t - current_time), abs(next_t - next_current_time))
            if num_steps > 0:
                for step in range(1, num_steps + 1):
                    ic(step, num_steps)
                    if len(all_coordinates) >= frame_count:
                        print(
                            f"Sobrepasado el número de anotaciones para los frames que existen. Num anotaciones: {len(all_coordinates)}, Num frames: {frame_count}")
                        break

                    if step == num_steps:
                        all_coordinates.append(next_yolo)
                    elif next_e and not prev_e: # Disabled annotation
                        all_coordinates.append(None)
                    else:
                        mid_x = get_mid_value(prev_x, next_x, num_steps, step)
                        mid_y = get_mid_value(prev_y, next_y, num_steps, step)
                        mid_w = get_mid_value(prev_w, next_w, num_steps, step)
                        mid_h = get_mid_value(prev_h, next_h, num_steps, step)
                        all_coordinates.append(get_yolo_value(mid_x, mid_y, mid_w, mid_h))

                    current_time = frame_times[len(all_coordinates) - 1]
                    ic(current_time)
                    visualize_calcul_video(cap_calc, label, next_yolo, prev_yolo, all_coordinates[-1])
            else:
                print(f"Hay mas anotaciones que frames. next_t {next_t}, current_time: {current_time}")
                ic(all_coordinates[len(all_coordinates) - 1], next_yolo)
                all_coordinates[len(all_coordinates) - 1] = next_yolo
        prev_x = next_x
        prev_y = next_y
        prev_w = next_w
        prev_h = next_h
        prev_t = next_t
        prev_e = next_e
        prev_yolo = next_yolo
    while len(all_coordinates) < frame_count:
        if next_e:
            all_coordinates.append(all_coordinates[len(all_coordinates) - 1])
        else:
            all_coordinates.append(None)
            current_time = frame_times[len(all_coordinates) - 1]
            visualize_calcul_video(cap_calc, label, next_yolo, prev_yolo, all_coordinates[-1])
    ic(len(all_coordinates), frame_count)
    if not len(all_coordinates) == frame_count:
        raise ValueError(
            f"No coincide el numero de anoaciones con el numero de frames del video. Num anotaciones: {len(all_coordinates)}, Num frames: {frame_count}")
    return all_coordinates, label


def initialize_calcul_video(path_video_source):
    cap_calc = None
    if config.getboolean('video_calculations', 'show'):
        cap_calc = cv2.VideoCapture(str(path_video_source))
    return cap_calc


def visualize_calcul_video(cap, label, next_box, prev_box, calculated_box):
    if config.getboolean('video_calculations', 'show'):
        success, img = cap.read()
        ic(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        if success:
            bboxes = [None, None, None]
            if config.getboolean('video_calculations', 'show_next_bbox'):
                bboxes[0] = next_box
            if config.getboolean('video_calculations', 'show_prev_bbox'):
                bboxes[1] = prev_box
            if config.getboolean('video_calculations', 'show_current_bbox'):
                bboxes[2] = calculated_box
            video_category_ids = ['1', '2', '3']
            category_id_to_name = {'1': 'next', '2': 'prev', '3': label}
            visualize(img, bboxes, video_category_ids, category_id_to_name)
        else:
            pass


def get_mid_value(pre_label, next_label, num_steps, step):
    mid = next_label
    if next_label > pre_label:
        step_x = (next_label - pre_label) / num_steps
        mid = pre_label + (step_x * step)
    elif next_label < pre_label:
        step_x = (pre_label - next_label) / num_steps
        mid = pre_label - (step_x * step)
    return mid


def get_yolo_value(next_x, next_y, next_w, next_h):
    yolo_x = (next_x + (next_w / 2)) / 100
    yolo_y = (next_y + (next_h / 2)) / 100
    yolo_w = next_w / 100
    yolo_h = next_h / 100
    return [yolo_x, yolo_y, yolo_w, yolo_h]


def init_destination_directories(path_destination_raw):
    path_destination = Path(path_destination_raw)
    path_dst_with_number = create_destination_directory(path_destination)
    dst_images_path = {TRAIN_DIRECTORY: (create_image_destination_subdirectory(path_dst_with_number, TRAIN_DIRECTORY)),
                       VALID_DIRECTORY: (create_image_destination_subdirectory(path_dst_with_number, VALID_DIRECTORY)),
                       TEST_DIRECTORY: (create_image_destination_subdirectory(path_dst_with_number, TEST_DIRECTORY))}
    dst_labels_path = {TRAIN_DIRECTORY: (create_label_destination_subdirectory(path_dst_with_number, TRAIN_DIRECTORY)),
                       VALID_DIRECTORY: (create_label_destination_subdirectory(path_dst_with_number, VALID_DIRECTORY)),
                       TEST_DIRECTORY: (create_label_destination_subdirectory(path_dst_with_number, TEST_DIRECTORY))}
    path_data_yaml = path_dst_with_number / f"data.yaml"
    return dst_images_path, dst_labels_path, path_data_yaml


def get_full_name_by_type(dst_path, num_frame, distribution_frames, video_name, file_extension):
    file_name = f"{video_name}_{int(num_frame):04}{file_extension}"
    destinations = [TRAIN_DIRECTORY, VALID_DIRECTORY, TEST_DIRECTORY]
    destination = None
    for dest_dir in destinations:
        if num_frame in distribution_frames[dest_dir]:
            destination = dest_dir
            break
    if destination is None:
        raise ValueError(f"Frame no asignado a ningun tipo (train, valid, test). num_frame: {num_frame}")
    full_name = dst_path[destination] / file_name
    return full_name


def create_destination_directory(path_destination):
    num_path_dst = 0;
    path_dst_with_number = path_destination
    while os.path.exists(path_dst_with_number):
        num_path_dst += 1
        path_dst_with_number = path_destination.parent / f"{path_destination.name}_{num_path_dst:02}"
    os.makedirs(path_dst_with_number)
    return path_dst_with_number


def create_image_destination_subdirectory(path_dst_with_number, type_directory):
    dst_train_path = path_dst_with_number / type_directory
    dst_train_images_path = dst_train_path / IMAGES_DIRECTORY
    if not os.path.exists(dst_train_path):
        os.makedirs(dst_train_path)
    os.makedirs(dst_train_images_path)
    return dst_train_images_path


def create_label_destination_subdirectory(path_dst_with_number, type_directory):
    dst_train_path = path_dst_with_number / type_directory
    dst_train_labels_path = dst_train_path / LABELS_DIRECTORY
    if not os.path.exists(dst_train_path):
        os.makedirs(dst_train_path)
    os.makedirs(dst_train_labels_path)
    return dst_train_labels_path

def distribute_frames_by_type(frame_count, perc_train, perc_valid, perc_test):
    all_frames = [int(i) for i in range(frame_count)]
    random.shuffle(all_frames)
    perc_train_val = int(perc_train * frame_count / 100)
    perc_valid_val = int(perc_valid * frame_count / 100)
    perc_test_val = int(perc_test * frame_count / 100)
    distribution_frames = {TEST_DIRECTORY: all_frames[0:perc_test_val],
                           VALID_DIRECTORY: all_frames[perc_test_val: perc_test_val + perc_valid_val],
                           TRAIN_DIRECTORY: all_frames[perc_test_val + perc_valid_val: frame_count]}
    ic(len(distribution_frames[TRAIN_DIRECTORY]), len(distribution_frames[VALID_DIRECTORY]), len(distribution_frames[TEST_DIRECTORY]))
    ic(len(distribution_frames[TRAIN_DIRECTORY]) + len(distribution_frames[VALID_DIRECTORY]) + len(distribution_frames[TEST_DIRECTORY]))
    return distribution_frames


def write_image_file_yolov8(img, video_name, num_frame, dst_images_path, distribution_frames):
    full_img_name = get_full_name_by_type(dst_images_path, num_frame, distribution_frames, video_name, IMAGE_EXTENSION)
    ic(str(full_img_name))
    cv2.imwrite(str(full_img_name), img)


def write_label_file_yolov8(all_category_ids, bboxes, video_name, num_frame, dst_labels_path, distribution_frames):
    full_lbl_name = get_full_name_by_type(dst_labels_path, num_frame, distribution_frames, video_name, LABEL_EXTENSION)
    ic(str(full_lbl_name))
    f = open(full_lbl_name, "a")
    for id_frame, bbox in zip(all_category_ids, bboxes):
        if bbox is not None:
            f.write(f"{id_frame} {'%.8f'%(bbox[0])} {'%.8f'%(bbox[1])} {'%.8f'%(bbox[2])} {'%.8f'%(bbox[3])}\n")
    f.close()


def write_data_yaml_yolov8(path_data_yaml, path_google_collab_raw, category_id_to_name):
    path_google_collab = PurePosixPath(path_google_collab_raw)
    name_directory = path_data_yaml.parents[0].name
    train_images_path = PurePosixPath().joinpath(name_directory, TRAIN_DIRECTORY, IMAGES_DIRECTORY)
    valid_images_path = PurePosixPath().joinpath(name_directory, VALID_DIRECTORY, IMAGES_DIRECTORY)
    test_images_path = PurePosixPath().joinpath(name_directory, TEST_DIRECTORY, IMAGES_DIRECTORY)
    data = {'path': str(path_google_collab),
            'train': str(train_images_path),
            'val': str(valid_images_path),
            'test=': str(test_images_path),
            'nc': len(category_id_to_name),
            'names': category_id_to_name
            }
    ic(data)
    with open(path_data_yaml, 'w') as file:
        outputs = yaml.dump(data, file, sort_keys=False)
        ic(outputs)


def has_bboxes(bboxes):
    has_bbox = False
    for label in bboxes:
        if label is not None:
            has_bbox = True
            break;
    return has_bbox


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id, color_bbox in zip(bboxes, category_ids, LIST_BOX_COLOR[:len(category_ids)]):
        # ic(bbox, category_id, category_id_to_name)
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name, color=color_bbox)

    cv2.imshow("Image", img)
    # cv2.waitKey(0)
    cv2.waitKey(config.getint('video', 'wait_key'))


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    if bbox is not None:
        """Visualizes a single bounding box on the image"""
        x_min, y_min, w, h = bbox
        img_h, img_w = img.shape[:2]
        x_min, x_max, y_min, y_max = get_coordinates(x_min - (w / 2), img_w), get_coordinates(x_min + (w / 2), img_w), \
            get_coordinates(y_min - (h / 2), img_h), get_coordinates(y_min + (h / 2), img_h)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
    return img


def get_coordinates(value_label, img_size):
    coordinate = int(value_label * img_size)
    if coordinate < 0:
        coordinate = 0
    if coordinate > img_size:
        coordinate = img_size
    return coordinate


def set_config():
    global config
    config = configparser.ConfigParser()
    config.add_section('yolov8')
    config.set('yolov8', '#perc_train', 'Percentage of images that we will use for training.')
    config.set('yolov8', 'perc_train', '70')
    config.set('yolov8', '#perc_valid', 'Percentage of images that we will use for validating.')
    config.set('yolov8', 'perc_valid', '20')
    config.set('yolov8', '#perc_valid', 'Percentage of images that we will use for testing.')
    config.set('yolov8', 'perc_test', '10')
    config.add_section('results')
    config.set('results', '#write_frame_video', 'To write a video frame image to disk')
    config.set('results', 'write_frame_video', 'True')
    config.set('results', '#write_annotation', 'To write a annotation bboxes to disk')
    config.set('results', 'write_annotation', 'True')
    config.add_section('video')
    config.set('video', '#wait_key 1', 'default value, video non stop')
    config.set('video', '#wait_key -1', 'wait for key')
    config.set('video', '#wait_key', 'Value of cv2.waitKey for the video output')
    config.set('video', 'wait_key', '1')
    config.add_section('video_result')
    config.set('video_result', '#show', 'Show the output video results')
    config.set('video_result', 'show', 'True')
    config.add_section('video_calculations')
    config.set('video_calculations', '#show', 'Show the output video calculations')
    config.set('video_calculations', 'show', 'False')
    config.set('video_calculations', '#show_next_bbox', 'Show bbox of next annotation')
    config.set('video_calculations', 'show_next_bbox', 'False')
    config.set('video_calculations', '#show_prev_bbox', 'Show bbox of previous annotation')
    config.set('video_calculations', 'show_prev_bbox', 'False')
    config.set('video_calculations', '#show_current_bbox', 'Show bbox of current annotation')
    config.set('video_calculations', 'show_current_bbox', 'True')
    config.add_section('ic')
    config.set('ic', '#disable', 'Disable output results')
    config.set('ic', 'disable', 'True')
    # write the configuration to a file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


def get_config():
    global config
    config = configparser.ConfigParser()
    config.read('config.ini')


if __name__ == '__main__':
    set_config()
    get_config()
    # convert(r"C:\Python\Dtasets\Pigeons",
    #         r"C:\Python\label_studio_video_json_to_yolov8_frame_by_frame\Pigeons006.json",
    #         r"C:\Python\label_studio_video_json_to_yolov8_frame_by_frame\yolo",
    #         r"/content/drive/MyDrive/Datasets/")
    convert(r"C:\Python\Dtasets\Pigeons",
            r"C:\Python\label_studio_video_json_to_yolov8_frame_by_frame\project-4-at-2023-03-31-07-42-b307dc76.json",
            r"C:\Python\GDrive\Datasets\yolo",
            r"/content/drive/MyDrive/Datasets/")
