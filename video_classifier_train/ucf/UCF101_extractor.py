import cv2
import os

MAX_NB_CLASSES = 4
INPUT_DATA_DIR_PATH = '../very_large_data/UCF-101'
OUTPUT_DATA_DIR_PATH = '../very_large_data/UCF-101-Frames'


def extract_images(video_input_file_path, image_output_dir_path):
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        cv2.imwrite(image_output_dir_path + os.path.sep + "frame%d.jpg" % count, image)  # save frame as JPEG file
        count = count + 1


def scan_and_extract_images():

    if not os.path.exists(OUTPUT_DATA_DIR_PATH):
        os.makedirs(OUTPUT_DATA_DIR_PATH)

    dir_count = 0
    for f in os.listdir(INPUT_DATA_DIR_PATH):
        file_path = INPUT_DATA_DIR_PATH + os.path.sep + f
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = OUTPUT_DATA_DIR_PATH + os.path.sep + output_dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                output_image_folder_path = output_dir_path + os.path.sep + ff.split('.')[0]
                if not os.path.exists(output_image_folder_path):
                    os.makedirs(output_image_folder_path)
                extract_images(video_file_path, output_image_folder_path)
        if dir_count == MAX_NB_CLASSES:
            break


def main():
    print(cv2.__version__)
    scan_and_extract_images()


if __name__ == '__main__':
    main()
