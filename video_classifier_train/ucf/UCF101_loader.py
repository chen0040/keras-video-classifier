import urllib.request
import os
import sys
import rarfile
import numpy as np

DATA_DIR_PATH = '../very_large_data'
UFC101_MODEL = DATA_DIR_PATH + "/medium.avi"
URL_LINK = 'http://crcv.ucf.edu/data/UCF101/UCF101.rar'


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_ucf():
    ucf_rar = DATA_DIR_PATH + '/UCF101.rar'

    if not os.path.exists(DATA_DIR_PATH):
        os.makedirs(DATA_DIR_PATH)

    if not os.path.exists(ucf_rar):
        print('ucf file does not exist, downloading from internet')
        urllib.request.urlretrieve(url=URL_LINK, filename=ucf_rar,
                                   reporthook=reporthook)

    print('unzipping ucf file')
    rar_ref = rarfile.RarFile(ucf_rar, 'r')
    rar_ref.extractall(DATA_DIR_PATH)
    rar_ref.close()


def load_ucf():
    if not os.path.exists(UFC101_MODEL):
        download_ucf()


class UCF101(object):

    def __init__(self):
        pass


def main():
    download_ucf()


if __name__ == '__main__':
    main()