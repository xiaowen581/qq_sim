import os


def write_file(data: list, path: str, mode: str):
    with open(path, encoding='utf-8', newline='\r\n', mode=mode) as f:
        f.writelines('\n'.join([item for item in data]) + '\n')


def file_list(file_dir):
    ret = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            ret.append(os.path.join(root, file))
    return ret


def read_file_as_str(path: str):
    with open(path, encoding='utf-8', mode='r') as f:
        data = f.readlines()
    return data

