import argparse
import json
import os
import functools
import pandas as pd
import soundfile
from tqdm import tqdm

from utils.utils import download, unpack
from utils.utils import add_arguments, print_arguments


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("dataset_path", default="dataset/cv-corpus-17.0/zh-CN", type=str, help="存放音频文件夹clip所在目录")
add_arg("annotation_path", default="dataset", type=str, help="存放音频标注文件的目录")
add_arg("language", default="zh-CN", type=str, help="语种")
args = parser.parse_args()


def create_annotation_text(dataset_path, annotation_path, language):
    print('Create Aishell annotation text ...')
    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)
    f_train = open(os.path.join(annotation_path, f'train_{language}.json'), 'w', encoding='utf-8')
    f_test = open(os.path.join(annotation_path, f'test_{language}.json'), 'w', encoding='utf-8')

    df_validated = pd.read_csv(os.path.join(dataset_path, 'validated.tsv'), sep='\t')
    df_test = pd.read_csv(os.path.join(dataset_path, 'test.tsv'), sep='\t')
    df_train = df_validated[~df_validated['path'].isin(df_test['path'])]

    # 训练集
    df_train_dict = dict(zip(df_train['path'], df_train['sentence']))
    lines = []
    for file in df_train_dict:
        line = {"audio": {"path": os.path.join(dataset_path, 'clips', file)}, "sentence": df_train_dict[file]}
        lines.append(line)
    # 添加音频时长
    for i in tqdm(range(len(lines))):
        audio_path = lines[i]['audio']['path']
        sample, sr = soundfile.read(audio_path)
        duration = round(sample.shape[-1] / float(sr), 2)
        lines[i]["duration"] = duration
        lines[i]["sentences"] = [{"start": 0, "end": duration, "text": lines[i]["sentence"]}]
    for line in lines:
        f_train.write(json.dumps(line, ensure_ascii=False) + "\n")
    # 测试集
    df_test_dict = dict(zip(df_test['path'], df_test['sentence']))
    lines = []
    for file in df_test_dict:
        line = {"audio": {"path": os.path.join(dataset_path, 'clips', file)}, "sentence": df_test_dict[file]}
        lines.append(line)
    # 添加音频时长
    for i in tqdm(range(len(lines))):
        audio_path = lines[i]['audio']['path']
        sample, sr = soundfile.read(audio_path)
        duration = round(sample.shape[-1] / float(sr), 2)
        lines[i]["duration"] = duration
        lines[i]["sentences"] = [{"start": 0, "end": duration, "text": lines[i]["sentence"]}]
    for line in lines:
        f_test.write(json.dumps(line,  ensure_ascii=False)+"\n")
    f_test.close()
    f_train.close()


def prepare_dataset(dataset_path, annotation_path, language):
    create_annotation_text(dataset_path, annotation_path, language)


def main():
    print_arguments(args)
    prepare_dataset(dataset_path=args.dataset_path,
                    annotation_path=args.annotation_path,
                    language=args.language)


if __name__ == '__main__':
    main()
