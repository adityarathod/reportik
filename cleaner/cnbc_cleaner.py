import os, re


def is_hidden_file(name: str):
    return name.startswith('.')


def create_separate_data():
    for data_type in ['documents', 'summaries']:
        files = open(data_type + '.txt', 'a+')
        for file in os.listdir('./' + data_type):
            if is_hidden_file(file):
                continue
            with open(os.path.join('./' + data_type, file)) as f:
                print('processing', f.name)
                contents = ' <NEWLINE> '.join(f.read().split('\n'))
                contents = re.sub(r'\.(?=[^ \W\d])', '. ', contents)
                files.write(contents + '\n')
    files.close()


def create_unified_data():
    files = open('dataset.txt', 'a+')
    for file in os.listdir('./documents'):
        if is_hidden_file(file):
            continue
        with open(os.path.join('./documents', file)) as doc:
            print('processing', file)
            with open(os.path.join('./summaries', file)) as summ:
                doc_contents = ' <NEWLINE> '.join(doc.read().split('\n'))
                doc_contents = re.sub(r'\.(?=[^ \W\d])', '. ', doc_contents)
                summ_contents = ' <NEWLINE> '.join(summ.read().split('\n'))
                summ_contents = re.sub(r'\.(?=[^ \W\d])', '. ', summ_contents)
                files.write(doc_contents + ('\t' * 5) + summ_contents)
    files.close()


if __name__ == '__main__':
    create_unified_data()
import os
import re


def is_hidden_file(name: str):
    return name.startswith('.')


def unify_files(base_folder='../data', subfolders=['documents', 'summaries']):
    for data_type in subfolders:
        files = open(data_type + '.txt', 'a+')
        for file in os.listdir(base_folder + data_type):
            if is_hidden_file(file):
                continue
            with open(os.path.join(base_folder + data_type, file)) as f:
                print('processing', f.name)
                contents = ' <NEWLINE> '.join(f.read().split('\n'))
                contents = re.sub(r'\.(?=[^ \W\d])', '. ', contents)
                files.write(contents + '\n')
        files.close()


if __name__ == '__main__':
    unify_files()
