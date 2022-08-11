from __future__ import unicode_literals, print_function, division
import os
import subprocess
import zipfile
import glob
from io import open
import unicodedata
import string
import torch

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def find_files(path):
    return glob.glob(path)

def unicode_to_ascii(s: str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

def read_lines(filename):
    """ read a file and split into lines """
    file = open(filename, encoding='utf-8')
    lines = file.read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

def letter_to_index(letter: str):
    """ find letter index from all_letters, e.g. 'a' = 0 """
    return all_letters.find(letter)

def letter_to_tensor(letter: str):
    """ just for demonstration, turn a letter into a <1 x n_letters> Tensor """
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line: str):
    """ turn a line into a <line_length x 1 x n_letters>, or an array of one-hot letter vectors """
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor
    

if os.path.exists('./data'):
    print('./data/ already exists')
else:
    subprocess.run("wget https://download.pytorch.org/tutorial/data.zip", shell=True, check=True)
    with zipfile.ZipFile("./data.zip") as zipfile:
        zipfile.extractall(".")

# print(find_files('data/names/*.txt'))
# print(unicode_to_ascii('Ślusàrski'))

# build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

print('--- example of Italian ---')
print(category_lines['Italian'][:5])