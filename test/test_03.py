# ！不要修改这里的内容
import os
import sys
import random

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from _03_Convolution_Neural_Network.Convolution_Neural_Network import main, read_data

_, val_dataset, _, _ = read_data()
model = main()
model = model.to('cpu')

def test():
    inputs, label = random.choice(val_dataset)
    inputs = inputs.view(1, -1)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    return label, predicted.item()

def test0():
    label, pred = test()
    assert label == pred

def test1():
    label, pred = test()
    assert label == pred

def test2():
    label, pred = test()
    assert label == pred

def test3():
    label, pred = test()
    assert label == pred

def test4():
    label, pred = test()
    assert label == pred

def test5():
    label, pred = test()
    assert label == pred

def test6():
    label, pred = test()
    assert label == pred

def test7():
    label, pred = test()
    assert label == pred

def test8():
    label, pred = test()
    assert label == pred

def test9():
    label, pred = test()
    assert label == pred