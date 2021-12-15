import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os
import torch.nn.functional as F
import torch.optim as optim
import h5py
import sys
import pandas as pd


USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'


def channel2sudoku(input):
    """
    Converts a 9x9x9 1/0 Sudoku input into a 9x9 1~9 Sudoku puzzle
    :param input: 9x9x9 1/0 Tensor
    :return: 9x9 1~9 Tensor Sudoku puzzle
    """
    output = torch.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            for channel in range(9):
                if input[channel, i, j]:
                    output[i, j] = channel + 1

    return output


def sudoku2channel(input):
    """
    Converts a 9x9 1~9 Sudoku puzzle into a 9x9x9 1/0 channel for training
    :param input: 9x9 1~9 Tensor
    :return: 9x9x9 1/0 Tensor
    """
    output = torch.zeros((9, 9, 9))
    for i in range(9):
        for j in range(9):
            output[input[i, j] - 1, i, j] = 1
    return output


def get_data():
    data = np.zeros((1000000, 2, 81), np.int32)
    for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
        quiz, solution = line.split(",")
        if i > 100:
            break
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            data[i, 0, j] = q
            data[i, 1, j] = s

    data = data.reshape((-1, 2, 1, 9, 9))
    train_data = data[:30, :, :, :]
    test_data = data[30:, :, :, :]
    #
    print(train_data[10, 0, :, :])
    print(train_data[10, 1, :, :])

    return train_data, test_data
