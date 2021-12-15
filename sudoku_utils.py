import torch

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