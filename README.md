# CSE 490G1 Final Project

Jiajie Shi & Yuan Wang
https://docs.google.com/document/d/1tsHxEA_iEM9FBZ9OcbB-oPmc2vU8F7VHZRUm2XDIwkY/edit?usp=sharing


## Abstract

Sudoku puzzle is a NP-complete problem. The default method for Sudoku solving algorithm uses backtracking to try out every possibility, which has exponential runtime. We want to know how well can we approach the result of these NP-complete problems using neural networks and deep learning in a polynomial runtime. 

In this project, we trained multiple convolutional/step-by-step neural network to solve Sudoku puzzles. 

We want to train a convolutional / LSTM network to solve Sudoku puzzles. The default method for Sudoku solving algorithm uses backtracking to try out every possibility. We want the network to solve the puzzle in a progressive manner, making one decision at a time without going back.

## Problem statement

Sudoku is a famous game where you are given a 9x9 grid puzzle with some initial numbers and you need to fill out the blanks so that each row, column and 3x3 sub areas in the puzzle only have the number 1 to 9 appearing once within it. 

The general N scaled Sudoku is thought to be NP-complete(1). That’s where we lay our initial inspiration for this project. We want to see what we can do when trying to solve NP problems using a neural network. The challenge is that NP problems are often without a fixed length of input, while the standard Sudoku is only constant 9x9. Finding standard Sudoku dataset is quite easy while it seems that a generalized Sudoku puzzle dataset is hard to find. Also, it is hard to train neural networks without a fixed size input and attain a good result. For the scale of this class’s final project, we thus decided to just get our feet wet in attempting to train a neural network on the standard Sudoku puzzles and see if and how it works well or not.

## Related work

We found a dataset from kaggle(2). It has 1 million puzzles and solutions generated by algorithms. They are stored as strings of numbers from the first grid left to right and top to bottom to the last where 0 represents blanks in a csv file. We then converted to numpy and with our customized data loader, eventually to pytorch tensors that are 9 channels of 9x9 matrix where each channel represents one solution layer. For example, if for a grid A we have channel 3 marked as 1 and all other channels as 0, the Sudoku puzzle or solution for this grid is the number 3. We will explain more into why we decide to represent the data this way later. We used the first 800 thousands of puzzles and solutions for training and the last 200 thousands for testing.

## Methodology 

After laying out the structure for converting the data to pytorch tensors, training, and evaluating, we fine-tuned the meta-parameters to a reasonable place for us to explore different structures of the neural network. We trained our network with 5 epochs for the scale of this project. The first epoch is set to have a learning rate of 0.1, momentum of 0.9 and decay of 0.0(that is the same for all epochs). We realized that because of the unique logic of solving Sudoku, the network tends to not pick up how to tone itself to get close at all at the beginning and thus we need a very aggressive learning rate at the beginning. We tried 0.2 at the beginning but as our architecture of the network grew, that became too unstable and we settled with 0.1 at the end. The second and third epochs have a learning rate of 0.02 and momentum of 0.9. This is the main training stage. The last two epochs have a learning rate of 0.01 and momentum of 0.5 to fine tune the network.

## Experiments/evaluation 

For evaluation we just used the cross entropy method. Our network outputs a 9 channel 9x9 matrix and it is compared to the solution of the same size where I explained earlier that each grid has one channel set to 1 to represent what number it should be. We choose the 9x9x9 input, output and label size because in Sudoku numbers actually are indistinct. A “9” is not “bigger” than a “1”. They only mark the exclusiveness of other digits. Changing 9 numbers to 9 colors and you will still play the same standard Sudoku. For this reason we think that 9x9x9 represents the essence of the data better than just the 9x9 and each matrix grid has a value representing the number. We tried to use the input as a 9x9 and the result was worse as expected and we ever since conformed to our current 9x9x9 model.

For testing, since we have the 9 channels to represent different numbers to fill in, we just do argmax to find what number has the biggest chance to be the correct number for this grid. We do this both to the label and the output and compare the accuracy. Therefore the accuracy we defined is the accuracy of each grid being correct.

## Results 

![alt text](https://keep.google.com/u/0/media/v2/1pLIEYxLpjFq8Uj5k2Reju-dhHvVL3gGOP4d0YWpHF9Zo2oxzdYcpFxlbdqFRN58/1r2niSi2sGrZhNsyuDLlSN0_E8VCszjfiL7H4pk2EaVZ6F3caXsJiaP06iZWiiA?sz=512&accept=image%2Fgif%2Cimage%2Fjpeg%2Cimage%2Fjpg%2Cimage%2Fpng%2Cimage%2Fwebp)

## Examples 

images/text/live demo, anything to show off your work (note, demos get some extra credit in the rubric)

## Video

a 2-3 minute long video where you explain your project and the above information
