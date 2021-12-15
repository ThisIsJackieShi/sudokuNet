# CSE 490G1 Final Project - SudokuNet

Jiajie Shi & Yuan Wang
https://docs.google.com/document/d/1tsHxEA_iEM9FBZ9OcbB-oPmc2vU8F7VHZRUm2XDIwkY/edit?usp=sharing


## Abstract

Sudoku is a famous game where you are given a 9x9 grid puzzle with some initial numbers and you need to fill out the blanks so that each row, column and 3x3 sub areas in the puzzle only have the number 1 to 9 appearing once within it. The general N scaled Sudoku is thought to be NP-complete(1). 

That’s where we lay our initial inspiration for this project. The default method for Sudoku solving algorithm uses backtracking to try out every possibility. We want to know how well we can approximate the result of these NP-complete problems using neural networks and deep learning in a polynomial runtime. 

In this project, we trained multiple convolutional/step-by-step neural networks to solve Sudoku puzzles. For the convolutional network, we considered solving the entire Sudoku puzzle at once. For the step-by-step approach, we want the network to solve the puzzle in a progressive manner like humans, making one most confident decision at a time.

## Problem statement

We analyzed the Sudoku puzzle and how neural networks can be used to solve it. We encountered a few challenges as stated below. Some high-level target related concepts are:

- we will try to solve the 9x9 standard Sudoku puzzle; 
- we will use training to increase the accuracy instead of to construct an exact solution; 
- and we will use different architectures to solve the problem. 

The first challenge is that NP problems are often without a fixed length of input, while the standard Sudoku is only constant 9x9. Finding standard Sudoku dataset is quite easy while it seems that a generalized Sudoku puzzle dataset is hard to find. Also, it is hard to train neural networks without a fixed size input and attain a good result. For the scale of this class’s final project, we thus decided to just get our feet wet in attempting to train a neural network on the standard Sudoku puzzles and see if and how it works well or not.

The second challenge is that neural networks often give the most probable solution instead of the concrete solution. This means at best we can only approximate the exact solution, and we will almost definitely incur some inaccuracy. We think what we can do is to train the neural network to minimize this error. 

The third challenge is that multiple architectures make sense in solving the Sudoku puzzle. Convolutional networks are a natural approach as the Sudoku puzzle resembles images; fully connected layers also make sense since we do need information about the numbers in the same column, row, and block; lastly, we also want to try if a step-by-step human-like approach can yield better results. Thus, we will try out different possibilities as listed here. 

## Related work

We are inspired by Verma’s approach to solve Sudoku puzzles using neural networks [3]. 

We also did some research to see that Sudoku puzzles are actually NP-complete [1]. 

Lastly, We found a dataset from Kaggle [2]. It has 1 million puzzles and solutions generated by algorithms. They are stored as strings of numbers from the first grid left to right and top to bottom to the last where 0 represents blanks in a csv file. We used this dataset to train and to test our model. 

## Methodology 

### Dataset

We used the dataset from Kaggle [2]. We converted the csv file to numpy and with our customized data loader, and eventually to a 1/0 Pytorch tensor that is a 9x9x9 matrix, where each channel represents one solution layer. For example, if for (i, j) in matrix A we have channel 3 marked as 1 and all other channels as 0, the Sudoku puzzle has 3 at (i, j) (See graph below). We will explain more into why we decide to represent the data this way later. 

The Kaggle dataset contains 1,000,000 Sudoku puzzles and solutions. We used the first 200,000 of puzzles and solutions for training and the following 10,000 for testing (since training on the entire dataset is too slow).

<img src="dataset.jpg" alt="dataset" style="zoom: 67%;" />

### Network Structure

We designed three types of networks.

The first kind is some simple convolutional networks that use some convolutional layers, and potentially with a fully connected layer at the beginning or the end. This network solves the entire Sudoku puzzle by only looking once. It takes a 9x9x9 input, and produce a 9x9x9 output, representing the value assigned to each number at each position (i, j). For training, we will calculate the 2D cross entropy loss for each cell with respect to the 9x9 label. For testing, we will take the argmax for output[:, i, j] to determine which number has been picked by the neural network for (i, j). 

The second kind is a simple fully connected network. It flattens the 9x9x9 input to some hidden layers, then produces the 9x9x9 output, representing the value assigned to each number at position (i, j). Similarly, this network also only looks at the input once. The training and testing methods are similar to that of the convolutional network.

The third kind is a step-by-step convolutional network. It takes a 9x9x9 input and produces a 729 vector, representing the value assigned to each number of each position. However, the loss calculation, the training, and the testing are quite different from that of the origin. For loss calculation, it is calculating whether the output number matches the solution at (i, j). For training, the loss calculation is repeated for 9*9 times. After each position has been picked, the mask is changed so that the position (i, j) is not considered in following iterations. The correct value at (i, j) is updated into the input, and the input is feed into the network again. For testing, a similar process is used. The main difference is that since there is no correct value at (i, j), the proposed value at (i, j) is updated into the input. See graph below.

<img src="network_structure_step.jpg" alt="network structure" style="zoom: 33%;" />

### Tuning

After laying out the structure for converting the data to Pytorch tensors, training, and evaluating, we fine-tuned the meta-parameters to a reasonable place for us to explore different structures of the neural network. We trained our network with 5 epochs for the scale of this project. The first epoch is set to have a learning rate of 0.1, momentum of 0.9 and decay of 0.0(that is the same for all epochs). We realized that because of the unique logic of solving Sudoku, the network tends to not pick up how to tone itself to get close at all at the beginning and thus we need a very aggressive learning rate at the beginning. We tried 0.2 at the beginning but as our architecture of the network grew, that became too unstable and we settled with 0.1 at the end. The second and third epochs have a learning rate of 0.02 and momentum of 0.9. This is the main training stage. The last two epochs have a learning rate of 0.01 and momentum of 0.5 to fine tune the network.

## Experiments/evaluation 

For evaluation we just used the cross entropy method. Our network outputs a 9 channel 9x9 matrix and it is compared to the solution of the same size where I explained earlier that each grid has one channel set to 1 to represent what number it should be. We choose the 9x9x9 input, output and label size because in Sudoku numbers actually are indistinct. A “9” is not “bigger” than a “1”. They only mark the exclusiveness of other digits. Changing 9 numbers to 9 colors and you will still play the same standard Sudoku. For this reason we think that 9x9x9 represents the essence of the data better than just the 9x9 and each matrix grid has a value representing the number. We tried to use the input as a 9x9 and the result was worse as expected and we ever since conformed to our current 9x9x9 model.

For testing, since we have the 9 channels to represent different numbers to fill in, we just do argmax to find what number has the biggest chance to be the correct number for this grid. We do this both to the label and the output and compare the accuracy. Therefore the accuracy we defined is the accuracy of each grid being correct.

## Results

We experimented with different structures of the neural network by changing the ordering of the fully connected layer and the convolutional layer. 

The worst result we got from these experiments is with only convolutional layers. The convolutional layers include 4 layers: from 9 to 16, 16 to 32, 32 to 16, 16 to 9 channels and each of them have kernel size of 3 and stride of 1. The choice of kernel size is related to the subarea part of Sudoku puzzles. These convolutional layers setup is the same for all four of our network structures. For only using the convolutional layer, the test accuracy ends up with 64%. It is expected for this model to be the lowest as it didn’t account the dependency of rows and columns in Sudoku but only the 9-grid subareas.

Our model that performed the best is one fully connected layer from 9x9x9 to 9x9x9 and then the convolutional layers which have a test accuracy as large as 86%. This is because the model allows both the calculation of the dependency of rows and columns and 9-grid subareas. Interestingly, compared to a different model where we had convolutional layers first and then the fully connected layer, which is a more standard structure for usage of convolutional layers, this model with fully connected layer first is more accurate by 3%. The model with convolutional layers then fully connected layers has a test accuracy of 83%.
This also makes sense because we should already processed our Sudoku matrix so we include the information of rows and columns dependency before we use convolutional layers that can account for the subarea dependency, but also lose some information of the relationships within rows and columns.

We also tried the fully connected layers by itself but repeated 3 times. This one gives an accuracy of 72% which is still much better than the convolutional layers as fully connected layers can still learn subarea dependency of Sudoku. It is still not as good as using both kinds as convolutional layers are more specified in subarea dependency and are more efficient.

As of the step-by-step network, the training is very slow. We eventually trained the network on 10,000 puzzles and got an accuracy of 48%. We suspect that this is because the network has not been fully trained. Unfortunately, we do not have a loss graph of this training. 


FC 999 to 999, Conv 9-16-32-16-9: 86%

![unnamed (1)](https://user-images.githubusercontent.com/47728497/146096424-0252fcbd-0776-429f-a10f-095ded3feb28.png)

Conv 9-16-32-16-9: 64%

![unnamed (2)](https://user-images.githubusercontent.com/47728497/146096425-ee32d052-afe1-4a97-b6be-17f7521e2881.png)

Conv 9-16-32-16-9, FC 999 to 999: 83%

![unnamed (3)](https://user-images.githubusercontent.com/47728497/146096426-fed3e550-dfe6-42b8-9940-dcac85523815.png)

FC999 x3: 72%

![unnamed (4)](https://user-images.githubusercontent.com/47728497/146096427-d5a64b34-f784-4c36-909c-1bdd1c89ca9f.png)



## Examples 

**Convolutional Network with Fully Connected Layer**

Input:

<img src="example_conv_input.jpg" alt="image-20211214203815887" style="zoom:67%;" />

Output:

<img src="example_conv_output.jpg" alt="image-20211214204041332" style="zoom:67%;" />

Solution:

<img src="example_conv_solution.jpg" alt="image-20211214203859956" style="zoom:67%;" />

**Step-by-Step Network**

Input:

<img src="example_step_input.jpg" alt="image-20211214192456583" style="zoom:67%;" />

Output:

<img src="example_step_output.jpg" alt="image-20211214192616637" style="zoom:67%;" />

Solution:

<img src="example_step_solution.jpg" alt="image-20211214192537108" style="zoom:67%;" />

## Video

a 2-3 minute long video where you explain your project and the above information

## References

1. Haythorpe M. Reducing the generalised Sudoku problem to the Hamiltonian cycle problem. *AKCE International Journal of Graphs and Combinatorics.* 2016;13(3):272-282. doi:10.1016/j.akcej.2016.10.001
2. Park K. 1 million Sudoku games. Kaggle. https://www.kaggle.com/bryanpark/sudoku. Published December 29, 2016. Accessed December 14, 2021. 
3. Verma, Shiva. “Solving Sudoku with Convolution Neural Network: Keras.” *Solving Sudoku with Convolution Neural Network | Keras*, Towards Data Science, 5 Oct. 2021, towardsdatascience.com/solving-sudoku-with-convolution-neural-network-keras-655ba4be3b11. 

