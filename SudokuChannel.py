from sudoku_utils import *

class sudokuDatasetChannel(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        feat = self.data[idx, 0, :, :]
        label = self.data[idx, 1, :, :]
        processed_label = np.zeros((9, 9, 9))
        processed_feat = np.zeros((9, 9, 9))

        for i in range(9):
            for j in range(9):
                processed_label[label[0, i, j] - 1, i, j] = 1
                if feat[0, i, j]:
                    processed_feat[feat[0, i, j] - 1, i, j] = 1

        return (torch.from_numpy(processed_feat).float(), torch.from_numpy(processed_label).float())


class sudokuNetChannel(nn.Module):
    def __init__(self):
        super(sudokuNetChannel, self).__init__()
        self.conv1 = nn.Conv2d(9, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 9, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(9 * 9 * 9, 9 * 9 * 9)
        self.accuracy = None

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = x.reshape((-1, 9, 9, 9))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        return x

    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction)
        return loss_val


def train(net, dataloader, epochs=1, lr=0.01, momentum=0.9, decay=0.0, verbose=1):
    net.to(DEVICE)
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    for epoch in range(epochs):
        sum_loss = 0.0
        for i, batch in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            losses.append(loss.item())
            sum_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                if verbose:
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
    return losses


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    blank_cnt = 0

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)

            '''
            blank_mask = torch.zeros([output.shape[0], 1, 9, 9]).floats()
            for k in range(output.shape[0]):
              for i in range(9):
                for j in range(9):
                  is_blank = True
                  for n in range(9):
                    if data[k, n, i, j] != 0:
                      is_blank = False
                      break;
                  if is_blank:
                    blank_mask[k, 0, i, k] = 1.0
                    blank_cnt += 1
            '''

            x = torch.zeros([output.shape[0], 1, 9, 9])
            for k in range(output.shape[0]):
                for i in range(9):
                    for j in range(9):
                        x[k, 0, i, j] = torch.argmax(output[k, :, i, j]) + 1

            y = torch.zeros([label.shape[0], 1, 9, 9])
            for k in range(label.shape[0]):
                for i in range(9):
                    for j in range(9):
                        y[k, 0, i, j] = torch.argmax(label[k, :, i, j]) + 1

            correct_mask = x.eq(y.view_as(x))

            num_correct = (correct_mask).sum().item()

            correct += num_correct

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset) / 81
    # test_accuracy = 100. * correct / blank_cnt

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset) * 81, test_accuracy))
    return test_loss, test_accuracy



def main():

    train_data, test_data = get_data()

    train_loader_channel = DataLoader(sudokuDatasetChannel(train_data), batch_size=16, shuffle=True)
    test_loader_channel = DataLoader(sudokuDatasetChannel(test_data), batch_size=1)
    print(train_loader_channel.dataset[0])

    print('Using device', DEVICE)
    model = sudokuNetChannel()
    model.load_state_dict(torch.load('sudokuNetChannel.pt'))
    model.to(DEVICE)

    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, momentum=0.9)
    # train_loss = []
    # train_loss += train(model, train_loader_channel, lr=0.1)
    # train_loss += train(model, train_loader_channel, lr=0.05)
    # train_loss += train(model, train_loader_channel, lr=0.05)
    # train_loss += train(model, train_loader_channel, lr=0.01, momentum=0.5)
    # train_loss += train(model, train_loader_channel, lr=0.01, momentum=0.5)


    # torch.save(model.state_dict(), "/gdrive/MyDrive/490g1/sudokuNetChannel.pt")

    test(model, test_loader_channel)


if __name__ == '__main__':
    main()