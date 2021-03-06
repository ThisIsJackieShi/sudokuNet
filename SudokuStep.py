from sudoku_utils import *



class SudokuStepNet(nn.Module):
    def __init__(self):
        super(SudokuStepNet, self).__init__()
        self.conv1 = nn.Conv2d(9, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 9, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(9 * 9 * 9, 9 * 9 * 9)
        self.accuracy = None

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x

    def loss(self, x, label, mask):
        x = F.softmax(x, dim=1)
        x = x * mask
        chosen = torch.argmax(x, dim=1)
        loss = torch.tensor(0., device=DEVICE)

        for n in range(x.size(dim=0)):
            k = chosen[n] // 81
            pos = chosen[n] % 81
            if label[n, pos] + 1 != k:
                loss += x[n, chosen[n]]
            else:
                loss += (1 - x[n, chosen[n]])

        return loss, chosen

    def chosen(self, x, mask):
        x = F.softmax(x, dim=1)
        x = x * mask
        return torch.argmax(x, dim=1)


def train_step(net, dataloader, epochs=1, lr=0.01, momentum=0.9, decay=0.0, verbose=1, limit=None):
    net.to(DEVICE)
    losses = []
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    for epoch in range(epochs):
        sum_loss = 0.0
        for i, batch in enumerate(dataloader, 0):
            if limit and i >= limit:
                break

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            used_mask = torch.ones(size=(inputs.size(dim=0), 9 * 9 * 9), dtype=torch.float16, device=DEVICE)

            for _ in range(9 * 9):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)

                # mask selected portions
                loss, chosen = net.loss(outputs, labels, used_mask)

                loss.backward()
                optimizer.step()

                for n in range(inputs.size(dim=0)):
                    chosen_num = chosen[n] // 81 + 1
                    chosen_pos = chosen[n] % 81
                    for num in range(9):
                        used_mask[n, chosen_pos + num * 81] = 0
                    inputs[n, int(labels[n, chosen_pos]) - 1, chosen_pos // 9, chosen_pos % 9] = 1

                print(loss.item())
                # print statistics
                losses.append(loss.item())
                sum_loss += loss.item()
            if i % 100 == 99 and verbose:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
    return losses


def test_step(model, test_loader, limit=None):
    model.eval()
    test_loss = 0
    correct_cnt = 0

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if limit and batch_idx >= limit:
                break

            data, label = data.to(DEVICE), label.to(DEVICE)

            used_mask = torch.ones(size=(data.size(dim=0), 9 * 9 * 9), dtype=torch.float16, device=DEVICE)

            for _ in range(9 * 9):
                x = model(data)
                chosen = model.chosen(x, used_mask)
                for n in range(data.size(dim=0)):
                    chosen_num = chosen[n] // 81
                    chosen_pos = chosen[n] % 81
                    chosen_i = chosen_pos // 9
                    chosen_j = chosen_pos % 9
                    for num in range(9):
                        if data[n, num, chosen_i, chosen_j]:
                            chosen_num = num
                    for num in range(9):
                        used_mask[n, chosen_pos + num * 81] = 0
                    data[n, chosen_num, chosen_pos // 9, chosen_pos % 9] = 1

            outputs = torch.zeros(label.size()).to(DEVICE)
            for n in range(data.size(dim=0)):
                for pos in range(9 * 9):
                    for chosen in range(9):
                        if data[n, chosen, pos // 9, pos % 9] == 1:
                            outputs[n, pos] = chosen + 1

            correct = outputs.eq(label)
            correct_cnt += correct.sum().item()

    test_accuracy = 100. * correct / len(test_loader.dataset) / 81
    print(f'accuracy: {test_accuracy}, correct: {correct_cnt} / {len(test_loader.dataset) * 81}')


class SudokuStepDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        feat = self.data[idx, 0, :, :]
        label = self.data[idx, 1, :, :]
        processed_feat = np.zeros((9, 9, 9))

        for i in range(9):
            for j in range(9):
                if feat[0, i, j]:
                    processed_feat[feat[0, i, j] - 1, i, j] = 1

        return (torch.from_numpy(processed_feat).float(), torch.from_numpy(label).float().flatten())


def main():
    train_data, test_data = get_data()

    train_dataset = SudokuStepDataset(train_data)
    test_dataset = SudokuStepDataset(test_data)
    print(train_dataset[10])
    train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=2)

    net = SudokuStepNet()
    net.load_state_dict(torch.load('step.pt'))

    net.to(DEVICE)
    train_step(net, train_data_loader)
    torch.save(net.state_dict())
    test_step(net, test_data_loader)


if __name__ == '__main__':
    main()
