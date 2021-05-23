"""
CNN architecture with PyTorch
https://kaunild.github.io/experiments/experiments-1/
https://jovian.ai/himani007/logistic-regression-fer
"""
from preprocess import *
from architectures import *

input_size = 48 * 48


def accuracy(output, labels):
    predictions, preds = torch.max(output, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def get_model():
    if MODEL_NAME == 'letnet5':
        m = LeNet5()
    elif MODEL_NAME == 'simple':
        m = custom1()
    else:
        m = custom2()
    return m.to(device)


model = get_model()
criterion = nn.CrossEntropyLoss()  # loss function
#TODO: check which optimizer works best
LR = 0.001
#optimizer = torch.optim.SGD(model.parameters(), lr=LR)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#optimizer = optim.Adadelta(model.parameters(), lr=LR, rho=0.95, eps=1e-08)


def train():
    loss_history = []
    acc_history = []
    running_loss = 0.0
    for epoch in range(NUM_EPOCHS):
        print('Epoch ', epoch + 1, 'of ', NUM_EPOCHS)

        # TRAIN MODEL
        for i, data in enumerate(train_loader, 0):
            # get the training data
            X, label = data
            X, label = X.to(device), label.to(device)

            # forward pass
            pred = model.forward(X)

            # backward pass
            loss = criterion(pred, label)
            # print current loss
            running_loss += float(loss.item())
            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called.
            optimizer.zero_grad()
            loss.backward()

            # optimize with backprop
            optimizer.step()
            del data,X,label

        # STORE CURRENT ACCURACY
        _pred = torch.tensor([])
        _label = torch.tensor([])
        _J = 0
        for X, label in test_loader:
            # calculate the accuracy on test set and print
            X, label = X.to(device), label.to(device)
            pred = model.forward(X)
            _pred = torch.cat((_pred.to(device), pred))
            _label = torch.cat((_label.to(device), label))
            _J += float(criterion(pred, label).item())

        print(' Loss: ', _J)
        loss_history.append(_J)
        print(' Accuracy: ', accuracy(_pred, _label).item())
        acc_history.append(accuracy(_pred, _label).item())

    # SAVE THE MODEL
    torch.save(model.state_dict(), SAVE_PATH)
    print("Training finished")

    # PLOT ACCURACY
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(loss_history)
    plt.subplot(1, 2, 2)
    plt.plot(acc_history)
    plt.title('Accuracy')
    plt.show()


def test():
    with torch.no_grad():
        model.load_state_dict(torch.load(SAVE_PATH))
        model.eval()
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model.forward(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            if labels.size()[0] == BATCH_SIZE:
                for i in range(BATCH_SIZE):
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(10):
            n_correct = n_class_correct[i]
            n_samples = n_class_samples[i]
            if n_correct != 0:
                acc = 100.0 * n_correct / n_samples
                print(f'Accuracy of {classes[i]}: {acc} %')


def visualize_filter(model):
    """
    :param kernel: model.conv1[0]
    nn.Conv2d(3, 1, 3).weight.data.clone()
    :return:
    """
    def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
        """
        https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorchhttps://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch
        """
        n,c,w,h = tensor.shape

        if allkernels: tensor = tensor.view(n*c, -1, w, h)
        elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

        rows = np.min((tensor.shape[0] // nrow + 1, 64))
        grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
        plt.figure( figsize=(nrow,rows) )
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

    kernels = model.conv1[0].weight.data.clone()
    print(kernels.shape)
    visTensor(kernels, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    plt.show()


def print_architecture(model):
    [print(x) for x in model.children()]
