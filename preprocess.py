"""
Fer2013 Dataset

The Fer2013 dataset contains 35,887 grayscale images of faces with 48*48 pixels.
7 categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral

https://jovian.ai/himani007/logistic-regression-fer
"""
from config import *

print("Preprocess starts")


class FileReader:
    def __init__(self, csv_file_name):
        self._csv_file_name = csv_file_name

    def read(self):
        self._data = pd.read_csv(self._csv_file_name)


file_reader = FileReader(FER_PATH)
file_reader.read()
columns = file_reader._data.columns.values
classes = sorted(file_reader._data['emotion'].unique())

# distribution = file_reader._data.groupby('Usage')['emotion'].value_counts().to_dict()


class FER2013Dataset(Dataset):
    """FER2013 Dataset"""

    def __init__(self, X, Y, transform=None):
        """
        Args:
            X (np array): Nx1x32x32.
            Y (np array): Nx1.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self._X = X
        self._y = Y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        """
        if self.transform:
            return {'inputs': self.transform(self._X[idx]), 'labels': self._y[idx]}
        return {'inputs': self._X[idx], 'labels': self._y[idx]}
        """
        # one_hot_encoding = [torch.tensor([1.0,0.0,0.0,0.0,0.0,0.0,0.0]),torch.tensor([0.0,1.0,0.0,0.0,0.0,0.0,0.0]),torch.tensor([0.0,0.0,1.0,0.0,0.0,0.0,0.0]),torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,0.0]),torch.tensor([0.0,0.0,0.0,0.0,1.0,0.0,0.0]),torch.tensor([0.0,0.0,0.0,0.0,0.0,1.0,0.0]),torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,1.0])]
        if self.transform:
            return self.transform(torch.from_numpy(self._X[idx]).float()), self._y[idx]
        return torch.from_numpy(self._X[idx]).float(), self._y[idx]


class Data:
    """
        Initialize the Data utility.
        :param data:
                    a pandas DataFrame containing data from the
                    FER2013 dataset.
        :type file_path:
                    DataFrame
        class variables:
        _x_train, _y_train:
                    Training data and corresopnding labels
        _x_test, _y_test:
                    Testing data and corresopnding labels
        _x_valid, _y_validation:
                    Validation/Development data and corresopnding labels

    """

    def __init__(self, data):
        self._x_train, self._y_train = [], []
        self._x_test, self._y_test = [], []
        self._x_valid, self._y_valid = [], []

        for xdx, x in enumerate(tqdm(data.values)):
            pixels = []
            label = None
            for idx, i in enumerate(x[1].split(' ')):
                pixels.append(int(i))
            pixels = np.array(pixels).reshape((1, 48, 48))

            if x[2] == 'Training':
                self._x_train.append(pixels)
                self._y_train.append(int(x[0]))
            elif x[2] == 'PublicTest':
                self._x_test.append(pixels)
                self._y_test.append(int(x[0]))
            else:
                self._x_valid.append(pixels)
                self._y_valid.append(int(x[0]))
        self._x_train, self._y_train = np.array(self._x_train).reshape((len(self._x_train), 1, 48, 48)), \
                                       np.array(self._y_train, dtype=np.int64)
        self._x_test, self._y_test = np.array(self._x_test).reshape((len(self._x_test), 1, 48, 48)), \
                                     np.array(self._y_test, dtype=np.int64)
        self._x_valid, self._y_valid = np.array(self._x_valid).reshape((len(self._x_valid), 1, 48, 48)), \
                                       np.array(self._y_valid, dtype=np.int64)


data = Data(file_reader._data)

preprocess = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(6),
    transforms.ColorJitter()
])

train_set = FER2013Dataset(np.asarray(data._x_train, dtype=np.single), data._y_train, transform=preprocess)
test_set = FER2013Dataset(np.asarray(data._x_valid, dtype=np.single), data._y_valid)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

print("Preprocess finished")
