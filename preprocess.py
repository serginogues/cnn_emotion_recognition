"""
Fer2013 Dataset
https://jovian.ai/himani007/logistic-regression-fer
"""
from config import *

data_df = pd.read_csv(FER_PATH)

Labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}
train_df = data_df[data_df['Usage'] == 'Training']
valid_df = data_df[data_df['Usage'] == 'PublicTest'].reset_index(drop=True)
test_df = data_df[data_df['Usage'] == 'PrivateTest'].reset_index(drop=True)


def show_image(df, idx):
    print('expression: ', df.iloc[idx])
    image = np.array([[int(i) for i in x.split()] for x in df.loc[idx, ['pixels']]])
    print(image.shape)
    image = image.reshape(48, 48)
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.show()


# show_image(train_df, 101)


class expressions(Dataset):
    def __init__(self, df, transforms_=None):
        self.df = df
        self.transforms = transforms_

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        image, label = np.array([x.split() for x in self.df.loc[index, ['pixels']]]), row['emotion']
        # image = image.reshape(1,48,48)
        image = np.asarray(image).astype(np.uint8).reshape(48, 48, 1)
        # image = np.reshape(image,(1,48,48))

        if self.transforms:
            image = self.transforms(image)

        return image.clone().detach(), label

    # return torch.tensor(image,dtype = torch.float), label


stats = ([0.5], [0.5])
train_tsfm = transforms.Compose([
    transforms.ToPILImage(),
    # T.RandomHorizontalFlip(), #--> only required to prevent over fitting
    # T.RandomRotation(10),     #-->   "  "
    transforms.ToTensor(),
    transforms.Normalize(*stats, inplace=True)
])
valid_tsfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(*stats, inplace=True)
])

train_ds = expressions(train_df, train_tsfm)
valid_ds = expressions(valid_df, valid_tsfm)
test_ds = expressions(test_df, valid_tsfm)

batch_size = 400
train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                      num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size * 2,
                      num_workers=2, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size * 2,
                     num_workers=2, pin_memory=True)

print("*******Preprocess finished******")
