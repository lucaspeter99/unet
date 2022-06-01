import torch
import nibabel as nib
import os
from torch.utils.data import Dataset
from torch.distributions import Uniform, Normal
from scipy.ndimage import zoom
from numpy import random, expand_dims
import numpy





class NiftyDataset(Dataset):
    in_channels = None
    out_channels = None
    size = 50
    features = 48
    mean_std = {'Colon': {'mean': -508.939852322062, 'std': 498.74947146061044},
                }

    def __init__(self, root, alpha=0.2, sigma=0.1, training = True , rotate=True, deform=True, set_norm=True, verbose=True, onlyRotDefSpecials = False , specials = [], start = 0):
        self.items = []
        self.alpha = alpha
        self.sigma = sigma
        self.current_id = 0
        self.root = root
        self.rotate = rotate
        self.deform = deform
        self.set_norm = set_norm
        self.verbose = verbose
        self.specials = specials
        self.onlyRotDefSpecials = onlyRotDefSpecials
        self.training = training
        self._walk(start)

    def _walk(self, start):
        self.current_id = 0
        if self.training:
            x_path = os.path.join(self.root, "imagesTr")
            y_path = os.path.join(self.root, "labelsTr")
            files = os.listdir(x_path)
            files = [i for i in files if "_colon" not in i]
            for file in files[start:]:
                x_data_path = os.path.join(x_path, file)
                y_data_path = os.path.join(y_path, file)
                if "_colon" not in file:
                    self._load_dataset(x_data_path, y_data_path, self.alpha, self.sigma, self.rotate, self.deform)
                    self.current_id += 1
        else:
            x_path = os.path.join(self.root, "imagesTs")
            files = os.listdir(x_path)
            files = [i for i in files if "_colon" not in i]
            for file in files[start:]:
                if "_colon" not in file:
                    self._load_dataset(os.path.join(x_path, file), None, self.alpha, self.sigma, self.rotate, self.deform)
                    self.current_id += 1


    def _load_dataset(self, x_path: str, y_path: str, alpha: float, sigma: float, rotate: bool, deform: bool) -> None:
        raise NotImplementedError('Subclasses of NiftyDataset must implement _load_dataset().')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        # data = self.items[item]
        # data = self._rotate(data) if self.rotate else data
        # data = self._deform(data, alpha=self.alpha, sigma=self.sigma) if self.deform else data
        # return data
        return self.items[item]

    def _load(self, path):
        if self.verbose:
            print(path)
        data = nib.load(path).get_fdata()
        data[numpy.isnan(data)] = 0.0
        if len(data.shape) == 3:
            data = expand_dims(data, 3)
        shape = data.shape
        data = zoom(data, (self.size / shape[0], self.size / shape[1], self.size / shape[2], 1.0))
        return torch.tensor(data)[1:-1, 1:-1, 1:-1].permute(3, 0, 1, 2)

    def _append_item(self, x, y):
        self.items.append((x, y))
        if self.in_channels is None and hasattr(x, 'shape'):
            self.in_channels = x.shape[0]
        if self.out_channels is None and hasattr(y, 'shape'):
            self.out_channels = y.shape[0]

    def _deform(self, data, alpha, sigma):  # a.k.a. elastic transform
        f = self.features
        k = 5  # kernsel size
        uni = Uniform(low=-1.0, high=1.0).sample((1, 1, f, f, f))
        gaussian = Normal(loc=0.0, scale=sigma).sample((1, 1, k, k, k))
        displacement = torch.conv3d(uni, gaussian, padding=2, stride=1).reshape(1, f, f, f)  # convolve and reshape to correct output
        displacement = torch.mul(displacement.double(), alpha)  # scaling factor
        x = torch.add(data[0], displacement)
        return x, data[1]

    def stats(self):
        for i in range(len(self.items)):
            x, y = self.items[i]
            print(f'x of item {i} - mean: {x.mean()} std: {x.std()} min: {x.min()} max: {x.max()}')

    def normalize(self, data, pattern=None):  # normalize values to (0, 1)
        data = torch.tensor(data) if type(data) is not torch.Tensor else data

        # normalize data (standard score)
        mean = self.mean_std[pattern]['mean'] if pattern and self.set_norm else data.mean()
        std = self.mean_std[pattern]['std'] if pattern and self.set_norm else data.std()
        data = (data - mean) / std

        # transform data to (0, 1)
        min = data.min()
        max = data.max()
        data = (data - min) / (max - min)

        return data

    @staticmethod
    def _get_filename(path, pattern):
        for filename in os.listdir(path):
            pattern = pattern.replace('Resliced', 'Reslised')  # adjust for 'Reslised'-typo in filenames
            if pattern in filename:
                return os.path.join(path, filename)
        raise RuntimeError(f'Filename not found. \npath:{path}\npattern: {pattern}')

    @staticmethod
    def _rotate(data):
        turns = random.randint(0, 4)
        rot_dims = random.choice([1, 2, 3], 2, replace=False).tolist()  # replace=False prevents duplicates
        flip_dims = random.choice([1, 2, 3], 2, replace=False).tolist()
        return tuple(item.rot90(turns, rot_dims).flip(flip_dims) for item in data)

    @staticmethod
    def contains_nan(data):
        return bool(torch.isnan(data).any())

    @staticmethod
    def substitute_nan(data, value):  # in place!
        data[torch.isnan(data)] = value


def info(file):
    print(f'min: {file.min()} max: {file.max()} mean: {file.mean()}')


class Colon(NiftyDataset):

    def _load_dataset(self, x_path, y_path, alpha, sigma, rotate, deform):
        x = self._load(x_path)
        x = self.normalize(x, 'Colon')
        if y_path is not None:
            y = self._load(y_path)
        else:
            y = torch.zeros(x.shape)
        self._append_item(x, y)
        if rotate:
            (x_prime, y_prime) = self._rotate((x, y))
            self._append_item(x_prime, y_prime)
        if deform:
            (x_prime, y_prime) = self._deform((x, y), alpha=alpha, sigma=sigma)
            self._append_item(x_prime, y_prime)
        if deform and rotate:
            (x_prime, y_prime) = self._rotate((x, y))
            (x_prime, y_prime) = self._deform((x_prime, y_prime), alpha=alpha, sigma=sigma)
            self._append_item(x_prime, y_prime)




