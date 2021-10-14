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
    mean_std = {'NormalizedCT-A': {'mean': 51.77077879286174, 'std': 546.2797761573629},
                'NormalizedCT-N': {'mean': 21.371641051425293, 'std': 474.9535209654172},
                'NormalizedCT-P_CBF': {'mean': 33.32211903417765, 'std': 73.49196844364432},
                'NormalizedCT-P_CBV': {'mean': 36.55955152858375, 'std': 82.69093995223194},
                'NormalizedCT-P_Tmax': {'mean': 0.46827028390412445, 'std': 1.0379624833429706},
                'NormalizedLesion': {'mean': 0.017087332687265363, 'std': 0.09865254668591036},
                'ReslicedCT-A': {'mean': -346.58975009333517, 'std': 609.1569996430978},
                'ReslicedCT-N': {'mean': -277.30447384156696, 'std': 533.8605525109376},
                'ReslicedCT-P_CBF': {'mean': 13.547389079882384, 'std': 50.063475027883655},
                'ReslicedCT-P_CBV': {'mean': 14.811534005216009, 'std': 55.86918868839639},
                'ReslicedCT-P_Tmax': {'mean': 0.18934590876457733, 'std': 0.7000650897555616},
                'ReslicedLesion': {'mean': 0.00664131865894179, 'std': 0.06290848792331691}
                }

    def __init__(self, root, alpha=0.2, sigma=0.1, rotate=True, deform=True, set_norm=True, verbose=True, onlyRotDefSpecials = False , specials = [], start = 0):
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

        self._walk(start)

    def _walk(self, start):
        self.current_id = 0
        for path in os.listdir(self.root)[start:]:
            path = os.path.join(self.root, path)
            if os.path.isdir(path):
                print(self.current_id)
                if self.onlyRotDefSpecials:
                    if self.current_id in self.specials:
                        self._load_dataset(path, alpha=self.alpha, sigma=self.sigma, rotate=True, deform=True)
                        print('special detected')
                    else:
                        self._load_dataset(path, alpha=self.alpha, sigma=self.sigma, rotate=False, deform=False)
                else:
                    self._load_dataset(path, alpha=self.alpha, sigma=self.sigma, rotate=self.rotate, deform=self.deform)
                self.current_id += 1

    def _load_dataset(self, path: str, alpha: float, sigma: float, rotate: bool, deform: bool) -> None:
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


# Normalized

class NormalizedAngioNative(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x1 = self._load(self._get_filename(path, 'NormalizedCT-A'))
        x1 = self.normalize(x1, 'NormalizedCT-A')
        x2 = self._load(self._get_filename(path, 'NormalizedCT-N'))
        x2 = self.normalize(x2, 'NormalizedCT-N')
        x = torch.cat((x1, x2), 0)
        y = self._load(self._get_filename(path, 'NormalizedLesion'))
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


class NormalizedPerfusionCBV(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x = self._load(self._get_filename(path, 'NormalizedCT-P_CBV'))
        x = self.normalize(x, 'NormalizedCT-P_CBV')
        y = self._load(self._get_filename(path, 'NormalizedLesion'))
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


class NormalizedPerfusionCBF(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x = self._load(self._get_filename(path, 'NormalizedCT-P_CBF'))
        x = self.normalize(x, 'NormalizedCT-P_CBF')
        y = self._load(self._get_filename(path, 'NormalizedLesion'))
        self._append_item(x,y)
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


class NormalizedPerfusionTmax(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x = self._load(self._get_filename(path, 'NormalizedCT-P_Tmax'))
        x = self.normalize(x, 'NormalizedCT-P_Tmax')
        y = self._load(self._get_filename(path, 'NormalizedLesion'))
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


class NormalizedPerfusionAll(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x1 = self._load(self._get_filename(path, 'NormalizedCT-P_CBV'))
        x1 = self.normalize(x1, 'NormalizedCT-P_CBV')
        x2 = self._load(self._get_filename(path, 'NormalizedCT-P_CBF'))
        x2 = self.normalize(x2, 'NormalizedCT-P_CBF')
        x3 = self._load(self._get_filename(path, 'NormalizedCT-P_Tmax'))
        x3 = self.normalize(x3, 'NormalizedCT-P_Tmax')
        x = torch.cat((x1, x2, x3), 0)
        y = self._load(self._get_filename(path, 'NormalizedLesion'))
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


class NormalizedAllModalities(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x1 = self._load(self._get_filename(path, 'NormalizedCT-A'))
        x1 = self.normalize(x1, 'NormalizedCT-A')
        x2 = self._load(self._get_filename(path, 'NormalizedCT-N'))
        x2 = self.normalize(x2, 'NormalizedCT-N')
        x3 = self._load(self._get_filename(path, 'NormalizedCT-P_CBV'))
        x3 = self.normalize(x3, 'NormalizedCT-P_CBV')
        x4 = self._load(self._get_filename(path, 'NormalizedCT-P_CBF'))
        x4 = self.normalize(x4, 'NormalizedCT-P_CBF')
        x5 = self._load(self._get_filename(path, 'NormalizedCT-P_Tmax'))
        x5 = self.normalize(x5, 'NormalizedCT-P_Tmax')
        x = torch.cat((x1, x2, x3, x4, x5), 0)
        y = self._load(self._get_filename(path, 'NormalizedLesion'))
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

class NormalizedAngioNativeTmax(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x1 = self._load(self._get_filename(path, 'NormalizedCT-A'))
        x1 = self.normalize(x1, 'NormalizedCT-A')
        x2 = self._load(self._get_filename(path, 'NormalizedCT-N'))
        x2 = self.normalize(x2, 'NormalizedCT-N')
        x3 = self._load(self._get_filename(path, 'NormalizedCT-P_Tmax'))
        x3 = self.normalize(x3, 'NormalizedCT-P_Tmax')
        x = torch.cat((x1, x2, x3), 0)
        y = self._load(self._get_filename(path, 'NormalizedLesion'))
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


# Resliced

class ReslicedAngioNative(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x1 = self._load(self._get_filename(path, 'ReslicedCT-A'))
        x1 = self.normalize(x1, 'ReslicedCT-A')
        x2 = self._load(self._get_filename(path, 'ReslicedCT-N'))
        x2 = self.normalize(x2, 'ReslicedCT-N')
        x = torch.cat((x1, x2), 0)
        y = self._load(self._get_filename(path, 'ReslicedLesion'))
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


class ReslicedPerfusionCBV(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x = self._load(self._get_filename(path, 'ReslicedCT-P_CBV'))
        x = self.normalize(x, 'ReslicedCT-P_CBV')
        y = self._load(self._get_filename(path, 'ReslicedLesion'))
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


class ReslicedPerfusionCBF(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x = self._load(self._get_filename(path, 'ReslicedCT-P_CBF'))
        x = self.normalize(x, 'ReslicedCT-P_CBF')
        y = self._load(self._get_filename(path, 'ReslicedLesion'))
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


class ReslicedPerfusionTmax(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x = self._load(self._get_filename(path, 'ReslicedCT-P_Tmax'))
        x = self.normalize(x, 'ReslicedCT-P_Tmax')
        y = self._load(self._get_filename(path, 'ReslicedLesion'))
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


class ReslicedPerfusionAll(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x1 = self._load(self._get_filename(path, 'ReslicedCT-P_CBV'))
        x1 = self.normalize(x1, 'ReslicedCT-P_CBV')
        x2 = self._load(self._get_filename(path, 'ReslicedCT-P_CBF'))
        x2 = self.normalize(x2, 'ReslicedCT-P_CBF')
        x3 = self._load(self._get_filename(path, 'ReslicedCT-P_Tmax'))
        x3 = self.normalize(x3, 'ReslicedCT-P_Tmax')
        x = torch.cat((x1, x2, x3), 0)
        y = self._load(self._get_filename(path, 'ReslicedLesion'))
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


class ReslicedAllModalities(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x1 = self._load(self._get_filename(path, 'ReslicedCT-A'))
        x1 = self.normalize(x1, 'ReslicedCT-A')
        x2 = self._load(self._get_filename(path, 'ReslicedCT-N'))
        x2 = self.normalize(x2, 'ReslicedCT-N')
        x3 = self._load(self._get_filename(path, 'ReslicedCT-P_CBV'))
        x3 = self.normalize(x3, 'ReslicedCT-P_CBV')
        x4 = self._load(self._get_filename(path, 'ReslicedCT-P_CBF'))
        x4 = self.normalize(x4, 'ReslicedCT-P_CBF')
        x5 = self._load(self._get_filename(path, 'ReslicedCT-P_Tmax'))
        x5 = self.normalize(x5, 'ReslicedCT-P_Tmax')
        x = torch.cat((x1, x2, x3, x4, x5), 0)
        y = self._load(self._get_filename(path, 'ReslicedLesion'))
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

class ReslicedAngioNativeTmax(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x1 = self._load(self._get_filename(path, 'ReslicedCT-A'))
        x1 = self.normalize(x1, 'ReslicedCT-A')
        x2 = self._load(self._get_filename(path, 'ReslicedCT-N'))
        x2 = self.normalize(x2, 'ReslicedCT-N')
        x3 = self._load(self._get_filename(path, 'ReslicedCT-P_Tmax'))
        x3 = self.normalize(x3, 'ReslicedCT-P_Tmax')
        x = torch.cat((x1, x2, x3), 0)
        y = self._load(self._get_filename(path, 'ReslicedLesion'))
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

