import torch
import nibabel as nib
import os

from skimage.transform import resize
from torch.utils.data import Dataset
from torch.distributions import Uniform, Normal
from scipy.ndimage import zoom
from numpy import random, expand_dims
import numpy
import pandas as pd

# data set implementation, mainly from carlo
class NiftyDataset(Dataset):
    in_channels = None
    out_channels = None
    dimensionality = None
    size = 50
    features = 48
    # this was for an old way to normalize data i think
    # mean_std = {'NormalizedCT-A': {'mean': 51.77077879286174, 'std': 546.2797761573629},
    #             'NormalizedCT-N': {'mean': 21.371641051425293, 'std': 474.9535209654172},
    #             'NormalizedCT-P_CBF': {'mean': 33.32211903417765, 'std': 73.49196844364432},
    #             'NormalizedCT-P_CBV': {'mean': 36.55955152858375, 'std': 82.69093995223194},
    #             'NormalizedCT-P_Tmax': {'mean': 0.46827028390412445, 'std': 1.0379624833429706},
    #             'NormalizedLesion': {'mean': 0.017087332687265363, 'std': 0.09865254668591036},
    #             'ReslicedCT-A': {'mean': -346.58975009333517, 'std': 609.1569996430978},
    #             'ReslicedCT-N': {'mean': -277.30447384156696, 'std': 533.8605525109376},
    #             'ReslicedCT-P_CBF': {'mean': 13.547389079882384, 'std': 50.063475027883655},
    #             'ReslicedCT-P_CBV': {'mean': 14.811534005216009, 'std': 55.86918868839639},
    #             'ReslicedCT-P_Tmax': {'mean': 0.18934590876457733, 'std': 0.7000650897555616},
    #             'ReslicedLesion': {'mean': 0.00664131865894179, 'std': 0.06290848792331691}
    #             }

    #init function for class
    def __init__(self, root, alpha=0.2, sigma=0.1, rotate=True, deform=True, set_norm=True, verbose=True, use_medical_data = True, onlyRotDefSpecials = False , specials = [], start = 0, load_old_lesions = True, use_windowing = True):
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
        self.use_mecial_data = use_medical_data
        self.load_old_lesions = load_old_lesions
        self.use_windowing = use_windowing
        if use_medical_data:
            #load demographics files
            if 'll337mdee' in os.getcwd():
                df = pd.read_excel('./lesions/demographics/demographics_1.xlsx')
                df2 = pd.read_excel('./lesions/demographics/demographics_2.xlsx')
                df3 = pd.read_excel('./lesions/demographics/demographics_3.xlsx')
            else:
                df = pd.read_excel('./demographics/demographics_1.xlsx')
                df2 = pd.read_excel('./demographics/demographics_2.xlsx')
                df3 = pd.read_excel('./demographics/demographics_3.xlsx')
            df = pd.concat([df, df2, df3]) # combine dataframes
            df.set_index("patients_id", inplace=True)
            #convert data to proper formats and types, normalize relevant columns
            #age
            df['age_at_onset'] = self._normalize_column(df['age_at_onset'])
            #nihss
            df['nihss_sum_initial'] = self._normalize_column(df['nihss_sum_initial'])

            #time to ct
            df['lsw_to_ct'] = df['lsw_to_ct'].astype('str')
            df['lsw_to_ct'] = df['lsw_to_ct'].apply(self._get_seconds)
            df['lsw_to_ct'] = self._normalize_column(df['lsw_to_ct'])

            #tici scale: convert to succesful(1), if 2b to 3 else 0
            df['TICI'] = df['TICI'].astype('str')
            df['TICI'] = df['TICI'].replace(['0', '1', '2a'], 0)
            df['TICI'] = df['TICI'].replace(['2b', '2c', '3'], 1)


            #convert char to binary
            df['sex'] = df['sex'].replace('m', 1)
            df['sex'] = df['sex'].replace('f', 0)
            self.demographics = df

        #load dataset
        self._walk(start)

    def _walk(self, start):
        self.current_id = 0
        for patient in os.listdir(self.root)[start:]: # for every patient from startpoint onwards (startpoint is mainly for testing with smaller datasets)
            path = os.path.join(self.root, patient)
            print(patient)
            if os.path.isdir(path): #if path exists
                print(self.current_id)
                if self.onlyRotDefSpecials: # for data augmentation, not used currently
                    if self.current_id in self.specials:
                        self._load_dataset(path, alpha=self.alpha, sigma=self.sigma, rotate=True, deform=True)
                        print('special detected')
                    else:
                        self._load_dataset(path, alpha=self.alpha, sigma=self.sigma, rotate=False, deform=False)
                else:
                    self._load_dataset(path, alpha=self.alpha, sigma=self.sigma, rotate=self.rotate, deform=self.deform) #load patient
                self.current_id += 1

    def _load_dataset(self, path: str, alpha: float, sigma: float, rotate: bool, deform: bool) -> None: #implemented for each subclass
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
            print(path.encode('utf-8', 'replace').decode()) # encode and then decode to fix path to utf8 conversion error in python
        data = nib.load(path).get_fdata()
        data[numpy.isnan(data)] = 0.0
        shape = data.shape
        if self.dimensionality is None:
            self.dimensionality = shape
        elif self.dimensionality != shape:
            print("mismatch in dim of dataset")
        data = torch.tensor(resize(data, (self.size, self.size , self.size ), order=0)) #resize image
        data = torch.flip(data, dims = [0]) #flip one dimension
        data = torch.unsqueeze(data[1:-1, 1:-1, 1:-1], dim=0) #shave off edge and add zeroed out dimension in front (for combinding different modalities, as they are concatenated in this dim)

        return data

    def _load_med_data(self, patient): # load tensor with the med data in them
        #load relevant data from dataframe
        age = self.demographics.loc[int(patient)]['age_at_onset']
        nihss = self.demographics.loc[int(patient)]['nihss_sum_initial']
        time_to_ct = self.demographics.loc[int(patient)]['lsw_to_ct']
        tici = self.demographics.loc[int(patient)]['TICI']
        sex = self.demographics.loc[int(patient)]['sex']
        data = [tici, age, nihss, time_to_ct, sex]
        #create tensors and concat
        x_med_data = torch.full([1,48,48,48], float(tici))
        for item in data[-4:]:
            new_x_med_data = torch.full([1,48,48,48,], float(item))
            x_med_data = torch.cat((x_med_data, new_x_med_data), 0)
        return x_med_data

    def _append_item(self, x, y, id):
        self.items.append((x, y, id))
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
            x, y, _ = self.items[i]
            print(f'x of item {i} - mean: {x.mean()} std: {x.std()} min: {x.min()} max: {x.max()}')

    def normalize(self, data, pattern=None):  # normalize values to (0, 1)
        data = torch.tensor(data) if type(data) is not torch.Tensor else data

        # # normalize data (standard score)
        # # mean = self.mean_std[pattern]['mean'] if pattern and self.set_norm else data.mean()
        # # std = self.mean_std[pattern]['std'] if pattern and self.set_norm else data.std()
        # mean = data.mean()
        # std = data.std()
        # data = (data - mean) / std

        # transform data to (0, 1)
        min = data.min()
        max = data.max()
        data = (data - min) / (max - min)

        return data

    #windowing
    def window(self, data, type):

        if 'CT-A' in type:
            data[data > 350] = 350
            data[data < 0] = 0
        elif 'CT-N' in type:
            data[data > 70] = 70
            data[data < 0] = 0
        elif 'CT-P_CBV' in type:
            data[data < 0] = 0
        elif 'CT-P_CBF' in type:
            data[data < 0] = 0
        elif 'CT-P_Tmax' in type:
            data[data < 0] = 0
        else:
            print("Error applying windowing function!")



        return data

    #load ground truths
    def loadys(self, dir, pattern):
        # combine ground truths, form 2/3s consesus
        paths = self._get_filenames(dir, pattern)
        y = None
        for path in paths:
            y_ = self._load(path)
            y_ = (y_ > 0.5).float() # this tensor consists of 0s and 1s
            if y is not None:
                y = torch.add(y, y_) # add two tensors element wise
            else:
                y = y_
        if y is None:
            print("Couldn't load all ys correctly")
        else:
            y = (y > 1.5).float() # 2s and 3s should be 1, 0 and 1 should be 0, as 2/3 consensus is expected
        return y

    @staticmethod
    def _normalize_column(column):
        column = (column - column.mean()) / column.std()
        column = (column - column.min()) / (column.max() - column.min())
        return column

    @staticmethod
    def _get_seconds(time):
        time = str(time)
        if '1900-01-01' in time:
            print(time)
            time = time[10:]
        array = time.split(":")
        seconds = int(array[0])*60*60 + int(array[1])*60 + int(array[2])
        return seconds

    @staticmethod
    def _get_filename(path, pattern):
        for filename in os.listdir(path):
            pattern = pattern.replace('Resliced', 'Reslized')  # adjust for 'Reslised'-typo in filenames
            if pattern in filename:
                return os.path.join(path, filename)
        raise RuntimeError(f'Filename not found. \npath:{path}\npattern: {pattern}')

    @staticmethod
    def _get_filenames(path, pattern, load_old_infarct = False):
        paths = []
        try:
            for filename in os.listdir(path):
                pattern = pattern.replace('Resliced', 'Reslized')  # adjust for 'Reslised'-typo in filenames
                if pattern in filename:
                    if load_old_infarct:
                        if "_sd2.nii" in filename:
                            paths.append(os.path.join(path, filename))
                    else:
                        if "_sd2.nii" not in filename:
                            paths.append(os.path.join(path, filename))
            return paths
        except:
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


#TODO: only ReslicedAllModalities implementation is currently complete

# Normalized

class NormalizedAngioNative(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x1 = self._load(self._get_filename(path, 'NormalizedCT-A'))
        x1 = self.normalize(x1, 'NormalizedCT-A')
        x2 = self._load(self._get_filename(path, 'NormalizedCT-N'))
        x2 = self.normalize(x2, 'NormalizedCT-N')
        x = torch.cat((x1, x2), 0)
        # load all y's
        y = self.loadys(path, 'NormalizedLesion')
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
        # load all y's
        y = self.loadys(path, 'NormalizedLesion')
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
        # load all y's
        y = self.loadys(path, 'NormalizedLesion')
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
        # load all y's
        if self.use_mecial_data:
            patient = path.split('/')[-1]
            x_med_data = self._load_med_data(patient).double()
            x = torch.cat((x_med_data, x), 0)
        # load all y's
        print("--------------------------------------------")
        y = self.loadys(path, 'ReslicedLesion')
        print("--------------------------------------------")
        id = path.split('/')[-1]
        self._append_item(x, y, id)
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
        # load all y's
        y = self.loadys(path, 'NormalizedLesion')
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
        # x1 = self._load(self._get_filename(path, 'NormalizedCT-A'))
        # x1 = self.normalize(x1, 'NormalizedCT-A')
        # x2 = self._load(self._get_filename(path, 'NormalizedCT-N'))
        # x2 = self.normalize(x2, 'NormalizedCT-N')
        # x3 = self._load(self._get_filename(path, 'NormalizedCT-P_CBV'))
        # x3 = self.normalize(x3, 'NormalizedCT-P_CBV')
        # x4 = self._load(self._get_filename(path, 'NormalizedCT-P_CBF'))
        # x4 = self.normalize(x4, 'NormalizedCT-P_CBF')
        # x5 = self._load(self._get_filename(path, 'NormalizedCT-P_Tmax'))
        # x5 = self.normalize(x5, 'NormalizedCT-P_Tmax')
        # x = torch.cat((x1, x2, x3, x4, x5), 0)
        # # load all y's
        # y = self.loadys(path, 'NormalizedLesion')
        # self._append_item(x, y)
        # if rotate:
        #     (x_prime, y_prime) = self._rotate((x, y))
        #     self._append_item(x_prime, y_prime)
        # if deform:
        #     (x_prime, y_prime) = self._deform((x, y), alpha=alpha, sigma=sigma)
        #     self._append_item(x_prime, y_prime)
        # if deform and rotate:
        #     (x_prime, y_prime) = self._rotate((x, y))
        #     (x_prime, y_prime) = self._deform((x_prime, y_prime), alpha=alpha, sigma=sigma)
        #     self._append_item(x_prime, y_prime)
        modalities = ['NormalizedCT-A', 'NormalizedCT-N', 'NormalizedCT-P_CBV', 'NormalizedCT-P_CBF', 'NormalizedCT-P_Tmax']
        data = []
        old_infarct_present = False
        # consider old infarcts
        if self.load_old_lesions:
            old_infarcts = self._get_filenames(path, 'NormalizedLesion',
                                               load_old_infarct=True)  # get filenames of all old infarcts
            if len(old_infarcts) > 0:  # if old infarct present save it
                print('Adding old infarct...')
                old_infarct = self._load(old_infarcts[0])
                old_infarct = (old_infarct > 0.5).float()
                old_infarct_present = True

        for mod in modalities:  # load each modality
            x_ = self._load(self._get_filename(path, mod))
            if self.use_windowing:  # apply windowing
                x_ = self.window(x_, mod)
            x_ = self.normalize(x_, mod)  # normalize
            if old_infarct_present:  # masking if old infarct is present
                x_ = (x_ * (
                            old_infarct.int() * -1 + 1)).float()  # inverting binary mask and then multiplying element-wise
                # as old infarcts are saved as 1s, but for elementwise mult with mask these voxels need to be zero and the rest 1
            data.append(x_)  # append to dataset

        x = torch.cat(data, 0).double()
        if self.use_mecial_data:  # add med data too
            patient = path.split('/')[-1]
            x_med_data = self._load_med_data(patient).double()
            x = torch.cat((x_med_data, x), 0)
        # load all y's
        print("--------------------------------------------")
        y = self.loadys(path, 'NormalizedLesion')
        # masking if old infarct is present
        if old_infarct_present:
            y = (y * (old_infarct.int() * -1 + 1)).float()  # inverting binary mask and then multiplying element-wise
        print("--------------------------------------------")
        id = path.split('/')[-1]
        self._append_item(x, y, id)

class NormalizedAngioNativeTmax(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x1 = self._load(self._get_filename(path, 'NormalizedCT-A'))
        x1 = self.normalize(x1, 'NormalizedCT-A')
        x2 = self._load(self._get_filename(path, 'NormalizedCT-N'))
        x2 = self.normalize(x2, 'NormalizedCT-N')
        x3 = self._load(self._get_filename(path, 'NormalizedCT-P_Tmax'))
        x3 = self.normalize(x3, 'NormalizedCT-P_Tmax')
        x = torch.cat((x1, x2, x3), 0)
        # load all y's
        y = self.loadys(path, 'ReslicedLesion')
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
        # load all y's
        y = self.loadys(path, 'ReslicedLesion')
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
        # load all y's
        y = self.loadys(path, 'ReslicedLesion')
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
        # load all y's
        y = self.loadys(path, 'ReslicedLesion')
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
        # load all y's
        y = self.loadys(path, 'ReslicedLesion')
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
        # load all y's
        y = self.loadys(path, 'ReslicedLesion')
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
        modalities = ['ReslicedCT-A', 'ReslicedCT-N','ReslicedCT-P_CBV', 'ReslicedCT-P_CBF','ReslicedCT-P_Tmax']
        data = []
        old_infarct_present = False
        # consider old infarcts
        if self.load_old_lesions:
            old_infarcts = self._get_filenames(path, 'ReslicedLesion', load_old_infarct=True) # get filenames of all old infarcts
            if len(old_infarcts) > 0: # if old infarct present save it
                print('Adding old infarct...')
                old_infarct = self._load(old_infarcts[0])
                old_infarct = (old_infarct > 0.5).float()
                old_infarct_present = True

        for mod in modalities: # load each modality
            x_ = self._load(self._get_filename(path, mod))
            if self.use_windowing: #apply windowing
                x_ = self.window(x_, mod)
            x_ = self.normalize(x_, mod) #normalize
            if old_infarct_present:#masking if old infarct is present
                x_ = (x_ * (old_infarct.int()*-1+1)).float()    # inverting binary mask and then multiplying element-wise
                                                                # as old infarcts are saved as 1s, but for elementwise mult with mask these voxels need to be zero and the rest 1
            data.append(x_) #append to dataset

        x = torch.cat(data,0).double()
        if self.use_mecial_data: # add med data too
            patient = path.split('/')[-1]
            x_med_data = self._load_med_data(patient).double()
            x = torch.cat((x_med_data, x), 0)
        # load all y's
        print("--------------------------------------------")
        y = self.loadys(path, 'ReslicedLesion')
        # masking if old infarct is present
        if old_infarct_present:
            y = (y * (old_infarct.int() * -1 + 1)).float()  # inverting binary mask and then multiplying element-wise
        print("--------------------------------------------")
        id = path.split('/')[-1]
        self._append_item(x, y, id)

class ReslicedAngioNativeTmax(NiftyDataset):

    def _load_dataset(self, path, alpha, sigma, rotate, deform):
        x1 = self._load(self._get_filename(path, 'ReslicedCT-A'))
        x1 = self.normalize(x1, 'ReslicedCT-A')
        x2 = self._load(self._get_filename(path, 'ReslicedCT-N'))
        x2 = self.normalize(x2, 'ReslicedCT-N')
        x3 = self._load(self._get_filename(path, 'ReslicedCT-P_Tmax'))
        x3 = self.normalize(x3, 'ReslicedCT-P_Tmax')
        x = torch.cat((x1, x2, x3), 0)
        #load all y's
        y = self.loadys(path, 'ReslicedLesion')
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