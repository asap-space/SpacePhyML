import numpy as np
from torch import flatten
import sys

class Normalize():
    def __init__ (self, mean, std):
        self.mean = mean
        self.std = std

    def __call__ (self, sample):
        return (((sample[0] - self.mean)/self.std), *sample[1:])

class Threshold():
    def __init__ (self, thresholds=None):
        self.thresholds = thresholds

    def __call__ (self, sample):
        x = sample[0]

        threshold_low_index = np.where( x < self.thresholds[0])
        x[threshold_low_index] = self.thresholds[0]

        threshold_high_index = np.where( x > self.thresholds[1])
        x[threshold_high_index] = self.thresholds[1]

        return (x, *sample[1:])

class LogNorm():
    def __init__ (self, normalization=None):
        self.normalization = normalization
        if normalization:
            self.threshold = np.power(10.0, normalization)

    def __call__ (self, sample):
        x = sample[0]
        # Replace zeros with lowest or threshold value.
        if self.normalization is None:
            zero_index = np.where( x == 0)
            min_value =  x[np.where( x>0)].min()
            x[zero_index] = min_value
        else:
            threshold_low_index = np.where( x < self.threshold[0])
            x[threshold_low_index] = self.threshold[0]

            threshold_high_index = np.where( x > self.threshold[1])
            x[threshold_high_index] = self.threshold[1]

        #Take log10 of data
        x = np.log10( x)
        if self.normalization is None:
            x -=  x.min()
            x /=  x.max()
        else:
            x -= self.normalization[0]
            x /= (self.normalization[1] - self.normalization[0])

        return (x, *sample[1:])
class Flatten():
    """
    Filter for flattening the data
    """
    def __call__ (self, sample):
        return (sample[0].reshape(-1), *sample[1:])

class Log10():
    """
    Filter for taking the log10 of all non zero values in an array.
    """
    def __call__ (self, sample):
        non_zero_indexes = np.where(sample[0]!=0)

        #Talk log10 of all non_zero values
        sample[0][non_zero_indexes] = np.log10(sample[0][non_zero_indexes])
        return sample


class Roll():
    def __init__ (self, shift = 16, axis = -3):
        self.shift = shift
        self. axis = axis

    def __call__ (self, sample):
        x = sample[0]

        #Roll along Phi
        x = np.roll( x, self.shift, axis=self.axis)

        return (x, *sample[1:])

class MoveAxis():
    def __init__ (self, src = -1, dst = -3):
        self.src = src
        self.dst = dst

    def __call__ (self, sample):
        return (np.moveaxis(sample[0],self.src,self.dst),
                *sample[1:])

class Sum3D():
    def __call__(self, sample):

        x = sample[0]
        x = np.concatenate((np.sum(x, axis=(1,2)),
                            np.sum(x, axis=(0,2)),
                            np.sum(x, axis=(0,1))))

        return (x, *sample[1:])

class Sum():
    def __init__ (self, axis = -1):
        self.axis = axis

    def __call__ (self, sample):
        x = sample[0]

        x = np.sum(x, axis=self.axis)

        return (x, *sample[1:])

class Mean():
    def __init__ (self, axis = -1):
        self.axis = axis

    def __call__ (self, sample):
        x = sample[0]

        x = np.mean(x, axis=self.axis)

        return (x, *sample[1:])



class Transform_seq():
    def __init__ (self, transforms):
        self.transforms = transforms

    def __call__ (self, sample):
        for trans in self.transforms:
            sample = trans(sample)
        return sample



def to_tensor(sample):
    return (torch.from_numpy(sample[0]),
           torch.tensor(sample[1]).long())

def get_data_conv_func(dtype):

    if dtype == torch.float32:
        return lambda x: to_float(torch.float32, x)

    if dtype == torch.float16:
        return lambda x: to_float(torch.float16, x)

    if dtype == torch.bfloat16:
        return lambda x: to_float(torch.bfloat16, x)

    return lambda x: to_float(torch.float32, x)

def to_float(dtype, sample):
    return (sample[0].to(dtype),
            sample[1])

class Transform_seq():
    def __init__ (self, transforms):
        self.transforms = transforms

    def __call__ (self, sample):
        for trans in self.transforms:
            sample = trans(sample)
        return sample



def to_tensor(sample):
    return (torch.from_numpy(sample[0]),
           torch.tensor(sample[1]).long())

def get_data_conv_func(dtype):

    if dtype == torch.float32:
        return lambda x: to_float(torch.float32, x)

    if dtype == torch.float16:
        return lambda x: to_float(torch.float16, x)

    if dtype == torch.bfloat16:
        return lambda x: to_float(torch.bfloat16, x)

    return lambda x: to_float(torch.float32, x)

def to_float(dtype, sample):
    return (sample[0].to(dtype),
            sample[1])
