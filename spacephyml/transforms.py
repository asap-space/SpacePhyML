"""
Different useful transforms.
"""
import numpy as np

class TransformBase():
    """
    Base class for the transforms
    """

    def __call__(self, sample):
        return self.forward(sample)

class Compose(TransformBase):
    """
    Compose multiple transforms into one.
    """
    def __init__(self,*transforms):
        self.transforms = transforms

    def forward(self, sample):
        """
        Forward call.
        """
        for trans in self.transforms:
            sample = trans(sample)

        return sample

class ZScoreNorm(TransformBase):
    """
    Calculate the Z-Score norm using specified mean and std.
    """
    def __init__ (self, mean, std):
        self.mean = mean
        self.std = std

    def forward(self, sample):
        """
        Forward call.
        """
        return (((sample[0] - self.mean)/self.std), *sample[1:])

class Threshold(TransformBase):
    """
    Threshold the sample.
    """
    def __init__ (self, thresholds=None):
        self.thresholds = thresholds

    def forward(self, sample):
        """
        Forward call.
        """
        x = sample[0]

        threshold_low_index = np.where( x < self.thresholds[0])
        x[threshold_low_index] = self.thresholds[0]

        threshold_high_index = np.where( x > self.thresholds[1])
        x[threshold_high_index] = self.thresholds[1]

        return (x, *sample[1:])


class LogNorm(TransformBase):
    """
    Log Normalize the data between a given range
    """
    def __init__ (self, normalization=None):
        self.normalization = normalization

    def forward(self, sample):
        """
        Forward call.
        """
        x = np.log10(sample[0])

        if self.normalization is None:
            x -=  x.min()
            x /=  x.max()
        else:
            x -= self.normalization[0]
            x /= (self.normalization[1] - self.normalization[0])

        return (x, *sample[1:])

class Flatten(TransformBase):
    """
    Filter for flattening the data
    """
    def forward(self, sample):
        """
        Forward call.
        """
        return (sample[0].reshape(-1), *sample[1:])

class Log10(TransformBase):
    """
    Calculate the log10 of all non zero values in the sample.
    """
    def forward(self, sample):
        """
        Forward call.
        """
        non_zero_indexes = np.where(sample[0]!=0)

        #Talk log10 of all non_zero values
        sample[0][non_zero_indexes] = np.log10(sample[0][non_zero_indexes])
        return sample


class Roll(TransformBase):
    """
    Perform a roll along a axis.
    """

    def __init__ (self, shift = 16, axis = -3):
        self.shift = shift
        self.axis = axis

    def forward(self, sample):
        """
        Forward call.
        """
        x = sample[0]

        #Roll along Phi
        x = np.roll( x, self.shift, axis=self.axis)

        return (x, *sample[1:])

class MoveAxis(TransformBase):
    """
    Move around the axis in the sample.
    """
    def __init__ (self, src = -1, dst = -3):
        self.src = src
        self.dst = dst

    def forward(self, sample):
        """
        Forward call.
        """
        return (np.moveaxis(sample[0],self.src,self.dst),
                *sample[1:])

class Sum(TransformBase):
    """
    Callculate the sum along specified axis.
    """
    def __init__ (self, axis = -1):
        self.axis = axis

    def forward(self, sample):
        """
        Forward call.
        """
        x = sample[0]

        x = np.sum(x, axis=self.axis)

        return (x, *sample[1:])

class Mean(TransformBase):
    """
    Callculate the mean along specified axis.
    """
    def __init__ (self, axis = -1):
        self.axis = axis

    def forward(self, sample):
        """
        Forward call.
        """
        x = sample[0]

        x = np.mean(x, axis=self.axis)

        return (x, *sample[1:])
