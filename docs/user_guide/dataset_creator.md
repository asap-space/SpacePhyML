# Dataset Creator
The dataset creator enables the creation of new datasets based on existing data and label
sources. The data set creator can be used either with the command line interface (CLI)
described below or in an python script using the API described [here](../reference/datasets/creator.md).

## CLI

The CLI for the dataset creator can be accessed using with the command

```
spacephyml create [-h] [--label_source {Olshevsky,Unlabeled}]
                         [--start START] [--end END] [--force] [--clean]
                         [--samples SAMPLES] [--resample RESAMPLE]
                         [--var {...}]
                         output
```

The only required argument is the path to the output file (`output`). The output file
can be either in CSV or Feather format and is determined by the file extension of the given
output file. By default, a new dataset will not be created if the output file exists. This
can be forced by setting the flag `--force`.

Currently the creator supports creating datasets based of labels from
[Olshevsky](#olshevsky-labels) or creating an unlabeled dataset. Creating a dataset based on
the Olshevsky labels currently does not support resampling and can only be done using
data from the FPI instrument on MMS1, see the variable [list below](#supported-variables)

The `--start` and `--end` flag gives the time range for creating the dataset.

When creating a labeled data set, you have the option of creating a clean data set without
and unknown labels. This done by setting the `--clean` flag. You can also select how many
samples of each label you want by setting the `--samples` flag.

When creating an unlabeled dataset, the dataset can be resampled to a sampling frequency specified by the `--resample` flag. This flag follow the rules from the [pandas resample function](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html).

### Supported variables
Controlled by the `--var` flag.

| Varible               | Instrument |
| -------               | ---------- |
| mms1_dis_dist_fast                | FPI |
| mms1_dis_energyspectr_omni_fast   | FPI |
| mms1_dis_bulkv_gse_fast           | FPI |
| mms1_dis_numberdensity_fast       | FPI |
| mms1_dis_temppara_fast            | FPI |
| mms1_dis_tempperp                 | FPI |

## Loading the dataset

Once created the dataset can be loaded using one of the general loaders.

| Mission | Resampled | Dataset Class |
| ------- | --------- | ------------- |
| MMS | Yes | [PandasDataset](../reference/datasets/general/pandas.md#PandasDataset) |
| MMS | No | [ExternalMMSData](../reference/datasets/general/mms.md#ExternalMMSData) |

## Olshevsky labels
These labels are from the work of Olshevsky, et al.[^1] who labeled data from the MMS1 spacecraft for November and December 2017. The labeled data is from the Earth's dayside and is labeled as one of the regions in the table below.

| Label | Region  |
| :---: | ------- |
| -1    | Undefined/Unknown |
| 0     | Solar Wind (SW) |
| 1     | Ion foreshock (IF) |
| 2     | Magnetosheath (MSH) |
| 3     | Magnetosphere (MSP) |

[^1]: Olshevsky, V., *et al.* (2021). Automated classification of plasma regions using 3D particle energy distributions. Journal of Geophysical Research: Space Physics, [https://doi.org/10.1029/2021JA029620](https://doi.org/10.1029/2021JA029620)

