# Quick Start

## Installation
To use SpacePhyML you can install the package using pip.

```
pip install git+https://github.com/Jonah-E/SpacePhyML.git
```

## Usage

Create a dataset through the commandline tool using the dataset generator.

```
spacephyml create my_dataset.csv
```

```
from spacephyml.datasets import MMSDataset

dataset = MMSDataset('my_dataset.csv')
```
