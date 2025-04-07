# Welcome to SpacePhyML

SpacePhyML is a framework for working with space physics datasets. It aims to bring togheter already existing datasets for ease of usage and enable the simple creation of new datasets in the future. It is built on the torch dataset class but can be used in Tensorflow.

You can find the full documentation at [https://jonah-e.github.io/SpacePhyML/](https://jonah-e.github.io/SpacePhyML/).

## Quick Start

### Installation
To use SpacePhyML you can install the package using pip.

```
pip install git+https://github.com/Jonah-E/SpacePhyML.git
```

### Usage

Create a dataset through the commandline tool using the dataset generator.

```
spacephyml create my_dataset.csv
```

```
from spacephyml.datasets.general.mms import ExternalMMSData

dataset = ExternalMMSData('my_dataset.csv')
```

## Automatics in Space Exploration (ASAP)

<img src="assets/asap_brand_logo.png" alt="ASAP Logo"; style="float:right; margin:10px" width="300px">

The ASAP project is dedicated to designing and developing advanced algorithms that leverage artificial intelligence to automate onboard operations for space missions. These algorithms are specifically tailored for implementation on onboard processors, enhancing the efficiency, autonomy, and reliability of space systems. By integrating AI techniques, the project aims to reduce manual intervention, optimize mission performance, and enable smarter decision-making in real-time operational scenarios.

Futher information can be found on the ASAP Project website, and the latest updates are posted on the project LinkedIn page:</br>
Website: [asap-space.eu](https://asap-space.eu)</br>
LinkedIn: [linkedin.com/company/asap-space/](https://www.linkedin.com/company/asap-space/)</br>
Youtube: [youtube.com/@ASAP-space-eu](https://www.youtube.com/@ASAP-space-eu)

---

<img src="docs/assets/Flag_of_Europe.svg" alt="Flag" style="float:left; margin:10px" width="50px">

*ASAP has received funding from the European Union’s HORIZON Research and Innovation Action under the Grant Agreement No 101082633. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union.*

---

