"""
Classifying MMS dayside space plasma regions

"""
import torch
from torch.utils.data import DataLoader
from spacephyml.datasets.general.mms import ExternalMMSData
from spacephyml.models.mms import PCReduced
from spacephyml.datasets.creator import create_dataset
from spacephyml.transforms import MMS1IonDistLabeled_Transform

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

create_dataset('./mms_region.csv',
               trange=['2017-12-04/05:00:00','2017-12-04/15:00:00'],
               clean=False, label_source='Olshevsky',
               var_list=['mms1_dis_dist_fast'])

dataset = ExternalMMSData('./mms_region.csv',
                          transform = MMS1IonDistLabeled_Transform())

model = PCReduced('s42').to(device)

labels = {'human': [], 'classifier': [], 'epoch': []}
for x, l, epoch in DataLoader(dataset, batch_size=32):
    lc = model(x.to(device)).to('cpu')
    labels['human'].extend(l)
    labels['classifier'].extend(lc)
    labels['epoch'].extend(epoch)
