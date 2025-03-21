"""
Classifying MMS dayside space plasma regions

"""
from torch.utils.data import DataLoader
from spacephyml.datasets.general.mms import ExternalMMSData
from spacephyml.models.mms import PCReduced
from spacephyml.creator import create_dataset
from spacephyml.transforms import MMS1IonDistLabeled_Transform

create_dataset('./mms_region.csv',
               trange=['2017-12-04/05:00:00','2017-12-04/15:00:00'],
               clean=False, label_source='Olshevsky',
               var_list=['mms1_dis_dist_fast'])

dataset = ExternalMMSData('./mms_region.csv',
                          transform = MMS1IonDistLabeled_Transform())
model = PCReduced()

labels = {'human': [], 'classifier': []}
for x,l, _, in DataLoader(dataset, batch_size=32, shuffle = True):
    lc = model(x)
    labels['human'].extend(l)
    labels['classifier'].extend(lc)
