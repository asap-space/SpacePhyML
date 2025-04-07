"""
Creates a new dataset of MMS plasma distribution data
using Olshevsky labels.
"""
from spacephyml.datasets.creator import create_dataset

create_dataset('./mms_region.csv',
               trange=['2017-12-04/05:00:00','2017-12-04/15:00:00'],
               clean=False, label_source='Olshevsky',
               var_list=['mms1_dis_dist_fast'])
