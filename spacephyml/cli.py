"""
Script for creating dataset based on exisiting labels.
"""
import tempfile
from argparse import ArgumentParser
from .datasets.creator import create_dataset

def pars_args():
    """
    Parse commandline arguments.
    """
    parser = ArgumentParser()

    actions = parser.add_subparsers()

    create = actions.add_parser('create', help = 'Create a dataset')
    create.add_argument('--config', default=None,
                        choices=['orbit', 'orbitclean', 'nov', 'nov_full','dec'])
    create.add_argument('--start', default='2017-11-01')
    create.add_argument('--end', default='2017-11-31')
    create.add_argument('--force', action='store_true', default=False)
    create.add_argument('--clean', action='store_true', default=False)
    create.add_argument('--samples', default=0)
    create.add_argument('output', nargs='?', default=None)

    args = parser.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f" {arg}: {getattr(args,arg)}")

    return args

def main():
    ARGS = pars_args()

    labels_dl_dir = tempfile.gettempdir() + '/mms_labels/'

    YEAR = 2017
    SAMPLED = False
    CLEAN = False
    if ARGS.config == 'orbit':
        MONTH = 12
        TRANGE=['2017-12-03','2017-12-07']
        SAMPLED = False
    elif ARGS.config == 'orbitclean':
        MONTH = 12
        TRANGE=['2017-12-03','2017-12-07']
        CLEAN = True
        SAMPLED = False
    elif ARGS.config == 'dec':
        CLEAN = True
        SAMPLED = True
        MONTH = 12
        TRANGE=['2017-12-01','2017-12-31']
    elif ARGS.config == 'nov_full':
        CLEAN = False
        SAMPLED = False
        MONTH = 11
        TRANGE=['2017-11-01','2017-11-30']
    else:
        CLEAN = True
        SAMPLED = True
        MONTH = 11
        TRANGE=['2017-11-01','2017-11-30']

    if ARGS.output is None:
        ARGS.output = f'./dataset_{ARGS.trange}_{YEAR}_{MONTH}'
        ARGS.output += '_clean.csv' if CLEAN else '.csv'

    create_dataset(ARGS.output, TRANGE, SAMPLED, CLEAN)

if __name__ == "__main__":
    main()
