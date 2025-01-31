"""
Script for creating dataset based on exisiting labels.
"""
from argparse import ArgumentParser
from .datasets.creator import create_dataset, _VAR_TO_FILE_INFO

def create_action(args):
    """
    Run the create action.
    """
    if args.var is None:
        args.var = ['mms1_dis_dist_fast']

    trange = [args.start, args.end]
    kwargs = {
        'force' : args.force,
        'samples' : args.samples,
        'clean' : args.clean,
        'var_list' : args.var,
    }

    create_dataset(args.output, args.label_source, trange, **kwargs)

def pars_args():
    """
    Parse commandline arguments.
    """
    parser = ArgumentParser()

    actions = parser.add_subparsers(dest="command")

    create = actions.add_parser('create', help = 'Create a dataset')
    create.add_argument('--label_source', default='Olshevsky',
                        choices=['Olshevsky', 'Unlabeled'])
    create.add_argument('--start', default='2017-11-01',
                        help ='Start date, format YYYY-MM-DD')
    create.add_argument('--end', default='2017-11-30',
                        help ='End date, format YYYY-MM-DD')
    create.add_argument('--force', action='store_true', default=False)
    create.add_argument('--clean', action='store_true', default=False)
    create.add_argument('--samples', default=0)
    create.add_argument('--var',
                        action = 'append',
                        choices=_VAR_TO_FILE_INFO.keys())
    create.add_argument('output')

    args = parser.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f" {arg}: {getattr(args,arg)}")

    return args

def main():
    """
    Main function for the SpacePhyML CLI.
    """
    args = pars_args()
    if args.command == 'create':
        create_action(args)

if __name__ == "__main__":
    main()
