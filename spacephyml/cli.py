"""
Script for creating dataset based on exisiting labels.
"""
from argparse import ArgumentParser
from .datasets.creator import create_dataset

def create_action(args):
    """
    Run the create action.
    """

    sampled = args.sampled
    clean = args.clean
    trange = [args.start, args.end]

    if args.config:
        if args.config == 'dec':
            clean = True
            sampled = True
            trange=['2017-12-01','2017-12-31']
        elif args.config == 'nov':
            clean = False
            sampled = False
            trange=['2017-11-01','2017-11-30']

    create_dataset(args.output, trange, sampled, clean)

def pars_args():
    """
    Parse commandline arguments.
    """
    parser = ArgumentParser()

    actions = parser.add_subparsers(dest="command")

    create = actions.add_parser('create', help = 'Create a dataset')
    create.add_argument('--config', default=None,
                        choices=['nov', 'dec'])
    create.add_argument('--start', default='2017-11-01',
                        help ='Start date, format YYYY-MM-DD')
    create.add_argument('--end', default='2017-11-31',
                        help ='End date, format YYYY-MM-DD')
    create.add_argument('--force', action='store_true', default=False)
    create.add_argument('--clean', action='store_true', default=False)
    create.add_argument('--samples', default=0)
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
