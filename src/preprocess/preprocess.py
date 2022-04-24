import sys
import argparse
import pandas as pd
import configparser
from metadata.const import features


def get_config(section):
    config = configparser.RawConfigParser()
    config.read('../settings.ini')
    return dict(config.items(section))


def main(args):
    df = pd.read_csv(args.file, dtype={features['similar_http']: str})
    df.dropna(axis=0, inplace=True)
    df.to_csv(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess data')
    args = parser.add_argument(
        '--file',
        help='data file')

    args = parser.add_argument(
        '--output',
        help='output file, if not mentioned, it will override the file')

    args = parser.parse_args()

    if not args.file:
        parser.print_help()
        sys.exit(1)
    if not args.output:
        args.output = args.file
    main(args)
