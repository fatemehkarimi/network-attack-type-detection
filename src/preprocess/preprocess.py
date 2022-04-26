import sys
import argparse
import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from metadata.const import features, categorical_features, class_label


def get_config(section):
    config = configparser.RawConfigParser()
    config.read('../settings.ini')
    return dict(config.items(section))


def get_box_whisker_range(col):
    q1 = col.quantile(q=0.25)
    q3 = col.quantile(q=0.75)
    iqr = q3 - q1
    min_range = q1 - 1.5 * iqr
    max_range = q3 + 1.5 * iqr

    return min_range, max_range


def remove_outlier(df, feature):
    min_range, max_range = get_box_whisker_range(df[feature])
    df.loc[
        lambda x: (x[feature] < min_range)
                    | (x[feature] > max_range),
        feature] = np.nan
    df.dropna(axis=0, inplace=True)


def remove_zero_variance(df, features):
    const_filter = VarianceThreshold()
    const_filter.fit(df)
    cols = const_filter.get_support(indices=True)
    return df.iloc[:, cols]


def plot_correlation(correlation, filename, labels=None):
    fig = plt.figure(figsize=(10.41, 7.29))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlation, vmin=-1, vmax=1)
    fig.colorbar(cax)
    if labels:
        ticks = np.arange(0, len(labels), 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    plt.savefig(filename)


def main(args):
    df = pd.read_csv(args.file, dtype={features['similar_http']: str})
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)

    numeric_features = [f for f in features.values()
                            if f not in categorical_features]
    # for f in numeric_features:
        # remove_outlier(df, f)

    df.loc[df[class_label] == 'BENIGN', class_label] = 0
    df.loc[df[class_label] != 0, class_label] = 1
    
    # df.drop(categorical_features, axis=1, inplace=True)
    # df = remove_zero_variance(df, numeric_features)

    # corr_matrix = numeric_df.corr(method='pearson')
    # redundent_features = set()
    # for i in range(len(corr_matrix.columns)):
    #     for j in range(i):
    #         if abs(corr_matrix.iloc[i, j]) > 0.8:
    #             redundent_features.add(corr_matrix.columns[i])

    # print(redundent_features)
    # plot_correlation(corr_matrix, 'numeric-correlation.png')
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
