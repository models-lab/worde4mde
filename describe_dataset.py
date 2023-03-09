from argparse import ArgumentParser

from modelset_evaluation.evaluation_classification_clustering import set_up_modelset


def main(args):
    modelset_df, dataset = set_up_modelset(args)

    print(f'Number of models: {len(modelset_df)}')
    print(f'Number of categories: {len(modelset_df.category.unique())}')
    print(f'Avg number of elements: {modelset_df.elements.mean():.2f}')
    print(f'Avg number of classes: {modelset_df.classes.mean():.2f}')
    print(f'Avg number of attributes: {modelset_df.attributes.mean():.2f}')
    print(f'Avg number of references: {modelset_df.references.mean():.2f}')
    print(f'Avg number of packages: {modelset_df.packages.mean():.2f}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to describe modelset')
    parser.add_argument('--model_type', default='ecore',
                        help='ecore or uml',
                        choices=['ecore', 'uml'])
    parser.add_argument('--remove_duplicates', help='Remove duplicate models', action='store_true')
    parser.add_argument('--min_occurrences_per_category', help='Min occurences per category.', type=int, default=10)
    parser.add_argument("--t0", dest="t0", help="t0 threshold.", type=float, default=0.8)
    parser.add_argument("--t1", dest="t1", help="t1 threshold.", type=float, default=0.7)

    args = parser.parse_args()
    main(args)