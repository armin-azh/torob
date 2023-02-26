from argparse import Namespace, ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from utils import read_json_lines, JSONLinesWriter, save_args
from normalizer import *
from sklearn.model_selection import KFold


def main(args: Namespace) -> None:
    root: Path = args.root
    save_path: Path = args.out
    save_path.mkdir(parents=True, exist_ok=True)

    # Paths
    product_info_path = root.joinpath('products-info_v1.jsonl')
    search_data_path = root.joinpath('torob-search-data_v1.jsonl')
    test_offline_path = root.joinpath('test-offline-data_v1.jsonl')

    save_args(save_path.joinpath('preprocess-parameters.txt'), args)

    # Search data
    print("Aggregating searches based on raw query...")
    agg_searches = defaultdict(lambda: dict(results=Counter(), clicks=Counter()))
    for search in tqdm(read_json_lines(str(search_data_path))):
        agg_searches[search['raw_query']]['results'].update(search['result'])
        agg_searches[search['raw_query']]['clicks'].update(search['clicked_result'])

    print('Writing aggregated searches into file...')
    with JSONLinesWriter(save_path.joinpath('aggregate_search_data.jsonl')) as out_file:
        for raw_query, stats in tqdm(agg_searches.items()):
            results, results_count = list(zip(*stats['results'].most_common()))
            clicks, clicks_count = list(zip(*stats['clicks'].most_common()))
            record = {
                'raw_query': raw_query,
                'raw_query_normalized': normalize_text(raw_query),
                'results': results,
                'results_count': results_count,
                'clicks': clicks,
                'clicks_count': clicks_count,
            }
            out_file.write_record(record)

    print("Finished aggregating searches.")
    print(f'Number of aggregate search records: {len(agg_searches)}')
    print(f"The aggregated searches data were stored in '{save_path.joinpath('aggregate_search_data.jsonl')}'.")

    if args.k_fold:
        print('K-Fold ...')
        kf_save_path = save_path.joinpath('searches-fold')
        kf_save_path.mkdir(parents=True, exist_ok=True)

        searches_key = list(agg_searches.keys())
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.fold_state)
        for foldIdx, (trainIdx, testIdx) in enumerate(kf.split(np.arange(len(searches_key)))):
            kf_dir = kf_save_path.joinpath(f'fold-{foldIdx + 1}')
            kf_dir.mkdir(parents=True, exist_ok=True)

            # Train
            with JSONLinesWriter(kf_dir.joinpath('aggregate_search_train.jsonl')) as out_file:
                for qIdx in tqdm(trainIdx):
                    raw_query = searches_key[qIdx]
                    stats = agg_searches[raw_query]
                    results, results_count = list(zip(*stats['results'].most_common()))
                    clicks, clicks_count = list(zip(*stats['clicks'].most_common()))
                    record = {
                        'raw_query': raw_query,
                        'raw_query_normalized': normalize_text(raw_query),
                        'results': results,
                        'results_count': results_count,
                        'clicks': clicks,
                        'clicks_count': clicks_count,
                    }
                    out_file.write_record(record)

            # Test
            with JSONLinesWriter(kf_dir.joinpath('aggregate_search_test.jsonl')) as out_file:
                for qIdx in tqdm(testIdx):
                    raw_query = searches_key[qIdx]
                    stats = agg_searches[raw_query]
                    results, results_count = list(zip(*stats['results'].most_common()))
                    clicks, clicks_count = list(zip(*stats['clicks'].most_common()))
                    record = {
                        'raw_query': raw_query,
                        'raw_query_normalized': normalize_text(raw_query),
                        'results': results,
                        'results_count': results_count,
                        'clicks': clicks,
                        'clicks_count': clicks_count,
                    }
                    out_file.write_record(record)

    # Products
    print('Preprocessing products...')
    count = 0
    with JSONLinesWriter(str(save_path.joinpath('aggregate_product_data.jsonl'))) as out_file:
        for product in tqdm(read_json_lines(product_info_path)):
            titles = product['titles']
            titles_concat_normalized = normalize_text(" ".join(titles))
            titles_words_set = set(titles_concat_normalized.split())
            titles_words_concat = " ".join(titles_words_set)

            record = {
                'id': product['id'],
                'title_normalized': titles_words_concat,
            }
            out_file.write_record(record)
            count += 1
    print('Finished preprocessing products.')
    print(f'Number of processed products: {count}')
    print(f"The processed products data were stored in '{save_path.joinpath('aggregate_product_data.jsonl')}'")

    print('Preprocessing test queries...')
    count = 0
    with JSONLinesWriter(str(save_path.joinpath('aggregate_test_query.jsonl'))) as out_file:
        for test_sample in tqdm(read_json_lines(test_offline_path)):
            normalized_query = normalize_text(test_sample['raw_query'])
            record = {
                'raw_query_normalized': normalized_query,
            }
            count += 1
            out_file.write_record(record)
    print('Finished preprocessing test queries.')
    print(f'Number of processed test queries: {count}')
    print(f"The processed test queries were stored in '{save_path.joinpath('aggregate_test_query.jsonl')}'")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', help='Contest data root', type=Path, default='data/contest')
    parser.add_argument('--out', help='Save mediums', type=Path, default='experimental/exp1')

    # K-fold
    parser.add_argument('--k-fold', '--k_fold', help='Enable k-fold', action='store_true', default=True)
    parser.add_argument('--n-folds', '--n_folds', help='No.Folds', type=int, default=10)
    parser.add_argument('--fold-state', '--fold_state', help='Fold State', type=int, default=2023)

    main(args=parser.parse_args())
