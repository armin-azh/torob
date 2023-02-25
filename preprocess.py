from argparse import Namespace, ArgumentParser
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from utils import read_json_lines, JSONLinesWriter
from normalizer import *


def main(args: Namespace) -> None:
    root: Path = args.root
    save_path: Path = args.out
    save_path.mkdir(parents=True, exist_ok=True)

    # Paths
    product_info_path = root.joinpath('products-info_v1.jsonl')
    search_data_path = root.joinpath('torob-search-data_v1.jsonl')
    test_offline_path = root.joinpath('test-offline-data_v1.jsonl')

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
    main(args=parser.parse_args())
