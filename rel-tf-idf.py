from argparse import Namespace, ArgumentParser
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from utils import read_json_lines, JSONLinesWriter, save_args
from normalizer import *
import gc

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def main(args: Namespace) -> None:
    root: Path = args.root
    save_path: Path = args.out
    save_path.mkdir(parents=True, exist_ok=True)

    # Paths
    product_info_path = root.joinpath('aggregate_product_data.jsonl')
    search_data_path = root.joinpath('aggregate_search_data.jsonl')
    test_offline_path = root.joinpath('aggregate_test_query.jsonl')

    save_args(save_path.joinpath('rel-tf-idf-parameters.txt'), args)

    searches_df = pd.DataFrame(read_json_lines(search_data_path, n_lines=10000))
    products_df = pd.DataFrame(read_json_lines(product_info_path))
    products_id_to_idx = dict((p_id, idx) for idx, p_id in enumerate(products_df['id']))

    random_projection_mat = np.random.rand(args.vocab_size, args.embedding)
    vectors = TfidfVectorizer(max_features=args.vocab_size, lowercase=True, use_idf=True)
    products_tfidf = vectors.fit_transform(products_df['title_normalized'])

    products_projected = products_tfidf.dot(random_projection_mat)

    del products_tfidf
    gc.collect()

    queries_tfidf = vectors.transform(searches_df['raw_query_normalized'])
    queries_projected = queries_tfidf.dot(random_projection_mat)

    del queries_tfidf
    gc.collect()

    # Create .dat file
    with open(str(save_path.joinpath('train.dat')),'w') as file:
        for qid,search in enumerate(searches_df.itertuples(index=False)):
            clicks = dict(zip(search.clicks,search.clicks_count))
            for can_product_id in search.results:
                if can_product_id is None:
                    continue
                cand_score = clicks.get(can_product_id,0)
                cand_score = np.log2(cand_score+1)

                p_idx = products_id_to_idx[can_product_id]
                features = np.concatenate((products_projected[p_idx], queries_projected[qid]))
                features = np.around(features, 3)
                print(features.shape)





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', help='Contest data root', type=Path, default='experimental/exp1')
    parser.add_argument('--out', help='Save mediums', type=Path, default='experimental/exp1/tf-idf')

    # Hyperparameter
    parser.add_argument('--vocab-size', '--vocab_size', help='Vocab size', type=int, default=4096)
    parser.add_argument('--embedding', help='Embedding size', type=int, default=256)

    # K-fold
    parser.add_argument('--k-fold', '--k_fold', help='Enable k-fold', action='store_true', default=True)
    main(args=parser.parse_args())
