from argparse import Namespace, ArgumentParser
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from utils import read_json_lines, JSONLinesWriter, save_args
from normalizer import *
import gc
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def main(args: Namespace) -> None:
    root: Path = args.root.joinpath('searches-fold')

    product_info_path = args.root.joinpath('aggregate_product_data.jsonl')
    test_query_path = args.root.joinpath('aggregate_test_query.jsonl')

    save_args(args.root.joinpath('rel-tf-idf-parameters.txt'), args)

    products_df = pd.DataFrame(read_json_lines(product_info_path))
    products_id_to_idx = dict((p_id, idx) for idx, p_id in enumerate(products_df['id']))

    if args.root.joinpath('random-proj-mat.npy').is_file():
        print('Random Project Mat is loaded')
        random_projection_mat = np.load(str(args.root.joinpath('random-proj-mat.npy')))
    else:
        random_projection_mat = np.random.rand(args.vocab_size, args.embedding)
        np.save(str(args.root.joinpath('random-proj-mat.npy')), random_projection_mat)

    # if args.root.joinpath('product-feature.npy').is_file():
    #     print('Product features mat is loaded')
    #     products_projected = np.load(str(args.root.joinpath('product-feature.npy')))
    # else:
    vectors = TfidfVectorizer(max_features=args.vocab_size, lowercase=True, use_idf=True)
    products_tfidf = vectors.fit_transform(products_df['title_normalized'])
    products_projected = products_tfidf.dot(random_projection_mat)
    np.save(str(args.root.joinpath('product-feature.npy')), products_projected)
    del products_tfidf
    gc.collect()

    if not args.root.joinpath('test-feature.npy').is_file():
        test_offline_queries_df = pd.DataFrame(read_json_lines(test_query_path))
        queries_test_tfidf = vectors.transform(test_offline_queries_df['raw_query_normalized'])
        queries_test_projected = queries_test_tfidf.dot(random_projection_mat)
        np.save(str(args.root.joinpath('test-feature.npy')), queries_test_projected)
        del queries_test_tfidf  # Free up memory.
        gc.collect()

    fold_paths = list(root.glob('*')) if not args.fold_idx else [args.fold_idx]
    for fold_path in tqdm(fold_paths, total=len(fold_paths)):
        search_data_path = fold_path.joinpath('aggregate_search_train.jsonl')
        searches_df = pd.DataFrame(read_json_lines(search_data_path))
        queries_tfidf = vectors.transform(searches_df['raw_query_normalized'])
        queries_projected = queries_tfidf.dot(random_projection_mat)

        del queries_tfidf
        gc.collect()

        # Create .dat file
        with open(str(fold_path.joinpath('train.dat')), 'w') as file:
            for qid, search in enumerate(searches_df.itertuples(index=False)):
                clicks = dict(zip(search.clicks, search.clicks_count))
                print("MalcomX")
                for can_product_id in search.results[:200]:
                    if can_product_id is None:
                        continue
                    cand_score = clicks.get(can_product_id, 0)
                    cand_score = np.log2(cand_score + 1)

                    p_idx = products_id_to_idx[can_product_id]
                    features = np.concatenate((products_projected[p_idx], queries_projected[qid]))
                    features = np.around(features, 3)
                    file.write(
                        f"{cand_score} qid:{qid} "
                        + " ".join([f"{i}:{s}" for i, s in enumerate(features)])
                        + "\n"
                    )

    # Paths
    # product_info_path = root.joinpath('aggregate_product_data.jsonl')
    # search_data_path = root.joinpath('aggregate_search_data.jsonl')
    # test_offline_path = root.joinpath('aggregate_test_query.jsonl')
    #
    # save_args(save_path.joinpath('rel-tf-idf-parameters.txt'), args)
    #
    # searches_df = pd.DataFrame(read_json_lines(search_data_path, n_lines=10000))
    # products_df = pd.DataFrame(read_json_lines(product_info_path))
    # products_id_to_idx = dict((p_id, idx) for idx, p_id in enumerate(products_df['id']))
    #
    # random_projection_mat = np.random.rand(args.vocab_size, args.embedding)
    # vectors = TfidfVectorizer(max_features=args.vocab_size, lowercase=True, use_idf=True)
    # products_tfidf = vectors.fit_transform(products_df['title_normalized'])
    #
    # products_projected = products_tfidf.dot(random_projection_mat)
    #
    # del products_tfidf
    # gc.collect()
    #
    # queries_tfidf = vectors.transform(searches_df['raw_query_normalized'])
    # queries_projected = queries_tfidf.dot(random_projection_mat)
    #
    # del queries_tfidf
    # gc.collect()
    #
    # # Create .dat file
    # with open(str(save_path.joinpath('train.dat')),'w') as file:
    #     for qid,search in enumerate(searches_df.itertuples(index=False)):
    #         clicks = dict(zip(search.clicks,search.clicks_count))
    #         for can_product_id in search.results:
    #             if can_product_id is None:
    #                 continue
    #             cand_score = clicks.get(can_product_id,0)
    #             cand_score = np.log2(cand_score+1)
    #
    #             p_idx = products_id_to_idx[can_product_id]
    #             features = np.concatenate((products_projected[p_idx], queries_projected[qid]))
    #             features = np.around(features, 3)
    #             print(features.shape)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', help='Contest data root', type=Path, default='experimental/exp1')
    parser.add_argument('--fold-idx','--fold_idx',help='fold path',type=Path,default='/home/samtech/MalcomX/torob/experimental/exp1/searches-fold/fold-1')

    # Hyperparameter
    parser.add_argument('--vocab-size', '--vocab_size', help='Vocab size', type=int, default=4096)
    parser.add_argument('--embedding', help='Embedding size', type=int, default=256)

    main(args=parser.parse_args())
