import sys
import os
import logging
import json
import random
import numpy as np
from tqdm import tqdm

import faiss
from sklearn.cluster import KMeans
from transformers import HfArgumentParser, set_seed


from arguments import HardNegativeDataArguments
from modeling import ModelForInference
from utils import create_faiss_index, faiss_search


# Setup logging
log_levels = {
    "debug": logging.DEBUG,         # 10
    "info": logging.INFO,           # 20
    "warning": logging.WARNING,     # 30
    "error": logging.ERROR,         # 40
    "critical": logging.CRITICAL,   # 50
}
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s]  %(message)s",
    # format="[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s",
    # format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=log_levels['info'],
)



############################
# Get negative ids by method
############################
# Get negative ids
def get_negative_ids(
    all_candidate_ids,
    num_negatives,
    method,
    train_data,
    corpus,
    corpus_embedding=None,
    num_clusters=None,
    lambda_=None,
    seed=42,
):
    
    if method == 'topk':
        print(f'[{method=}] Use top-k negatives.')
    elif method == 'sample':
        print(f'[{method=}]: Use sampled negatives from the search range.')
    elif method == 'cluster':
        print(f'[{method=}, {num_clusters=}, {lambda_=}]: Use sampled negatives from clusters.')
    else:
        raise Exception("Should specify the method from [topk, sample, cluster]")
    
    all_negtative_ids = []
    for i, row in tqdm(enumerate(train_data), total=len(train_data), desc='Row'):
        negtative_ids = []
        candidate_ids = all_candidate_ids[i]
        for j in candidate_ids:
            if j == -1:
                raise Exception(f"No hard negatives found for the {i}-th data!")
            # make sure hard negatives are not in positive examples and the query itself
            if corpus[j] not in row['positives'] and corpus[j] != row['query']:
                negtative_ids.append(j)
        
        if len(negtative_ids) < num_negatives:
            raise Exception("No enough negative samples! Consider to increase the search range.")
        
        # 1. simply use topk hard negatives
        if method == 'topk':
            negtative_ids = negtative_ids[:num_negatives]
        
        # 2. random sampled hard negatives from the search range
        elif method == 'sample':
            negtative_ids = random.sample(negtative_ids, num_negatives)
        
        # 3. sample hard negatives by cluster
        elif method == 'cluster':
            
            # assert num_clusters >= num_negatives, "num_clusters should >= num_negatives"
            
            matrix = []
            for j in negtative_ids:          # negative ids
                matrix.append(corpus_embedding[j])    # embeddings of one negative
            matrix = np.asarray(matrix, dtype=np.float32)
            
            # KMeans clustering
            kmeans = KMeans(n_clusters=num_clusters, init="k-means++", random_state=seed)
            kmeans.fit(matrix)
            labels = kmeans.labels_
            
            # sample by cluster-adjusted weight
            weights = []
            visited = [0] * num_clusters
            for j in range(len(negtative_ids)):
                cluster_label = labels[j]
                k = visited[cluster_label]
                probablity = lambda_ ** k   # `alpha` has no impact in the new setting
                weights.append(probablity)  # probablity weight in sampling
                visited[cluster_label] += 1 # visited count +1

            weights = np.array(weights).astype('float64')   # precision issue
            weights /= weights.sum()  # normalize

            # sample by normalized weights
            negtative_ids = np.random.choice(
                negtative_ids, size=num_negatives, replace=False, p=weights
            )
        
        # append negativge_ids from each row
        all_negtative_ids.append(negtative_ids)
    
    return all_negtative_ids


############################
# Save generated hard negatives
############################
def save_data(
    output_file,
    all_negative_ids,
    train_data,
    corpus
):

    with open(output_file, 'w') as f:

        for i, row in enumerate(train_data):
            
            d = {}
            d['query'] = row['query']
            # sample one positive only (note: may change to multiple samples later)
            d['positives'] = [random.choice(row['positives'])]
            # sampled negatives
            d['negatives'] = [corpus[j] for j in all_negative_ids[i]]
            
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    logger.info(f"Saved to file `{output_file}`\n")



############################
# Main for generating hard negatives
############################
def find_hard_negatives(
    model, 
    input_file, 
    output_prefix, 
    max_query_length,
    max_passage_length,
    num_negatives, 
    search_range, 
    method,
    batch_size,
    num_clusters,
    lambda_,
    seed,
):
    
    # Get methods
    if isinstance(method, str):
        methods = [s.strip() for s in method.split(',')]
        methods = [s for s in methods if s in ['topk', 'sample', 'cluster']]

    # if method in ['topk', 'sample', 'cluster']:
    #     # given one method
    #     methods = [method]

    if not method or not methods:
        # else, use all methods
        methods = ['topk', 'sample', 'cluster']
    
    logger.info(f'Methods used to mine hard negatives: {methods=}')
    
    
    # Get corpus and query 
    corpus, query = [], []
    train_data = []
    for line in open(input_file):
        d = json.loads(line.strip())
        
        # append positives
        positives = d['positives']['text']
        assert isinstance(positives, list)
        
        corpus.extend(positives)    # all positives to corpus
        # pos = random.choice(positives) # sample one positive (do it later)
        
        train_data.append(
            {
                'query': d['query']['text'],
                
                # 'positives': d['positives']['text'],

                # use the fixed postive sample for all data in this setting?
                # sample one, before we proceed
                'positives': [random.choice(d['positives']['text'])],
            }
        )
        
        if 'negatives' in d:
            corpus.extend(d['negatives']['text'])
        
        # query.append(d['query'])      # TODO: wrong!!!!!!!
        query.append(d['query']['text'])      # use query text only


    corpus = list(set(corpus))  # remove duplicates if any
    

    # Encode query and corpus sentences
    logger.info(f'Embedding query ({len(query)=}, {batch_size=}, {max_query_length=}):')
    query_embedding = model.encode(query, batch_size=batch_size, max_length=max_query_length)

    logger.info(f'Embedding corpus ({len(corpus)=}, {batch_size=}, {max_passage_length=}):')
    corpus_embedding = model.encode(corpus, batch_size=batch_size, max_length=max_passage_length)
    
    
    # Crete faiss index
    logger.info('Create Faiss index and search:')
    faiss_index = create_faiss_index(corpus_embedding)
    logger.info(f"Number of items in faiss index: {faiss_index.ntotal}\n")
    
    if isinstance(search_range, str):   # search range should be [int, int]
        search_range = [int(x) for x in search_range.split('-')]
    
    # Faiss search
    all_scores, all_indices = faiss_search(
        faiss_index, 
        query_embedding,
        topk=search_range[-1], 
        batch_size=batch_size, 
    )
    
    assert len(all_indices) == len(train_data)

    # all candidate ids from search_range
    all_candidate_ids = [x[search_range[0]:search_range[1]] for x in all_indices]
    
    # =================
    # Save to file
    # =================
    
    if lambda_ is not None:
        # given one lambda
        lambdas = [lambda_]
    else:
        # else, use lambdas [0.1-0.9]
        lambdas = [x/10.0 for x in range(1,10)][::-1]
    
    
    for method in methods:
        for lambda_ in lambdas:

            if method in ['topk', 'sample']:
                # lambda_ is not for topk and sample mwthods"
                # output_file = output_prefix + f"{method}_{args.search_range}" + ".jsonl"
                output_file = os.path.join(output_prefix, f"{method}.jsonl")
            
            else:   # cluster based method
                # output_file = output_prefix + f"{method}_{args.search_range}_{num_clusters}_{lambda_}" + ".jsonl"
                output_file = os.path.join(output_prefix, f"{method}{int(lambda_*10)}.jsonl")
                
            
            # get negative ids
            all_negative_ids = get_negative_ids(
                all_candidate_ids,
                num_negatives,
                method,
                train_data,
                corpus,
                corpus_embedding,
                num_clusters=num_clusters,
                lambda_=lambda_,
                seed=seed,
            )
            
            # output_file
            save_data(
                output_file,
                all_negative_ids,
                train_data,
                corpus
            )
            
            # only run once for topk and sample (lambda_ is not for them)
            if method in ['topk', 'sample']:
                break
    
    return





if __name__ == '__main__':
    
    parser = HfArgumentParser([HardNegativeDataArguments])
    args = parser.parse_args_into_dataclasses()[0]
    
    if args.log_level in log_levels:
        logger.setLevel(log_levels[args.log_level])
    
    logger.info(f"Hard negative data arguments:\n  {args.to_json_string()}")

    # set seed
    set_seed(args.seed)
    
    os.makedirs(args.output_prefix, exist_ok=True)

    # Save config info for the generated data
    # fpath = args.output_file.rsplit('.', 1)[0] + '-config.json'
    # fpath = args.output_prefix + f'[{args.search_range}]-config.json'
    fpath = os.path.join(args.output_prefix,f'config.json')

    with open(fpath, 'w') as f:
        f.write(args.to_json_string())
    
    model = ModelForInference(
        args.model_name_or_path, 
        device=args.device,
        use_fp16=args.fp16,
        use_bf16=args.bf16,
    )
    

    find_hard_negatives(
        model,
        input_file=args.input_file,
        output_prefix=args.output_prefix,
        max_query_length=args.max_query_length,
        max_passage_length=args.max_passage_length,
        num_negatives=args.num_negatives,
        search_range=args.search_range,
        method=args.method,
        batch_size=args.batch_size,
        
        num_clusters=args.num_clusters,
        lambda_=args.lambda_,
        seed=args.seed
    )
    