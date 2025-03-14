import argparse
import json
import random


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--input_file', default=None, type=str,
        help="The input file to mine hard negatives from."
            "This script will retrieve top-k documents for each query, "
            "and random sample negatives from the top-k documents "
            "(not including the positive documents)."
    )

    parser.add_argument(
        '--output_file', default=None, type=str,
        help="The output file to store the mined hard negatives"
    )

    parser.add_argument(
        '--num_negatives', default=15, type=int, 
        help='The number of negatives'
    )

    return parser.parse_args()


def find_random_neg(
    input_file, 
    output_file, 
    num_negatives, 
):
    corpus = []
    queries = []
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
                'positives': d['positives']['text'],
            }
        )
        
        if 'negatives' in d:
            corpus.extend(d['negatives']['text'])
        
        queries.append(d['query']['text'])
        
    # remove duplicates
    corpus = list(set(corpus))
    
    
    for i, data in enumerate(train_data):
        query = data['query']
        # inxs = all_inxs[i][sample_range[0]:sample_range[1]]
        inxs = []
        # get k negative samples per query
        while len(inxs) < num_negatives:
            inx = random.choice(range(len(corpus)))     # random one
            # make sure hard negatives are not positive examples and the query itself
            if inx not in inxs and corpus[inx] not in data['positives'] and corpus[inx] != query:
                inxs.append(inx)
        
        data['negatives'] = [corpus[inx] for inx in inxs]

    with open(output_file, 'w') as f:
        for data in train_data:
            
            # sample one positive only (note: may change to multiple samples later)
            # data['positives'] = [random.choice(data['positives'])]

            # retain all the positives
            
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Write to file: {output_file=}")


if __name__ == '__main__':
    args = get_args()
    print(f"{args=}\n")
    
    find_random_neg(
        input_file=args.input_file,
        output_file=args.output_file,
        num_negatives=args.num_negatives,
    )
    