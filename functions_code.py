import torch

def get_pooling_techniques(poolings_args):

    simple_poolings = ['CLS', 'AVG', 'SUM', 'MAX']
    simple_ns_poolings = ['AVG-NS', 'SUM-NS', 'MAX-NS'] 
    two_tokens_poolings = ['CLS+AVG', 'CLS+SUM', 'CLS+MAX', 'CLS+AVG-NS', 'CLS+SUM-NS', 'CLS+MAX-NS', 
                           'AVG+SUM', 'AVG+MAX', 'AVG+AVG-NS', 'AVG+SUM-NS', 'AVG+MAX-NS', 
                           'SUM+MAX', 'SUM+AVG-NS', 'SUM+SUM-NS', 'SUM+MAX-NS', 
                           'MAX+AVG-NS', 'MAX+SUM-NS', 'MAX+MAX-NS', 
                           'AVG-NS+SUM-NS', 'AVG-NS+MAX-NS', 
                           'SUM-NS+MAX-NS']

    three_tokens_poolings = ['CLS+AVG+SUM', 'CLS+AVG+MAX', 'CLS+AVG+AVG-NS', 'CLS+AVG+SUM-NS', 'CLS+AVG+MAX-NS', 
                             'CLS+SUM+MAX', 'CLS+SUM+AVG-NS', 'CLS+SUM+SUM-NS', 'CLS+SUM+MAX-NS', 'CLS+MAX+AVG-NS', 
                             'CLS+MAX+SUM-NS', 'CLS+MAX+MAX-NS', 'CLS+AVG-NS+SUM-NS', 'CLS+AVG-NS+MAX-NS', 'CLS+SUM-NS+MAX-NS', 
                             'AVG+SUM+MAX', 'AVG+SUM+AVG-NS', 'AVG+SUM+SUM-NS', 'AVG+SUM+MAX-NS', 'AVG+MAX+AVG-NS', 
                             'AVG+MAX+SUM-NS', 'AVG+MAX+MAX-NS', 'AVG+AVG-NS+SUM-NS', 'AVG+AVG-NS+MAX-NS', 'AVG+SUM-NS+MAX-NS', 
                             'SUM+MAX+AVG-NS', 'SUM+MAX+SUM-NS', 'SUM+MAX+MAX-NS', 'SUM+AVG-NS+SUM-NS', 'SUM+AVG-NS+MAX-NS', 
                             'SUM+SUM-NS+MAX-NS', 'MAX+AVG-NS+SUM-NS', 'MAX+AVG-NS+MAX-NS', 'MAX+SUM-NS+MAX-NS', 'AVG-NS+SUM-NS+MAX-NS']

    four_tokens_poolings = ['CLS+AVG+SUM+MAX', 'CLS+AVG+SUM+AVG-NS', 'CLS+AVG+SUM+SUM-NS', 'CLS+AVG+SUM+MAX-NS', 'CLS+AVG+MAX+AVG-NS', 
                            'CLS+AVG+MAX+SUM-NS', 'CLS+AVG+MAX+MAX-NS', 'CLS+AVG+AVG-NS+SUM-NS', 'CLS+AVG+AVG-NS+MAX-NS', 'CLS+AVG+SUM-NS+MAX-NS', 
                            'CLS+SUM+MAX+AVG-NS', 'CLS+SUM+MAX+SUM-NS', 'CLS+SUM+MAX+MAX-NS', 'CLS+SUM+AVG-NS+SUM-NS', 'CLS+SUM+AVG-NS+MAX-NS', 
                            'CLS+SUM+SUM-NS+MAX-NS', 'CLS+MAX+AVG-NS+SUM-NS', 'CLS+MAX+AVG-NS+MAX-NS', 'CLS+MAX+SUM-NS+MAX-NS', 'CLS+AVG-NS+SUM-NS+MAX-NS', 
                            'AVG+SUM+MAX+AVG-NS', 'AVG+SUM+MAX+SUM-NS', 'AVG+SUM+MAX+MAX-NS', 'AVG+SUM+AVG-NS+SUM-NS', 'AVG+SUM+AVG-NS+MAX-NS', 
                            'AVG+SUM+SUM-NS+MAX-NS', 'AVG+MAX+AVG-NS+SUM-NS', 'AVG+MAX+AVG-NS+MAX-NS', 'AVG+MAX+SUM-NS+MAX-NS', 'AVG+AVG-NS+SUM-NS+MAX-NS', 
                            'SUM+MAX+AVG-NS+SUM-NS', 'SUM+MAX+AVG-NS+MAX-NS', 'SUM+MAX+SUM-NS+MAX-NS', 'SUM+AVG-NS+SUM-NS+MAX-NS', 'MAX+AVG-NS+SUM-NS+MAX-NS']
    
    poolings_large = ['CLS+AVG+SUM+AVG-NS', 'CLS+AVG+SUM+SUM-NS', 'CLS+AVG+AVG-NS+SUM-NS',  'CLS+SUM+AVG-NS+SUM-NS', 'AVG+SUM+AVG-NS+SUM-NS'] + ['CLS+AVG+SUM', 'CLS+AVG+AVG-NS', 'CLS+AVG+SUM-NS', 'CLS+SUM+AVG-NS', 'CLS+SUM+SUM-NS', 'CLS+AVG-NS+SUM-NS', 'AVG+SUM+AVG-NS', 'AVG+SUM+SUM-NS', 'AVG+AVG-NS+SUM-NS' 'SUM+AVG-NS+SUM-NS'] + ['CLS+AVG', 'CLS+SUM', 'CLS+AVG-NS', 'CLS+SUM-NS', 'AVG+SUM', 'AVG+AVG-NS', 'AVG+SUM-NS', 'SUM+AVG-NS', 'SUM+SUM-NS', 'AVG-NS+SUM-NS'] + ['AVG-NS', 'SUM-NS', 'MAX-NS']  + ['CLS', 'AVG', 'SUM', 'MAX']

    pooling_prefixs = []
    
    if poolings_args[0] == 'all':
        pooling_prefixs = simple_poolings + simple_ns_poolings + two_tokens_poolings + three_tokens_poolings + four_tokens_poolings
        return pooling_prefixs
    
    if poolings_args[0] == 'best':
        pooling_prefixs = two_tokens_poolings + three_tokens_poolings + four_tokens_poolings
        return pooling_prefixs
    
    if poolings_args[0] == 'large':
        return poolings_large

    if 'simple' in poolings_args:
        pooling_prefixs = simple_poolings
        return pooling_prefixs
    if 'simple-ns' in poolings_args:
        pooling_prefixs = simple_ns_poolings
        return pooling_prefixs
    if 'two' in poolings_args:
        pooling_prefixs = two_tokens_poolings
        return pooling_prefixs
    if 'three' in poolings_args:
        pooling_prefixs = three_tokens_poolings
        return pooling_prefixs  
    if 'four' in poolings_args:
        pooling_prefixs = four_tokens_poolings
        return pooling_prefixs     
    else:
        return poolings_args

def get_list_layers(final_layer, initial_layer, agg_layers_args):

    list_lyrs_agg_sum = ["SUM-7-11", "SUM-6-10", "SUM-5-9", "SUM-6-11", "SUM-5-10", "SUM-5-11", "SUM-4-10", "SUM-4-11"] + ["SUM-7-12", "SUM-7-10", "SUM-8-11", "SUM-9-12", "SUM-7-9", "SUM-8-10", "SUM-9-11", "SUM-10-12", "SUM-7-8", "SUM-8-9",  "SUM-9-10", "SUM-10-11", "SUM-11-12", "SUM-1-12"] 
    list_lyrs_agg_avg = ["AVG-7-11", "AVG-6-10", "AVG-5-9", "AVG-6-11", "AVG-5-10", "AVG-5-11", "AVG-4-10", "AVG-4-11"] + ["AVG-7-12", "AVG-7-10", "AVG-8-11", "AVG-9-12", "AVG-7-9", "AVG-8-10", "AVG-9-11", "AVG-10-12", "AVG-7-8", "AVG-8-9",  "AVG-9-10", "AVG-10-11", "AVG-11-12", "AVG-1-12"]
    list_lyrs_agg = list_lyrs_agg_sum + list_lyrs_agg_avg

    list_lyrs_agg_sum_large = ["SUM-14-15", "SUM-15-16", "SUM-16-17", "SUM-17-18", "SUM-14-16", "SUM-15-17",  "SUM-16-18", "SUM-14-17", "SUM-15-18"] 
    list_lyrs_agg_avg_large = ["AVG-14-15", "AVG-15-16", "AVG-16-17", "AVG-17-18", "AVG-14-16", "AVG-15-17",  "AVG-16-18", "AVG-14-17", "AVG-15-18"] 
    
    lyrs = []
        
    if agg_layers_args[0] == 'ALL':
        for i in range(initial_layer, final_layer):
            lyrs.append(f"LYR-{i+1}")
        lyrs += list_lyrs_agg
        return lyrs
    
    if agg_layers_args[0] == 'SUMAGGLAYERS':
        return list_lyrs_agg_sum
    
    if agg_layers_args[0] == 'AVGAGGLAYERS':
        return list_lyrs_agg_avg
    
    if agg_layers_args[0] == 'SUMAGGLAYERSLARGE':
        return list_lyrs_agg_sum_large
    
    if agg_layers_args[0] == 'AVGAGGLAYERSLARGE':
        return list_lyrs_agg_avg_large
       
    if agg_layers_args[0] == 'LYR':
        for i in range(initial_layer, final_layer):
            lyrs.append(f"LYR-{i+1}")
        return lyrs
    
    if agg_layers_args[0] == 'BEST':
        return ["BEST"]
    
    else:
        return agg_layers_args

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def batcher(params, batch):
    sentences = [' '.join(sent) for sent in batch]
    return params['encoder']._encode(sentences, params.current_task)

