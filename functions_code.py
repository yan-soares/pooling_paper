import torch

def get_pooling_techniques(poolings_args, agg_layers_args):

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
                             'SUM+MAX+AVG-NS', 'SUM+MAX+SUM-NS', 'SUM+MAX+MAX-NS', 'SUM+AVG-NS+SUM-NS', 'SUM+AVG-NS+MAX-NS', 'SUM+SUM-NS+MAX-NS', 
                             'MAX+AVG-NS+SUM-NS', 'MAX+AVG-NS+MAX-NS', 'MAX+SUM-NS+MAX-NS', 
                             'AVG-NS+SUM-NS+MAX-NS']
    
    two_tokens_poolings_new = ['AVG+SUM', 'AVG+MAX', 'AVG+AVG-NS', 'AVG+SUM-NS', 'AVG+MAX-NS',
                               'SUM+MAX', 'SUM+AVG-NS', 'SUM+SUM-NS', 'SUM+MAX-NS',
                               'MAX+AVG-NS', 'MAX+SUM-NS', 'MAX+MAX-NS',
                               'AVG-NS+SUM-NS', 'AVG-NS+MAX-NS',
                               'SUM-NS+MAX-NS']
    three_tokens_poolings_new = ['CLS+AVG+SUM', 'CLS+AVG+MAX', 'CLS+SUM+MAX', 'CLS+AVG-NS+SUM-NS', 'CLS+AVG-NS+MAX-NS', 'CLS+SUM-NS+MAX-NS',
                                 'AVG+SUM+MAX', 'AVG+SUM+AVG-NS', 'AVG+SUM+SUM-NS', 'AVG+SUM+MAX-NS', 'AVG+MAX+AVG-NS', 
                                 'AVG+MAX+SUM-NS', 'AVG+MAX+MAX-NS', 'AVG+AVG-NS+SUM-NS', 'AVG+AVG-NS+MAX-NS', 'AVG+SUM-NS+MAX-NS', 
                                 'SUM+MAX+AVG-NS', 'SUM+MAX+SUM-NS', 'SUM+MAX+MAX-NS', 'SUM+AVG-NS+SUM-NS', 'SUM+AVG-NS+MAX-NS', 'SUM+SUM-NS+MAX-NS', 
                                 'MAX+AVG-NS+SUM-NS', 'MAX+AVG-NS+MAX-NS', 'MAX+SUM-NS+MAX-NS', 
                                 'AVG-NS+SUM-NS+MAX-NS']
    
    two_and_three_new = two_tokens_poolings_new + three_tokens_poolings_new

    two_tokens_best = ['CLS+AVG', 'CLS+SUM', 'CLS+AVG-NS', 'CLS+SUM-NS', 'AVG+SUM', 'AVG+AVG-NS', 'AVG+SUM-NS', 'SUM+AVG-NS', 'SUM+SUM-NS', 'AVG-NS+SUM-NS']
    three_tokens_best = ['CLS+AVG+SUM','CLS+AVG+AVG-NS', 'CLS+AVG+SUM-NS', 'CLS+SUM+AVG-NS', 'CLS+SUM+SUM-NS', 'CLS+AVG-NS+SUM-NS', 'AVG+SUM+AVG-NS', 'AVG+SUM+SUM-NS', 'AVG+AVG-NS+SUM-NS', 'SUM+AVG-NS+SUM-NS']

    
    pooling_prefixs = []
    
    if poolings_args[0] == 'all':
        pooling_prefixs = simple_poolings + simple_ns_poolings + two_tokens_poolings + three_tokens_poolings
        return pooling_prefixs

    if 'simple' in poolings_args:
        pooling_prefixs += simple_poolings
    if 'simple-ns' in poolings_args:
        pooling_prefixs += simple_ns_poolings
    if 'two' in poolings_args:
        pooling_prefixs += two_tokens_poolings
    if 'three' in poolings_args:
        pooling_prefixs += three_tokens_poolings  
    if 'twonew' in poolings_args:
        pooling_prefixs += two_tokens_poolings_new
    if 'three_new' in poolings_args:
        pooling_prefixs += three_tokens_poolings_new
    if 'two_and_three_new' in poolings_args:
        pooling_prefixs += two_and_three_new

    if agg_layers_args[0] == 'BEST':
        pooling_prefixs = two_tokens_best + three_tokens_best
        return pooling_prefixs

    return pooling_prefixs      

def get_list_layers(final_layer, initial_layer, agg_layers_args):

    list_lyrs_agg_sum = ["SUM-7-12", "SUM-7-10", "SUM-8-11", "SUM-9-12", "SUM-7-9", "SUM-8-10",  "SUM-9-11", "SUM-10-12", "SUM-7-8", "SUM-8-9",  "SUM-9-10", "SUM-10-11", "SUM-11-12", "SUM-1-12"] 
    list_lyrs_agg_avg = ["AVG-7-12", "AVG-7-10", "AVG-8-11", "AVG-9-12", "AVG-7-9", "AVG-8-10",  "AVG-9-11", "AVG-10-12", "AVG-7-8", "AVG-8-9",  "AVG-9-10", "AVG-10-11", "AVG-11-12", "AVG-1-12"]

    list_lyrs_agg_sum_new = ["SUM-7-9", "SUM-8-10",  "SUM-9-11", "SUM-10-12", "SUM-7-8", "SUM-8-9",  "SUM-9-10", "SUM-10-11", "SUM-11-12"] 
    list_lyrs_agg_avg_new = ["AVG-7-9", "AVG-8-10",  "AVG-9-11", "AVG-10-12", "AVG-7-8", "AVG-8-9",  "AVG-9-10", "AVG-10-11", "AVG-11-12"]

    list_lyrs_agg_sum_last = ["SUM-7-12", "SUM-7-10", "SUM-8-11", "SUM-9-12", "SUM-1-12"] 
    list_lyrs_agg_avg_last = ["AVG-7-12", "AVG-7-10", "AVG-8-11", "AVG-9-12", "AVG-1-12"]

    list_lyrs_agg = list_lyrs_agg_sum + list_lyrs_agg_avg
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
    
    if agg_layers_args[0] == 'NEWSUMAGGLAYERS':
        return list_lyrs_agg_sum_new
    
    if agg_layers_args[0] == 'NEWAVGAGGLAYERS':
        return list_lyrs_agg_avg_new
    
    if agg_layers_args[0] == 'LASTSUMAGGLAYERS':
        return list_lyrs_agg_sum_last
    
    if agg_layers_args[0] == 'LASTAVGAGGLAYERS':
        return list_lyrs_agg_avg_last
       
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

