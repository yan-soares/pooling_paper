import torch

def get_agg_base():
    list_sum_agg = []
    list_avg_agg = []

    ranges = list(range(1, 13))

    slices = {}
    for size in range(2, 13):  # De 2 até 12
        slices[size] = [f"SUM-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_sum_agg+=groups

    slices = {}
    for size in range(2, 13):  # De 2 até 12
        slices[size] = [f"AVG-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_avg_agg+=groups
    
    return list_sum_agg, list_avg_agg

def get_pooling_techniques(poolings_args, name_agg):

    simple_poolings = ['CLS', 'AVG', 'SUM', 'MAX']
    simple_ns_poolings = ['AVG-NS', 'SUM-NS', 'MAX-NS'] 
    simple_nostop_poolings = ['AVG-NOSTOP', 'SUM-NOSTOP', 'MAX-NOSTOP']
    simple_ns_nostop_poolings = ['AVG-NS-NOSTOP', 'SUM-NS-NOSTOP', 'MAX-NS-NOSTOP']
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
        pooling_prefixs = simple_poolings + simple_ns_poolings + simple_nostop_poolings + simple_ns_nostop_poolings + two_tokens_poolings + three_tokens_poolings + four_tokens_poolings
        return pooling_prefixs
    
    if poolings_args[0] == 'best':
        pooling_prefixs = two_tokens_poolings + three_tokens_poolings + four_tokens_poolings
        return pooling_prefixs
    
    if poolings_args[0] == 'best_new':
        pooling_prefixs = ["CLS+AVG", "CLS+SUM", "CLS+AVG-NS", "CLS+SUM-NS", "AVG+SUM", "AVG+AVG-NS", "AVG+SUM-NS", "SUM+AVG-NS",
                           "SUM+SUM-NS", "AVG-NS+SUM-NS", "CLS+AVG+SUM", "CLS+AVG+AVG-NS", "CLS+AVG+SUM-NS", "CLS+SUM+AVG-NS", "CLS+SUM+SUM-NS", 
                           "CLS+AVG-NS+SUM-NS", "AVG+SUM+AVG-NS", "AVG+SUM+SUM-NS", "AVG+AVG-NS+SUM-NS", "SUM+AVG-NS+SUM-NS"]
        return pooling_prefixs
    
    if poolings_args[0] == 'large':
        return poolings_large

    if 'simple' in poolings_args:
        pooling_prefixs += simple_poolings
        #return pooling_prefixs
    if 'simple-ns' in poolings_args:
        pooling_prefixs += simple_ns_poolings
        #return pooling_prefixs
    if 'simple-nostop' in poolings_args:
        pooling_prefixs += simple_nostop_poolings
        #return pooling_prefixs
    if 'simple-ns-nostop' in poolings_args:
        pooling_prefixs += simple_ns_nostop_poolings
        #return pooling_prefixs
    if 'two' in poolings_args:
        pooling_prefixs += two_tokens_poolings
        #return pooling_prefixs
    if 'three' in poolings_args:
        pooling_prefixs += three_tokens_poolings
        #return pooling_prefixs  
    if 'four' in poolings_args:
        pooling_prefixs += four_tokens_poolings
        #return pooling_prefixs     
    
    
    if len(pooling_prefixs) > 0:
        return pooling_prefixs
    else:
        return poolings_args

def get_list_layers(final_layer, initial_layer, agg_layers_args):

    list_lyrs_agg_sum_last = ["SUM-5-6", "SUM-6-7", "SUM-7-8", "SUM-8-9", "SUM-9-10", "SUM-10-11", "SUM-11-12"] + ["SUM-5-7", "SUM-6-8", "SUM-7-9", "SUM-8-10", "SUM-9-11", "SUM-10-12"] + ["SUM-5-8", "SUM-6-9", "SUM-7-10", "SUM-8-11", "SUM-9-12"] + ["SUM-5-9", "SUM-6-10", "SUM-7-11", "SUM-8-12"] + ["SUM-5-10", "SUM-6-11", "SUM-7-12"] + ["SUM-5-11", "SUM-6-12"] + ["SUM-5-12"]
    list_lyrs_agg_avg_last = ["AVG-5-6", "AVG-6-7", "AVG-7-8", "AVG-8-9", "AVG-9-10", "AVG-10-11", "AVG-11-12"] + ["AVG-5-7", "AVG-6-8", "AVG-7-9", "AVG-8-10", "AVG-9-11", "AVG-10-12"] + ["AVG-5-8", "AVG-6-9", "AVG-7-10", "AVG-8-11", "AVG-9-12"] + ["AVG-5-9", "AVG-6-10", "AVG-7-11", "AVG-8-12"] + ["AVG-5-10", "AVG-6-11", "AVG-7-12"] + ["AVG-5-11", "AVG-6-12"] + ["AVG-5-12"]
    list_lyrs_agg_sum, list_lyrs_agg_avg = get_agg_base()
    list_lyrs_agg = list_lyrs_agg_sum + list_lyrs_agg_avg

    list_lyrs_last = list_lyrs_agg_sum_last + list_lyrs_agg_avg_last

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
    
    if agg_layers_args[0] == 'SUMAGGLAYERSLAST':
        return list_lyrs_agg_sum_last
    
    if agg_layers_args[0] == 'AVGAGGLAYERSLAST':
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

