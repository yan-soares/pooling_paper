import torch

def get_pooling_techniques(poolings_args, agg_layers_args):
    
    simple_poolings = ['CLS', 'AVG', 'SUM', 'MAX']
    simple_ns_poolings = ['AVG-NS', 'SUM-NS', 'MAX-NS']
    two_tokens_poolings = ['CLS+AVG', 'CLS+SUM', 'CLS+MAX', 'CLS+AVG-NS', 'CLS+SUM-NS', 'CLS+MAX-NS']
    three_tokens_poolings = ['CLS+AVG+AVG-NS', 'CLS+AVG+SUM-NS', 'CLS+AVG+MAX-NS', 
                            'CLS+SUM+AVG-NS', 'CLS+SUM+SUM-NS', 'CLS+SUM+MAX-NS', 
                            'CLS+MAX+AVG-NS', 'CLS+MAX+SUM-NS', 'CLS+MAX+MAX-NS']
    pooling_prefixs = []

    if agg_layers_args[0] == 'BEST':
        pooling_prefixs = two_tokens_poolings + three_tokens_poolings
        return pooling_prefixs
    
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
    else:
        pooling_prefixs += poolings_args

    return pooling_prefixs      

def get_pooling_strategies_with_layers(qtd_layers, pooling_techniques, initial_layer, agg_layers_args):

    temp_lyr = []
    lyrs = []

    #ONE LAYER STRATEGIES
    LYR = []
    for i in range(initial_layer, qtd_layers):
        temp_lyr.append(f"LYR-{i+1}")
        for p in pooling_techniques:
            LYR.append(p + f"_LYR-{i+1}")
    
    #AGGREGATE LAYERS STRATEGIES
    SUML4L = []
    AVGL4L = []
    SUML6L = []
    AVGL6L = []
    SUML4BL = []
    AVGL4BL = []
    SUMALL = []
    AVGALL = []
    BEST = []
    for p in pooling_techniques:
        SUML4L.append(p + f"_SUML4L") 
        AVGL4L.append(p + f"_AVGL4L") 
        SUML6L.append(p + f"_SUML6L") 
        AVGL6L.append(p + f"_AVGL6L") 
        SUML4BL.append(p + f"_SUML4BL")
        AVGL4BL.append(p + f"_AVGL4BL") 
        SUMALL.append(p + f"_SUMALL")
        AVGALL.append(p + f"_AVGALL") 
        BEST.append(p + f"_BEST") 

    ####GET POOLINGS + LAYERS        
    if agg_layers_args[0] == 'ALL':
        lyrs = temp_lyr + ["SUML4L"] + ["AVGL4L"] + ["SUML6L" ]+ ["AVGL6L"] + ["SUML4BL"] + ["AVGL4BL"] + ["SUMALL"] + ["AVGALL"]
        return LYR + SUML4L + AVGL4L + SUML6L + AVGL6L + SUML4BL + AVGL4BL + SUMALL + AVGALL, lyrs
    if agg_layers_args[0] == 'AGGLAYERS':
        lyrs = ["SUML4L"] + ["AVGL4L"] + ["SUML6L" ]+ ["AVGL6L"] + ["SUML4BL"] + ["AVGL4BL"] + ["SUMALL"] + ["AVGALL"]
        return SUML4L + AVGL4L + SUML6L + AVGL6L + SUML4BL + AVGL4BL + SUMALL + AVGALL, lyrs
    if agg_layers_args[0] == 'BEST':
        return BEST, ["BEST"]
    
    pooling_strategies_with_layers = []

    if 'LYR' in agg_layers_args:
        pooling_strategies_with_layers += LYR
        lyrs += temp_lyr
    if 'SUML4L' in agg_layers_args:
        pooling_strategies_with_layers += SUML4L
        lyrs += ["SUML4L"]
    if 'AVGL4L' in agg_layers_args:
        pooling_strategies_with_layers += AVGL4L
        lyrs += ["AVGL4L"]
    if 'SUML6L' in agg_layers_args:
        pooling_strategies_with_layers += SUML6L
        lyrs += ["SUML6L"]
    if 'AVGL6L' in agg_layers_args:
        pooling_strategies_with_layers += AVGL6L
        lyrs += ["AVGL6L"]
    if 'SUML4BL' in agg_layers_args:
        pooling_strategies_with_layers += SUML4BL
        lyrs += ["SUML4BL"]
    if 'AVGL4BL' in agg_layers_args:
        pooling_strategies_with_layers += AVGL4BL
        lyrs += ["AVGL4BL"]      
    if 'SUMALL' in agg_layers_args:
        pooling_strategies_with_layers += SUMALL
        lyrs += ["SUMALL"] 
    if 'AVGALL' in agg_layers_args:
        pooling_strategies_with_layers += AVGALL  
        lyrs += ["AVGALL"] 
    
    return pooling_strategies_with_layers, lyrs

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def batcher(params, batch):
    sentences = [' '.join(sent) for sent in batch]
    return params['encoder']._encode(sentences)

def batcher_generate_all(params, batch):
    sentences = [' '.join(sent) for sent in batch]
    return params['encoder']._encode_generate_all(sentences, params.current_task)

