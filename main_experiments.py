import senteval
from transformers import AutoTokenizer, AutoModelForMaskedLM, DebertaV2Model, DebertaV2Tokenizer, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AutoModel
import torch
import argparse
import pandas as pd
import logging
import os
import functions_code

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class SentenceEncoder:
    def __init__(self, model_name, device):
        self.device = device
        self.size_embedding = None
        self.pooling_strategy = None
        self.print_best_layers = None

        self.general_embeddings = {}
        self.list_poolings = None
        self.list_layers = None
        self.actual_layer = None

        if model_name == 'bert-base' or  model_name == 'bert-large':
            if model_name == 'bert-base':
                self.name_model = 'google-bert/bert-base-uncased'
                self.qtd_layers = 12
            if model_name == 'bert-large':
                self.name_model = 'google-bert/bert-large-uncased'
                self.qtd_layers = 24
            self.tokenizer = BertTokenizer.from_pretrained(self.name_model)
            self.model = BertModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'roberta-base' or  model_name == 'roberta-large':
            if model_name == 'roberta-base':
                self.name_model = 'FacebookAI/roberta-base'
                self.qtd_layers = 12
            if model_name == 'roberta-large':
                self.name_model = 'FacebookAI/roberta-large'
                self.qtd_layers = 24
            self.tokenizer = RobertaTokenizer.from_pretrained(self.name_model)
            self.model = RobertaModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'deberta-base' or model_name == 'deberta-large':
            if model_name == 'deberta-base':
                self.name_model = 'microsoft/deberta-v3-base'
                self.qtd_layers = 12
            if model_name == 'deberta-large':
                self.name_model = 'microsoft/deberta-v3-large'
                self.qtd_layers = 24
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.name_model)
            self.model = DebertaV2Model.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'angle-base' or model_name == 'angle-large':       
            if model_name == 'angle-base':
                self.name_model = 'SeanLee97/angle-bert-base-uncased-nli-en-v1'
                self.qtd_layers = 12
            if model_name == 'angle-large':
                self.name_model = 'WhereIsAI/UAE-Large-V1'
                self.qtd_layers = 24            
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            self.model = AutoModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device) 

        if model_name == 'allmpnet':
            self.name_model = 'sentence-transformers/all-mpnet-base-v2'
            self.qtd_layers = 12
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            self.model = AutoModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)
    
    def _encode(self, sentences, current_task, batch_size=2048): 
        tokens = self.tokenizer(
            sentences, padding="longest", truncation=True, return_tensors="pt", max_length = self.model.config.max_position_embeddings
        )

        tokens = {key: val.to(self.device) for key, val in tokens.items()}

        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_tokens = {key: val[i:i+batch_size] for key, val in tokens.items()}
            with torch.no_grad(), torch.amp.autocast('cuda'):
                output = self.model(**batch_tokens)
                embeddings = self._apply_pooling(output, batch_tokens['attention_mask'])

                del batch_tokens, output
                torch.cuda.empty_cache()

                all_embeddings.append(embeddings)           

        self.size_embedding = all_embeddings[0].shape   

        final_embeddings = torch.cat(all_embeddings, dim=0).to('cpu').numpy()
        return final_embeddings
   
    def _mean_pooling_exclude_cls_sep(self, output, attention_mask):
 
        # Exclui o CLS removendo o primeiro token
        embeddings = output[:, 1:-1, :]  # Remove CLS (primeiro) e SEP (último)
        attention_mask = attention_mask[:, 1:-1]  # Remove CLS e SEP na máscara também

        # Expande a máscara para corresponder ao tamanho dos embeddings
        expanded_mask = attention_mask.unsqueeze(-1)  # (batch_size x seq_len-2 x 1)

        # Aplica a máscara para excluir padding
        masked_embeddings = embeddings * expanded_mask

        # Soma os embeddings válidos e calcula a média
        sum_embeddings = masked_embeddings.sum(dim=1)  # Soma ao longo da sequência (dim=1)
        valid_token_counts = expanded_mask.sum(dim=1)  # Soma a quantidade de tokens válidos
        mean_pooled_embeddings = sum_embeddings / valid_token_counts.clamp(min=1e-9)  # Evita divisão por zero

        return mean_pooled_embeddings
    
    def _sum_pooling_exclude_cls_sep(self, output, attention_mask):

        # Exclui o CLS removendo o primeiro token e exclui o SEP removendo o último token
        embeddings = output[:, 1:-1, :]  # Remove CLS (primeiro) e SEP (último)
        attention_mask = attention_mask[:, 1:-1]  # Remove CLS e SEP na máscara também

        # Expande a máscara para corresponder ao tamanho dos embeddings
        expanded_mask = attention_mask.unsqueeze(-1)  # (batch_size x seq_len-2 x 1)

        # Aplica a máscara para excluir padding
        masked_embeddings = embeddings * expanded_mask

        # Soma os embeddings válidos ao longo da sequência
        sum_pooled_embeddings = masked_embeddings.sum(dim=1)

        return sum_pooled_embeddings

    def _max_pooling_exclude_cls_sep(self, output, attention_mask):

        # Exclui o CLS removendo o primeiro token e exclui o SEP removendo o último token
        embeddings = output[:, 1:-1, :]  # Remove CLS (primeiro) e SEP (último)
        attention_mask = attention_mask[:, 1:-1]  # Remove CLS e SEP na máscara também

        # Expande a máscara para corresponder ao tamanho dos embeddings
        expanded_mask = attention_mask.unsqueeze(-1)  # (batch_size x seq_len-2 x 1)

        # Substitui tokens de padding por um valor muito pequeno (-inf) para ignorá-los
        masked_embeddings = embeddings.masked_fill(expanded_mask == 0, float('-inf'))

        # Seleciona o valor máximo ao longo da sequência
        max_pooled_embeddings = masked_embeddings.max(dim=1).values

        return max_pooled_embeddings
    
    def _simple_pooling(self, hidden_state, attention_mask, name_pooling):

         match name_pooling:
             
            case "CLS":
                return hidden_state[:, 0, :]
            
            case "AVG":
                return ((hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
            
            case "SUM":
                return (hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
            
            case "MAX":
                return torch.max(hidden_state.masked_fill(attention_mask.unsqueeze(-1).expand(hidden_state.size()).float() == 0, -1e9), dim=1)[0]
             
            case "AVG-NS":
                return self._mean_pooling_exclude_cls_sep(hidden_state, attention_mask)
            
            case "SUM-NS":
                return self._sum_pooling_exclude_cls_sep(hidden_state, attention_mask)
            
            case "MAX-NS":
                return self._max_pooling_exclude_cls_sep(hidden_state, attention_mask)  

    def _get_pooling_result(self, hidden_state, attention_mask, name_pooling, name_agg):

        name_pooling_split = name_pooling.split('+')
        self.print_best_layers =  "NORMAL"

        match len(name_pooling_split):

            case 1:
                return self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0])
            
            case 2:
                return torch.cat(
                (
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0]),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1])
                ), 
                dim=1)
            
            case 3:
                return torch.cat(
                (
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0]),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1]),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[2])
                ), 
                dim=1)
            
            case 4:
                return torch.cat(
                (
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0]),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1]),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[2]),
                    self._simple_pooling(hidden_state, attention_mask, name_pooling_split[3])
                ), 
                dim=1)

    def get_best_pooling(self, hidden_state, attention_mask, name_pooling, name_agg):

        if name_agg == 'BEST':
            match self.name_model:
                case 'sentence-transformers/all-mpnet-base-v2':
                    SUM_7_12_hidden = torch.stack(hidden_state[7:13], dim=0).sum(dim=0) 
                    AVG_1_12_hidden = torch.stack(hidden_state[1:13], dim=0).mean(dim=0)
                    self.print_best_layers =  "cls_SUM-7-12_avg_SUM-7-12_sum_AVG-1-12_avgns_SUM-7-12_sumns_AVG-1-12"

                    cls_result = SUM_7_12_hidden[:, 0, :]
                    avg_result = ((SUM_7_12_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                    sum_result = (AVG_1_12_hidden * attention_mask.unsqueeze(-1)).sum(dim=1)
                    avg_ns_result = self._mean_pooling_exclude_cls_sep(SUM_7_12_hidden, attention_mask)
                    sum_ns_result = self._sum_pooling_exclude_cls_sep(AVG_1_12_hidden, attention_mask)
                case 'microsoft/deberta-v3-base':
                    SUM_11_12_hidden = torch.stack(hidden_state[11:13], dim=0).sum(dim=0) 
                    SUM_7_10_hidden = torch.stack(hidden_state[7:11], dim=0).sum(dim=0) 
                    SUM_8_11_hidden = torch.stack(hidden_state[8:12], dim=0).sum(dim=0)
                    LYR_9 = hidden_state[9]
                    self.print_best_layers =  "cls_SUM-11-12_avg_SUM-7-10_sum_LYR-9_avgns_SUM-8-11_sumns_LYR-9"

                    cls_result = SUM_11_12_hidden[:, 0, :]
                    avg_result = ((SUM_7_10_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                    sum_result = (LYR_9 * attention_mask.unsqueeze(-1)).sum(dim=1)
                    avg_ns_result = self._mean_pooling_exclude_cls_sep(SUM_8_11_hidden, attention_mask)
                    sum_ns_result = self._sum_pooling_exclude_cls_sep(LYR_9, attention_mask)

        match name_pooling:
       
            case "CLS+AVG":
                return torch.cat((cls_result, avg_result), dim=1)
            case "CLS+SUM":
                return torch.cat((cls_result, sum_result), dim=1)
            case "CLS+AVG-NS":
                return torch.cat((cls_result, avg_ns_result), dim=1)
            case "CLS+SUM-NS":
                return torch.cat((cls_result, sum_ns_result), dim=1)
            case "AVG+SUM":
                return torch.cat((avg_result, sum_result), dim=1)
            case "AVG+AVG-NS":
                return torch.cat((avg_result, avg_ns_result), dim=1)
            case "AVG+SUM-NS":
                return torch.cat((avg_result, sum_ns_result), dim=1)
            case "SUM+AVG-NS":
                return torch.cat((sum_result, avg_ns_result), dim=1)
            case "SUM+SUM-NS":
                return torch.cat((sum_result, sum_ns_result), dim=1)
            case "AVG-NS+SUM-NS":
                return torch.cat((avg_ns_result, sum_ns_result), dim=1)

            case "CLS+AVG+SUM":
                return torch.cat((cls_result, avg_result, sum_result), dim=1)
            case "CLS+AVG+AVG-NS":
                return torch.cat((cls_result, avg_result, avg_ns_result), dim=1)
            case "CLS+AVG+SUM-NS":
                return torch.cat((cls_result, avg_result, sum_ns_result), dim=1)
            case "CLS+SUM+AVG-NS":
                return torch.cat((cls_result, sum_result, avg_ns_result), dim=1)
            case "CLS+SUM+SUM-NS":
                return torch.cat((cls_result, sum_result, sum_ns_result), dim=1)
            case "CLS+AVG-NS+SUM-NS":
                return torch.cat((cls_result, avg_ns_result, sum_ns_result), dim=1)
            case "AVG+SUM+AVG-NS":
                return torch.cat((avg_result, sum_result, avg_ns_result), dim=1)
            case "AVG+SUM+SUM-NS":
                return torch.cat((avg_result, sum_result, sum_ns_result), dim=1)
            case "AVG+AVG-NS+SUM-NS":
                return torch.cat((avg_result, avg_ns_result, sum_ns_result), dim=1)
            case "SUM+AVG-NS+SUM-NS":
                return torch.cat((sum_result, avg_ns_result, sum_ns_result), dim=1)
            
    def _apply_pooling(self, output, attention_mask):  

        hidden_states = output.hidden_states
        name_pooling = self.pooling_strategy.split("_")[0]
        name_agg = self.pooling_strategy.split("_")[-1]

        if name_agg == 'BEST':
             return self.get_best_pooling(hidden_states, attention_mask, name_pooling, name_agg)

        if name_agg.startswith("LYR"):
            layer_idx = int(name_agg.split('-')[-1])   
            LYR_hidden =  hidden_states[layer_idx]            
            return self._get_pooling_result(LYR_hidden, attention_mask, name_pooling, "LYR")        
        else:        
            name_agg_type = name_agg.split("-")[0]
            agg_initial_layer = int(name_agg.split("-")[1])
            agg_final_layer = int(name_agg.split("-")[2])
            
            match name_agg_type:  

                case "SUM":
                    return self._get_pooling_result(torch.stack(hidden_states[agg_initial_layer:agg_final_layer+1], dim=0).sum(dim=0), attention_mask, name_pooling, name_agg)
                    
                case "AVG":
                    return self._get_pooling_result(torch.stack(hidden_states[agg_initial_layer:agg_final_layer+1], dim=0).mean(dim=0), attention_mask, name_pooling, name_agg)                  
        
    def _strategies_pooling_list (self, initial_layer_args, final_layer_args, poolings_args, agg_layers_args):
        
        #POOLING
        pooling_techniques = functions_code.get_pooling_techniques(poolings_args, agg_layers_args)
        
        #LAYERS
        if initial_layer_args is not None:
            initial_layer = initial_layer_args
        else:
            initial_layer = int(self.qtd_layers / 2)

        if final_layer_args is not None:
            final_layer = final_layer_args
        else:
            final_layer = int(self.qtd_layers)

        list_lyrs = functions_code.get_list_layers(final_layer, initial_layer, agg_layers_args)

        #STRATEGIES CONCAT
        pooling_strategies = []
        for l in list_lyrs:
            for p in pooling_techniques:
                pooling_strategies.append(p + "_" + l) 

        #RETURN
        return pooling_strategies, pooling_techniques, list_lyrs

def run_senteval(model_name, tasks, epochs, nhid_number, initial_layer_args, final_layer_args, poolings_args, agg_layers_args, type_task, batch_args, optim_args, kfold_args):

    results_general = {}

    device = functions_code.get_device()
    print(f"\nExecuting Device: {device}")
    
    encoder = SentenceEncoder(model_name, device)
    pooling_strategies, list_poolings, list_layers = encoder._strategies_pooling_list(initial_layer_args, final_layer_args, poolings_args, agg_layers_args)

    #GET ALL EMBEDDINGS
    print("LISTA DE POOLINGS: ", list_poolings)
    print("LISTA DE LAYERS: ", list_layers)
    
    for pooling in pooling_strategies:
        encoder.pooling_strategy = pooling
        print(f"Running: Model={encoder.name_model}, Pooling={encoder.pooling_strategy}")
        if type_task == 'cl':
            senteval_params = {
                'task_path': 'data',
                'usepytorch': True,
                'kfold': kfold_args,
                'classifier': {
                    'nhid': nhid_number,
                    'optim': optim_args,
                    'batch_size': batch_args,
                    'tenacity': 5,
                    'epoch_size': epochs
                },
                'encoder': encoder
            }
        else:
             senteval_params = {
                'task_path': 'data',
                'usepytorch': True,
                'kfold': 10,
                'encoder': encoder
            }
        se = senteval.engine.SE(senteval_params, functions_code.batcher)
        results_general[pooling] = se.eval(tasks)
        results_general[pooling]['out_vec_size'] = encoder.size_embedding
        results_general[pooling]['qtd_layers'] = encoder.qtd_layers
        results_general[pooling]['best_layers'] = encoder.print_best_layers
        print(encoder.size_embedding)
        print(f"\nBEST LAYERS: {encoder.print_best_layers}")
                
    return results_general

def tasks_run(models_args, epochs_args, nhid_args, main_path, initial_layer_args, final_layer_args, poolings_args, agg_layers_args, filename_task, tasks_list, type_task, batch_args, optim_args, kfold_args):
    path_created = main_path + '/' + filename_task
    os.makedirs(path_created, exist_ok=True)

    logging.basicConfig(
        filename=path_created + '/' + filename_task + '_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    results_data = []

    for model_name in models_args:
        print(f"\nExecuting Model: {model_name}")
        results = run_senteval(model_name, tasks_list, epochs_args, nhid_args, initial_layer_args, final_layer_args, poolings_args, agg_layers_args, type_task, batch_args, optim_args, kfold_args)
        for pooling, res in results.items():
            if type_task == 'cl':
                dict_results = [res.get(task, {}) for task in tasks_list]
            elif type_task == 'si':
                dict_results = [res.get(task, {}).get('all', 0) for task in tasks_list[:5]] + [res.get(task, {}) for task in tasks_list[-2:]]
            
            results_data.append({
                "model": model_name,
                "pooling": pooling,
                "out_vec_size": res.get('out_vec_size'),
                "best_layers": res.get('best_layers'),
                "epochs": epochs_args,
                "nhid": nhid_args,
                "qtd_layers": res.get('qtd_layers'),              
                **{task: dict_results[i] for i, task in enumerate(tasks_list)}
            })
        
        final_df1 = pd.DataFrame(results_data)
        final_df1.to_csv(path_created + '/' + filename_task + '_intermediate.csv', index=False)
                    
    final_df = pd.DataFrame(results_data)
    final_df.to_csv(path_created + '/' + filename_task + '.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description="SentEval Experiments")
    parser.add_argument("--task_type", type=str, required=True, choices=['classification', 'similarity'], help="Tipo de tarefa (classification ou similarity)")
    parser.add_argument("--models", type=str, required=True, help="Modelos separados por vírgula (sem espaços)")
    parser.add_argument("--epochs", type=int, required=True, help="Número máximo de épocas do classificador linear")
    parser.add_argument("--batch", type=int, required=True, help="Batch Size do classificador")
    parser.add_argument("--kfold", type=int, required=True, help="KFold para validação")
    parser.add_argument("--optim", type=str, required=True, help="otimizador do classificador")
    parser.add_argument("--nhid", type=int, required=True, help="Numero de camadas ocultas (0 = Logistic Regression, 1 ou mais = MLP)")
    parser.add_argument("--initial_layer", type=int, help="Camada inicial para execução dos experimentos (default metade superior)")
    parser.add_argument("--final_layer", type=int, help="Camada inicial para execução dos experimentos (default metade superior)")
    parser.add_argument("--poolings", type=str, required=True, default="all", help="Poolings separados por virgula (sem espacos) ou simple, simple-ns, two, three")
    parser.add_argument("--agg_layers", type=str, required=True, default="ALL", help="agg layers separados por virgula (sem espacos)")
    parser.add_argument("--tasks", type=str, help="tasks separados por virgula (sem espacos)")
    args = parser.parse_args()

    task_type_args = args.task_type 
    models_args = args.models.split(",")        
    epochs_args = args.epochs 
    batch_args = args.batch 
    kfold_args = args.kfold     
    optim_args = args.optim 
    nhid_args = args.nhid
    initial_layer_args = args.initial_layer 
    final_layer_args = args.final_layer
    poolings_args = args.poolings.split(",")
    agg_layers_args = args.agg_layers.split(",")  

    main_path = '../pooling_paper_results/main_experiments_tables_final_07022025'   

    initial_layer_args_print = args.initial_layer if args.initial_layer is not None else "default"
    final_layer_args_print = args.final_layer if args.final_layer is not None else "default"

    filename_task = ('_models_' + '&'.join([st for st in models_args]) + 
                     '_epochs_' + str(epochs_args) + 
                     '_batch_' + str(batch_args) +
                     '_kfold_' + str(kfold_args) +
                     '_optim_' + str(optim_args) +
                     '_nhid_' + str(nhid_args) + 
                     '_initiallayer_' + str(initial_layer_args_print) + 
                     '_finallayer_' + str(final_layer_args_print) +
                     '_pooling_' + '&'.join([st for st in poolings_args]) + 
                     '_agglayers_' + '&'.join([st for st in agg_layers_args])
                     )

    if task_type_args == "classification":      
        filename_cl = "cl" + filename_task
        classification_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']        
        classification_tasks = args.tasks.split(",") if args.tasks is not None else classification_tasks
        tasks_run(models_args, epochs_args, nhid_args, main_path, initial_layer_args, final_layer_args, poolings_args, agg_layers_args, filename_cl, classification_tasks, 'cl', batch_args, optim_args, kfold_args)

    elif task_type_args == "similarity":
        filename_si = "si" + filename_task
        similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        similarity_tasks = args.tasks.split(",") if args.tasks is not None else similarity_tasks
        tasks_run(models_args, epochs_args, nhid_args, main_path, initial_layer_args, final_layer_args, poolings_args, agg_layers_args, filename_si, similarity_tasks, 'si', batch_args, optim_args, kfold_args)

if __name__ == "__main__":
    main()