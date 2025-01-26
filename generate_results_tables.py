import argparse
import pandas as pd
import os
import shutil

MAIN_PATH = "tables_main_experiments"
FINAL_RESULTS_PATH = 'tables_final_results'

cl_paths = [p for p in os.listdir(MAIN_PATH) if p.startswith('cl_')]
si_paths = [p for p in os.listdir(MAIN_PATH) if p.startswith('si_')]

columns_tasks_cl = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
columns_tasks_si = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']

main_colunas = ['Modelo', 'Pooling', 'AGG', 'LAYER', 'Epochs', 'out_vec_size', 'qtd_layers', 'Nhid']
ordem_colunas_cl = main_colunas + columns_tasks_cl 
ordem_colunas_si = main_colunas + columns_tasks_si 

def parse_dict_with_eval(value):
    try:
        if isinstance(value, str):
            value = value.replace('np.float64', 'float')
            return eval(value)
        return {}
    except Exception as e:
        return {}

def tables_classification(cl_paths, columns_tasks_cl, ordem_colunas_cl):
    for clp in cl_paths:
        path_cl = MAIN_PATH + '/' + clp
        path_cl_acc = path_cl + "/" + "cl_acc"
        path_cl_devacc = path_cl + "/" + "cl_devacc"
        os.makedirs(path_cl_acc, exist_ok=True)
        os.makedirs(path_cl_devacc, exist_ok=True)

        if [f for f in os.listdir(path_cl) if f.endswith('_intermediate.csv')]:
            os.remove(MAIN_PATH + '/' + clp + '/' + [f for f in os.listdir(path_cl) if f.endswith('_intermediate.csv')][0])

        cl_file_name = [f for f in os.listdir(path_cl) if f.endswith('.csv')][0]

        caminho_arquivo_cl = os.path.join(path_cl, cl_file_name)
        data = pd.read_csv(caminho_arquivo_cl, encoding="utf-8", on_bad_lines="skip")

        devacc_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling'], 'Epochs': data['Epochs'], 'out_vec_size': data['out_vec_size'], 'qtd_layers': data['qtd_layers'], 'Nhid': data['Nhid']}
        acc_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling'], 'Epochs': data['Epochs'], 'out_vec_size': data['out_vec_size'], 'qtd_layers': data['qtd_layers'], 'Nhid': data['Nhid']}

        for task in columns_tasks_cl:
            devacc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('devacc', None))
            acc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('acc', None))

        devacc_table = pd.DataFrame(devacc_data)
        acc_table = pd.DataFrame(acc_data)        

        devacc_table[['AGG', 'LAYER']] = devacc_table['Pooling'].str.split('_', expand=True)
        acc_table[['AGG', 'LAYER']] = acc_table['Pooling'].str.split('_', expand=True)

        devacc_table = devacc_table[ordem_colunas_cl]
        acc_table = acc_table[ordem_colunas_cl]  

        devacc_table['Avg'] = devacc_table[columns_tasks_cl].mean(axis=1)
        acc_table['Avg'] = acc_table[columns_tasks_cl].mean(axis=1)

        devacc_table.to_csv(os.path.join(path_cl_devacc, cl_file_name.split('.csv')[0] + '_processado_devacc.csv'))
        acc_table.to_csv(os.path.join(path_cl_acc, cl_file_name.split('.csv')[0]) + '_processado_acc.csv')

        os.makedirs(MAIN_PATH + '/processados/' + clp, exist_ok=True)
        shutil.copy(caminho_arquivo_cl, MAIN_PATH + '/processados/' + clp)
        shutil.move(MAIN_PATH + '/' + clp, FINAL_RESULTS_PATH + '/' + clp)

def tables_similarity(si_paths, columns_tasks_si, ordem_colunas_si):
   for slp in si_paths:
        path_si = MAIN_PATH + '/' + slp
        path_si_pearson = path_si + "/" + "si_pearson"
        path_si_spearman = path_si + "/" + "si_spearman"
        os.makedirs(path_si_pearson, exist_ok=True)
        os.makedirs(path_si_spearman, exist_ok=True)

        if [f for f in os.listdir(path_si) if f.endswith('_intermediate.csv')]:
            os.remove(MAIN_PATH + '/' + slp + '/' + [f for f in os.listdir(path_si) if f.endswith('_intermediate.csv')][0])

        si_file_name = [f for f in os.listdir(path_si) if f.endswith('.csv')][0]

        caminho_arquivo_si = os.path.join(path_si, si_file_name)
        data = pd.read_csv(caminho_arquivo_si, encoding="utf-8", on_bad_lines="skip")

        pearson_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling'], 'Epochs': data['Epochs'], 'out_vec_size': data['out_vec_size'], 'qtd_layers': data['qtd_layers'], 'Nhid': data['Nhid']}
        spearman_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling'], 'Epochs': data['Epochs'], 'out_vec_size': data['out_vec_size'], 'qtd_layers': data['qtd_layers'], 'Nhid': data['Nhid']}

        for task in columns_tasks_si:
            pearson_data[task] = data[task].apply(lambda x: (parse_dict_with_eval(x).get('pearson', None).get('mean', None)) * 100)
            spearman_data[task] = data[task].apply(lambda x: (parse_dict_with_eval(x).get('spearman', None).get('mean', None)) * 100)

        pearson_table = pd.DataFrame(pearson_data)
        spearman_table = pd.DataFrame(spearman_data)        

        pearson_table[['AGG', 'LAYER']] = pearson_table['Pooling'].str.split('_', expand=True)
        spearman_table[['AGG', 'LAYER']] = spearman_table['Pooling'].str.split('_', expand=True)

        pearson_table = pearson_table[ordem_colunas_si]
        spearman_table = spearman_table[ordem_colunas_si]  

        pearson_table['Avg'] = pearson_table[columns_tasks_si].mean(axis=1)
        spearman_table['Avg'] = spearman_table[columns_tasks_si].mean(axis=1)

        pearson_table.to_csv(os.path.join(path_si_pearson, si_file_name.split('.csv')[0] + '_processado_pearson.csv'))
        spearman_table.to_csv(os.path.join(path_si_spearman, si_file_name.split('.csv')[0]) + '_processado_spearman.csv')

        os.makedirs(MAIN_PATH + '/processados/' + slp, exist_ok=True)
        shutil.copy(caminho_arquivo_si, MAIN_PATH + '/processados/' + slp)
        shutil.move(MAIN_PATH + '/' + slp, FINAL_RESULTS_PATH + '/' + slp)

def main():
    parser = argparse.ArgumentParser(description="SentEval Experiments")
    parser.add_argument("--task_type", type=str, required=True, default="classification", help="Tipo de tarefa (classification ou similarity)")
    args = parser.parse_args()

    task_type_args = args.task_type 

    if task_type_args == "classification":
        tables_classification(cl_paths, columns_tasks_cl, ordem_colunas_cl)

    elif task_type_args == "similarity":
        tables_similarity(si_paths, columns_tasks_si, ordem_colunas_si)

if __name__ == "__main__":
    main()