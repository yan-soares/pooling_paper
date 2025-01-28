import pandas as pd
import math

main_path = "../pooling_paper_results/google_drive_results"
main_path_base = "../pooling_paper_results/google_drive_results/base"
main_path_large = "../pooling_paper_results/google_drive_results/large"

#####FILES BASE
files_base_acc = [
    "/home/yandellwsl/pooling_paper_results/tables_final_results/cl_models_allmpnet_epochs_1_batch_1024_kfold_10_optim_adam_nhid_0_initiallayer_default_finallayer_default_pooling_all_agglayers_AGGLAYERS/cl_acc/cl_models_allmpnet_epochs_1_batch_1024_kfold_10_optim_adam_nhid_0_initiallayer_default_finallayer_default_pooling_all_agglayers_AGGLAYERS_processado_acc.csv",
    "/home/yandellwsl/pooling_paper_results/tables_final_results/cl_models_deberta-base_epochs_1_batch_1024_kfold_10_optim_adam_nhid_0_initiallayer_default_finallayer_default_pooling_all_agglayers_AGGLAYERS/cl_acc/cl_models_deberta-base_epochs_1_batch_1024_kfold_10_optim_adam_nhid_0_initiallayer_default_finallayer_default_pooling_all_agglayers_AGGLAYERS_processado_acc.csv"
]
files_base_devacc = [
    "/home/yandellwsl/pooling_paper_results/tables_final_results/cl_models_allmpnet_epochs_1_batch_1024_kfold_10_optim_adam_nhid_0_initiallayer_default_finallayer_default_pooling_all_agglayers_AGGLAYERS/cl_devacc/cl_models_allmpnet_epochs_1_batch_1024_kfold_10_optim_adam_nhid_0_initiallayer_default_finallayer_default_pooling_all_agglayers_AGGLAYERS_processado_devacc.csv",
    "/home/yandellwsl/pooling_paper_results/tables_final_results/cl_models_deberta-base_epochs_1_batch_1024_kfold_10_optim_adam_nhid_0_initiallayer_default_finallayer_default_pooling_all_agglayers_AGGLAYERS/cl_devacc/cl_models_deberta-base_epochs_1_batch_1024_kfold_10_optim_adam_nhid_0_initiallayer_default_finallayer_default_pooling_all_agglayers_AGGLAYERS_processado_devacc.csv"
]
######FILES LARGE
files_large_acc = []
files_large_devacc = []


#JOINS TABLES BASE
if len(files_base_acc) > 0 and len(files_base_devacc) > 0:

    #ACC
    dataframes_base_acc = [pd.read_csv(file) for file in files_base_acc]
    combined_df_base_acc = pd.concat(dataframes_base_acc, ignore_index=True)
    combined_df_base_acc.to_csv(main_path_base + "/resultados_cl_base_acc.csv", index=False)

    combined_df_base_acc = combined_df_base_acc.map(
        lambda x: f"{x:.2f}".replace(".", ",") if isinstance(x, (float, int)) else x
    )
    combined_df_base_acc.to_csv(main_path_base + "/resultados_cl_base_acc_google_drive.csv", index=False, sep=";")

    #DEVACC
    dataframes_base_devacc = [pd.read_csv(file) for file in files_base_devacc]
    combined_df_base_devacc = pd.concat(dataframes_base_devacc, ignore_index=True)
    combined_df_base_devacc.to_csv(main_path_base + "/resultados_cl_base_devacc.csv", index=False)

    combined_df_base_devacc = combined_df_base_devacc.map(
        lambda x: f"{x:.2f}".replace(".", ",") if isinstance(x, (float, int)) else x
    )
    combined_df_base_devacc.to_csv(main_path_base + "/resultados_cl_base_devacc_google_drive.csv", index=False, sep=";")