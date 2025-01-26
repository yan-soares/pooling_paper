import pandas as pd
import math

main_path = "./"

files_base = [
    "tables_final_results/planilha_geral.csv"
]

files_large = [
    "tables_final_results/cl_nhid_0_models_allmini6&allmpnet_epochs_1/cl_acc/cl_nhid_0_models_allmini6&allmpnet_epochs_1_processado_acc.csv",
]

df_baseline_base = "baselines/base_cl.tsv"
df_baseline_large = "baselines/large_cl.tsv"

#BASE
df_baseline = pd.read_csv(df_baseline_base, sep="\t")
dataframes = [pd.read_csv(file) for file in files_base] + [df_baseline]
combined_df = pd.concat(dataframes, ignore_index=True)

combined_df.to_csv("resultados_cl_base_with_baseline.csv", index=False)

combined_df = combined_df.map(
    lambda x: f"{x:.2f}".replace(".", ",") if isinstance(x, (float, int)) else x
)

combined_df.to_csv("resultados_cl_base_with_baseline_google_drive.csv", index=False, sep=";")

#LARGE
df_baseline = pd.read_csv(df_baseline_large, sep="\t")
dataframes = [pd.read_csv(file) for file in files_large] + [df_baseline]
combined_df = pd.concat(dataframes, ignore_index=True)

combined_df.to_csv("resultados_cl_large_with_baseline.csv", index=False)

combined_df = combined_df.map(
    lambda x: f"{x:.2f}".replace(".", ",") if isinstance(x, (float, int)) else x
)

combined_df.to_csv("resultados_cl_large_with_baseline_google_drive.csv", index=False, sep=";")