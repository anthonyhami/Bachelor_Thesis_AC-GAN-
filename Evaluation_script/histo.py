import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def return_df_hist( df, REAL_OR_FAKE,  bins=20, density=False) :
    df_hist = pd.DataFrame( columns =  ["sample_ID", "DNA_meth_bin", "num_CpGs", "Real_or_Fake"])
    bin_edges_1=np.arange(0, 105, 5)
    for index, row in df.iterrows():
        hist_list, bin_edges_list = np.histogram(row.values.tolist(), bins=bin_edges_1*0.01, density=density)
        for hist, bin_edges in zip(hist_list, bin_edges_1):
            df_hist.loc[len(df_hist)] = [class_name + "_" + str(index), bin_edges*0.01, hist, REAL_OR_FAKE]
    return df_hist



df_real = pd.read_csv("976.csv", sep=";", header=None)  # TODO: HIER AUS ÄNDERN
df= pd.read_csv("Dx_Families.csv",sep=";")
df_Dx_families_summary= pd.read_csv("Dx_families_summary.csv",sep=";")

for index_family in range(39):
    df_fake = pd.read_csv("fake_orginal_" + str(index_family) +".csv", sep=";", header=None)

    class_name = df_Dx_families_summary[df_Dx_families_summary["Dx_num"] == index_family]["Dx"].values[0]

    filtered_df = df[df['Dx']==class_name].index.tolist()
    df_filtered_class_name=df_real.iloc[filtered_df,:]
    df_filtered_class_name


    df_hist_real= return_df_hist( df_filtered_class_name, REAL_OR_FAKE = "real")
    df_hist_fake= return_df_hist( df_fake, REAL_OR_FAKE = "fake" )
    df_hist_all = df_hist_real.append(df_hist_fake, ignore_index=True)

    # change numberic type
    df_hist_all.dtypes
    df_hist_all.DNA_meth_bin = df_hist_all.DNA_meth_bin.astype(float)
    df_hist_all.num_CpGs = df_hist_all.num_CpGs.astype(float)
    df_hist_all.dtypes


    plt.figure(figsize=(5
        ,5))

    sns_plot = sns.lineplot(
        data=df_hist_all, x="DNA_meth_bin", y="num_CpGs", hue="Real_or_Fake", err_style="bars", ci=95,
        markers=True, dashes=False
    )

    plt.savefig("lineplot_" + class_name +".png")
    plt.clf()
