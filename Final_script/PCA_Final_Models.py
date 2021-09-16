import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py

from sklearn.metrics import confusion_matrix, classification_report


# """
def centroid(PCA1, PCA2):

    _len = len(PCA1)
    _x = sum(PCA1) / _len
    _y = sum(PCA2) / _len
    return (_x, _y)


def return_df_centroids(df_grouped):

    # iterate over each group
    centroid_x, centroid_y, centroid_name = [], [], []
    for group_name, df_group in df_grouped:

        print("group_name ===", group_name)
        print(df_group)

        PCA1 = df_group["principal component 1"].values
        PCA2 = df_group["principal component 2"].values

        centroid_group = centroid(PCA1, PCA2)
        centroid_x.append(centroid_group[0])
        centroid_y.append(centroid_group[1])
        centroid_name.append(group_name)
    return centroid_x, centroid_y, centroid_name





# """



def plot_all_points(
    Dx, finalDf, y_true_name, fakeresult, class_index,
):

    # all colors
    unique_colour = Dx.iloc[:, 0].unique()
    print("unique_colour----------")
    print(unique_colour)
    print("len(unique_colour--------------")
    print(len(unique_colour))
    # colors = [np.random.rand(3,) for i in range(len(unique_colour))]
    colors = [
        "red" if i == y_true_name else "gray" for i in unique_colour
    ]
    data_color = []
    for i, j in zip(unique_colour, colors):
        print("%20s, %5s" % (i, j))
        data_color.append([i, j])

    df_colors = pd.DataFrame(data_color, columns=["unique_colour", "colors"])
    df_colors

    targets = Dx.to_numpy()

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=30)
    ax.set_ylabel("Principal Component 2", fontsize=30)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_title("2 component PCA", fontsize=30,fontweight='bold')


    count_red_points = 0
    for target in targets:
        color = df_colors[df_colors["unique_colour"] == target[0]]["colors"]  # [0]
        color = color.values
        indicesToKeep = finalDf.iloc[:, -1] == target[0]
        if color == "red":

            ax.scatter(
                finalDf.loc[indicesToKeep, "principal component 1"],
                finalDf.loc[indicesToKeep, "principal component 2"],
                color="red",
                s=50,
                label=target,
            )
            count_red_points = count_red_points +1
        else:
            ax.scatter(
                finalDf.loc[indicesToKeep, "principal component 1"],
                finalDf.loc[indicesToKeep, "principal component 2"],
                color="blue",
                s=50,
                marker="+",
                label=target,
            )

        #print("********************")

    for i in range(fakeresult.shape[0]):
        ax.scatter(fakeresult[i, 0], fakeresult[i, 1], c="green", s=400, marker="*")

    ax.grid()
    unique_Dx = Dx.iloc[:, 0].unique()

    # remove the red class from unique_Dx in order to assign the correct RED color for it
    indices = np.where(unique_Dx==y_true_name)
    unique_Dx = np.delete(unique_Dx, indices)


    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    for i in by_label:
        print (i)
    plt.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1.05, 1.0),loc='upper left',fontsize=15)
    plt.savefig("PCA_all_new/all_points_%s.png" % (str(class_index)),bbox_inches='tight')



def euclidean_distance(points, single_point):
    dist = (points - single_point) ** 2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    return dist


def plot_centroids_and_points(
    all_centroids, df_centroids, y_true_name, y_pred, class_index, y_centroid=None,
):

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_title("2 component PCA", fontsize=20)

    unique_colour = df_centroids.iloc[:, 0].unique()
    colors = [
        "red" if i == y_true_name else "gray" for i in unique_colour
    ]

    data_color = []
    for i, j in zip(unique_colour, colors):
        #print("%20s, %5s" % (i, j))
        data_color.append([i, j])

    df_colors = pd.DataFrame(data_color, columns=["unique_colour", "colors"])
    df_colors

    for index, row_centroid in df_centroids.iterrows():
        PCA1 = row_centroid["centroid_x"]
        PCA2 = row_centroid["centroid_y"]
        centroid_name = row_centroid["centroid_name"]
        color = df_colors[df_colors["unique_colour"] == row_centroid["centroid_name"]][
            "colors"
        ]
        color = color.values
        

        ax.scatter(PCA1, PCA2, color=color, s=50)

    # plot the prediction point
    if len(y_pred) == 1:  # only one point
        ax.scatter(y_pred[0][0], y_pred[0][1], color="blue", s=50, marker="+")
    else:  # more than one point
        y_pred_x = [i[0] for i in y_pred]
        y_pred_y = [i[1] for i in y_pred]
        ax.scatter(y_pred_x, y_pred_y, color="blue", s=50, marker="+")

    plt.savefig("PCA_centroids_new/all_centroids_%s.png" % (str(class_index)))
    # plt.show()


def nearest_centroid(df_centroids, distances, y_true_name):
    df_distances = pd.DataFrame(columns=["centroid_name", "distance"])
    df_distances["centroid_name"] = df_centroids["centroid_name"].values
    df_distances["distance"] = distances

    df_distances = df_distances.sort_values(by=["distance"])
    

    nearest_position = np.where(df_distances["centroid_name"] == y_true_name)[0] + 1

    return nearest_position[0], df_distances[df_distances["centroid_name"] == y_true_name]["distance"].values[0]


def main():

    df_all_end_results = pd.DataFrame(
        columns=["Class", "Precison", "Recal", "F1", "Accuracy", "NearestNeigbors"]
    )

    
    h5f = h5py.File('betaValues.h5','r')
    x= h5f['dataset'][:]
    x= np.array((h5f["dataset"][:]),dtype=np.float32)
    indece=pd.read_csv('cg_all_position_sorted_new.csv',sep=";")
    indece_list=indece.values.tolist()
    x=x[:, [indece_list]]
    x=x.reshape((2801,976))
    h5f.close()

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=["principal component 1", "principal component 2"],
    )

    Dx = pd.read_csv("Dx_Families_new.csv", sep=";", header=None)
    Dx.columns = ["Dx"]
    finalDf = pd.concat([principalDf, Dx], axis=1)
    finalDf.to_csv("finalpca.csv", sep=";", header=None, index=False)

    df_grouped = finalDf.groupby("Dx")
    centroid_x, centroid_y, centroid_name = return_df_centroids(df_grouped)
    df_centroids = pd.DataFrame(
        list(zip(centroid_name, centroid_x, centroid_y)),
        columns=["centroid_name", "centroid_x", "centroid_y"],
    )

    #print(df_centroids)

    targets = Dx.to_numpy()
    y_true_all_class_names = ['ANA PA', 'ATRT', 'CHGL', 'CHORDM', 'CN', 'CNS NB', 'CONTR', 'CPH', 'DLGNT', 'DMG', 'EFT', 'ENB', 'EPN', 'ETMR', 'EWS', 'GBM', 'HGNET', 'HMB', 'IDH', 'IHG', 'LGG', 'LIPN', 'LYMPHO', 'MB', 'MELAN', 'MELCYT', 'MNG', 'PGG', 'PIN T', 'PITAD', 'PITUI', 'PLASMA', 'PLEX', 'PTPR', 'PXA', 'RETB', 'SCHW', 'SFT HMPC', 'SUBEPN']

    df_summary = pd.DataFrame(
        columns=[
            "class_index",
            "fake_index",
            "real_name",
            "pred_name",
            "number_of_nearest_classes",
            "distance",
            "min_raw_row_num",
            "min_raw_class_name"
        ]
    )

    for class_index in range(39):

        file_name = "fake_orginal_%s.csv" % (
            str(class_index)
        )
        fake = pd.read_csv(file_name, sep=";", header=None)  # TODO: HIER AUS ÄNDERN

        fakeresult = pca.transform(fake)
    
        y_true_name = y_true_all_class_names[class_index]  # "b'PTPR, A'"
        plot_all_points(Dx, finalDf, y_true_name, fakeresult, class_index)

        all_centroids = df_centroids[["centroid_x", "centroid_y"]].values

        principalDf_values=principalDf.values

        y_pred_index_all, y_pred_name_all, nearest_position_all = [], [], []

        y_distances_raw_all,y_distances_raw_name_all=[],[]

        index_fake_point = 1
        
        for fake_point in fakeresult:
            distances_raw = euclidean_distance(principalDf_values, fake_point)

            distances = euclidean_distance(all_centroids, fake_point)
            
            y_distances_raw_all.append(np.argmin(distances_raw))
            #print(Dx.iloc[np.argmin(distances_raw),0])
            y_distances_raw_name_all.append( Dx.iloc[np.argmin(distances_raw),0])


            y_pred_index_all.append(np.argmin(distances))
            y_pred_name_all.append(
                df_centroids.iloc[np.argmin(distances)]["centroid_name"]
            )

            nearest_position, distance_nearest_position = nearest_centroid(
                df_centroids, distances, y_true_name
            )

            nearest_position_all.append(nearest_position)

            df_summary.loc[len(df_summary)] = [
                class_index,
                index_fake_point,
                y_true_name,
                df_centroids.iloc[np.argmin(distances)]["centroid_name"],
                nearest_position,
                distance_nearest_position,
                np.argmin(distances_raw),
                Dx.iloc[np.argmin(distances_raw),0],


            ]
            index_fake_point = index_fake_point + 1

        # calculate the nearest position for the centroid of the fakeresult
        fakeresult_x = [i[0] for i in fakeresult]
        fakeresult_y = [i[1] for i in fakeresult]

        fakeresult_centroid = centroid(fakeresult_x, fakeresult_y)
        distances = euclidean_distance(all_centroids, fakeresult_centroid)

        nearest_position, distance_nearest_position = nearest_centroid(
            df_centroids, distances, y_true_name
        )
        nearest_position_all.append(nearest_position)

        df_summary.loc[len(df_summary)] = [
            class_index,
            index_fake_point,
            y_true_name,
            df_centroids.iloc[np.argmin(distances)]["centroid_name"],
            nearest_position,
            distance_nearest_position,
            "",
            "",
        ]
        print("df_summary")
        print(df_summary)



        y_true = df_centroids.loc[df_centroids["centroid_name"] == y_true_name]
        y_pred = fakeresult.copy()
        y_pred_name = df_centroids.iloc[np.argmin(distances)]["centroid_name"]
        


        plot_centroids_and_points(
            all_centroids,
            df_centroids,
            y_true_name,
            y_pred,
            # y_centroid,
            class_index,
        )


        #print("y_true == ", y_true)
        y_true = y_true[["centroid_x", "centroid_y"]].values
        #print("y_true == ", y_true)
        #print("y_true == ", y_true[0])

        y_true = [list(y_true[0]) for i in range(len(fakeresult))]  # ["b'PTPR,A'"]
        y_true_name = [y_true_name] * len(fakeresult)  # ["b'PTPR,A'"]
        y_pred = fakeresult.copy()  # [y_pred_name]

    
        labels = list(set(y_true_name + y_pred_name_all))  # ["ant", "bird", "cat"]
        
        conf_mat = confusion_matrix(
            y_true_name, y_pred_name_all, labels=labels
        )
        class_report = classification_report(
            y_true_name, y_pred_name_all, output_dict=True
        )
        precision = class_report[y_true_all_class_names[class_index]]["precision"]
        recall = class_report[y_true_all_class_names[class_index]]["recall"]
        f1_score = class_report[y_true_all_class_names[class_index]]["f1-score"]
        accuracy = class_report["accuracy"]
        # ["Class", "Precison", "Recal", "F1", "Accuracy", "NearestNeigbors"]
        df_all_end_results.loc[len(df_all_end_results)] = [
            y_true_all_class_names[class_index],
            precision,
            recall,
            f1_score,
            accuracy,
            " / ".join(set(y_pred_name_all)),
        ]

    print(df_all_end_results)

    df_all_end_results.to_csv("df_all_end_results_new.csv", sep=";", index=False)

    df_summary.to_csv("df_summary_new.csv", sep=";", index=False)




def plot_all_points_3models(
    fakeresult_model1, fakeresult_model2, fakeresult_model3,
    Dx, finalDf, y_true_name, class_index,output_dir_name,
):

    # all colors
    unique_colour = Dx.iloc[:, 0].unique()
    print("unique_colour----------")
    print(unique_colour)
    print("len(unique_colour--------------")
    print(len(unique_colour))
    
    colors = [
        "red" if i == y_true_name else "gray" for i in unique_colour
    ]
    data_color = []
    for i, j in zip(unique_colour, colors):
        print("%20s, %5s" % (i, j))
        data_color.append([i, j])

    df_colors = pd.DataFrame(data_color, columns=["unique_colour", "colors"])
    df_colors

    targets = Dx.to_numpy()
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=30)
    ax.set_ylabel("Principal Component 2", fontsize=30)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_title("2 component PCA", fontsize=30,fontweight='bold')


    count_red_points = 0
    for target in targets:
        
        color = df_colors[df_colors["unique_colour"] == target[0]]["colors"]  # [0]
        
        color = color.values
        

        indicesToKeep = finalDf.iloc[:, -1] == target[0]
        

        if color == "red":

            ax.scatter(
                finalDf.loc[indicesToKeep, "principal component 1"],
                finalDf.loc[indicesToKeep, "principal component 2"],
                color="red",
                s=50,
                label=target,
            )
            count_red_points = count_red_points +1
        else:
            ax.scatter(
                finalDf.loc[indicesToKeep, "principal component 1"],
                finalDf.loc[indicesToKeep, "principal component 2"],
                color="blue",
                s=50,
                marker="+",
                label=target,
            )



    for i in range(fakeresult_model1.shape[0]):
        
        ax.scatter(fakeresult_model1[i, 0], fakeresult_model1[i, 1], c="green", s=300, marker="*")

    for i in range(fakeresult_model2.shape[0]):
        
        ax.scatter(fakeresult_model2[i, 0], fakeresult_model2[i, 1], c="black", s=300, marker="*")

    for i in range(fakeresult_model3.shape[0]):

        ax.scatter(fakeresult_model3[i, 0], fakeresult_model3[i, 1], c="orange", s=300, marker="*")


    ax.grid()
    unique_Dx = Dx.iloc[:, 0].unique()

    
    indices = np.where(unique_Dx==y_true_name)
    unique_Dx = np.delete(unique_Dx, indices)


    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    for i in by_label:
        print (i)
    l1 = plt.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1.05, 1.0),loc='upper left',fontsize=15)

    

    # add legend for the 3 models
    import matplotlib.patches as mpatches

    model1_patch = mpatches.Patch(color='green', label='Testing')
    model2_patch = mpatches.Patch(color='black', label='Wasserstein')
    model3_patch = mpatches.Patch(color='orange', label='Early Stopping')

    l2 = plt.legend(handles=[model1_patch, model2_patch, model3_patch], loc=4)

    plt.gca().add_artist(l1) # add l1 as a separate artist to the axes


    
    plt.savefig("PCA_all_new/all_points_%s.png" % (str(class_index)),bbox_inches='tight')
    # plt.show()


def main2():

    df_all_end_results = pd.DataFrame(
        columns=["Class", "Precison", "Recal", "F1", "Accuracy", "NearestNeigbors"]
    )

    
    h5f = h5py.File('betaValues.h5','r')
    x= h5f['dataset'][:]
    x= np.array((h5f["dataset"][:]),dtype=np.float32)
    indece=pd.read_csv('cg_all_position_sorted_new.csv',sep=";")
    indece_list=indece.values.tolist()
    x=x[:, [indece_list]]
    x=x.reshape((2801,976))
    h5f.close()

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=["principal component 1", "principal component 2"],
    )

    Dx = pd.read_csv("Dx_Families_new.csv", sep=";", header=None)
    Dx.columns = ["Dx"]

    finalDf = pd.concat([principalDf, Dx], axis=1)
    finalDf.to_csv("finalpca.csv", sep=";", header=None, index=False)

    df_grouped = finalDf.groupby("Dx")
    centroid_x, centroid_y, centroid_name = return_df_centroids(df_grouped)
    df_centroids = pd.DataFrame(
        list(zip(centroid_name, centroid_x, centroid_y)),
        columns=["centroid_name", "centroid_x", "centroid_y"],
    )

    

    targets = Dx.to_numpy()

    y_true_all_class_names = ['ANA PA', 'ATRT', 'CHGL', 'CHORDM', 'CN', 'CNS NB', 'CONTR', 'CPH', 'DLGNT', 'DMG', 'EFT', 'ENB', 'EPN', 'ETMR', 'EWS', 'GBM', 'HGNET', 'HMB', 'IDH', 'IHG', 'LGG', 'LIPN', 'LYMPHO', 'MB', 'MELAN', 'MELCYT', 'MNG', 'PGG', 'PIN T', 'PITAD', 'PITUI', 'PLASMA', 'PLEX', 'PTPR', 'PXA', 'RETB', 'SCHW', 'SFT HMPC', 'SUBEPN']

    df_summary = pd.DataFrame(
        columns=[
            "class_index",
            "fake_index",
            "real_name",
            "pred_name",
            "number_of_nearest_classes",
            "distance",
            "min_raw_row_num",
            "min_raw_class_name"
        ]
    )

    for class_index in range(39):


        exp = 0
        output_dir_name = "exp_" + str(exp)


        #-------------------------------------LOAD all 3 models

        file_name = "fake_orginal_%s.csv" % (
            str(class_index)
        )
        fake_model1 = pd.read_csv("testing/" + output_dir_name +"/"+ file_name, sep=";", header=None)

        fakeresult_model1 = pca.transform(fake_model1)
        
        fake_model2 = pd.read_csv("Wasserstein/" + output_dir_name +"/"+ file_name, sep=";", header=None)  # TODO: HIER AUS ÄNDERN

        fakeresult_model2 = pca.transform(fake_model2)
        
        fake_model3 = pd.read_csv("early/" + output_dir_name +"/"+ file_name, sep=";", header=None)  # TODO: HIER AUS ÄNDERN

        fakeresult_model3 = pca.transform(fake_model3)
        
        #-------------------------------------

        y_true_name = y_true_all_class_names[class_index]  # "b'PTPR, A'"



        plot_all_points_3models(
            fakeresult_model1, fakeresult_model2, fakeresult_model3,
            Dx, finalDf, y_true_name, class_index, output_dir_name
        )




main2()
