import os
import pandas as pd

#pip3 install torch torchvision

os.chdir('./datasets/multi-class')

def fileload(type,images,labels):

    type_image = str(type)+str('_images')
    type_label = str(type)+str('_labels')

    num_data = len(images[type_image])
    num_feature = len(images[type_image][0])

    # create cloumns name
    columns = []

    # for i in range(784):
    for i in range(num_feature):
        feature = "F" + str(i)
        columns.append(feature)

    columns.append('label')

    df= pd.DataFrame(columns=columns)

    # print(len(columns))

    # for j in range(1000):
    for j in range(num_data):
        to_append = []

        for i in range(num_feature):
            to_append.append(images[type_image][j][i])

        to_append.append(labels[type_label][0][j])
        df.loc[j] = to_append


    #filename = str(type) + str('.csv')


    #df.to_csv(filename)

    #print(type,"data is saved in ",type,'.csv file')

    return df








