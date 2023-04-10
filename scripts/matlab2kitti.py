import scipy.io
import numpy as np
import pandas as pd
import fire

def load(path_to_mat):
    # MatLab labels looks like this: ['xctr', 'yctr', 'zctr', 'xlen', 'ylen', 'zlen', 'xrot', 'yrot', 'zrot']
    mat = scipy.io.loadmat(path_to_mat)
    labels = mat['labels'].astype(np.ndarray)
    df = pd.DataFrame(columns = ['File', 'Label', 'Type', 'Truncated', 'Occluded', 'Alpha', 'BBox', 'Dimensions', 'Location', 'Rotation'])
    for file_counter, file in enumerate(labels):
        for label_counter, label in enumerate(file[0].round(decimals=3)):
            new_row = [file_counter, label_counter, 'Pedestrian', 0.0, 0, 0, [0, 0, 50, 50], [label[3], label[4], label[5]], [label[0], label[1], label[2]], [0, 0, 0]]
            df.loc[len(df)] = new_row

    # Organizing by File and Label helps us in the writing
    return df.set_index(['File', 'Label'])

def create_kitti_labels(df):
    for file in range(len(df.groupby(level=0))):
        file_data = df.loc[[file]]
        with open('labels/{}.txt'.format(file), 'x') as txtfile:
            for label in range(len(file_data.groupby(level=1))):
                label_data = file_data.loc[file, label]
                str = f"{label_data['Type']} {label_data['Truncated']} {label_data['Occluded']} {label_data['Alpha']} {label_data['BBox']} {label_data['Dimensions']} {label_data['Location']} {label_data['Rotation']}\n"
                print(str.replace(', ', ' ').replace('[', '').replace(']', ''))
                txtfile.write(str.replace(', ', ' ').replace('[', '').replace(']', ''))

if __name__ == '__main__':
    df = load('labels.mat')
    create_kitti_labels(df)