import numpy as np

# creat an empty ndarray
r_npy = np.empty([25608, 3, 150, 27, 1])

# read npy
npy = np.load("train_data_joint.npy")

# read original sequence
with open("uni_trainlist01.csv") as file:
    old_seq = np.loadtxt(file, str, delimiter=',', usecols=(0))
    old_seq = old_seq.tolist()

# read target sequence
with open("NMFs_CSL_train_split_rgb.txt") as target_file:
    target_seq = np.loadtxt(target_file, str, delimiter=' ', usecols=(0))
    target_seq = target_seq.tolist()

# assign the values
for i in old_seq:
    for j in target_seq:
        if i == j:
            # the print can be used to check the indexes' comparison
            # print(old_seq.index(i), target_seq.index(j))
            r_npy[target_seq.index(j)] = npy[old_seq.index(i)]

# save as npy
np.save('r_npy.npy', r_npy)
