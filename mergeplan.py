# mergeplan.py


import numpy as np
import math


file_offsets = [      0,   20374,   40748,   61122,   81496,  101870,  122244,
        142618,  162992,  183366,  203740,  224114,  244488,  264862,
        285236,  305610,  325984,  346358,  366732,  387106,  407480,
        427854,  448228,  468602,  488976,  509350,  529724,  550098,
        570472,  590846,  611220,  631594,  651968,  672342,  692716,
        713090,  733464,  753838,  774212,  794586,  814960,  835334,
        855708,  876082,  896456,  916830,  937204,  957578,  977952,
        998326, 1018700, 1039074, 1059448, 1079822, 1100196, 1120570,
       1140944, 1161318, 1181692, 1202066, 1222440, 1242814, 1263188,
       1283562, 1303936, 1324310, 1344684, 1365058, 1385432, 1405806,
       1426180, 1446554, 1466928, 1487302, 1507676, 1528050]

batch_size = 5_000


def get_batch_histories(file_offsets, batch_size, tiny=False):
    batch_histories = []
    num_batches = math.ceil(file_offsets[-1] / batch_size)

    for batch_index in range(num_batches):
        abs_idx = batch_index * batch_size # absolute *event* index
        file_idx = np.searchsorted(file_offsets, abs_idx, side="right") - 1

        rel_idx = abs_idx - file_offsets[file_idx]
        stop_idx = min(rel_idx + batch_size, file_offsets[file_idx + 1] - file_offsets[file_idx])

        # file_idx, batch_file_size, batch_index
        batch_histories.append((file_idx, stop_idx - rel_idx, batch_index))

        if False:
            # print all the variables
            print("abs_idx:", abs_idx)
            print("file_idx:", file_idx)
            print("self.file_offsets[file_idx]:", self.file_offsets[file_idx])
            print("rel_idx:", rel_idx, "stop_idx:", stop_idx)
            print("X.shape:", X.shape)
            print("y.shape:", y.shape)

    if not tiny:
        batch_histories = [
            bh for bh in batch_histories if bh[1] < batch_size
        ]

    return batch_histories

def build_merge_plan(file_offsets, batch_size):
    batch_histories = get_batch_histories(file_offsets, batch_size, tiny = True)



if __name__ == '__main__':
    print(file_offsets)
    print(len(file_offsets))
    print(batch_size)

    batch_histories = get_batch_histories(file_offsets, batch_size)
    for batch_history in batch_histories[:10]:
        print(batch_history)
    