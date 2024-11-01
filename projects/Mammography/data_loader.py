import numpy as np
import scipy

def load_clinical_data(data_root,
                       splits=['train','val','test'],
                       mode='clinical-only'):
    if mode == 'clinical-only':
        print("load clinical only features")
        X = {split: np.load(f"{data_root}/{split}_x_preope.npy")
             for split in splits}
        print("X shape:", X['train'].shape)
    else:
        print("load clinical and mammo features")
        X={split: np.concatenate([np.load(f"{data_root}/{split}_x_preope.npy", allow_pickle=True),np.load(f"{data_root}/{split}_feat_5.npy",allow_pickle=True),
                                  ],axis=1)
        for split in splits}
        print("X shape:", X['train'].shape)
    #TODO softmax
    Y={split: np.squeeze(np.load(f"{data_root}/{split}_y.npy")) for split in splits}
    fortnr = {split: np.squeeze(np.load(f"{data_root}/{split}_fortnr.npy",allow_pickle=True)) for split in splits}

    nan_masks={split:np.isnan(Y[split]) for split in splits}
    Y={split:Y[split][~nan_masks[split]]for split in splits}
    X={split:X[split][~nan_masks[split]]for split in splits}
    fortnr={split:fortnr[split][~nan_masks[split]]for split in splits}

    return X,Y, fortnr

def load_clinical_data_test(data_root,
                       splits=['test'],
                       mode='clinical-only'):
    if mode == 'clinical-only':
        print("load clinical only features")
        X = {split: np.load(f"{data_root}/{split}_x_preope.npy")
             for split in splits}
        print("X shape:", X[splits[0]].shape)
    else:
        print("load clinical and mammo features")
        X={split: np.concatenate([np.load(f"{data_root}/{split}_x_preope.npy", allow_pickle=True),np.load(f"{data_root}/{split}_feat_5.npy",allow_pickle=True),
                                  ],axis=1)
        for split in splits}
        print("X shape:", X[splits[0]].shape)

    Y={split: np.squeeze(np.load(f"{data_root}/{split}_y.npy")) for split in splits}
    fortnr={split: np.squeeze(np.load(f"{data_root}/{split}_fortnr.npy",allow_pickle=True)) for split in splits}

    nan_masks={split:np.isnan(Y[split]) for split in splits}
    Y={split:Y[split][~nan_masks[split]]for split in splits}
    X={split:X[split][~nan_masks[split]]for split in splits}
    fortnr={split:fortnr[split][~nan_masks[split]]for split in splits}

    return X,Y,fortnr
