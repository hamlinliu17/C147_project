from torch.utils.data import Dataset
import numpy as np

def moving_average(x, w):
    """
    Simple 1d moving average using numpy

    :input x: 1d np array
    :input w: size of moving average filter
    returns np array with applied moving average
    """
    return np.convolve(x, np.ones(w), 'valid') / w


class eegData(Dataset):
    """
    PyTorch style dataset to be loaded into torch data loader for training

    provides simple functionality

    """
    def __init__(self, data_file_name, label_file_name, validation_size=0.1, preprocessing_params={}):
        """
        :input data_file_name: file path of the data
        :input label_file_name: file path of the labels
        :input validation_size: size of validation (percentage given to validation)
        :input preprocessing_params:
            'subsample': int on the size of step of the subsampling
            'mov_avg': int on the size of the moving average window
            'trim': how many of the last indices will be trimmed off
        """
        subsample = preprocessing_params.get('subsample', 1) # we can increase our trial count
        mov_avg_window = preprocessing_params.get('mov_avg', 1) # limit ourselves to 2115
        trimming = preprocessing_params.get('trim', 0) # how much you want to trim
        eeg_data = np.load(data_file_name)
        label_data = np.load(label_file_name) - 769
        

        # remove the last x amount of time steps
        trimmed_indices = eeg_data.shape[2] - trimming 
        eeg_data = eeg_data[:, :, :trimmed_indices].copy()

        #begin_subsample
        stack_eeg_data = []
        stack_label_data = []
        for i in range(subsample):
            sampled_eeg_data = eeg_data[:, :, i::subsample].copy()
            stack_label_data.append(label_data.copy())
            stack_eeg_data.append(sampled_eeg_data)

        eeg_data = np.vstack(stack_eeg_data)
        label_data = np.concatenate(stack_label_data)


        # begin shuffling to get validation and training data
        trials = eeg_data.shape[0]
        valid_size = int(trials * validation_size)
        shuffled_indices = np.arange(trials)
        np.random.shuffle(shuffled_indices)
        validation_data = eeg_data[shuffled_indices[:valid_size]]
        validation_labels = label_data[shuffled_indices[:valid_size]]
        eeg_data = eeg_data[shuffled_indices[valid_size:]]
        label_data = label_data[shuffled_indices[valid_size:]]


        # begin applying moving_average
        eeg_data = np.apply_along_axis(func1d=moving_average, axis=2, arr=eeg_data, w=mov_avg_window)


        self.eeg_data = eeg_data 
        self.label_data = label_data
        self.mov_avg_window = mov_avg_window 
        self.trim = trimming 
        self.sampling = subsample

        self.validation_data = validation_data
        self.validation_labels = validation_labels

    def __len__(self):
        assert self.eeg_data.shape[0] == self.label_data.shape[0]
        return self.eeg_data.shape[0]


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.numpy()

        sample = {}
        sample['data'] = self.eeg_data[idx]
        sample['label'] = self.label_data[idx]

        return sample
