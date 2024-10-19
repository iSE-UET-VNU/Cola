import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

class Generator(object):
    def __init__(self, dataset_name:str, noise_type:str = 'sym', error_rate:float = 0.25, encode_model:str = 'clip', sampling_rate:float = 1, random_state:float = 0):
        valid_noise_types = ['sym', 'asym', 'ins']
        if noise_type not in valid_noise_types:
            raise ValueError(f"Invalid noise type: {noise_type}. Expected one of {valid_noise_types}.")
        if not (0 <= error_rate <= 1):
            raise ValueError(f"Error rate should be between 0 and 1. Got {error_rate}.")
        self.dataset_name = dataset_name
        self.noise_type = noise_type
        self.error_rate = error_rate
        self.encode_model = encode_model
        self.sampling_rate = sampling_rate
        self.random_state = random_state
        np.random.seed(random_state)
        self.data, self.clean_label = self.init_data()
        self.n_label = self.data['label'].nunique()
        self.corrupted_label_index = self.data[self.data['label'] != raw_labels].index

    def init_data(self):
        
        global processed_samples, raw_labels
        global noisy_sample_indices
        global n_processed_samples
 
        
        df = pd.DataFrame(np.load(f'Data/{self.dataset_name}/{self.encode_model}.npy'))
        clean_labels = np.load(f'Data/{self.dataset_name}/labels.npy').flatten()
        
        df['clean_label'] = clean_labels
        n_unique_labels = df['clean_label'].nunique()

        if self.error_rate == 0:
            df.rename(columns={'clean_label': 'label'}, inplace=True)
            return df, df['label'].values
        
        if self.sampling_rate != 1:
            df = self.sample_data(df, self.sampling_rate, n_unique_labels)
        
        if self.noise_type == 'ins':
            noisy_labels, actual_noise_rate = self.gen_instance_noise(clean_labels=df['clean_label'].values, noise_rate=self.error_rate, n_class=n_unique_labels, train_data=df.values)
        elif self.noise_type == 'sym':
            noisy_labels, actual_noise_rate = self.gen_symmetric_noise(clean_labels=df['clean_label'].values, noise_rate=self.error_rate, n_class=n_unique_labels)
        elif self.noise_type == 'asym':
            noisy_labels, actual_noise_rate = self.gen_asymmetric_noise(clean_labels=df['clean_label'].values, noise_rate=self.error_rate, n_class=n_unique_labels)
        
        df['label'] = noisy_labels

        processed_samples = df
        raw_labels = df.pop('clean_label').tolist()
        noisy_sample_indices = df[df['label'] != raw_labels].index
        n_processed_samples = df.shape[0]

        print(f'Actual noise rate: {round(actual_noise_rate, 2)}')
        print(f'# of processed samples: {n_processed_samples}')
        print(f'# of noisy samples: {len(noisy_sample_indices)}')
        # print(f'Indexes of noisy samples: {noisy_sample_indices}')

        return df, raw_labels

    def sample_data(df, sampling_rate, n_unique_labels):
        num_samples_per_label = int(df.shape[0] * sampling_rate / n_unique_labels)
        selected_samples = pd.DataFrame()
        for label in range(n_unique_labels):
            label_samples = df[df['clean_label'] == label].sample(num_samples_per_label)
            selected_samples = pd.concat([selected_samples, label_samples])
        
        return selected_samples.reset_index(drop=True)
    
    def multiclass_generate(self, y, P):
        assert P.shape[0] == P.shape[1], "Transition matrix P should be square."
        assert np.max(y) < P.shape[0], "Label indices should be within the range of P."
        assert (P >= 0.0).all(), "Transition matrix P should not contain negative values."

        n_samples = y.shape[0]
        noisy_labels = y.copy()
        rng = np.random.RandomState(self.random_state)

        for idx in range(n_samples):
            current_class = y[idx]
            flipped = rng.multinomial(1, P[current_class, :], 1)[0]
            noisy_labels[idx] = np.where(flipped == 1)[0]

        print("Transition Matrix (in %):\n", np.round(P * 100, 2))
        return noisy_labels

    def gen_asymmetric_noise(self, clean_labels, noise_rate, n_class):

        assert noise_rate > 0, "noise_rate must be greater than 0"

        P = np.eye(n_class)    
        for i in range(n_class - 1):
            P[i, i], P[i, i + 1] = 1.0 - noise_rate, noise_rate
        P[n_class - 1, n_class - 1], P[n_class - 1, 0] = 1.0 - noise_rate, noise_rate

        noisy_labels = self.multiclass_generate(clean_labels, P=P)

        actual_noise_rate = (noisy_labels != clean_labels).mean()
        return noisy_labels, actual_noise_rate


    def gen_symmetric_noise(self, clean_labels, noise_rate, n_class):

        assert noise_rate > 0, "noise_rate must be greater than 0"
        
        P = np.full((n_class, n_class), noise_rate / (n_class - 1)) 
        np.fill_diagonal(P, 1.0 - noise_rate)
        noisy_labels = self.multiclass_generate(clean_labels, P=P)
        actual_noise_rate = (noisy_labels != clean_labels).mean()
        return noisy_labels, actual_noise_rate

    def gen_instance_noise(self, clean_labels, noise_rate, n_class, train_data):

        assert noise_rate > 0, "noise_rate must be greater than 0"
        n_class = np.max(clean_labels) + 1

        noise_rates = np.random.normal(loc=noise_rate, scale=0.1, size=1000000)
        valid_noise_rates = [n for n in noise_rates if 0 < n < 1]
        weights = [np.random.normal(loc=0, scale=1, size=(train_data.shape[1], n_class)) for _ in range(n_class)]

        noisy_labels = []
        transition_matrix = np.zeros((n_class, n_class))

        for i, sample in enumerate(train_data):
            class_weights = np.matmul(sample, weights[clean_labels[i]])
            class_weights[clean_labels[i]] = -1e6
            prob_dist = valid_noise_rates[i] * F.softmax(torch.tensor(class_weights), dim=0).numpy()
            prob_dist[clean_labels[i]] = 1 - valid_noise_rates[i]

            noisy_label = np.random.choice(np.arange(n_class), p=prob_dist)
            noisy_labels.append(noisy_label)
            transition_matrix[clean_labels[i], noisy_label] += 1

        overall_noise_rate = 1 - (torch.tensor(clean_labels) == torch.tensor(noisy_labels)).float().mean().item()
        transition_matrix /= transition_matrix.sum(axis=1)[:, None]
        
        print("Transition Matrix (in %):\n", np.round(transition_matrix * 100, 1))
        return noisy_labels, overall_noise_rate

