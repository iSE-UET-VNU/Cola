import time
import os
import sys
from clothingData import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ErrorLabelDetection import *
from FeatureExtraction import *

def run_process():
    if not os.path.exists("Data/clothing/clean_labels.npy"):
        batch_size = 64
        data_root = '/drive2/lnduyphong/clothing/'
        dataset = Clothing(root=data_root, img_transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        extract_clothing_img(data_loader, 'facebook/dinov2-large')

    root = 'Data/clothing/'
    data = pd.DataFrame(np.load(root + 'embeddings.npy'))
    clean_labels = np.load(root + 'clean_labels.npy').flatten()
    noise_labels = np.load(root + 'noise_labels.npy').flatten()
    data['labels'] = noise_labels
    corrupted_index = data[noise_labels != clean_labels].index
    
    print(f'Actual noise rate: {round(len(corrupted_index) / len(data), 2)}')
    print(f'# of processed samples: {len(data)}')
    print(f'# of noise samples: {len(corrupted_index)}')
    
    detector = Detector(data, corrupted_index, 14, 'Clothing1M')
    start = time.time()
    detector.local_detection(k_neighbors=41)
    detector.global_detection(5)
    print(f"Run time: {time.time() - start}")
    
    
if __name__ == '__main__':
    run_process()
