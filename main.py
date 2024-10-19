import argparse
from ErrorLabelDetection import Detector
from NoiseGeneration import Generator
import FeatureExtraction as encode
import time

def run_process(dataset_name, noise_type, error_rate, knn_k, n_iterations, sampling_rate, encode_model):
    encode_model, dataset_name = encode.extract_img_embedding(dataset_name=dataset_name, batch_size=64, encode_model=encode_model)
    data = Generator(dataset_name, noise_type, error_rate, encode_model, sampling_rate)
    detector = Detector(data.data, data.corrupted_label_index, data.n_label, dataset_name)
    start = time.time()
    detector.local_detection(k_neighbors=knn_k)
    detector.global_detection(n_iterations)
    print(f"Run time: {time.time() - start}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some data with noise.")
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='Dataset on Huggingface to use (e.g cifa10, cifar100, ag_news)')
    parser.add_argument('--noise_type', type=str, default='sym', help='Type of noise (sym, asym, ins)')
    parser.add_argument('--error_rate', type=float, default=0.05, help='Error rate should be less than 1')
    parser.add_argument('--k', type=int, default=21, help='Parameter for KNN detection')
    parser.add_argument('--iteration', type=int, default=-1, help='Number of global iterations')
    parser.add_argument('--sampling', type=int, default=1, help='Sampling rate')
    parser.add_argument('--encode_model', type=str, default='facebook/dinov2-base', help='Embedding model to use')

    args = parser.parse_args()
    
    run_process(args.dataset_name, args.noise_type, args.error_rate, args.k, 
         args.iteration, args.sampling, args.encode_model)