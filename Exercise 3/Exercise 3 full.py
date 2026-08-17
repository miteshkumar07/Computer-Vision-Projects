import os
import gzip
import pickle
import argparse
import numpy as np
import cv2 as cv
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser(description="Writer Retrieval Exercise")
    #parser.add_argument('--train_dir', type=str, required=True, 
    #                    help='Path to the directory of training data')
    #parser.add_argument('--test_dir', type=str, required=True, 
    #                    help='Path to the directory of test data')
    parser.add_argument('--train_labels', type=str, required=True, 
                        help='Path to the training labels')
    parser.add_argument('--test_labels', type=str, required=True, 
                        help='Path to the test labels')
    parser.add_argument('--clusters', type=int, default=100, 
                        help='Number of clusters for the Codebook (k)')
    parser.add_argument('--svm_c', type=float, default=1000, 
                        help='C parameter for the Exemplar SVM')
    parser.add_argument('--samples', type=int, default=500000, 
                        help='Number of descriptors to sample for codebook generation')    
    # Individual Exercise
    parser.add_argument('--n_runs', type=int, default=5, 
                        help='Number of Multi-VLAD runs (Task G)')
    parser.add_argument('--pca_dim', type=int, default=1000, 
                        help='PCA target dimension (Task G)')
    parser.add_argument('--gamma', type=float, default=None, 
                        help='Gamma for Generalized Max Pooling')
    parser.add_argument('--raw_img_train', type=str, 
                        help='Path to folder of RAW training images (jpg/png)')
    parser.add_argument('--raw_img_test', type=str, 
                        help='Path to folder of RAW test images (jpg/png)')
    ######
    
    return parser.parse_args()

def get_file_paths(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl.gz')]
    return sorted(files)

def load_labels(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")        
    labels = []
    with open(path, 'r') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line:
                parts = clean_line.split(' ')
                label = int(parts[-1])
                labels.append(label)
    return np.array(labels)

# TASK A: Codebook Generation
# Generating the dictionary of K=100 visual words (cluster centers) from the training data.
def codebook_gen(file_paths, n_clusters, target_samples, seed):
    print(f"\n Task A: Generating Codebook with k={n_clusters} and seed={seed}")
    descriptors = []
    print("Loading descriptors for codebook generation...")
    for f_path in tqdm(file_paths):
        with gzip.open(f_path, 'rb') as d:
            desc = pickle.load(d, encoding='latin1')
            descriptors.append(desc)  
    descriptors = np.vstack(descriptors)
    if len(descriptors) > target_samples:
        indices = np.random.choice(len(descriptors), target_samples, replace=False)
        sampled = descriptors[indices]
    else:
        sampled = descriptors
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=3584, random_state=seed).fit(sampled)
    return kmeans.cluster_centers_

# TASK B: VLAD Encoding
# Aggregating the local descriptors into a single global vector per image.
def vlad_encod(file_paths, cluster_centers, gamma= None):
    print(f"\n Task B: Computing VLAD vectors for {len(file_paths)} images...")
    if gamma is None:
        print("Using Sum Pooling")
    else:
        print(f"Using Generalized Max Pooling (GMP) with gamma={gamma} ")
    vlad_vectors = []
    k = cluster_centers.shape[0]
    bf = cv.BFMatcher()
    if gamma is not None:
        ridge = Ridge(alpha= gamma, fit_intercept= False, solver= 'sparse_cg', max_iter= 500)
    for f_path in tqdm(file_paths):
        with gzip.open(f_path, 'rb') as d:
            local_descriptor = pickle.load(d, encoding='latin1')
            #VLAD Matrix K x D
            vlad_matrix = np.zeros((k, local_descriptor.shape[1]))
            matches = bf.knnMatch(local_descriptor, cluster_centers, k=1)
            labels = np.array([m[0].trainIdx for m in matches])
            # Calculates GMP encodings if gamma is passed
            if gamma is not None:
                unique_labels = np.unique(labels)
                for uni in unique_labels:
                    mask = (labels == uni)
                    residuals = local_descriptor[mask] - cluster_centers[uni]
                    n_res = residuals.shape[0]
                    X = np.ones(n_res)
                    ridge.fit(residuals, X)
                    vlad_matrix[uni] = ridge.coef_.flatten()
                vlad_vec = vlad_matrix.flatten()
                vlad_vectors.append(vlad_vec)
            else:
                # Calculates SUM encodings
                # Finds the handwriting difference from the standard shapes
                residuals = local_descriptor - cluster_centers[labels]
                np.add.at(vlad_matrix, labels, residuals)
                vlad_vec = vlad_matrix.flatten()
                vlad_vectors.append(vlad_vec)            
    return np.array(vlad_vectors)

# TASK C: Normalization

def normalize_vlad(vectors):
    print("\n Task C: Applying Power and L2 Normalization")
    # Power Normalization
    pow_norm = np.sign(vectors) * np.sqrt(np.abs(vectors))
    # L2 Normalization
    l2_norm = normalize(pow_norm, norm='l2', axis=1)
    return l2_norm

# Evaluation (Distances & mAP)

def calculate_distances(encodings):
    similarity = np.dot(encodings, encodings.T)
    distances = 1 - similarity
    np.fill_diagonal(distances, np.finfo(distances.dtype).max)
    return distances

def mAP(labels, distances):
    err = []    
    for i in range(len(labels)):
        query_label = labels[i]        
        relevance = (labels == query_label).astype(int)
        relevance[i] = 0
        query_dist = -distances[i]
        if np.sum(relevance) > 0:
            avg_prec = average_precision_score(relevance, query_dist)
            err.append(avg_prec)            
    return np.mean(err)

# TASK D: Exemplar SVM

def esvm(test_vectors, train_vectors, C):
    print(f"\n Task D: Training Exemplar SVMs (C={C})")
    
    # Negatives = The entire training set
    X_negatives = train_vectors
    y_negatives = np.zeros(X_negatives.shape[0])    
    esvm_weights = []    
    for i, query_vec in tqdm(enumerate(test_vectors), total=len(test_vectors)):
        # Positive = The single test vector
        X_positive = query_vec.reshape(1, -1)
        
        # Stack Data
        X = np.vstack((X_positive, X_negatives))
        # Label 1 for query, 0 for negatives
        y = np.concatenate(([1], y_negatives))
        clf = LinearSVC(C=C, class_weight='balanced', dual=True, max_iter=2000)
        clf.fit(X, y)
        # Using the learned weight vector (coef_) as the new descriptor
        w = clf.coef_
        w_normalized = normalize(w, norm='l2').flatten()
        esvm_weights.append(w_normalized)        
    return np.array(esvm_weights)


# Task E: Custum feature extraction
def computeDescs(data_dir, output_dir):
    "Extracting features as per exercise"
    print(f"Extracting features from {data_dir}")
    image_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith((".jpg", ".png"))])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for img in tqdm(image_files):
        # descriptor logic
        base_name = os.path.splitext(os.path.basename(img))[0]
        img = cv.imread(img, cv.IMREAD_GRAYSCALE)
        sift = cv.SIFT_create()
        keypoints = sift.detect(img, None)
        # Setting keypoints angles to 0
        for k in keypoints:
            k.angle = 0.0
        keypoints, descriptor = sift.compute(img, keypoints)
        desc_l1 = normalize(descriptor, norm= 'l1')
        desc_root = (np.sign(desc_l1) * np.sqrt(np.abs(desc_l1))).astype(np.float32)
        output_filename = f"{base_name}.pkl.gz"
        output_path = os.path.join(output_dir, output_filename)
        with gzip.open(output_path, 'wb') as f:
            pickle.dump(desc_root, f)

# Task G: Multi Vlad with PCA
def multi_vlad(train_data, test_data, n_runs, n_clusters, pca_dim, gamma):
    all_train = []
    all_test = []
    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}")
        # Generating the codebook with different seeds
        centers = codebook_gen(train_data, n_clusters, target_samples=500000, seed= i + 1)
        # Computing vlad encodings using sum pooling else using gmp if gamma is passed
        v_train = vlad_encod(train_data, centers, gamma=gamma)
        v_test = vlad_encod(test_data, centers, gamma=gamma)
        # Normalizing the encodings
        v_train = normalize_vlad(v_train)
        v_test = normalize_vlad(v_test)
        all_train.append(v_train)
        all_test.append(v_test)
    # Concatenating all the runs
    X_train = np.hstack(all_train)
    X_test = np.hstack(all_test)
    # Applying PCA Whitening
    pca = PCA(n_components= pca_dim, whiten= True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    # Normalizing
    X_train_final = normalize_vlad(X_train_pca)
    X_test_final = normalize_vlad(X_test_pca)
    return X_train_final, X_test_final

if __name__ == "__main__":
    args = parse_args()
    
    # Load Paths
    # print(f"Loading data from: {args.train_dir} and {args.test_dir}")
    # train_files = get_file_paths(args.train_dir)
    # test_files = get_file_paths(args.test_dir)
    
    train_labels = load_labels(args.train_labels)
    test_labels = load_labels(args.test_labels)
    
    #print(f"Found {len(train_files)} training files and {len(test_files)} test files.")
    
    print("Extracting custom features")
    # Extracting Custom features
    train_feat_extrac = computeDescs(args.raw_img_train, r"feat_extracted_icdar17_historicalwi\train")
    test_feat_extrac = computeDescs(args.raw_img_test, r"feat_extracted_icdar17_historicalwi\test")


    print("Generating codebook")
    # Codebook generationg
    train_feature_extracted = get_file_paths(r"feat_extracted_icdar17_historicalwi\train")
    test_feature_extracted = get_file_paths(r"feat_extracted_icdar17_historicalwi\test")
    cluster_center = codebook_gen(train_feature_extracted,
                                  n_clusters=args.clusters, 
                                  target_samples= 500000, seed= 0)
    
    print("Computing vlad using sum pooling")
    
    # Computing encodings using sum pooling
    train_vlad_sum = vlad_encod(train_feature_extracted, cluster_centers= cluster_center, gamma=None)
    test_vlad_sum = vlad_encod(test_feature_extracted, cluster_centers=cluster_center, gamma=None)

    print("Normalizing the encodings")
    train_vlad_sum_norm = normalize_vlad(train_vlad_sum)
    test_vlad_sum_norm = normalize_vlad(test_vlad_sum)

    print("Calculating mAP Score")
    dists_sum = calculate_distances(test_vlad_sum_norm)
    score_sum = mAP(test_labels, dists_sum)

    print(f"Vlad (Sum Pooling) mAP = {score_sum}")


    print("Vlad (SUM) + E-SVM")
    esvm_sum_feats = esvm(test_vlad_sum_norm, train_vlad_sum_norm, args.svm_c)
    dists_esvm_sum = calculate_distances(esvm_sum_feats)
    score_esvm_sum = mAP(test_labels, dists_esvm_sum)
    print(f"VLAD (Sum) + E-SVM mAP = {score_esvm_sum:}")










    print("Computing vlad using gmp")
    
    # Computing encodings using sum pooling
    train_vlad_gmp = vlad_encod(train_feature_extracted, cluster_centers= cluster_center, gamma=args.gamma)
    test_vlad_gmp= vlad_encod(test_feature_extracted, cluster_centers=cluster_center, gamma=args.gamma)

    print("Normalizing the encodings")
    train_vlad_gmp_norm = normalize_vlad(train_vlad_gmp)
    test_vlad_gmp_norm = normalize_vlad(test_vlad_gmp)

    print("Calculating mAP Score")
    dists_sum = calculate_distances(test_vlad_gmp_norm)
    score_gmp = mAP(test_labels, dists_sum)

    print(f"Vlad (GMP Pooling) mAP = {score_gmp}")


    print("Vlad (GMP) + E-SVM")
    esvm_gmp_feats = esvm(test_vlad_gmp_norm, train_vlad_gmp_norm, args.svm_c)
    dists_esvm_gmp = calculate_distances(esvm_gmp_feats)
    score_esvm_gmp = mAP(test_labels, dists_esvm_gmp)
    print(f"VLAD (GMP) + E-SVM mAP = {score_esvm_gmp:}")




    
    
    print("Multi Vlad (Sum Pooling)")
    multi_vlad_sum_train, multi_vlad_sum_test = multi_vlad(train_feature_extracted,test_feature_extracted,
                                                            n_runs= args.n_runs,
                                                            n_clusters= args.clusters,
                                                            gamma=None,
                                                            pca_dim=args.pca_dim)
    
    dists_mv = calculate_distances(multi_vlad_sum_test)
    score_mv_s = mAP(test_labels, dists_mv)
    print(f"Multi-VLAD (sum) mAP = {score_mv_s:}")


    print("Multi Vlad (Sum Pooling) + ESVM")
    esvm_mv_sum = esvm(multi_vlad_sum_test, multi_vlad_sum_train, args.svm_c)
    dist_esvm_mv_sum = calculate_distances(esvm_mv_sum)
    esmv_mv_sum_score = mAP(test_labels, dist_esvm_mv_sum)

    print(f"Multi-VLAD (sum) + E-SVM mAP = {esmv_mv_sum_score:}")



    print("Multi Vlad (GMP Pooling)")
    multi_vlad_gmp_train, multi_vlad_gmp_test = multi_vlad(train_feature_extracted,test_feature_extracted,
                                                            n_runs= args.n_runs,
                                                            n_clusters= args.clusters,
                                                            gamma=args.gamma,
                                                            pca_dim=args.pca_dim)
    
    dists_mv_gmp = calculate_distances(multi_vlad_gmp_test)
    score_mv_gmp = mAP(test_labels, dists_mv_gmp)
    print(f"Multi-VLAD (GMP) mAP = {score_mv_gmp:}")


    print("Multi Vlad (GMP Pooling) + ESVM")
    esvm_mv_gmp = esvm(multi_vlad_gmp_test, multi_vlad_gmp_train, args.svm_c)
    dist_esvm_mv_gmp = calculate_distances(esvm_mv_gmp)
    esmv_mv_gmp_score = mAP(test_labels, dist_esvm_mv_gmp)

    print(f"Multi-VLAD (GMP) + E-SVM mAP = {esmv_mv_gmp_score:}")


    print("Summary of Results:")
    print(f"Vlad Sum: {score_sum}")
    print(f"Vlad Sum + ESVM: {score_esvm_sum}")
    print(f"Vlad Gmp: {score_gmp}")
    print(f"Vlad Gmp + ESVM: {score_esvm_gmp}")
    print(f"Multi Vlad Sum: {score_mv_s}")
    print(f"Multi Vlad Sum + ESVM: {esmv_mv_sum_score}")
    print(f"Multi Vlad Gmp: {score_mv_gmp}")
    print(f"Multi Vlad Gmp + ESVM: {esmv_mv_gmp_score}")
