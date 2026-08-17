'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np
import os 
from sklearn import svm
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
import json

def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###

    # Generating image mask using the felzenswalb algo
    segs_felzen = skimage.segmentation.felzenszwalb(image= im_orig, scale = scale, sigma= sigma, min_size=min_size)
    # Expanding it to 3 channels
    segs_felzen_colored = np.expand_dims(segs_felzen, axis=-1)
    # Merging the mask to the original image
    im_orig = np.concatenate((im_orig, segs_felzen_colored), axis = -1)

    return im_orig

def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###
    h_r1 = np.array(r1['hist_c'])
    h_r2 = np.array(r2['hist_c'])
    sim_col = np.minimum(h_r1, h_r2).sum()

    return sim_col


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    ### YOUR CODE HERE ###
    h_r1 = np.array(r1['hist_t'])
    h_r2 = np.array(r2['hist_t'])
    sim_text = np.minimum(h_r1, h_r2).sum()

    return sim_text


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    ### YOUR CODE HERE ###
    sim_size = 1 - ((r1['size'] + r2['size']) / imsize)

    return sim_size


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###

    # Calculating the bounding box
    bb_min_x = np.minimum(r1['min_x'], r2['min_x'])
    bb_max_x = np.maximum(r1['max_x'], r2['max_x'])
    bb_min_y = np.minimum(r1['min_y'], r2['min_y'])
    bb_max_y = np.maximum(r1['max_y'], r2['max_y'])
    bbij = (bb_max_x - bb_min_x) * (bb_max_y - bb_min_y)
    
    sim_fill = 1.0 - ((bbij - r1['size'] - r2['size']) / imsize)
    return sim_fill

def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    hist = np.array([])
    ### YOUR CODE HERE ###

    temp_hist = []
    for channel in range(3):
        channel_data = img[:, channel]
        count, _ = np.histogram(channel_data, bins = BINS, range = (0.0, 1.0))
        temp_hist.append(count)
    hist = np.concatenate(temp_hist) # Concatenating allt he 3 channels
    hist = hist / np.sum(hist) # L1 Normalization

    return hist

def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ### YOUR CODE HERE ###
    for channel in range(img.shape[2]):
        ret[:, : , channel] = skimage.feature.local_binary_pattern(image=img[:, :, channel], P=8, R=1.0, method= 'uniform')

    return ret

def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.array([])
    ### YOUR CODE HERE ###
    
    temp_hist = []
    for channel in range(3):
        channel_data = img[:, channel]
        count, _ = np.histogram(channel_data, bins = BINS, range = (0.0, BINS))
        temp_hist.append(count)
    hist = np.concatenate(temp_hist) # Concatenating allt he 3 channels
    hist = hist / np.sum(hist) # L1 Normalization

    return hist


def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}
    ### YOUR CODE HERE ###
    img_rgb = img[:, :, :3]
    img_mask = img[:, :, 3]

    # Converting image to hsv
    hsv_img = skimage.color.rgb2hsv(img_rgb)

    # Calculating the texture gradient
    tex_grad = calc_texture_gradient(img_rgb)

    # Counting pixel positions
    labels = np.unique(img_mask)
    for label in labels:
        y_idx, x_idx = np.where(img_mask == label)
        min_x = np.min(x_idx)
        max_x = np.max(x_idx)
        min_y = np.min(y_idx)
        max_y = np.max(y_idx)

        mask = (img_mask == label)
        masked_hsv = hsv_img[mask]
        masked_tex = tex_grad[mask]
        R[label] = {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "size": len(masked_hsv),
            "labels": [label],
            "hist_c": calc_colour_hist(masked_hsv), # Calculating color histogram
            "hist_t": calc_texture_hist(masked_tex) # Calculating texture histogram
        }


    return R

def extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    list_of_regions = list(regions.items())
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###
    for i in range(len(list_of_regions)):
        for j in range(i+1, len(list_of_regions)):
            region_a = list_of_regions[i][1]
            region_b = list_of_regions[j][1]
            if intersect(region_a, region_b):
                neighbours.append((list_of_regions[i], list_of_regions[j]))


    return neighbours

def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    ### YOUR CODE HERE

    # New Bounding box
    rt["min_x"] = min(r1["min_x"], r2["min_x"])
    rt["min_y"] = min(r1["min_y"], r2["min_y"])
    rt["max_x"] = max(r1["max_x"], r2["max_x"])
    rt["max_y"] = max(r1["max_y"], r2["max_y"])

    rt["size"] = new_size
    
    # Merging colot histograms usign weighted average
    rt["hist_c"] = (r1["size"] * r1["hist_c"] + 
                    r2["size"] * r2["hist_c"]) / new_size
    
    # Merging texture histograms usign weighted average
                    
    rt["hist_t"] = (r1["size"] * r1["hist_t"] + 
                    r2["size"] * r2["hist_t"]) / new_size
    
    # Merging Labels
    rt["labels"] = r1["labels"] + r2["labels"]

    return rt



def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        ### YOUR CODE HERE ###
        sim_to_remove = []
        for key, value in S.items():
            if (i in key) or (j in key):
                sim_to_remove.append(key)


        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###
        for rem in sim_to_remove:
            del S[rem]


        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###
        for k in sim_to_remove:
            if k == (i, j):
                continue
            for n in k:
                if n == i or n == j:
                    continue
                S[(t, n)] = calc_sim(R[t], R[n], imsize)


    # Task 8: Generating the final regions from R
    regions = []
    ### YOUR CODE HERE ###
    for k, r in R.items():
        # The output format requires 'rect' as (x, y, width, height)
        regions.append({
            'rect': (
                r['min_x'], 
                r['min_y'], 
                r['max_x'] - r['min_x'] + 1 , 
                r['max_y'] - r['min_y'] + 1
            ),
            'size': r['size'],
            'labels': r['labels']
        })


    return image, regions


####### Individual Exercise ###########

def feature_extractor():
    # Using pretrained resnet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Removing the last classification layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

pretrained_model = feature_extractor()
# preprocessing the data
preprocess_data = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def calculate_iou(boxA, boxB):
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB
    # converting (x,y,w,h) to (x1,y1,x2,y2)
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    # Calculating intersection
    xA = max(box1[0], box2[0]) 
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    # Calculating union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union = box1_area + box2_area - intersection_area
    # Calculating iou
    iou = intersection_area / union
    return iou

def extract_features(img):
    # Extracting feature
    preprocessed_img = preprocess_data(img).unsqueeze(0)
    with torch.no_grad():
        extracted_features = pretrained_model(preprocessed_img)
    return extracted_features.numpy().flatten()

def load_ground_truth(json_path):
    # Loading the annotations and mapping ground truth to image_id
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    # Mapping images to id
    file_to_id = {img['file_name']: img['id'] for img in json_data['images']}
    ground_truth_map = {}
    for i in json_data['annotations']:
        img_id = i['image_id']
        if img_id not in ground_truth_map:
            ground_truth_map[img_id] = []
        ground_truth_map[img_id].append(i['bbox'])
    return file_to_id, ground_truth_map

# Using different scales to calculate the proposals
def generate_proposals_multiscale(image):
    proposals = []

    for s in [100, 300, 500, 800]: 
        _, regions = selective_search(image, scale=s, sigma=0.9, min_size=50)
        
        for r in regions:
            x, y, w, h = r['rect']
            if w > 20 and h > 20: # Filtering small boxes
                proposals.append([x, y, w, h])
    
    # Removing any duplicates
    proposals = [list(x) for x in set(tuple(x) for x in proposals)]
    return proposals

def train_pipeline():
    base_dir = "../data/balloon_dataset"
    dataset = [
        ('train', '_annotations.coco.json'),
        ('valid', '_annotations.coco.json')
    ]
    
    Results_dir = "results"
    if not os.path.exists(Results_dir):
        os.makedirs(Results_dir)
        
    Proposals_path = os.path.join(Results_dir, 'proposals.pkl')

    # Task 5.2.1: Generating proposals
    if not os.path.exists(Proposals_path):
        print("Generating Proposals (Multi-Scale)...")
        proposals = {}
        
        for data, _ in dataset:
            data_type = os.path.join(base_dir, data)
            print(f"Processing {data_type} ")
            if os.path.exists(data_type):
                for img in os.listdir(data_type):
                    if not img.lower().endswith(('.png', '.jpg', '.jpeg')): 
                        continue
                    img_path = os.path.join(data_type, img)
                    image = skimage.io.imread(img_path)
                    # Use new multiscale function
                    img_proposals = generate_proposals_multiscale(image)
                    proposals[img] = img_proposals

        print("Saving proposals")
        with open(Proposals_path, 'wb') as f:
            pickle.dump(proposals, f)
    else:
        print("Loading proposals from file...")
        with open(Proposals_path, 'rb') as f:
            proposals = pickle.load(f)

    # Task 5.2.2: Creating positive and negative samples 
    tp = 0.5 
    tn = 0.3    
 
    x_train = []
    y_train = []
    
    print('Creating Positive and Negative Samples')
    
    for data, annot in dataset:
        data_type = os.path.join(base_dir, data)
        annotations_file = os.path.join(data_type, annot)
        
        if not os.path.exists(annotations_file): continue
            
        file_to_id, ground_truth_map = load_ground_truth(annotations_file)
        
        for img_name in os.listdir(data_type):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): 
                continue    
            img_path = os.path.join(data_type, img_name)               
            if not os.path.exists(img_path): 
                continue
            image = skimage.io.imread(img_path)            
            img_id = file_to_id.get(img_name)
            ground_truths = ground_truth_map.get(img_id, [])

            # Adding ground truth to the positive samples
            for gt_box in ground_truths:
                x, y, w, h = [int(v) for v in gt_box]
                patch = image[y:y+h, x:x+w]
                if patch.size > 0 and w > 5 and h > 5:
                    feature = extract_features(patch)
                    x_train.append(feature)
                    y_train.append(1) # Label 1 is Balloon
            
            # Adding Selective Search Proposals to the positive samples
            if img_name in proposals:
                region_rects = proposals[img_name]
                for rect in region_rects:
                    x, y, w, h = rect
                    max_iou = 0
                    for gt_box in ground_truths:
                        iou = calculate_iou(rect, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                    
                    label = None
                    if max_iou >= tp:
                        label = 1
                    elif max_iou < tn:
                        label = 0
                    
                    if label is not None:
                        patch = image[int(y):int(y+h), int(x):int(x+w)]
                        if patch.size > 0 and patch.shape[0] > 5 and patch.shape[1] > 5:
                            feature = extract_features(patch)
                            x_train.append(feature)
                            y_train.append(label)

    print(f"Total samples: Positive Samples: {sum(y_train)}, Negative Samples: {len(y_train) - sum(y_train)}")

    # Balancing the samples
    if sum(y_train) > 0:
        pos_count = sum(y_train)
        neg_count = len(y_train) - pos_count
        
        # Keeping negatives to max 2x positives
        if neg_count > pos_count * 2:
            neg_indices = [i for i, label in enumerate(y_train) if label == 0]
            pos_indices = [i for i, label in enumerate(y_train) if label == 1]
            
            keep_neg_indices = np.random.choice(neg_indices, size=pos_count * 2, replace=False)
            all_indices = list(pos_indices) + list(keep_neg_indices)
            
            x_train = [x_train[i] for i in all_indices]
            y_train = [y_train[i] for i in all_indices]
            
            print(f"After balancing: Positive Samples: {sum(y_train)}, Negative Samples: {len(y_train)-sum(y_train)}")

    # Task 5.2.3: Training SVM 
    print(f"Training RBF SVM on {len(x_train)} samples...")
    if len(x_train) > 0:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        
        # Using rbf kernel
        clf = SVC(kernel='rbf', gamma='scale', C=1.0, probability=True, class_weight='balanced')
        clf.fit(x_train_scaled, y_train)
        
        model_path = os.path.join(Results_dir, "svm_model.pkl")
        scalar_path = os.path.join(Results_dir, "scalar.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        with open(scalar_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"SVM (RBF) trained and saved to {model_path}")
if __name__ == "__main__":
    train_pipeline()