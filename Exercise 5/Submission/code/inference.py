import os
import json
import pickle
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import selective_search
from selective_search import calculate_iou, generate_proposals_multiscale

TEST_DIR = os.path.join("../data/balloon_dataset", "test")
ANNOT_FILE = os.path.join(TEST_DIR, r"_annotations.coco.json")
MODEL_PATH = os.path.join("results", "svm_model.pkl")
SCALER_PATH = os.path.join("results", "scalar.pkl")

def nms(boxes, overlap_thresh=0.3):
    if len(boxes) == 0: return []
    boxes = np.array(boxes)
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    score = boxes[:,4]
    area = (boxes[:,2] + 1) * (boxes[:,3] + 1)
    idxs = np.argsort(score)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].tolist()

def feature_extractor():
    # Using pretrained resnet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Removing the last classification layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

pretrained_model = feature_extractor()

preprocess_data = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(img):
    # Extracting feature
    preprocessed_img = preprocess_data(img).unsqueeze(0)
    with torch.no_grad():
        extracted_features = pretrained_model(preprocessed_img)
    return extracted_features.numpy().flatten()

def detect_objects(image, clf, scalar):
    # Generating proposals 
    raw_proposals = generate_proposals_multiscale(image)
    detections = []

    # classifying
    for rect in raw_proposals:
        x, y, w, h = rect
        
        y_min = max(0, int(y))
        x_min = max(0, int(x))
        y_max = min(image.shape[0], int(y+h))
        x_max = min(image.shape[1], int(x+w))
        
        patch = image[y_min:y_max, x_min:x_max]
        
        if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2: 
            continue
        
    
        feat = extract_features(patch).reshape(1, -1)
        feat_scaled = scalar.transform(feat)
        
        # Using RBF SVM for prediction
        prediction = clf.predict(feat_scaled)[0]
        
        if prediction == 1:
            score = clf.decision_function(feat_scaled)[0]
            detections.append([float(x), float(y), float(w), float(h), float(score)])
    
            
    return raw_proposals, detections

# Task 5.2.5: Evaluation using COCO mAP and MABO
def main():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f: clf = pickle.load(f)

    if not os.path.exists(SCALER_PATH):
        print("Scaler file not found.")
        return
    
    print(f"Loading scaler from {SCALER_PATH}...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load Ground Truth using COCO API
    cocoGt = COCO(ANNOT_FILE)
    
    imgIds = cocoGt.getImgIds()
    coco_results = [] 

    all_best_overlaps = []

    print(f"Task 5.2.5: Evaluating on {len(imgIds)} images...")
    
    for img_id in imgIds:
        img_info = cocoGt.loadImgs(img_id)[0]
        filename = img_info['file_name']
        img_path = os.path.join(TEST_DIR, filename)
        
        if not os.path.exists(img_path): continue
        
        image = skimage.io.imread(img_path)
        raw_proposals, detections = detect_objects(image, clf, scaler)

        # MABO Calculation
        gt_anns = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=img_id))
        gt_boxes = [ann['bbox'] for ann in gt_anns]
        if len(gt_boxes) > 0 and len(raw_proposals) > 0:
            for gt_box in gt_boxes:
                best_iou = 0.0
                for proposal in raw_proposals:
                    iou = calculate_iou(proposal, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                all_best_overlaps.append(best_iou)
        
        for det in detections:
            coco_results.append({
                'image_id': img_id,
                'category_id': 1, 
                'bbox': [det[0], det[1], det[2], det[3]],
                'score': det[4]
            })

        print(f"Processed {filename}: Found {len(detections)} detections")
        if img_id == imgIds[2]:
            clean_boxes = nms(detections)
            gt_anns = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=img_id))
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image)
            for ann in gt_anns:
                box = ann['bbox']
                ax.add_patch(mpatches.Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor='lime', linewidth=2, label='Ground Truth'))
            for box in clean_boxes:
                ax.add_patch(mpatches.Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor='red', linewidth=2, label='Prediction'))
            plt.title(f"Detections")
            plt.show()

    if len(all_best_overlaps) > 0:
        mabo = np.mean(all_best_overlaps)
        print("Evaluation Results")
        print(f"MABO (Mean Average Best Overlap): {mabo:}")
    
    # COCO Evaluation
    if len(coco_results) > 0:
        print("Running COCO Evaluation...")
        cocoDt = cocoGt.loadRes(coco_results)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        print("\nCOCO METRICS")
        print(f"COCO mAP@[0.50:0.95]:      {cocoEval.stats[0]:}")
        print(f"COCO mAP@0.50:             {cocoEval.stats[1]:}")
        print(f"COCO mAP@0.75:             {cocoEval.stats[2]:}")
    else:
        print("\nNo detections found")
    
if __name__ == "__main__":
    main()