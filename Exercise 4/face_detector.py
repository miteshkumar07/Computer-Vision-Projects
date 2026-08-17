import cv2
from mtcnn import MTCNN
import numpy as np
import os
import copy


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=20, tm_threshold=0.7, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

	# Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.tm_window_size = tm_window_size

    # Track a face in a new image using template matching.
    def track_face(self, image):
        # Get the first image
        if self.reference is None:
            self.reference = self.detect_face(image)
            if self.reference is None:
                return None
        
        else:
            # Get the region of interest
            self.rectangle= copy.copy(self.reference["rect"])
            region_of_interest = self.crop_face(image, self.rectangle)
            
            # Get the template
            intial_template = self.crop_face(self.reference["image"], self.reference["rect"])
            
            # Template matching
            result = cv2.matchTemplate(region_of_interest, intial_template, cv2.TM_CCOEFF_NORMED)
            
            # Get the max value and location
            min_value, max_value, min_location, max_location = cv2.minMaxLoc(result)
            
            if max_value < self.tm_threshold:
                # reintialize the reference
                self.reference = self.detect_face(image)
                if self.reference is None:
                    return None
            else:
                # Update the rectangle
                x,y = max_location
                x+=self.rectangle[0]
                y+=self.rectangle[1]
                top_left_corner = (x,y)
                bottom_right_corner= (self.rectangle[2],self.rectangle[3])
                face_rectangle= (top_left_corner[0],top_left_corner[1],bottom_right_corner[0],bottom_right_corner[1])
                
                # Align the face
                aligned_image= self.align_face(image, face_rectangle)
                self.reference = {"rect": face_rectangle, "image": image, "aligned": aligned_image, "response": max_value}
                
        return self.reference

    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        detections = self.detector.detect_faces(image)
        if not detections:
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]
    
    def image_load(self,folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img_rgb is not None:
                images.append(img_rgb)
        return images


