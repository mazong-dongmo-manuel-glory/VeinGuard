import cv2
import numpy as np
import io
import sys
import os

# Ensure we can import pbbm even if launched from another dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pbbm import generate_pbbm, extract_lbp, match_pbbm_translate

def _bytes_to_image(image_bytes):
    """
    Decodes image bytes to a numpy array (grayscale)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img

def _image_to_lbp(image_np):
    """
    Extracts LBP from an image array, bypassing file extraction from pbbm.py
    """
    if image_np is None:
        return None
        
    _, thresh = cv2.threshold(image_np, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    roi = image_np[y:y+h, x:x+w]
    roi = cv2.resize(roi, (96, 64), interpolation=cv2.INTER_AREA)
    roi_eq = cv2.equalizeHist(roi)
    
    return extract_lbp(roi_eq)

def enroll_user(image_bytes_list):
    """
    Takes a list of image bytes, extracts LBPs, and generates a PBBM mask.
    Returns (reference_lbp_bytes, pbbm_mask_bytes) or (None, None).
    """
    lbps = []
    for img_b in image_bytes_list:
        np_img = _bytes_to_image(img_b)
        lbp = _image_to_lbp(np_img)
        if lbp is not None:
            lbps.append(lbp)
            
    if len(lbps) < 2:
        return None, None
        
    ref_lbp, pbbm_mask = generate_pbbm(lbps)
    if ref_lbp is None or pbbm_mask is None:
        return None, None
        
    return ref_lbp.tobytes(), pbbm_mask.tobytes()

def verify_user(test_image_bytes, ref_lbp_bytes, pbbm_mask_bytes, threshold=0.15):
    """
    Verifies a test image against the reference LBP and mask bytes.
    Returns (is_match, score).
    """
    np_img = _bytes_to_image(test_image_bytes)
    test_lbp = _image_to_lbp(np_img)
    if test_lbp is None:
        return False, 1.0
        
    # Reconstruct arrays from bytes
    ref_lbp = np.frombuffer(ref_lbp_bytes, dtype=np.uint8).reshape((64, 96))
    pbbm_mask = np.frombuffer(pbbm_mask_bytes, dtype=np.uint8).reshape((64, 96))
    
    score = match_pbbm_translate(test_lbp, ref_lbp, pbbm_mask)
    return (score <= threshold), score
