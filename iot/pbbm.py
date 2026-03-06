import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Paramètres standards de l'article de Yang (2012)
ROI_WIDTH = 96
ROI_HEIGHT = 64
LBP_RADIUS = 1
LBP_POINTS = 8

def extract_roi(image_path):
    """
    Étape 1: Prétraitement & Extraction de la Région d'Intérêt (ROI).
    """
    img = cv2.imread(image_path, 0)
    if img is None:
        return None

    # L'article mentionne un recadrage des bords avec l'opérateur Sobel + Normalisation
    # Simplification de l'extraction des contours du doigt:
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
        
    # Prendre le contour le plus grand (le doigt)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Recadrer la ROI grossière
    roi = img[y:y+h, x:x+w]
    
    # Normaliser la taille en 96x64 comme décrit dans l'article PBBM
    roi = cv2.resize(roi, (ROI_WIDTH, ROI_HEIGHT), interpolation=cv2.INTER_AREA)
    
    # Normalisation de gris (histogram equalization)
    roi_eq = cv2.equalizeHist(roi)
    return roi_eq

def extract_lbp(roi):
    """
    Étape 2: Extraction du Local Binary Pattern.
    """
    if roi is None:
        return None
        
    # Utilise l'algorithme LBP 'default' qui encode la texture locale en octets
    lbp = local_binary_pattern(roi, LBP_POINTS, LBP_RADIUS, method='default')
    
    # Retourne une matrice 96x64 contenant des entiers de 0 à 255 (8 bits)
    return lbp.astype(np.uint8)

def generate_pbbm(user_lbps):
    """
    Étape 3: Formation du Mask Personalized Best Bit Map (PBBM).
    user_lbps: Liste des LBP (images 96x64 de 8 bits) d'entrainement pour UN sujet.
    """
    n_samples = len(user_lbps)
    if n_samples < 2:
        return None, None
        
    # On choisit l'une des images comme LBP de référence (ex: la première)
    reference_lbp = user_lbps[0]
    
    # Initialisation du masque PBBM. 
    # Au départ, on considère que tous les 8 bits de tous les pixels sont "bons" (masque = 255).
    pbbm_mask = np.full((ROI_HEIGHT, ROI_WIDTH), 255, dtype=np.uint8)
    
    # On compare la référence avec toutes les autres images d'entrainement
    for i in range(1, n_samples):
        # L'article mentionne un alignement de la ROI, pour l'instant on fait un XOR direct.
        # XOR (OU Exclusif) : 0 = bits identiques, 1 = bits différents (bruit)
        diff_bits = cv2.bitwise_xor(reference_lbp, user_lbps[i])
        
        # Le masque s'affine en éliminant les bits instables. 
        # Si un bit a différé dans l'image 'i', NOT(diff_bits) mettra un 0 à cet endroit.
        # En accumulant les masques par un AND, le bit restera 0 pour toujours.
        pbbm_mask = cv2.bitwise_and(pbbm_mask, cv2.bitwise_not(diff_bits))

    return reference_lbp, pbbm_mask

def match_pbbm_translate(test_lbp, ref_lbp, pbbm_mask):
    """
    Étape 4: Matching PBBM via XOR avec Alignement Intelligent par Blocs.
    Les doigts étant élastiques, une corrélation de phase globale ne suffit pas.
    On découpe l'image en blocs pour tolérer les déformations non-linéaires.
    """
    if test_lbp is None or ref_lbp is None or pbbm_mask is None:
        return 1.0
        
    valid_bits = np.unpackbits(pbbm_mask).sum()
    if valid_bits == 0:
        return 1.0
        
    h, w = test_lbp.shape
    
    # On découpe l'image en 4 blocs verticaux (les veines s'étirant souvent dans cette direction)
    n_blocks = 4
    block_w = w // n_blocks
    
    total_mismatches = 0
    max_shift = 8
    
    for i in range(n_blocks):
        x_start = i * block_w
        x_end = (i + 1) * block_w if i < n_blocks - 1 else w
        
        test_block = test_lbp[:, x_start:x_end]
        ref_block = ref_lbp[:, x_start:x_end]
        pbbm_block = pbbm_mask[:, x_start:x_end]
        
        # Corrélation de phase locale
        shift, _ = cv2.phaseCorrelate(test_block.astype(np.float32), ref_block.astype(np.float32))
        dx, dy = shift
        
        dx = max(min(dx, max_shift), -max_shift)
        dy = max(min(dy, max_shift), -max_shift)
        
        # Alignement local
        bh, bw = test_block.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_test_block = cv2.warpAffine(test_block, M, (bw, bh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Erreur bit-à-bit locale
        error_map = cv2.bitwise_xor(shifted_test_block, ref_block)
        filtered_error = cv2.bitwise_and(error_map, pbbm_block)
        
        total_mismatches += np.unpackbits(filtered_error).sum()
        
    return total_mismatches / valid_bits

if __name__ == '__main__':
    print("--- Test de l'Algorithme PBBM (Yang 2012) ---")
    
    # Nous allons tester l'enrôlement de la personne "001" (doigt gauche)
    files_train = [f'data/001/left/index_{i}.bmp' for i in range(1, 4)] # Images 1,2,3 pour entrainement
    file_match = 'data/001/left/index_4.bmp' # Image 4 pour teste du propriétaire
    file_impost = 'data/001/left/middle_1.bmp' # Doigt du milieu = Imposteur
    file_impost2 = 'data/002/left/index_1.bmp' # Doigt de la personne 2
    
    print("Enregistrement du profil LBP et création de la carte PBBM pour l'Utilisateur 001...")
    train_lbps = []
    for f in files_train:
        roi = extract_roi(f)
        lbp = extract_lbp(roi)
        if lbp is not None:
            train_lbps.append(lbp)
            
    # Entrainement PBBM
    ref_lbp, pbbm_mask = generate_pbbm(train_lbps)
    total_bits = 96 * 64 * 8
    best_bits = np.unpackbits(pbbm_mask).sum()
    print(f"-> Sur {total_bits} bits disponibles, {best_bits} bits se sont révélés 'stables' et fiables.")
    
    print("\n--- Test de Vérification ---")
    
    # 1. Matcher Vrai Utilisateur
    roi_match = extract_roi(file_match)
    lbp_match = extract_lbp(roi_match)
    score_match = match_pbbm_translate(lbp_match, ref_lbp, pbbm_mask)
    print(f"Score d'erreur (Vrai User - Index_4) : {score_match:.5f}")
    
    # 2. Matcher Imposteur
    roi_imp = extract_roi(file_impost)
    lbp_imp = extract_lbp(roi_imp)
    score_imp = match_pbbm_translate(lbp_imp, ref_lbp, pbbm_mask)
    print(f"Score d'erreur (Imposteur - Middle_1): {score_imp:.5f}")
    
    roi_imp2 = extract_roi(file_impost2)
    lbp_imp2 = extract_lbp(roi_imp2)
    score_imp2 = match_pbbm_translate(lbp_imp2, ref_lbp, pbbm_mask)
    print(f"Score d'erreur (Imposteur - User002): {score_imp2:.5f}")
    
    # L'EER pour LBP de Yang est autour de 0.003
    SEUIL_ACCEPTATION = 0.15000 
    print(f"\nSeuil d'erreur maximum d'acceptation : {SEUIL_ACCEPTATION}")
    
    print("\nDécisions :")
    print(f"Index 4 (User)   : {'REJETÉ' if score_match > SEUIL_ACCEPTATION else 'ACCEPTÉ \u2705'}")
    print(f"Middle 1 (Fake)  : {'REJETÉ \u2705' if score_imp > SEUIL_ACCEPTATION else 'ACCEPTÉ'}")
    print(f"User 002 (Fake)  : {'REJETÉ \u2705' if score_imp2 > SEUIL_ACCEPTATION else 'ACCEPTÉ'}")
