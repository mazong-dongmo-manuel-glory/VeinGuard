import os
import random
import time
import numpy as np
from pbbm import extract_roi, extract_lbp, generate_pbbm, match_pbbm_translate
from glob import glob

def evaluate_pbbm_dataset(data_dir, num_subjects=10):
    print(f"Chargement d'un échantillon de {num_subjects} sujets...")
    subjects = sorted(os.listdir(data_dir))
    subjects = [s for s in subjects if os.path.isdir(os.path.join(data_dir, s))]
    
    selected_subjects = random.sample(subjects, min(num_subjects, len(subjects)))
    
    # 1. Enrôlement : Extraire la PBBM et le LBP de réf pour chaque doigt (3 images d'entrainement)
    enrolled_profiles = {}
    test_images = {} # Images restantes (test set)
    
    count_train = 0
    t0 = time.time()
    for subject in selected_subjects:
        for hand in ['left', 'right']:
            hand_dir = os.path.join(data_dir, subject, hand)
            if not os.path.isdir(hand_dir): continue
            
            images = glob(os.path.join(hand_dir, '*.bmp'))
            fingers_dict = {}
            for img in images:
                finger_type = os.path.basename(img).split('_')[0]
                uid = f"{subject}_{hand}_{finger_type}"
                if uid not in fingers_dict: fingers_dict[uid] = []
                fingers_dict[uid].append(img)
                
            for uid, paths in fingers_dict.items():
                if len(paths) < 4: continue # Pas assez d'images pour PBBM
                
                # Prendre les 3 premières images pour entrainer le masque PBBM perso
                train_paths = paths[:3]
                test_paths = paths[3:]
                
                # Extraction
                train_lbps = []
                for p in train_paths:
                    roi = extract_roi(p)
                    lbp = extract_lbp(roi)
                    if lbp is not None: train_lbps.append(lbp)
                
                if len(train_lbps) == 3:
                    ref_lbp, pbbm_mask = generate_pbbm(train_lbps)
                    if ref_lbp is not None:
                        enrolled_profiles[uid] = (ref_lbp, pbbm_mask)
                        test_images[uid] = test_paths
                count_train += 1
                if count_train % 50 == 0:
                    print(f" ... {count_train} profils générés ({time.time()-t0:.1f}s)")
                    
    # 2. Évaluation des scores 
    intra_scores = []
    inter_scores = []
    
    print("\nÉvaluation des tests Intra/Inter...")
    uids = list(enrolled_profiles.keys())
    
    # Test Intra : Vrais utilisateurs (Images test contre Profil Entrainé)
    for uid, paths in test_images.items():
        ref_lbp, pbbm_mask = enrolled_profiles[uid]
        for p in paths:
            roi = extract_roi(p)
            lbp = extract_lbp(roi)
            if lbp is not None:
                err = match_pbbm_translate(lbp, ref_lbp, pbbm_mask)
                intra_scores.append(err)
                
    # Test Inter : Imposteurs
    # On teste les LBP d'autres utilisateurs sur les profils entrainés
    num_inter_tests = len(intra_scores) * 2
    
    for _ in range(num_inter_tests):
        uid_prof, uid_test = random.sample(uids, 2)
        if not test_images[uid_test]: continue
        
        path = random.choice(test_images[uid_test])
        roi = extract_roi(path)
        lbp = extract_lbp(roi)
        
        if lbp is not None:
            ref_lbp, pbbm_mask = enrolled_profiles[uid_prof]
            err = match_pbbm_translate(lbp, ref_lbp, pbbm_mask)
            inter_scores.append(err)
            
    return intra_scores, inter_scores

def compute_metrics(intra_scores, inter_scores):
    # Les scores PBBM sont des TAUX D'ERREUR (0.0 = parfait, 1.0 = faux).
    # On cherche le seuil d'erreur maximal toléré.
    thresholds = np.arange(0.0, 0.40, 0.005)
    best_thresh = 0
    min_diff = float('inf')
    best_far = 0
    best_frr = 0
    
    for t in thresholds:
        # Faux Rejet : L'erreur du vrai user est SUPÉRIEURE au seuil toléré
        frr = sum(1 for s in intra_scores if s > t) / max(1, len(intra_scores))
        # Fausse Acception : L'erreur de l'imposteur est INFÉRIEURE au seuil
        far = sum(1 for s in inter_scores if s <= t) / max(1, len(inter_scores))
        
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            best_thresh = t
            best_far = far
            best_frr = frr
            
    # Accuracy globale
    true_pos = sum(1 for s in intra_scores if s <= best_thresh)
    true_neg = sum(1 for s in inter_scores if s > best_thresh)
    total = len(intra_scores) + len(inter_scores)
    accuracy = (true_pos + true_neg) / total if total > 0 else 0
    
    return best_thresh, best_far, best_frr, accuracy

if __name__ == '__main__':
    data_dir = 'data'
    random.seed(42)
    
    # Évaluation sur une grosse portion du dataset (ex: 25 sujets)
    print("=== ÉVALUATION DE MASSE : MÉTHODE LBP + PBBM (Yang 2012) ===")
    intra, inter = evaluate_pbbm_dataset(data_dir, num_subjects=50) # Moitié du dataset
    
    print("\nCalcul des métriques (EER)...")
    print(f"Tests Vrais Utilisateurs : {len(intra)}")
    print(f"Tests d'Imposteurs       : {len(inter)}")
    
    best_t, far, frr, acc = compute_metrics(intra, inter)
    eer = (far + frr) / 2.0
    
    print("\n" + "="*50)
    print("RÉSULTATS DE L'ALGORITHME PBBM")
    print("="*50)
    print(f"Seuil d'erreur toléré (T) : {best_t:.4f} (Ratio de bits XOR)")
    print(f"Taux d'accep. (FAR)      : {far*100:.2f}%")
    print(f"Taux de rejet (FRR)      : {frr*100:.2f}%")
    print(f"Equal Error Rate (EER)   : {eer:.4f}")
    print(f"\nSCORE DE CONFIANCE (ACC) : {acc*100:.2f}%")
    print("="*50)
    
    print(f"Erreur moyenne (Vrais)   : {np.mean(intra):.4f}")
    print(f"Erreur moyenne (Impost.) : {np.mean(inter):.4f}")
