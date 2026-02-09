import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Charger le modèle que tu as entraîné
print("Chargement du modèle...")
model = load_model('mon_modele_chiffres.h5')

# 2. Configurer la webcam
cap = cv2.VideoCapture(0) # 0 est généralement la webcam par défaut
cap.set(3, 1280) # Largeur
cap.set(4, 720)  # Hauteur

def pretraiter_image(img_roi):
    """Prépare l'image détectée pour qu'elle ressemble aux données MNIST"""
    # MNIST est en noir et blanc (1 canal)
    # L'image doit être redimensionnée à 28x28 pixels
    img_roi = cv2.resize(img_roi, (28, 28))
    # Normalisation (0-1)
    img_roi = img_roi.astype('float32') / 255.0
    # Reshape pour correspondre à l'entrée du modèle (1 image, 28x28, 1 canal)
    img_roi = img_roi.reshape(1, 28, 28, 1)
    return img_roi

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- TRAITEMENT D'IMAGE ---
    
    # 1. Convertir en gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Flou pour réduire le bruit (grains)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Seuillage (Threshold) : C'est CRUCIAL.
    # On utilise THRESH_BINARY_INV car MNIST attend des chiffres BLANCS sur fond NOIR.
    # Mais tu écris probablement en NOIR sur feuille BLANCHE. L'inversion règle ça.
    _, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

    # 4. Trouver les contours (les formes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # On ignore les trop petits points (bruit)
        if area > 1000: 
            # Récupérer les coordonnées du rectangle autour du chiffre
            x, y, w, h = cv2.boundingRect(cnt)
            
            # On découpe la zone du chiffre (Region of Interest - ROI)
            # On ajoute un peu de marge (padding) pour ne pas coller au chiffre
            padding = 10
            # Sécurité pour ne pas sortir de l'image
            y_start = max(y - padding, 0)
            y_end = min(y + h + padding, frame.shape[0])
            x_start = max(x - padding, 0)
            x_end = min(x + w + padding, frame.shape[1])
            
            roi = thresh[y_start:y_end, x_start:x_end]
            
            if roi.size > 0: # Si la découpe a réussi
                # Prétraitement pour le modèle
                img_input = pretraiter_image(roi)
                
                # --- PRÉDICTION ---
                prediction = model.predict(img_input, verbose=0)
                chiffre_detecte = np.argmax(prediction) # L'indice de la plus haute probabilité
                confiance = np.max(prediction) # Le pourcentage de certitude
                
                # --- AFFICHAGE ---
                # Dessiner le rectangle vert
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Écrire le chiffre
                texte = f"{chiffre_detecte} ({int(confiance*100)}%)"
                cv2.putText(frame, texte, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher la vue webcam
    cv2.imshow('Reconnaissance de Chiffres - Quitter avec "q"', frame)
    # Afficher aussi ce que l'ordi "voit" (le seuillage) pour t'aider à débugger
    cv2.imshow('Vision Ordinateur (Noir/Blanc)', thresh) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()