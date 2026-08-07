# Modèle de prompt pour Claude

Ce prompt sert à demander un lot facile à importer et à auditer. Remplacez les
éléments entre crochets avant de l'envoyer. Une première série courte de 5 à
10 échantillons par classe est préférable afin de vérifier le format.

## Prompt

```text
Je réalise une expérience scientifique sur la reconnaissance de caractères.
Génère un dataset synthétique de matrices 28 × 28 pour les classes suivantes :
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

Objectif : obtenir [NOMBRE] variantes par classe qui imitent différents styles
manuscrits (inclinaison, épaisseur, largeur, hauteur et forme), tout en gardant
chaque caractère lisible et entièrement contenu dans l'image.

Contraintes strictes :
1. Retourne un unique document JSON valide, sans bloc Markdown ni commentaire.
2. Utilise exactement cette structure racine :
   {"format_version": 1, "samples": [...]}
3. Chaque objet de "samples" contient :
   - "id" : identifiant unique et stable ;
   - "label" : classe sous forme de chaîne ;
   - "split" : toujours "non_attribue" ;
   - "style" : courte description du style visé ;
   - "pixels" : exactement 28 lignes de exactement 28 entiers.
4. Chaque pixel vaut seulement 0 ou 1 : 0 pour le fond, 1 pour le trait.
5. Ne fournis ni prédiction, ni score, ni donnée MNIST recopiée.
6. Laisse une marge vide autour du caractère. Aucun trait ne doit toucher le bord.
7. Vérifie avant de répondre : dimensions 28 × 28, valeurs dans {0,1}, nombre
   d'échantillons et équilibre exact entre les classes.
8. Les variantes d'une même classe doivent être réellement différentes ; évite
   les doublons exacts.

Ajoute dans la réponse JSON un objet "generation_notes" à la racine contenant
la liste des styles demandés et les limites connues de cette génération.
```

## Après la réponse

1. Enregistrez le contenu brut dans un fichier `.json`, par exemple
   `donnees/synthetiques_a_importer/lot_claude_01.json`.
2. Ne corrigez pas silencieusement une matrice : laissez l'importeur consigner
   les erreurs dans `rapports/rejets_import.csv`.
3. Importez avec `--origine claude` et une description indiquant le prompt ou le
   lot utilisé.
4. Examinez la grille PNG et le rapport de validation avant tout entraînement.
5. Gardez ce dataset distinct de MNIST afin de comparer expérimentalement :
   MNIST seul, augmentations classiques et données générées.

Pour d'autres caractères, remplacez simplement la liste des classes. Les
étiquettes restent des chaînes (`"A"`, `"?"`, etc.). Si un caractère est ambigu
dans une police manuscrite, demandez à Claude de le signaler dans `style` plutôt
que de changer son étiquette.
