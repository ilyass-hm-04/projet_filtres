# Projet filtres

Application de traitement d'images avec interface graphique Tkinter. Elle
permet d'importer une image ou d'utiliser une caméra, puis de comparer l'image
originale et le résultat des filtres en temps réel.

## Fonctions

- Filtres de lissage : moyenneur, gaussien, médian et bilatéral.
- Ajout de bruits gaussien, sel et poivre, et uniforme.
- Détection de contours : Sobel, Laplacien et Canny.
- Pipelines de lissage suivis de Canny.
- Affichage des histogrammes et de l'erreur quadratique moyenne (MSE).

## Installation et lancement

Python 3.10 ou plus récent est recommandé. Sous Windows :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python vision_filters_app.py
```

Tkinter est fourni avec la plupart des installations Python pour Windows.
