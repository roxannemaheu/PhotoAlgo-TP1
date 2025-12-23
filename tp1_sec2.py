#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP1 Section 2: Dématriçage (Demosaicking)

Ce script:
1. Charge les mosaïques TIFF normalisées et les métadonnées depuis ./images_intermediaires_sec1/
2. Applique le dématriçage par interpolation bilinéaire (implémenté)
3. Applique le dématriçage Malvar-He-Cutler 2004 (À IMPLÉMENTER)
4. [Cycles supérieurs] Applique le dématriçage appris (À IMPLÉMENTER)
5. Génère des images de comparaison
6. Sauvegarde dans ./images_intermediaires_sec2/

Usage:
    python tp1_sec2.py --input-dir images_intermediaires_sec1 --output-dir images_intermediaires_sec2
"""

import numpy as np
import glob
import os
import time
from scipy.signal import convolve2d

from tp1_io import load_tiff, load_metadata, save_tiff16, save_jpeg, linear_to_srgb
from tp1_rapport import (
    html_document, section, subsection, figure, table, algorithm_box, save_report,
    create_demosaic_comparison_figure, create_difference_figure,
    find_edge_region, create_demosaic_zoom_figure
)


# =============================================================================
# Fonctions Utilitaires
# =============================================================================

def get_color_masks(pattern_2x2, H, W):
    """
    Créer des masques booléens pour les canaux R, G, B selon le motif de Bayer.
    
    Args:
        pattern_2x2: Motif de Bayer 2x2 (ex: [['R','G'],['G','B']])
        H, W: Dimensions de l'image
    
    Returns:
        masks: Dict avec les masques booléens pour 'R', 'G', 'B'
        positions: Dict avec les positions (i,j) dans le motif 2x2 pour chaque couleur
    """
    positions = {'R': [], 'G': [], 'B': []}
    for i in range(2):
        for j in range(2):
            positions[pattern_2x2[i][j]].append((i, j))
    
    masks = {}
    for color in 'RGB':
        mask = np.zeros((H, W), dtype=bool)
        for i, j in positions[color]:
            mask[i::2, j::2] = True
        masks[color] = mask
    
    return masks, positions


# =============================================================================
# Algorithmes de Dématriçage
# =============================================================================

def demosaic_bilinear(raw_data, pattern_2x2):
    """
    Dématriçage par interpolation bilinéaire.
    
    Approche simple: interpoler les valeurs de couleur manquantes
    en utilisant la moyenne des pixels voisins de la même couleur.
    
    Args:
        raw_data: Image mosaïque 2D normalisée [0, 1]
        pattern_2x2: Motif de Bayer 2x2 (ex: [['R','G'],['G','B']])
    
    Returns:
        Image RGB 3D [H, W, 3] normalisée [0, 1]
    """
    H, W = raw_data.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    masks, _ = get_color_masks(pattern_2x2, H, W)
    
    # Noyaux d'interpolation
    # Pour R et B: moyenne des 4 voisins diagonaux + 4 voisins orthogonaux
    kernel_full = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 4
    # Pour G: moyenne des 4 voisins orthogonaux uniquement
    kernel_cross = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32) / 4
    
    # Canal Rouge
    r_plane = raw_data * masks['R']
    rgb[:, :, 0] = convolve2d(r_plane, kernel_full, mode='same', boundary='symm')
    rgb[masks['R'], 0] = raw_data[masks['R']]  # Garder les valeurs originales
    
    # Canal Bleu
    b_plane = raw_data * masks['B']
    rgb[:, :, 2] = convolve2d(b_plane, kernel_full, mode='same', boundary='symm')
    rgb[masks['B'], 2] = raw_data[masks['B']]
    
    # Canal Vert
    g_plane = raw_data * masks['G']
    rgb[:, :, 1] = convolve2d(g_plane, kernel_cross, mode='same', boundary='symm')
    rgb[masks['G'], 1] = raw_data[masks['G']]
    
    return np.clip(rgb, 0, 1)


def demosaic_malvar(raw_data, pattern_2x2):
    """
    Dématriçage par la méthode Malvar-He-Cutler (2004).
    
    Interpolation corrigée par gradient qui réduit les artefacts de couleur.
    Référence: "High-Quality Linear Interpolation for Demosaicing of Bayer-Patterned Color Images"
    
    Args:
        raw_data: Image mosaïque 2D normalisée [0, 1]
        pattern_2x2: Motif de Bayer 2x2 (ex: [['R','G'],['G','B']])
    
    Returns:
        Image RGB 3D [H, W, 3] normalisée [0, 1]
    
    TODO: Implémenter l'algorithme Malvar-He-Cutler avec les noyaux 5×5
          décrits dans la Figure 2 de l'article.
          
    Indices:
    - Les noyaux sont définis pour différentes configurations:
      * G aux positions R/B
      * R aux positions G dans les rangées R
      * R aux positions G dans les rangées B  
      * R aux positions B
      * (et symétriquement pour B)
    - Les noyaux utilisent des corrections de gradient pour réduire les artefacts
    """
    H, W = raw_data.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    masks, positions = get_color_masks(pattern_2x2, H, W)
    
    # =========================================================================
    # TODO: Implémenter les noyaux Malvar-He-Cutler 5×5
    # =========================================================================
    # 
    # Exemple de structure pour les noyaux (à compléter avec les vraies valeurs):
    #
    # kernel_g_at_rb = np.array([
    #     [0,  0, -1,  0,  0],
    #     [0,  0,  2,  0,  0],
    #     [-1, 2,  4,  2, -1],
    #     [0,  0,  2,  0,  0],
    #     [0,  0, -1,  0,  0]
    # ], dtype=np.float32) / 8
    #
    # kernel_rb_at_g_same_row = ...
    # kernel_rb_at_g_same_col = ...
    # kernel_rb_at_opposite = ...
    #
    # Puis appliquer les convolutions appropriées selon les positions du motif.
    # =========================================================================
    
    # Pour l'instant, utilise l'interpolation bilinéaire comme placeholder
    print("    [ATTENTION] Malvar-He-Cutler non implémenté, utilisation de bilinéaire")
    return demosaic_bilinear(raw_data, pattern_2x2)


def demosaic_learned(raw_data, pattern_2x2, models=None):
    """
    Dématriçage par apprentissage automatique.
    
    [CYCLES SUPÉRIEURS UNIQUEMENT]
    
    Utilise un modèle entraîné (SGDRegressor, RandomForest, ou MLP) pour
    prédire les valeurs de couleur manquantes à partir de fenêtres locales.
    
    Args:
        raw_data: Image mosaïque 2D normalisée [0, 1]
        pattern_2x2: Motif de Bayer 2x2
        models: Dict avec les modèles entraînés pour chaque canal {'r': model, 'g': model, 'b': model}
    
    Returns:
        Image RGB 3D [H, W, 3] normalisée [0, 1]
    
    TODO: Implémenter le dématriçage appris:
    1. Charger les données d'entraînement depuis ./demosaic_dataset/
    2. Entraîner un modèle par canal (R, G, B)
    3. Pour chaque pixel, extraire une fenêtre locale (ex: 9×9)
    4. Utiliser le modèle pour prédire la valeur manquante
    
    Indices:
    - Vous pouvez utiliser sklearn.linear_model.SGDRegressor
    - Ou sklearn.ensemble.RandomForestRegressor
    - Ou sklearn.neural_network.MLPRegressor
    """
    # =========================================================================
    # TODO: Implémenter le dématriçage appris (cycles supérieurs)
    # =========================================================================
    
    print("    [ATTENTION] Dématriçage appris non implémenté")
    return demosaic_bilinear(raw_data, pattern_2x2)


# =============================================================================
# Métriques de Qualité
# =============================================================================

def compute_psnr(img1, img2):
    """Calculer le PSNR entre deux images."""
    mse = np.mean((img1 - img2) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)


def compute_ssim(img1, img2):
    """
    Calculer le SSIM simplifié entre deux images.
    
    Note: Pour une implémentation complète, utilisez skimage.metrics.structural_similarity
    """
    from skimage.metrics import structural_similarity as ssim
    return ssim(img1, img2, data_range=1.0, channel_axis=2 if img1.ndim == 3 else None)


# =============================================================================
# Génération du Rapport HTML
# =============================================================================

def generate_report(results, output_dir):
    """Générer le rapport HTML pour la section 2."""
    # Section des algorithmes
    algorithms = algorithm_box('A) Interpolation Bilinéaire',
        '<p>Moyenne des pixels voisins de la même couleur. Simple mais produit des artefacts aux contours.</p>')
    algorithms += algorithm_box('B) Malvar-He-Cutler (2004)',
        '<p>Interpolation corrigée par gradient. Réduit les artefacts de couleur aux contours. <strong>À IMPLÉMENTER</strong></p>')
    algorithms += algorithm_box('C) Dématriçage Appris',
        '<p>Modèle entraîné sur des fenêtres locales. Pour cycles supérieurs. <strong>À IMPLÉMENTER</strong></p>')

    content = section('Algorithmes implémentés', algorithms, icon='📘')

    for result in results:
        basename = result['basename']
        metrics = result.get('metrics', {})

        # Tableau des métriques
        rows = [
            ['Bilinéaire', f"{metrics.get('bilinear', {}).get('time', 0):.3f}", '-', '-'],
        ]
        if 'malvar' in metrics:
            rows.append(['Malvar-He-Cutler', f"{metrics['malvar'].get('time', 0):.3f}",
                        f"{metrics['malvar'].get('psnr_vs_bilinear', 0):.2f}",
                        f"{metrics['malvar'].get('ssim_vs_bilinear', 0):.4f}"])

        section_content = subsection('Comparaison côte à côte',
            figure(f"{basename}_comparison.png", "Comparaison des méthodes de dématriçage"))

        section_content += subsection('Métriques de qualité',
            table(['Méthode', 'Temps (s)', 'PSNR vs Bilinéaire', 'SSIM vs Bilinéaire'], rows))

        section_content += subsection('Zoom sur les artefacts',
            figure(f"{basename}_zoom.png", "Recadrages montrant les artefacts de contour"))

        content += section(basename, section_content)

    html = html_document('TP1 - Section 2', 'Dématriçage (Demosaicing)',
                         '🎨', content, accent_color='#778da9')

    save_report(html, os.path.join(output_dir, 'rapport_section2.html'))


# =============================================================================
# Traitement Principal
# =============================================================================

def process_mosaic_files(input_dir='images_intermediaires_sec1', output_dir='images_intermediaires_sec2',
                         enable_malvar=True, enable_learned=False):
    """Traiter tous les fichiers TIFF mosaïques et appliquer le dématriçage."""
    os.makedirs(output_dir, exist_ok=True)

    tiff_files = [f for f in sorted(glob.glob(os.path.join(input_dir, '*.tiff')))
                  if 'zoom' not in f]

    if not tiff_files:
        print(f"Aucun fichier TIFF trouvé dans {input_dir}/")
        return

    print(f"\n{'#'*60}")
    print("# Section 2: Dématriçage")
    print(f"{'#'*60}")
    print(f"\n{len(tiff_files)} fichier(s) TIFF trouvé(s)")

    results = []

    for tiff_path in tiff_files:
        basename = os.path.splitext(os.path.basename(tiff_path))[0]
        json_path = os.path.join(input_dir, f"{basename}.json")

        if not os.path.exists(json_path):
            print(f"  Ignoré {basename}: métadonnées non trouvées")
            continue

        print(f"\n{'='*60}")
        print(f"Traitement: {basename}")
        print('='*60)

        try:
            raw_data = load_tiff(tiff_path)
            metadata = load_metadata(json_path)
            pattern_2x2 = metadata['bayer_pattern_2x2']

            result = {'basename': basename, 'metrics': {}}

            # Bilinéaire (toujours exécuté)
            print("  [A] Dématriçage bilinéaire...")
            t0 = time.time()
            rgb_bilinear = demosaic_bilinear(raw_data, pattern_2x2)
            t_bilinear = time.time() - t0
            save_tiff16(rgb_bilinear, os.path.join(output_dir, f"{basename}_bilinear.tiff"))
            save_jpeg(rgb_bilinear, os.path.join(output_dir, f"{basename}_bilinear.jpg"))
            result['metrics']['bilinear'] = {'time': t_bilinear}

            # Malvar-He-Cutler
            rgb_malvar = None
            if enable_malvar:
                print("  [B] Dématriçage Malvar-He-Cutler...")
                t0 = time.time()
                rgb_malvar = demosaic_malvar(raw_data, pattern_2x2)
                t_malvar = time.time() - t0
                save_tiff16(rgb_malvar, os.path.join(output_dir, f"{basename}_malvar.tiff"))
                save_jpeg(rgb_malvar, os.path.join(output_dir, f"{basename}_malvar.jpg"))
                result['metrics']['malvar'] = {
                    'time': t_malvar,
                    'psnr_vs_bilinear': compute_psnr(rgb_malvar, rgb_bilinear),
                    'ssim_vs_bilinear': compute_ssim(rgb_malvar, rgb_bilinear)
                }

            # Dématriçage appris (cycles supérieurs)
            rgb_learned = None
            if enable_learned:
                print("  [C] Dématriçage appris...")
                t0 = time.time()
                rgb_learned = demosaic_learned(raw_data, pattern_2x2)
                t_learned = time.time() - t0
                save_tiff16(rgb_learned, os.path.join(output_dir, f"{basename}_learned.tiff"))
                save_jpeg(rgb_learned, os.path.join(output_dir, f"{basename}_learned.jpg"))

            # Figure de comparaison
            print("  Création des figures de comparaison...")
            images = {'Bilinéaire': rgb_bilinear}
            if rgb_malvar is not None:
                images['Malvar-He-Cutler'] = rgb_malvar
            if rgb_learned is not None:
                images['Appris'] = rgb_learned

            create_demosaic_comparison_figure(images,
                os.path.join(output_dir, f"{basename}_comparison.png"),
                linear_to_srgb, title=f"Comparaison - {basename}")

            # Figure de zoom
            edge_pos = find_edge_region(rgb_bilinear)
            center_pos = (rgb_bilinear.shape[0] // 2, rgb_bilinear.shape[1] // 2)
            create_demosaic_zoom_figure(images, edge_pos, center_pos,
                os.path.join(output_dir, f"{basename}_zoom.png"),
                linear_to_srgb, title=f"Zoom - {basename}")

            results.append(result)

        except Exception as e:
            print(f"\nErreur lors du traitement de {tiff_path}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        generate_report(results, output_dir)

    print(f"\n{'='*60}")
    print(f"Terminé! {len(results)} image(s) traitée(s) → {output_dir}/")
    print('='*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TP1 Section 2: Dématriçage')
    parser.add_argument('--input-dir', '-i', default='images_intermediaires_sec1')
    parser.add_argument('--output-dir', '-o', default='images_intermediaires_sec2')
    parser.add_argument('--no-malvar', action='store_true', help='Désactiver Malvar-He-Cutler')
    parser.add_argument('--learned', action='store_true', help='Activer le dématriçage appris (cycles supérieurs)')

    args = parser.parse_args()
    process_mosaic_files(args.input_dir, args.output_dir, 
                          enable_malvar=not args.no_malvar,
                          enable_learned=args.learned)
