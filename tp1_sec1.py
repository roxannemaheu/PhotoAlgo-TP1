#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP1 Section 1: Chargement et Compréhension des données brutes (RAW)

Ce script:
1. Charge les fichiers DNG
2. Extrait les métadonnées (motif de Bayer, profondeur de bits, niveaux noir/blanc, balance des blancs)
3. Normalise les données brutes à l'intervalle [0, 1]
4. Applique la rotation selon le tag EXIF d'orientation
5. Sauvegarde en TIFF 16 bits et les métadonnées en JSON
6. Génère un rapport HTML avec les figures et métadonnées

Usage:
    python tp1_sec1.py --input-dir tp1_data --output-dir images_intermediaires_sec1
"""

import numpy as np
import rawpy
import glob
import os
from PIL import Image

from tp1_io import save_tiff16, save_metadata
from tp1_rapport import (
    html_document, section, subsection, figure, table,
    metadata_grid, metadata_card, bayer_grid_html, wb_chips_html,
    matrix_html, formula_box, save_report,
    create_bayer_zoom_figure, find_interesting_region
)


# =============================================================================
# Rotation Bayer-Aware
# =============================================================================

def get_orientation(raw_img):
    """Obtenir l'information de rotation depuis l'attribut sizes.flip."""
    try:
        flip = raw_img.sizes.flip
        descriptions = {
            0: "Pas de rotation", 3: "Rotation 180°",
            5: "90° anti-horaire", 6: "90° horaire"
        }
        return flip, descriptions.get(flip, f"Inconnu: {flip}")
    except AttributeError:
        return 0, "Pas de rotation"


def rotate_bayer_image(image, flip_value, pattern_name):
    """
    Rotation de la mosaïque Bayer en préservant le motif.
    
    Chaque canal couleur 2x2 est pivoté indépendamment, puis réentrelacé
    aux mêmes positions pour que le motif de Bayer reste identique.
    """
    if flip_value == 0:
        return image, pattern_name

    H, W = image.shape
    H, W = (H // 2) * 2, (W // 2) * 2  # Assurer dimensions paires
    image = image[:H, :W]

    # Extraire 4 sous-images des positions Bayer 2x2
    channels = [image[i::2, j::2] for i in range(2) for j in range(2)]

    # Quantité de rotation: 3→180°(k=2), 5→90°CCW(k=1), 6→90°CW(k=3)
    k = {3: 2, 5: 1, 6: 3}.get(flip_value)
    if k is None:
        return image, pattern_name

    # Pivoter chaque canal et réentrelacer
    rotated = [np.rot90(ch, k=k) for ch in channels]
    new_h, new_w = rotated[0].shape

    output = np.zeros((new_h * 2, new_w * 2), dtype=image.dtype)
    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        output[i::2, j::2] = rotated[idx]

    return output, pattern_name


# =============================================================================
# Extraction des Métadonnées
# =============================================================================

def get_bayer_pattern(raw_img):
    """Obtenir le motif de Bayer (ex: 'RGGB') et tableau 2x2."""
    pattern = raw_img.raw_pattern[:2, :2]
    color_desc = raw_img.color_desc.decode('utf-8')

    pattern_2x2 = [[color_desc[pattern[i, j]] for j in range(2)] for i in range(2)]
    pattern_name = ''.join(pattern_2x2[i][j] for i in range(2) for j in range(2))

    return pattern_name, pattern_2x2


def extract_metadata(raw_img):
    """Extraire toutes les métadonnées pertinentes."""
    metadata = {}
    
    # Motif de Bayer
    pattern_name, pattern_2x2 = get_bayer_pattern(raw_img)
    metadata['bayer_pattern'] = pattern_name
    metadata['bayer_pattern_2x2'] = pattern_2x2
    
    # Niveaux de noir/blanc
    black_levels = list(raw_img.black_level_per_channel)
    metadata['black_level_per_channel'] = black_levels
    
    white_levels = raw_img.camera_white_level_per_channel
    if white_levels is None:
        white_levels = [raw_img.white_level] * 4
    metadata['white_level_per_channel'] = list(white_levels)
    
    # Inférer la profondeur de bits
    metadata['inferred_bit_depth'] = int(np.ceil(np.log2(max(white_levels) + 1)))
    
    # Balance des blancs et matrices de couleur
    metadata['camera_whitebalance'] = list(raw_img.camera_whitebalance)
    metadata['rgb_xyz_matrix'] = raw_img.rgb_xyz_matrix.tolist()
    
    try:
        metadata['color_matrix'] = raw_img.color_matrix.tolist()
    except AttributeError:
        metadata['color_matrix'] = None
    
    try:
        metadata['daylight_whitebalance'] = list(raw_img.daylight_whitebalance)
    except:
        metadata['daylight_whitebalance'] = None
    
    metadata['color_desc'] = raw_img.color_desc.decode('utf-8')
    
    # Orientation
    flip, description = get_orientation(raw_img)
    metadata['orientation_flip'] = flip
    metadata['orientation_description'] = description
    
    return metadata


def process_raw_file(filepath):
    """
    Traiter un fichier DNG: extraire métadonnées, normaliser, pivoter.
    
    Retourne:
        metadata: Dictionnaire avec toutes les métadonnées extraites
        normalized: Données d'image normalisées et pivotées [0, 1]
        srgb_preview: Aperçu sRGB généré par rawpy
    """
    print(f"\n{'='*60}")
    print(f"Traitement: {os.path.basename(filepath)}")
    print('='*60)
    
    with rawpy.imread(filepath) as raw_img:
        metadata = extract_metadata(raw_img)
        
        # Obtenir les données brutes
        raw_data = raw_img.raw_image_visible.copy()
        metadata['image_height'] = raw_data.shape[0]
        metadata['image_width'] = raw_data.shape[1]
        
        print(f"  Motif de Bayer: {metadata['bayer_pattern']}")
        print(f"  Taille: {metadata['image_width']} x {metadata['image_height']}")
        print(f"  Profondeur de bits: {metadata['inferred_bit_depth']} bits")
        
        # Normaliser à [0, 1]
        black_level = np.mean(metadata['black_level_per_channel'][:4])
        white_level = max(metadata['white_level_per_channel'])
        
        normalized = (raw_data.astype(np.float32) - black_level) / (white_level - black_level)
        normalized = np.clip(normalized, 0, 1)
        
        metadata['normalization'] = {
            'black_level_used': float(black_level),
            'white_level_used': float(white_level)
        }
        
        # Pivoter selon EXIF
        flip = metadata['orientation_flip']
        if flip != 0:
            normalized, _ = rotate_bayer_image(normalized, flip, metadata['bayer_pattern'])
            metadata['image_height_rotated'] = normalized.shape[0]
            metadata['image_width_rotated'] = normalized.shape[1]
            print(f"  Pivoté: {metadata['orientation_description']}")
        
        # Générer un aperçu sRGB avec rawpy
        srgb = raw_img.postprocess(
            gamma=(2.222, 4.5), no_auto_bright=False, 
            output_bps=8, use_camera_wb=True
        )
        
        return metadata, normalized, srgb


# =============================================================================
# Génération du Rapport HTML
# =============================================================================

def generate_report(results, output_dir):
    """Générer le rapport HTML pour la section 1."""
    content = ""
    
    for result in results:
        metadata = result['metadata']
        basename = result['basename']
        
        # Construire le contenu de la section
        cards = metadata_card('Motif de Bayer', metadata['bayer_pattern'],
                             bayer_grid_html(metadata['bayer_pattern_2x2']))
        cards += metadata_card('Profondeur de bits', f"{metadata['inferred_bit_depth']} bits")
        cards += metadata_card('Dimensions', f"{metadata['image_width']} × {metadata['image_height']} px")
        cards += metadata_card('Orientation (EXIF)', metadata['orientation_description'])
        
        section_content = subsection('Métadonnées extraites', metadata_grid(cards))
        
        # Tableau des niveaux noir/blanc
        bl = metadata['black_level_per_channel']
        wl = metadata['white_level_per_channel']
        section_content += subsection('Niveaux de noir et de blanc',
            table(['Paramètre', 'Canal 0', 'Canal 1', 'Canal 2', 'Canal 3'],
                  [['Niveau de noir'] + bl, ['Niveau de blanc'] + wl]))
        
        # Formule de normalisation
        section_content += subsection('Formule de normalisation',
            formula_box('normalized = (raw_value - black_level) / (white_level - black_level)'))
        
        # Balance des blancs
        wb = metadata.get('camera_whitebalance', [0, 0, 0, 0])
        section_content += subsection('Balance des blancs (caméra)', wb_chips_html(wb))
        
        # Matrices
        section_content += subsection('Matrice RGB-XYZ', matrix_html(metadata['rgb_xyz_matrix']))
        
        if metadata.get('color_matrix'):
            section_content += subsection('Matrice de couleur', matrix_html(metadata['color_matrix']))
        
        # Figure
        section_content += subsection('Région 16×16 de la mosaïque',
            figure(f"{basename}_zoom16x16.png",
                   "Zoom sur une région 16×16 montrant les valeurs normalisées et le motif de Bayer coloré."))
        
        content += section(basename, section_content)
    
    html = html_document('TP1 - Section 1', 
                         'Chargement et Compréhension des Données RAW',
                         '📷', content, accent_color='#4fc3f7')
    
    save_report(html, os.path.join(output_dir, 'rapport_section1.html'))


# =============================================================================
# Traitement Principal
# =============================================================================

def process_dng_files(input_dir='tp1_data', output_dir='images_intermediaires_sec1'):
    """Traiter tous les fichiers DNG dans le répertoire d'entrée."""
    os.makedirs(output_dir, exist_ok=True)
    
    dng_files = sorted(glob.glob(os.path.join(input_dir, '*.dng')))
    
    if not dng_files:
        print(f"Aucun fichier DNG trouvé dans {input_dir}/")
        return
    
    print(f"\n{'#'*60}")
    print("# Section 1: Chargement et Compréhension des Données RAW")
    print(f"{'#'*60}")
    print(f"\n{len(dng_files)} fichier(s) DNG trouvé(s)")
    
    results = []
    
    for filepath in dng_files:
        try:
            metadata, normalized, srgb_preview = process_raw_file(filepath)
            basename = os.path.splitext(os.path.basename(filepath))[0]
            
            # Sauvegarder les sorties
            save_tiff16(normalized, os.path.join(output_dir, f"{basename}.tiff"))
            save_metadata(metadata, os.path.join(output_dir, f"{basename}.json"))
            
            # Sauvegarder l'aperçu sRGB
            Image.fromarray(srgb_preview).save(
                os.path.join(output_dir, f"{basename}_srgb.jpg"), quality=95)
            
            # Créer la visualisation
            start_y, start_x = find_interesting_region(normalized)
            create_bayer_zoom_figure(normalized, metadata['bayer_pattern_2x2'],
                                     start_y, start_x,
                                     os.path.join(output_dir, f"{basename}_zoom16x16.png"),
                                     title=f"Image: {basename}")
            
            results.append({'basename': basename, 'metadata': metadata})
            
        except Exception as e:
            print(f"\nErreur lors du traitement de {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        generate_report(results, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Terminé! {len(results)} image(s) traitée(s) → {output_dir}/")
    print('='*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TP1 Section 1: Chargement des données RAW')
    parser.add_argument('--input-dir', '-i', default='tp1_data', help='Répertoire avec les fichiers DNG')
    parser.add_argument('--output-dir', '-o', default='images_intermediaires_sec1', help='Répertoire de sortie')

    args = parser.parse_args()
    process_dng_files(input_dir=args.input_dir, output_dir=args.output_dir)
