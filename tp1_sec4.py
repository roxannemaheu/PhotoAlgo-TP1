#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP1 Section 4: Mappage Tonal et Encodage d'Affichage

Ce script:
1. Charge les images XYZ depuis ./images_intermediaires_sec3/*_camera_xyz.tiff
2. Applique l'ajustement de luminosité (À IMPLÉMENTER - 99e percentile)
3. Applique le mappage tonal:
   - Linéaire (implémenté)
   - Reinhard (À IMPLÉMENTER)
   - Filmic (À IMPLÉMENTER)
   - [Cycles supérieurs] Local par filtre bilatéral (À IMPLÉMENTER)
4. Convertit XYZ vers sRGB linéaire (implémenté)
5. Applique l'OETF sRGB (implémenté)
6. Sauvegarde le JPEG final (implémenté)
7. Analyse les artefacts JPEG (À IMPLÉMENTER par l'étudiant)
8. Sauvegarde dans ./images_intermediaires_sec4/

Usage:
    python tp1_sec4.py --input-dir images_intermediaires_sec3 --output-dir images_intermediaires_sec4
"""

import numpy as np
import glob
import os
from PIL import Image

from tp1_io import (
    load_tiff, save_tiff16, linear_to_srgb, xyz_to_linear_srgb, quantize_to_8bit
)
from tp1_rapport import (
    html_document, section, subsection, figure, table,
    algorithm_box, formula_box, save_report,
    create_tonemapping_curves_figure, create_tonemapping_comparison_figure,
    create_oetf_comparison_figure, create_dynamic_range_figure
)


# =============================================================================
# Ajustement de Luminosité
# =============================================================================

def adjust_brightness(xyz_image, percentile=99):
    """
    Ajuster la luminosité de l'image en normalisant au percentile donné.
    
    Mesure le percentile spécifié du canal Y (luminance) et divise
    toute l'image par cette valeur pour normaliser la luminosité.
    
    Args:
        xyz_image: Image XYZ [H, W, 3]
        percentile: Percentile à utiliser pour la normalisation (défaut: 99)
    
    Returns:
        Image XYZ avec luminosité ajustée
    
    TODO: Implémenter l'ajustement de luminosité
    
    Indices:
    1. Extraire le canal Y (luminance): Y = xyz_image[:, :, 1]
    2. Filtrer les valeurs valides (Y > 0)
    3. Calculer le percentile spécifié des valeurs valides
    4. Diviser toute l'image par cette valeur
    5. Retourner l'image ajustée
    """
    # =========================================================================
    # TODO: Implémenter l'ajustement de luminosité par le 99e percentile
    # =========================================================================
    
    print("    [ATTENTION] Ajustement de luminosité non implémenté")
    return xyz_image.copy()


# =============================================================================
# Opérateurs de Mappage Tonal
# =============================================================================

def tonemap_linear(xyz_image):
    """
    Mappage tonal linéaire (identité) - pas de compression.
    
    Les valeurs > 1 seront clippées lors de la conversion finale.
    
    Args:
        xyz_image: Image XYZ [H, W, 3]
    
    Returns:
        Image XYZ (copie)
    """
    return xyz_image.copy()


def tonemap_reinhard(xyz_image):
    """
    Mappage tonal de Reinhard: L_out = L_in / (1 + L_in)
    
    Appliqué à Y (luminance), X et Z sont mis à l'échelle proportionnellement.
    
    Référence: "Photographic Tone Reproduction for Digital Images" (2002)
    
    Args:
        xyz_image: Image XYZ [H, W, 3]
    
    Returns:
        Image XYZ avec mappage tonal appliqué
    
    TODO: Implémenter l'opérateur de Reinhard
    
    Indices:
    1. Extraire le canal Y (luminance): Y = xyz_image[:, :, 1]
    2. Appliquer la formule: Y_mapped = Y / (1 + Y)
    3. Calculer le ratio: scale = Y_mapped / Y (attention aux divisions par zéro!)
    4. Appliquer ce ratio à X et Z également
    5. Retourner l'image résultante
    """
    # =========================================================================
    # TODO: Implémenter le mappage tonal de Reinhard
    # =========================================================================
    
    print("    [ATTENTION] Reinhard non implémenté, utilisation de linéaire")
    return tonemap_linear(xyz_image)


def tonemap_filmic(xyz_image, A=0.22, B=0.3, C=0.1, D=0.2, E=0.01, F=0.3,
                   exposure=2.0, white_point=11.2):
    """
    Mappage tonal Filmic (style Uncharted 2).
    
    Courbe en S avec toe (ombres) et shoulder (hautes lumières).
    
    La fonction de transfert est:
        f(x) = ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F
    
    Args:
        xyz_image: Image XYZ [H, W, 3]
        A, B, C, D, E, F: Paramètres de la courbe
        exposure: Multiplicateur d'exposition
        white_point: Point blanc pour normalisation
    
    Returns:
        Image XYZ avec mappage tonal appliqué
    
    TODO: Implémenter l'opérateur Filmic
    
    Indices:
    1. Définir la fonction curve(x) selon la formule ci-dessus
    2. Appliquer à Y * exposure
    3. Normaliser par curve(white_point)
    4. Appliquer le ratio à X et Z
    """
    # =========================================================================
    # TODO: Implémenter le mappage tonal Filmic
    # =========================================================================
    
    print("    [ATTENTION] Filmic non implémenté, utilisation de linéaire")
    return tonemap_linear(xyz_image)


def tonemap_bilateral(xyz_image, sigma_spatial=16, sigma_range=0.1, compression=0.5):
    """
    Mappage tonal local par filtre bilatéral.
    
    [CYCLES SUPÉRIEURS UNIQUEMENT]
    
    Décompose l'image en couche de base (basse fréquence) et couche de détail
    (haute fréquence). Compresse la couche de base tout en préservant les détails.
    
    Références:
    - Durand & Dorsey (2002)
    - Paris & Durand (2006)
    
    Args:
        xyz_image: Image XYZ [H, W, 3]
        sigma_spatial: Écart-type spatial du filtre
        sigma_range: Écart-type de plage du filtre
        compression: Facteur de compression pour la couche de base
    
    Returns:
        Image XYZ avec mappage tonal local appliqué
    
    TODO: Implémenter le mappage tonal local (cycles supérieurs)
    
    Indices:
    1. Convertir Y en log: log_Y = log(Y + epsilon)
    2. Appliquer un filtre bilatéral pour obtenir la couche de base
    3. Soustraire pour obtenir la couche de détail: detail = log_Y - base
    4. Compresser la couche de base: base_compressed = base * compression
    5. Recombiner: log_Y_new = base_compressed + detail
    6. Reconvertir: Y_new = exp(log_Y_new)
    """
    # =========================================================================
    # TODO: Implémenter le mappage tonal local (cycles supérieurs)
    # =========================================================================
    
    print("    [ATTENTION] Mappage tonal local non implémenté")
    return tonemap_linear(xyz_image)


# =============================================================================
# Sauvegarde d'Images
# =============================================================================

def save_jpeg(img_8bit, filepath, quality=95):
    """
    Sauvegarder une image en JPEG.
    
    Args:
        img_8bit: Image uint8 [H, W, 3]
        filepath: Chemin de sortie
        quality: Qualité JPEG (1-100, défaut: 95)
    """
    Image.fromarray(img_8bit, mode='RGB').save(filepath, 'JPEG', quality=quality)
    print(f"  Saved JPEG: {filepath}")


def save_png(img_8bit, filepath):
    """
    Sauvegarder une image en PNG (sans perte).
    
    Args:
        img_8bit: Image uint8 [H, W, 3]
        filepath: Chemin de sortie
    """
    Image.fromarray(img_8bit, mode='RGB').save(filepath, 'PNG')
    print(f"  Saved PNG: {filepath}")


# =============================================================================
# Analyse de Plage Dynamique
# =============================================================================

def analyze_dynamic_range(image_linear):
    """Analyser l'écrêtage des hautes lumières et l'écrasement des ombres."""
    lum = 0.2126 * image_linear[:,:,0] + 0.7152 * image_linear[:,:,1] + 0.0722 * image_linear[:,:,2]
    
    highlight_pct = np.sum(lum >= 0.99) / lum.size * 100
    shadow_pct = np.sum(lum <= 0.01) / lum.size * 100
    
    valid = lum[lum > 0]
    if len(valid) > 0:
        min_lum, max_lum = np.percentile(valid, 1), np.percentile(valid, 99)
        dr_stops = np.log2(max_lum / min_lum) if min_lum > 0 else 0
    else:
        dr_stops = 0
    
    return {
        'highlight_clipped_percent': highlight_pct,
        'shadow_crushed_percent': shadow_pct,
        'dynamic_range_stops': dr_stops,
        'min_luminance': float(np.min(lum)),
        'max_luminance': float(np.max(lum)),
        'mean_luminance': float(np.mean(lum))
    }


# =============================================================================
# Génération du Rapport HTML
# =============================================================================

def generate_report(results, output_dir):
    """Générer le rapport HTML pour la section 4."""
    algorithms = algorithm_box('A) Ajustement de luminosité',
        '<p>Division par le 99e percentile. <strong>À IMPLÉMENTER</strong></p>')
    algorithms += algorithm_box('B) Mappage tonal',
        '<p><b>Linéaire:</b> Pas de compression. <strong>IMPLÉMENTÉ</strong></p>'
        '<p><b>Reinhard:</b> <code>L_out = L_in / (1 + L_in)</code>. <strong>À IMPLÉMENTER</strong></p>'
        '<p><b>Filmic:</b> Courbe en S (toe + shoulder). <strong>À IMPLÉMENTER</strong></p>')
    
    algorithms += algorithm_box('C) Conversion XYZ → sRGB',
        '<p>Matrice standard D65 suivie de l\'OETF sRGB. <strong>IMPLÉMENTÉ</strong></p>')
    
    algorithms += algorithm_box('D) OETF sRGB',
        formula_box('sRGB = 1.055 × linéaire^(1/2.4) − 0.055') + '<p><strong>IMPLÉMENTÉ</strong></p>')
    
    algorithms += algorithm_box('E) Analyse des artefacts JPEG',
        '<p>Sauvegarde en différentes qualités et analyse des artefacts. <strong>À IMPLÉMENTER PAR L\'ÉTUDIANT</strong></p>')
    
    content = section('Concepts', algorithms, icon='📘')
    content += section('Courbes de mappage tonal',
        figure('tonemapping_curves.png', 'Comparaison des courbes de réponse'), icon='📈')
    
    for result in results:
        basename = result['basename']
        dr = result.get('dynamic_range', {})
        
        section_content = subsection('Comparaison des opérateurs',
            figure(f"{basename}_tonemapping_comparison.png",
                   "Comparaison: Linéaire, Reinhard, Filmic"))
        
        section_content += subsection('Avant/Après OETF',
            figure(f"{basename}_oetf_comparison.png",
                   "L'OETF encode les valeurs linéaires pour l'affichage"))
        
        section_content += subsection('Image finale',
            figure(f"{basename}_final.jpg",
                   "Image JPEG finale (qualité 95)"))
        
        section_content += subsection('Plage dynamique',
            figure(f"{basename}_dynamic_range.png", "Analyse des hautes lumières et ombres") +
            table(['Métrique', 'Valeur'], [
                ['Plage dynamique', f"{dr.get('dynamic_range_stops', 0):.1f} stops"],
                ['Hautes lumières écrêtées', f"{dr.get('highlight_clipped_percent', 0):.2f}%"],
                ['Ombres écrasées', f"{dr.get('shadow_crushed_percent', 0):.2f}%"]
            ]))
        
        content += section(basename, section_content)
    
    html = html_document('TP1 - Section 4', 'Mappage tonal et encodage d\'affichage',
                         '🎨', content, accent_color='#778da9')
    
    save_report(html, os.path.join(output_dir, 'rapport_section4.html'))


# =============================================================================
# Traitement Principal
# =============================================================================

def process_display_encoding(input_dir='images_intermediaires_sec3', 
                              output_dir='images_intermediaires_sec4',
                              input_suffix='_camera_xyz.tiff'):
    """Traiter les images XYZ avec mappage tonal et encodage d'affichage."""
    os.makedirs(output_dir, exist_ok=True)
    
    tiff_files = sorted(glob.glob(os.path.join(input_dir, f'*{input_suffix}')))
    
    if not tiff_files:
        print(f"Aucun fichier *{input_suffix} trouvé dans {input_dir}/")
        return
    
    print(f"\n{'#'*60}")
    print("# Section 4: Mappage Tonal et Encodage d'Affichage")
    print(f"{'#'*60}")
    print(f"\n{len(tiff_files)} fichier(s) trouvé(s)")
    
    # Générer la figure des courbes une seule fois
    create_tonemapping_curves_figure(os.path.join(output_dir, 'tonemapping_curves.png'))
    
    results = []
    
    for tiff_path in tiff_files:
        basename = os.path.basename(tiff_path).replace(input_suffix, '')
        
        print(f"\n{'='*60}")
        print(f"Traitement: {basename}")
        print('='*60)
        
        try:
            xyz_image = load_tiff(tiff_path)
            result = {'basename': basename}
            
            # Ajustement de luminosité (à implémenter par l'étudiant)
            print("  [0] Ajustement de luminosité...")
            xyz_image = adjust_brightness(xyz_image, percentile=99)
            
            # Comparaison des opérateurs de mappage tonal
            print("  [A] Comparaison du mappage tonal...")
            tonemap_funcs = {
                'Linéaire': tonemap_linear,
                'Reinhard': tonemap_reinhard,
                'Filmic': tonemap_filmic
            }
            srgb_results = create_tonemapping_comparison_figure(xyz_image,
                os.path.join(output_dir, f"{basename}_tonemapping_comparison.png"),
                tonemap_funcs, xyz_to_linear_srgb, linear_to_srgb,
                title=f"Mappage tonal - {basename}")
            
            # Utiliser linéaire pour la suite (ou Reinhard si implémenté)
            xyz_tonemapped = tonemap_linear(xyz_image)
            rgb_linear = xyz_to_linear_srgb(xyz_tonemapped)
            rgb_linear = np.clip(rgb_linear, 0, 1)
            srgb = linear_to_srgb(rgb_linear)
            
            # Sauvegarder les résultats
            for name, img in srgb_results.items():
                save_tiff16(img, os.path.join(output_dir, f"{basename}_{name.lower()}.tiff"))
            
            # Comparaison OETF
            print("  [B] Comparaison OETF...")
            create_oetf_comparison_figure(rgb_linear, srgb,
                os.path.join(output_dir, f"{basename}_oetf_comparison.png"),
                title=f"OETF sRGB - {basename}")
            
            # Sauvegarder l'image finale en JPEG
            print("  [C] Sauvegarde de l'image finale...")
            img_8bit = quantize_to_8bit(srgb)
            
            final_jpg = os.path.join(output_dir, f"{basename}_final.jpg")
            save_jpeg(img_8bit, final_jpg, quality=95)
            
            # TODO: L'étudiant doit implémenter l'analyse des artefacts JPEG
            # - Sauvegarder en différentes qualités (95, 75, 50, 25)
            # - Comparer avec PNG (sans perte)
            # - Visualiser les artefacts de compression
            # - Créer un graphique taille vs qualité
            print("  [!] Analyse JPEG à implémenter par l'étudiant")
            
            # Analyse de plage dynamique
            print("  [D] Analyse de plage dynamique...")
            dr_analysis = analyze_dynamic_range(rgb_linear)
            result['dynamic_range'] = dr_analysis
            print(f"    Plage dynamique: {dr_analysis['dynamic_range_stops']:.1f} stops")
            
            create_dynamic_range_figure(rgb_linear, srgb, dr_analysis,
                os.path.join(output_dir, f"{basename}_dynamic_range.png"),
                title=f"Plage dynamique - {basename}")
            
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
    
    parser = argparse.ArgumentParser(description='TP1 Section 4: Mappage Tonal et Encodage')
    parser.add_argument('--input-dir', '-i', default='images_intermediaires_sec3')
    parser.add_argument('--output-dir', '-o', default='images_intermediaires_sec4')
    parser.add_argument('--suffix', '-s', default='_camera_xyz.tiff')
    
    args = parser.parse_args()
    process_display_encoding(args.input_dir, args.output_dir, args.suffix)

