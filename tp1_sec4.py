#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP1 Section 4: Mappage Tonal et Encodage d'Affichage

Ce script:
1. Charge les images XYZ depuis ./images_intermediaires_sec3/*_camera_xyz.tiff
2. Applique l'ajustement de luminosité (À IMPLÉMENTER)
3. Applique le mappage tonal:
   - Linéaire (implémenté)
   - Reinhard (À IMPLÉMENTER)
4. Convertit XYZ vers sRGB linéaire (implémenté)
5. Applique l'OETF sRGB (implémenté)
6. Sauvegarde le JPEG final (implémenté)
7. Analyse les artefacts JPEG (À IMPLÉMENTER)
8. Sauvegarde dans ./images_intermediaires_sec4/

Usage:
    python tp1_sec4.py --input-dir images_intermediaires_sec3 --output-dir images_intermediaires_sec4
"""

import glob
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from tp1_io import (
    load_tiff,
    save_tiff16,
    linear_to_srgb,
    xyz_to_linear_srgb,
    quantize_to_8bit,
)
from tp1_rapport import (
    html_document,
    section,
    subsection,
    figure,
    table,
    algorithm_box,
    formula_box,
    save_report,
    comparison_grid,
    create_tonemapping_curves_figure,
    create_tonemapping_comparison_figure,
    create_oetf_comparison_figure,
    create_dynamic_range_figure, find_edge_region,
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
    """
    Y = xyz_image[:, :, 1]

    # Compute the percentile of luminance (excluding zeros/negatives)
    valid_Y = Y[Y > 0]
    if len(valid_Y) == 0:
        print("    Warning: No valid luminance values, skipping brightness adjustment")
        return xyz_image.copy()

    percentile_value = np.percentile(valid_Y, percentile)

    if percentile_value <= 0:
        print("    Warning: Percentile value <= 0, skipping brightness adjustment")
        return xyz_image.copy()

    # Divide the entire image by the percentile value
    adjusted = xyz_image / percentile_value

    print(f"    Brightness adjustment: divided by {percentile_value:.6f} (1st percentile)")

    return adjusted


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
    """

    # Copier l'image pour ne pas modifier l'entrée
    result = xyz_image.copy()

    # 1. Extraire le canal Y (luminance)
    Y = xyz_image[:, :, 1]

    # 2. Appliquer la formule de Reinhard sur Y
    Y_mapped = Y / (1.0 + Y)

    # 3. Calculer le facteur d'échelle (éviter division par zéro)
    scale = np.ones_like(Y)
    mask = Y > 0
    scale[mask] = Y_mapped[mask] / Y[mask]

    # 4. Appliquer ce ratio à X, Y et Z
    result[:, :, 0] *= scale  # X
    result[:, :, 1] = Y_mapped  # Y (remplacé directement)
    result[:, :, 2] *= scale  # Z

    return result


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
    Image.fromarray(img_8bit, mode="RGB").save(filepath, "JPEG", quality=quality)
    print(f"  Saved JPEG: {filepath}")


def save_png(img_8bit, filepath):
    """
    Sauvegarder une image en PNG (sans perte).

    Args:
        img_8bit: Image uint8 [H, W, 3]
        filepath: Chemin de sortie
    """
    Image.fromarray(img_8bit, mode="RGB").save(filepath, "PNG")
    print(f"  Saved PNG: {filepath}")


# =============================================================================
# Analyse des artefacts JPEG
# =============================================================================

def analyze_jpeg_artifacts(img_8bit, output_dir, basename, qualities=(95, 75, 50, 25)):
    """
    Analyse des artefacts JPEG pour une image linéaire sRGB.

    Args:
        img_8bit: Image sRGB uint8 [H, W, 3], 0-255
        output_dir: Répertoire où sauvegarder les JPEG/PNG
        basename: Nom de base pour les fichiers
        qualities: Liste de qualités JPEG à tester

    Returns:
        dict: informations sur la taille des fichiers et différences avec PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder PNG (sans perte)
    png_path = os.path.join(output_dir, f"{basename}_reference.png")
    save_png(img_8bit, png_path)

    sizes = {}
    mse_values = {}

    # Convertir img_8bit en float [0-1] pour le calcul du MSE
    img_float = img_8bit.astype(np.float32) / 255.0

    for q in qualities:
        jpeg_path = os.path.join(output_dir, f"{basename}_q{q}.jpg")
        save_jpeg(img_8bit, jpeg_path, quality=q)

        # Taille du fichier en Ko
        sizes[q] = os.path.getsize(jpeg_path) / 1024.0

        # Comparer avec PNG (quantitativement)
        jpeg_img = np.array(Image.open(jpeg_path), dtype=np.float32) / 255.0
        mse = np.mean((img_float - jpeg_img) ** 2)
        mse_values[q] = mse

    # Générer un graphique taille vs qualité
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    ax1.plot(list(sizes.keys()), list(sizes.values()), 'o-', color='tab:blue', label="Taille JPEG (Ko)")
    ax2.plot(list(mse_values.keys()), list(mse_values.values()), 's--', color='tab:red', label="MSE vs PNG")

    ax1.set_xlabel("Qualité JPEG")
    ax1.set_ylabel("Taille du fichier (Ko)", color='tab:blue')
    ax2.set_ylabel("MSE vs PNG", color='tab:red')
    ax1.set_xticks(qualities)
    ax1.set_title(f"Analyse des artefacts JPEG - {basename}")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{basename}_jpeg_analysis.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"  Analyse JPEG terminée → graphique sauvegardé: {plot_path}")

    return {
        "jpeg_sizes_Ko": sizes,
        "mse_vs_png": mse_values,
        "png_path": png_path,
        "plot_path": plot_path,
    }


def plot_global_jpeg_size_vs_quality(
        global_jpeg_sizes,
        global_png_sizes,
        output_path,
        title="Taille JPEG vs Qualité (moyenne sur toutes les images)",
):
    """
    Trace un graphique global taille JPEG vs qualité (moyenne sur toutes les images).

    Args:
        global_jpeg_sizes: dict {quality: [sizes_in_Ko]}
        output_path: chemin de sauvegarde du graphique
        title: titre optionnel
    """
    if not global_jpeg_sizes:
        print("  Aucune donnée JPEG globale à tracer.")
        return

    qualities = sorted(global_jpeg_sizes.keys())
    mean_sizes = [np.mean(global_jpeg_sizes[q]) for q in qualities]
    std_sizes = [np.std(global_jpeg_sizes[q]) for q in qualities]

    mean_png_size = np.mean(global_png_sizes)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        qualities,
        mean_sizes,
        "o-",
        color="tab:blue",
        label="Taille moyenne JPEG",
    )

    ax.fill_between(
        qualities,
        np.array(mean_sizes) - std_sizes,
        np.array(mean_sizes) + std_sizes,
        alpha=0.25,
        color="tab:blue",
        label="± écart-type",
    )

    # Ligne PNG de référence
    ax.axhline(
        mean_png_size,
        color="tab:green",
        linestyle="--",
        linewidth=2,
        label="PNG (sans perte, moyenne)",
    )

    ax.set_xlabel("Qualité JPEG")
    ax.set_ylabel("Taille du fichier (Ko)")
    ax.set_xticks(qualities)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"  Graphique global JPEG sauvegardé: {output_path}")


def create_jpeg_zoom_figure(images_dict, output_path, edge_pos=None, center_pos=None, zoom_size=150, title=""):
    """
    Crée une figure pour visualiser les artefacts JPEG avec 3 rangées :
    1) Image entière
    2) Région avec contours
    3) Centre de l'image

    Args:
        images_dict : dict {nom: chemin_fichier}
        output_path : chemin de sauvegarde
        edge_pos : (y, x) position centrale de la région avec contours (optionnel)
        center_pos : (y, x) position centrale de la région centrale (optionnel)
        zoom_size : taille des régions zoomées
        title : titre optionnel
    """
    num_images = len(images_dict)
    fig, axes = plt.subplots(3, num_images, figsize=(4 * num_images, 12))
    if num_images == 1:
        axes = np.expand_dims(axes, axis=1)  # pour que axes[i, j] fonctionne même avec 1 image

    def clean_axes(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for col, (name, path) in enumerate(images_dict.items()):
        img = np.array(Image.open(path))
        H, W = img.shape[:2]

        # 1) Image entière
        axes[0, col].imshow(img)
        axes[0, col].set_title(name, fontweight="bold")
        if col == 0:
            axes[0, col].set_ylabel("Image entière", fontweight="bold")
        clean_axes(axes[0, col])

        # 2) Région avec contours
        if edge_pos is not None:
            y, x = edge_pos
            y1, y2 = max(0, y - zoom_size // 2), min(H, y + zoom_size // 2)
            x1, x2 = max(0, x - zoom_size // 2), min(W, x + zoom_size // 2)
            axes[1, col].imshow(img[y1:y2, x1:x2], interpolation="nearest")
            if col == 0:
                axes[1, col].set_ylabel("Région avec contours", fontweight="bold")
            clean_axes(axes[1, col])

        # 3) Centre de l'image
        if center_pos is not None:
            y, x = center_pos
        else:
            y, x = H // 2, W // 2
        y1, y2 = max(0, y - zoom_size // 2), min(H, y + zoom_size // 2)
        x1, x2 = max(0, x - zoom_size // 2), min(W, x + zoom_size // 2)
        axes[2, col].imshow(img[y1:y2, x1:x2], interpolation="nearest")
        if col == 0:
            axes[2, col].set_ylabel("Centre de l'image", fontweight="bold")
        clean_axes(axes[2, col])

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    fig.subplots_adjust(left=0.06)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved JPEG zoom figure: {output_path}")


# =============================================================================
# Analyse de Plage Dynamique
# =============================================================================


def analyze_dynamic_range(image_linear):
    """Analyser l'écrêtage des hautes lumières et l'écrasement des ombres."""
    lum = (
            0.2126 * image_linear[:, :, 0]
            + 0.7152 * image_linear[:, :, 1]
            + 0.0722 * image_linear[:, :, 2]
    )

    highlight_pct = np.sum(lum >= 0.99) / lum.size * 100
    shadow_pct = np.sum(lum <= 0.01) / lum.size * 100

    valid = lum[lum > 0]
    if len(valid) > 0:
        min_lum, max_lum = np.percentile(valid, 1), np.percentile(valid, 99)
        dr_stops = np.log2(max_lum / min_lum) if min_lum > 0 else 0
    else:
        dr_stops = 0

    return {
        "highlight_clipped_percent": highlight_pct,
        "shadow_crushed_percent": shadow_pct,
        "dynamic_range_stops": dr_stops,
        "min_luminance": float(np.min(lum)),
        "max_luminance": float(np.max(lum)),
        "mean_luminance": float(np.mean(lum)),
    }


# =============================================================================
# Génération du Rapport HTML
# =============================================================================


def generate_report(results, output_dir):
    """
    Générer un rapport HTML template pour toutes les sections du TP1.
    
    Crée un rapport complet avec:
    - Section 1: Chargement et compréhension des données RAW
    - Section 2: Dématriçage (Demosaicking)
    - Section 3: Balance des Blancs (White Balance)
    - Section 4: Mappage tonal et encodage d'affichage
    
    Inclut toutes les figures générées et des espaces "À remplir" pour l'étudiant.
    """
    # Définir les répertoires de sortie pour chaque section
    # Si output_dir est "images_intermediaires_sec4", base_dir sera le répertoire parent
    if "images_intermediaires_sec" in os.path.basename(output_dir):
        base_dir = os.path.dirname(output_dir) or "."
    else:
        base_dir = output_dir

    sec1_dir = os.path.join(base_dir, "images_intermediaires_sec1")
    sec2_dir = os.path.join(base_dir, "images_intermediaires_sec2")
    sec3_dir = os.path.join(base_dir, "images_intermediaires_sec3")
    sec4_dir = output_dir

    # Obtenir la liste des basenames (noms de fichiers sans extension)
    basenames = [result["basename"] for result in results] if results else []

    # Si aucun résultat, chercher les fichiers dans les répertoires
    if not basenames:
        # Chercher dans sec1
        tiff_files = glob.glob(os.path.join(sec1_dir, "*.tiff"))
        basenames = [os.path.splitext(os.path.basename(f))[0] for f in tiff_files if "zoom" not in f]
        basenames = list(set(basenames))  # Dédupliquer

    # Limiter à 2 images d'exemple pour rendre le rapport plus court
    basenames = sorted(basenames)[:2]
    content = ""

    # =============================================================================
    # SECTION 1: Chargement et Compréhension des Données RAW
    # =============================================================================
    sec1_content = ""

    # Texte d'introduction pour la section 1
    sec1_content += subsection(
        "Introduction",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #4fc3f7;">'
        '<p style="color: #a0a0a0; font-style: italic;">À remplir: Décrivez ici votre compréhension du format RAW, '
        'du motif de Bayer, et de la normalisation des données brutes.</p>'
        '</div>'
    )

    for basename in basenames:
        sec1_img_content = ""

        # Figure: Zoom sur la mosaïque Bayer
        zoom_path = os.path.join(sec1_dir, f"{basename}_zoom16x16.png")
        if os.path.exists(zoom_path):
            sec1_img_content += subsection(
                f"Région 16×16 de la mosaïque - {basename}",
                figure(f"../images_intermediaires_sec1/{basename}_zoom16x16.png",
                       "Zoom sur une région 16×16 montrant les valeurs normalisées et le motif de Bayer coloré.")
            )

        if sec1_img_content:
            sec1_content += section(f"Image: {basename}", sec1_img_content)

    # Analyse et observations
    sec1_content += subsection(
        "Analyse et observations",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #4fc3f7;">'
        '<p style="color: #a0a0a0; font-style: italic;">À remplir: Décrivez vos observations sur les métadonnées extraites, '
        'le motif de Bayer, et la normalisation.</p>'
        '</div>'
    )

    content += section("Section 1: Chargement et Compréhension des Données RAW", sec1_content, icon="📷")

    # =============================================================================
    # SECTION 2: Dématriçage (Demosaicking)
    # =============================================================================
    sec2_content = ""

    # Texte d'introduction pour la section 2
    sec2_content += subsection(
        "Introduction",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #778da9;">'
        '<p style="color: #a0a0a0; font-style: italic;">À remplir: Expliquez le processus de dématriçage, '
        'les différences entre les méthodes bilinéaire et Malvar-He-Cutler, et les artefacts observés.</p>'
        '</div>'
    )

    for basename in basenames:
        sec2_img_content = ""

        # Figure: Comparaison des méthodes
        comp_path = os.path.join(sec2_dir, f"{basename}_comparison.png")
        if os.path.exists(comp_path):
            sec2_img_content += subsection(
                f"Comparaison des méthodes - {basename}",
                figure(f"../images_intermediaires_sec2/{basename}_comparison.png",
                       "Comparaison des méthodes de dématriçage")
            )

        # Figure: Zoom sur les artefacts
        zoom_path = os.path.join(sec2_dir, f"{basename}_zoom.png")
        if os.path.exists(zoom_path):
            sec2_img_content += subsection(
                f"Zoom sur les artefacts - {basename}",
                figure(f"../images_intermediaires_sec2/{basename}_zoom.png",
                       "Recadrages montrant les artefacts de contour")
            )

        if sec2_img_content:
            sec2_content += section(f"Image: {basename}", sec2_img_content)

    # Analyse et observations
    sec2_content += subsection(
        "Analyse et observations",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #778da9;">'
        '<p style="color: #a0a0a0; font-style: italic;">À remplir: Comparez les résultats des différentes méthodes de dématriçage. '
        'Discutez des métriques de qualité (PSNR, SSIM) et des temps d\'exécution. Identifiez les régions où les artefacts sont les plus visibles.</p>'
        '</div>'
    )

    content += section("Section 2: Dématriçage (Demosaicking)", sec2_content, icon="🎨")

    # =============================================================================
    # SECTION 3: Balance des Blancs (White Balance)
    # =============================================================================
    sec3_content = ""

    # Texte d'introduction pour la section 3
    sec3_content += subsection(
        "Introduction",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #e94560;">'
        '<p style="color: #a0a0a0; font-style: italic;">À remplir: Expliquez le concept de balance des blancs, '
        'les différents algorithmes implémentés (région neutre, Grey World, caméra), et leurs avantages/inconvénients.</p>'
        '</div>'
    )

    for basename in basenames:
        sec3_img_content = ""

        # Figure: Comparaison des méthodes
        comp_path = os.path.join(sec3_dir, f"{basename}_comparison.png")
        if os.path.exists(comp_path):
            sec3_img_content += subsection(
                f"Comparaison des méthodes - {basename}",
                figure(f"../images_intermediaires_sec3/{basename}_comparison.png",
                       "Comparaison des méthodes de balance des blancs")
            )

        # Figure: Conversion XYZ
        xyz_path = os.path.join(sec3_dir, f"{basename}_xyz_comparison.png")
        if os.path.exists(xyz_path):
            sec3_img_content += subsection(
                f"Conversion XYZ - {basename}",
                figure(f"../images_intermediaires_sec3/{basename}_xyz_comparison.png",
                       "Images converties en XYZ puis reconverties en sRGB")
            )

        if sec3_img_content:
            sec3_content += section(f"Image: {basename}", sec3_img_content)

    # Analyse et observations
    sec3_content += subsection(
        "Analyse et observations",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #e94560;">'
        '<p style="color: #a0a0a0; font-style: italic;">À remplir: Comparez les résultats des différentes méthodes de balance des blancs. '
        'Discutez des multiplicateurs calculés et de leur impact visuel. Expliquez la conversion vers l\'espace XYZ.</p>'
        '</div>'
    )

    content += section("Section 3: Balance des Blancs (White Balance)", sec3_content, icon="⚪")

    # =============================================================================
    # SECTION 4: Mappage Tonal et Encodage d'Affichage
    # =============================================================================
    sec4_content = ""

    # Texte d'introduction pour la section 4
    sec4_content += subsection(
        "Introduction",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #778da9;">'
        '<p style="color: #a0a0a0; font-style: italic;">À remplir: Expliquez le besoin du mappage tonal, '
        'les différents opérateurs (linéaire, Reinhard), l\'OETF sRGB, et l\'analyse de la plage dynamique.</p>'
        '</div>'
    )

    # Concepts et algorithmes
    algorithms = algorithm_box(
        "A) Ajustement de luminosité",
        "<p>Division par le 99e percentile. <strong>À IMPLÉMENTER</strong></p>",
    )
    algorithms += algorithm_box(
        "B) Mappage tonal",
        "<p><b>Linéaire:</b> Pas de compression.</p>"
        "<p><b>Reinhard:</b> <code>L_out = L_in / (1 + L_in)</code>. <strong>À IMPLÉMENTER</strong></p>",
    )
    algorithms += algorithm_box(
        "C) Conversion XYZ → sRGB",
        "<p>Matrice standard D65 suivie de l'OETF sRGB. <strong>IMPLÉMENTÉ</strong></p>",
    )
    algorithms += algorithm_box(
        "D) OETF sRGB",
        formula_box("sRGB = 1.055 × linéaire^(1/2.4) − 0.055")
        + "<p><strong>IMPLÉMENTÉ</strong></p>",
    )
    algorithms += algorithm_box(
        "E) Analyse des artefacts JPEG",
        "<p>Sauvegarde en différentes qualités et analyse des artefacts. <strong>À IMPLÉMENTER PAR L'ÉTUDIANT</strong></p>",
    )

    sec4_content += subsection("Concepts et algorithmes", algorithms)

    # Figure: Courbes de mappage tonal
    curves_path = os.path.join(sec4_dir, "tonemapping_curves.png")
    if os.path.exists(curves_path):
        sec4_content += subsection(
            "Courbes de mappage tonal",
            figure("tonemapping_curves.png", "Comparaison des courbes de réponse")
        )

    # Figures pour chaque image
    # Utiliser results si disponible, sinon utiliser basenames
    # Filtrer pour ne garder que les 2 images sélectionnées
    if results:
        images_to_process = [r for r in results if r["basename"] in basenames]
    else:
        images_to_process = [{"basename": bn} for bn in basenames]

    for result in images_to_process:
        basename = result["basename"]
        dr = result.get("dynamic_range", {})

        sec4_img_content = ""

        # Figure: Comparaison des opérateurs
        comp_path = os.path.join(sec4_dir, f"{basename}_tonemapping_comparison.png")
        if os.path.exists(comp_path):
            sec4_img_content += subsection(
                "Comparaison des opérateurs",
                figure(
                    f"{basename}_tonemapping_comparison.png",
                    "Comparaison: Linéaire, Reinhard",
                ),
            )

        # Figure: Avant/Après OETF
        oetf_path = os.path.join(sec4_dir, f"{basename}_oetf_comparison.png")
        if os.path.exists(oetf_path):
            sec4_img_content += subsection(
                "Avant/Après OETF",
                figure(
                    f"{basename}_oetf_comparison.png",
                    "L'OETF encode les valeurs linéaires pour l'affichage",
                ),
            )

        # Figure: Image finale
        final_path = os.path.join(sec4_dir, f"{basename}_final.jpg")
        if os.path.exists(final_path):
            sec4_img_content += subsection(
                "Image finale",
                figure(f"{basename}_final.jpg", "Image JPEG finale (qualité 95)"),
            )

        # Figure: Plage dynamique
        dr_path = os.path.join(sec4_dir, f"{basename}_dynamic_range.png")
        if os.path.exists(dr_path):
            dr_table = ""
            if dr:
                dr_table = table(
                    ["Métrique", "Valeur"],
                    [
                        [
                            "Plage dynamique",
                            f"{dr.get('dynamic_range_stops', 0):.1f} stops",
                        ],
                        [
                            "Hautes lumières écrêtées",
                            f"{dr.get('highlight_clipped_percent', 0):.2f}%",
                        ],
                        ["Ombres écrasées", f"{dr.get('shadow_crushed_percent', 0):.2f}%"],
                    ],
                )
            sec4_img_content += subsection(
                "Plage dynamique",
                figure(
                    f"{basename}_dynamic_range.png", "Analyse des hautes lumières et ombres"
                ) + dr_table,
            )

        if sec4_img_content:
            sec4_content += section(basename, sec4_img_content)

    # Analyse et observations
    sec4_content += subsection(
        "Analyse et observations",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #778da9;">'
        '<p style="color: #a0a0a0; font-style: italic;">À remplir: Comparez les résultats des différents opérateurs de mappage tonal. '
        'Discutez de l\'impact de l\'OETF sur l\'apparence de l\'image. Analysez la plage dynamique et les zones écrêtées/écrasées. '
        'Discutez des artefacts JPEG à différentes qualités.</p>'
        '</div>'
    )

    content += section("Section 4: Mappage Tonal et Encodage d'Affichage", sec4_content, icon="🎨")

    # =============================================================================
    # GRILLE DE COMPARAISON DES IMAGES FINALES
    # =============================================================================
    # Collecter toutes les images finales JPG de la section 4 et leurs références
    comparisons = []
    jpg_files = sorted(glob.glob(os.path.join(sec4_dir, "*_final.jpg")))

    for jpg_path in jpg_files:
        basename = os.path.basename(jpg_path).replace("_final.jpg", "")
        final_src = os.path.basename(jpg_path)

        # Chercher l'image de référence correspondante
        reference_src = None
        srgb_path = os.path.join(sec1_dir, f"{basename}_srgb.jpg")
        if os.path.exists(srgb_path):
            reference_src = f"../images_intermediaires_sec1/{basename}_srgb.jpg"

        if reference_src:
            comparisons.append({
                "basename": basename,
                "final_src": final_src,
                "reference_src": reference_src,
                "final_alt": f"Image finale - {basename}",
                "reference_alt": f"Référence sRGB - {basename}"
            })
        else:
            # Si pas de référence, ajouter quand même l'image finale seule
            comparisons.append({
                "basename": basename,
                "final_src": final_src,
                "reference_src": final_src,  # Dupliquer pour l'affichage
                "final_alt": f"Image finale - {basename}",
                "reference_alt": f"Image finale - {basename}"
            })

    if comparisons:
        grid_content = subsection(
            "Comparaison: Vos résultats vs Références sRGB",
            '<p style="color: #a0a0a0; margin-bottom: 20px;">Comparez vos images finales avec les aperçus sRGB générés par rawpy. Cliquez sur une image pour l\'agrandir.</p>'
        )
        grid_content += comparison_grid(comparisons)
        content += section("Comparaison des Images Finales", grid_content, icon="🖼️")

    # =============================================================================
    # CONCLUSION GÉNÉRALE
    # =============================================================================
    conclusion_content = subsection(
        "Conclusion",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ffd54f;">'
        '<p style="color: #a0a0a0; font-style: italic;">À remplir: Faites une synthèse de votre travail sur les quatre sections. '
        'Discutez des défis rencontrés, des apprentissages, et des améliorations possibles. '
        'Comparez vos résultats avec les images de référence.</p>'
        '</div>'
    )

    content += section("Conclusion", conclusion_content, icon="📝")

    # Générer le document HTML final
    html = html_document(
        "Rapport TP1 - Roxanne Maheu",
        "",
        "📸",
        content,
        accent_color="#778da9",
    )

    save_report(html, os.path.join(output_dir, "rapport_complet.html"))


# =============================================================================
# Traitement Principal
# =============================================================================


def process_display_encoding(
        input_dir="images_intermediaires_sec3",
        output_dir="images_intermediaires_sec4",
        input_suffix="_camera_xyz.tiff",
):
    """Traiter les images XYZ avec mappage tonal et encodage d'affichage."""
    os.makedirs(output_dir, exist_ok=True)

    tiff_files = sorted(glob.glob(os.path.join(input_dir, f"*{input_suffix}")))

    if not tiff_files:
        print(f"Aucun fichier *{input_suffix} trouvé dans {input_dir}/")
        return

    print(f"\n{'#' * 60}")
    print("# Section 4: Mappage Tonal et Encodage d'Affichage")
    print(f"{'#' * 60}")
    print(f"\n{len(tiff_files)} fichier(s) trouvé(s)")

    # Générer la figure des courbes une seule fois
    create_tonemapping_curves_figure(os.path.join(output_dir, "tonemapping_curves.png"))

    results = []

    global_jpeg_sizes = defaultdict(list)
    global_png_sizes = []

    for tiff_path in tiff_files:
        basename = os.path.basename(tiff_path).replace(input_suffix, "")

        print(f"\n{'=' * 60}")
        print(f"Traitement: {basename}")
        print("=" * 60)

        try:
            xyz_image = load_tiff(tiff_path)
            result = {"basename": basename}

            # Ajustement de luminosité (à implémenter par l'étudiant)
            print("  [0] Ajustement de luminosité...")
            xyz_image = adjust_brightness(xyz_image, percentile=99)

            # Comparaison des opérateurs de mappage tonal
            print("  [A] Comparaison du mappage tonal...")
            tonemap_funcs = {
                "Linéaire": tonemap_linear,
                "Reinhard": tonemap_reinhard,
            }
            srgb_results = create_tonemapping_comparison_figure(
                xyz_image,
                os.path.join(output_dir, f"{basename}_tonemapping_comparison.png"),
                tonemap_funcs,
                xyz_to_linear_srgb,
                linear_to_srgb,
                title=f"Mappage tonal - {basename}",
            )

            # Utiliser Reinhard pour la suite
            xyz_tonemapped = tonemap_reinhard(xyz_image)
            rgb_linear = xyz_to_linear_srgb(xyz_tonemapped)
            rgb_linear = np.clip(rgb_linear, 0, 1)
            srgb = linear_to_srgb(rgb_linear)

            # Sauvegarder les résultats
            for name, img in srgb_results.items():
                save_tiff16(
                    img, os.path.join(output_dir, f"{basename}_{name.lower()}.tiff")
                )

            # Comparaison OETF
            print("  [B] Comparaison OETF...")
            create_oetf_comparison_figure(
                rgb_linear,
                srgb,
                os.path.join(output_dir, f"{basename}_oetf_comparison.png"),
                title=f"OETF sRGB - {basename}",
            )

            # Sauvegarder l'image finale en JPEG
            print("  [C] Sauvegarde de l'image finale...")
            img_8bit = quantize_to_8bit(srgb)

            final_jpg = os.path.join(output_dir, f"{basename}_final.jpg")
            save_jpeg(img_8bit, final_jpg, quality=95)

            # Analyse des artefacts JPEG vs le png de référence
            print("  [D] Analyse des artefacts JPEG...")
            jpeg_analysis = analyze_jpeg_artifacts(img_8bit, output_dir, basename)

            for q, size in jpeg_analysis["jpeg_sizes_Ko"].items():
                global_jpeg_sizes[q].append(size)
            png_size = os.path.getsize(jpeg_analysis["png_path"]) / 1024.0
            global_png_sizes.append(png_size)

            zoom_images = {"PNG": jpeg_analysis["png_path"]}
            zoom_images.update({
                f"JPEG q{q}": os.path.join(output_dir, f"{basename}_q{q}.jpg")
                for q in jpeg_analysis["jpeg_sizes_Ko"].keys()
            })

            edge_pos = find_edge_region(img_8bit)
            center_pos = (img_8bit.shape[0] // 2, img_8bit.shape[1] // 2)

            create_jpeg_zoom_figure(
                zoom_images,
                os.path.join(output_dir, f"{basename}_jpeg_zoom.png"),
                edge_pos=edge_pos,
                center_pos=center_pos,
                zoom_size=150,
                title=f"Zoom sur artefacts JPEG - {basename}"
            )

            result["jpeg_analysis"] = jpeg_analysis

            # Analyse de plage dynamique
            print("  [E] Analyse de plage dynamique...")
            dr_analysis = analyze_dynamic_range(rgb_linear)
            result["dynamic_range"] = dr_analysis
            print(
                f"    Plage dynamique: {dr_analysis['dynamic_range_stops']:.1f} stops"
            )

            create_dynamic_range_figure(
                rgb_linear,
                srgb,
                dr_analysis,
                os.path.join(output_dir, f"{basename}_dynamic_range.png"),
                title=f"Plage dynamique - {basename}",
            )

            results.append(result)

        except Exception as e:
            print(f"\nErreur lors du traitement de {tiff_path}: {e}")
            import traceback

            traceback.print_exc()

    if global_jpeg_sizes:
        plot_global_jpeg_size_vs_quality(
            global_jpeg_sizes,
            global_png_sizes,
            os.path.join(output_dir, "jpeg_size_vs_quality_global_mean.png"),
        )

    if results:
        generate_report(results, output_dir)

    print(f"\n{'=' * 60}")
    print(f"Terminé! {len(results)} image(s) traitée(s) → {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TP1 Section 4: Mappage Tonal et Encodage"
    )
    parser.add_argument("--input-dir", "-i", default="images_intermediaires_sec3")
    parser.add_argument("--output-dir", "-o", default="images_intermediaires_sec4")
    parser.add_argument("--suffix", "-s", default="_camera_xyz.tiff")

    args = parser.parse_args()
    process_display_encoding(args.input_dir, args.output_dir, args.suffix)
