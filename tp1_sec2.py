#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP1 Section 2: Dématriçage (Demosaicking)

Ce script:
1. Charge les mosaïques TIFF normalisées et les métadonnées depuis ./images_intermediaires_sec1/
2. Applique le dématriçage par interpolation bilinéaire (À IMPLÉMENTER)
3. Applique le dématriçage Malvar-He-Cutler 2004 (À IMPLÉMENTER)
5. Génère des images de comparaison
6. Sauvegarde dans ./images_intermediaires_sec2/

Usage:
    python tp1_sec2.py --input-dir images_intermediaires_sec1 --output-dir images_intermediaires_sec2
"""

import glob
import os
import time

import numpy as np
from scipy.signal import convolve2d

from tp1_io import load_tiff, load_metadata, save_tiff16, save_jpeg, linear_to_srgb
from tp1_rapport import (
    html_document,
    section,
    subsection,
    figure,
    table,
    algorithm_box,
    save_report,
    create_demosaic_comparison_figure,
    find_edge_region,
    create_demosaic_zoom_figure,
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
    positions = {"R": [], "G": [], "B": []}
    for i in range(2):
        for j in range(2):
            positions[pattern_2x2[i][j]].append((i, j))

    masks = {}
    for color in "RGB":
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

    kernel_g = np.array(
        [[0, 0.25, 0],
         [0.25, 1, 0.25],
         [0, 0.25, 0]], dtype=np.float32
    )

    kernel_rb = np.array(
        [[0.25, 0.5, 0.25],
         [0.5, 1, 0.5],
         [0.25, 0.5, 0.25]], dtype=np.float32
    )

    for c, color in enumerate(["R", "G", "B"]):
        mask = masks[color].astype(np.float32)
        channel = raw_data * mask

        if color in ["R", "B"]:
            kernel = kernel_rb
        else:
            kernel = kernel_g

        num = convolve2d(channel, kernel, mode="same", boundary="symm")
        den = convolve2d(mask, kernel, mode="same", boundary="symm")
        interp = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        rgb[..., c] = interp

    return np.clip(rgb, 0.0, 1.0)


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
    """

    H, W = raw_data.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    masks, positions = get_color_masks(pattern_2x2, H, W)

    R_mask = masks["R"]
    G_mask = masks["G"]
    B_mask = masks["B"]

    # --- Noyaux Malvar (Fig. 2) ---
    kernel_g_at_rb = np.array([
        [0, 0, -1, 0, 0],
        [0, 0, 2, 0, 0],
        [-1, 2, 4, 2, -1],
        [0, 0, 2, 0, 0],
        [0, 0, -1, 0, 0]
    ], dtype=np.float32) / 8

    kernel_rb_at_g_same_row = np.array([
        [0, 0, 0.5, 0, 0],
        [0, -1, 0, -1, 0],
        [-1, 4, 5, 4, -1],
        [0, -1, 0, -1, 0],
        [0, 0, 0.5, 0, 0],
    ], dtype=np.float32) / 8.0

    kernel_rb_at_g_same_col = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, 4, -1, 0],
        [0.5, 0, 5, 0, 0.5],
        [0, -1, 4, -1, 0],
        [0, 0, -1, 0, 0],
    ], dtype=np.float32) / 8.0

    kernel_rb_at_opposite = np.array([
        [0, 0, -1.5, 0, 0],
        [0, 2, 0, 2, 0],
        [-1.5, 0, 6, 0, -1.5],
        [0, 2, 0, 2, 0],
        [0, 0, -1.5, 0, 0],
    ], dtype=np.float32) / 8.0

    # Copier valeurs connues
    rgb[..., 0][R_mask] = raw_data[R_mask]  # Rouge connu
    rgb[..., 1][G_mask] = raw_data[G_mask]  # Vert connu
    rgb[..., 2][B_mask] = raw_data[B_mask]  # Bleu connu

    # --- Interpolation du V aux positions R et B ---
    G_interp = convolve2d(raw_data, kernel_g_at_rb, mode="same", boundary="symm")
    rgb[..., 1][R_mask | B_mask] = G_interp[R_mask | B_mask]

    # --- Interpolation R et B aux positions V ---
    # On regarde dans quelle rangée on est (R ou B)
    is_G_on_R_row = G_mask & (np.roll(R_mask, shift=1, axis=1) | np.roll(R_mask, shift=-1, axis=1))
    is_G_on_B_row = G_mask & (np.roll(B_mask, shift=1, axis=1) | np.roll(B_mask, shift=-1, axis=1))

    # Convolutions pré-calculées
    R_same_row = convolve2d(raw_data, kernel_rb_at_g_same_row, mode="same", boundary="symm")
    R_same_col = convolve2d(raw_data, kernel_rb_at_g_same_col, mode="same", boundary="symm")

    B_same_row = R_same_row.copy()
    B_same_col = R_same_col.copy()

    # R aux positions G
    rgb[..., 0][is_G_on_R_row] = R_same_row[is_G_on_R_row]
    rgb[..., 0][is_G_on_B_row] = R_same_col[is_G_on_B_row]

    # B aux positions G
    rgb[..., 2][is_G_on_B_row] = B_same_row[is_G_on_B_row]
    rgb[..., 2][is_G_on_R_row] = B_same_col[is_G_on_R_row]

    # --- Interpolation R aux positions B et de B aux positons R ---

    RB_opposite = convolve2d(raw_data, kernel_rb_at_opposite, mode="same", boundary="symm")

    rgb[..., 0][B_mask] = RB_opposite[B_mask]  # R aux pixels B
    rgb[..., 2][R_mask] = RB_opposite[R_mask]  # B aux pixels R

    # Clamp pour sécurité numérique
    np.clip(rgb, 0, 1, out=rgb)

    return rgb


# =============================================================================
# Métriques de Qualité
# =============================================================================


def compute_psnr(img1, img2):
    """Calculer le PSNR entre deux images."""
    mse = np.mean((img1 - img2) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


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
    algorithms = algorithm_box(
        "A) Interpolation Bilinéaire",
        "<p>Moyenne des pixels voisins de la même couleur. Simple mais produit des artefacts aux contours.</p>",
    )
    algorithms += algorithm_box(
        "B) Malvar-He-Cutler (2004)",
        "<p>Interpolation corrigée par gradient. Réduit les artefacts de couleur aux contours. <strong>À IMPLÉMENTER</strong></p>",
    )
    algorithms += algorithm_box(
        "C) Dématriçage Appris",
        "<p>Modèle entraîné sur des fenêtres locales. Pour cycles supérieurs. <strong>À IMPLÉMENTER</strong></p>",
    )

    content = section("Algorithmes implémentés", algorithms, icon="📘")

    for result in results:
        basename = result["basename"]
        metrics = result.get("metrics", {})

        # Tableau des métriques
        rows = [
            [
                "Bilinéaire",
                f"{metrics.get('bilinear', {}).get('time', 0):.3f}",
                "-",
                "-",
            ],
        ]
        if "malvar" in metrics:
            rows.append(
                [
                    "Malvar-He-Cutler",
                    f"{metrics['malvar'].get('time', 0):.3f}",
                    f"{metrics['malvar'].get('psnr_vs_bilinear', 0):.2f}",
                    f"{metrics['malvar'].get('ssim_vs_bilinear', 0):.4f}",
                ]
            )

        section_content = subsection(
            "Comparaison côte à côte",
            figure(
                f"{basename}_comparison.png", "Comparaison des méthodes de dématriçage"
            ),
        )

        section_content += subsection(
            "Métriques de qualité",
            table(
                ["Méthode", "Temps (s)", "PSNR vs Bilinéaire", "SSIM vs Bilinéaire"],
                rows,
            ),
        )

        section_content += subsection(
            "Zoom sur les artefacts",
            figure(
                f"{basename}_zoom.png", "Recadrages montrant les artefacts de contour"
            ),
        )

        content += section(basename, section_content)

    # Discussion
    discussion_content = subsection(
        "Interprétation visuelle des images obtenues",
        """
        <p>
        Il y a peu de différences entre les résultats obtenus avec chacune des méthodes. 
        Ces minces différences ne sont perceptibles qu'en zoomant sur les zones avec de forts contrastes.
        On voit alors que l'extrapolation bilinéaire donne un résultat avec des contours plus adoucis, 
        alors que les contours obtenus avec Malvar sont plus définis.
        </p>
    
        <p>
        Peu importe la méthode utilisée, on voit parfois apparaitre des pixels de couleur (effet de Moiré) 
        à des endroits de haute luminosité (couleur blanche).
        L'interpolation bilinéaire permet de "lisser" ces pixels de couleur, et donc les atténue, 
        comparativement à la méthode Malvac-He-Cutler.
        </p>  
        """
    )

    discussion_content += subsection(
        "Interprétation des métriques: Temps",
        """
        <p>
        L'interpolation bilinéaire est toujours plus rapide que la méthode Malvar-He-Cutler, 
        probablement en raison des kernels de convolution, qui sont plus gros.
        Donc la faible amélioration de la qualité se fait au détriment de la vitesse.
        </p>  
        """
    )

    discussion_content += subsection(
        "Interprétation des métriques: PSNR",
        """        
        <p>
        Le PSNR est une métrique qui informe sur la différence de valeur pixel par pixel entre deux images. Il s'exprime en décibels (dB). 
        Plus la valeur est élevée, plus l'image traitée est proche de l'originale. 
        </p>
                
        <p>
        Selon la littérature, pour des données 8 bits, les valeurs de PSNR oscillent généralement entre 30 et 50 dB. 
        Pour des données 16 bits, les valeurs de PSNR oscillent généralement entre 60 et 80 dB. 
        Nos résultats vont de 40.82 dB à 57.08 dB, pour des images majoritairement de 12 et 14 bits, et une seule image à 16 bits (pelican).
        </p>
        
        <p> 
        Dans notre cas, comme le PSNR se calcule par rapport à l'image avec interpolation bilinéaire, mon interprétion de la métrique
        est que plus elle est faible, plus la différence entre les deux algorithmes est marquée (on ne peut que comparer des images ayant le même nombre de bits). La PSNR la plus élevée est pour pelican, ce qui est logique puisque J'ai toutefois eu 
        du mal à voir une corrélation entre la valeur de la métrique (qui varie entre 40.82 et 57.08) et la similarité entre les résultats.
        </p>
        """
    )

    discussion_content += subsection(
        "Interprétation des métriques: SSIM",
        """
        <p>
        Le SSIM repose sur un indice de similarité structurelle entre deux images, en intégrant le contraste de l'image, les différences structurelles et la luminosité.
        Plus il est près de 1, plus deux images sont similaires. Il est moins sensible au Gaussian noise, mais plus sensible à la compression.
        Dans notre cas, les valeurs de SSIM (Structural Similarity Index) sont très près de 1, ce qui indique que la structure globable 
        de l'image est presque autant préservée 
        ce qui indique des gains faibles au niveau de la réduction des artefacts de couleur et la netteté des contours.
        </p>
        """
    )

    discussion_content += subsection(
        "Référence",
        """
        <p>
        Référence pour la compréhension des métriques PSNR et SSIM: 
        Sara, U. , Akter, M. and Uddin, M. (2019) Image Quality Assessment through FSIM, SSIM, MSE and PSNR—A Comparative Study. Journal of Computer and Communications, 7, 8-18. doi: 10.4236/jcc.2019.73002.
        Disponible à https://www.scirp.org/journal/paperinformation?paperid=90911.
        </p>
        """
    )

    content += section(
        "Discussion",
        discussion_content,
        icon="📝",
    )

    html = html_document(
        "TP1 - Section 2",
        "Dématriçage (Demosaicing)",
        "🎨",
        content,
        accent_color="#778da9",
    )

    save_report(html, os.path.join(output_dir, "rapport_section2.html"))


# =============================================================================
# Traitement Principal
# =============================================================================


def process_mosaic_files(
        input_dir="images_intermediaires_sec1",
        output_dir="images_intermediaires_sec2",
        enable_malvar=True,
        enable_learned=False,
):
    """Traiter tous les fichiers TIFF mosaïques et appliquer le dématriçage."""
    os.makedirs(output_dir, exist_ok=True)

    tiff_files = [
        f
        for f in sorted(glob.glob(os.path.join(input_dir, "*.tiff")))
        if "zoom" not in f
    ]

    if not tiff_files:
        print(f"Aucun fichier TIFF trouvé dans {input_dir}/")
        return

    print(f"\n{'#' * 60}")
    print("# Section 2: Dématriçage")
    print(f"{'#' * 60}")
    print(f"\n{len(tiff_files)} fichier(s) TIFF trouvé(s)")

    results = []

    for tiff_path in tiff_files:
        basename = os.path.splitext(os.path.basename(tiff_path))[0]
        json_path = os.path.join(input_dir, f"{basename}.json")

        if not os.path.exists(json_path):
            print(f"  Ignoré {basename}: métadonnées non trouvées")
            continue

        print(f"\n{'=' * 60}")
        print(f"Traitement: {basename}")
        print("=" * 60)

        try:
            raw_data = load_tiff(tiff_path)
            metadata = load_metadata(json_path)
            pattern_2x2 = metadata["bayer_pattern_2x2"]

            result = {"basename": basename, "metrics": {}}

            # Bilinéaire (toujours exécuté)
            print("  [A] Dématriçage bilinéaire...")
            t0 = time.time()
            rgb_bilinear = demosaic_bilinear(raw_data, pattern_2x2)
            t_bilinear = time.time() - t0
            save_tiff16(
                rgb_bilinear, os.path.join(output_dir, f"{basename}_bilinear.tiff")
            )
            save_jpeg(
                rgb_bilinear, os.path.join(output_dir, f"{basename}_bilinear.jpg")
            )
            result["metrics"]["bilinear"] = {"time": t_bilinear}

            # Malvar-He-Cutler
            rgb_malvar = None
            if enable_malvar:
                print("  [B] Dématriçage Malvar-He-Cutler...")
                t0 = time.time()
                rgb_malvar = demosaic_malvar(raw_data, pattern_2x2)
                t_malvar = time.time() - t0
                save_tiff16(
                    rgb_malvar, os.path.join(output_dir, f"{basename}_malvar.tiff")
                )
                save_jpeg(
                    rgb_malvar, os.path.join(output_dir, f"{basename}_malvar.jpg")
                )
                result["metrics"]["malvar"] = {
                    "time": t_malvar,
                    "psnr_vs_bilinear": compute_psnr(rgb_malvar, rgb_bilinear),
                    "ssim_vs_bilinear": compute_ssim(rgb_malvar, rgb_bilinear),
                }

            # Figure de comparaison
            print("  Création des figures de comparaison...")
            images = {"Bilinéaire": rgb_bilinear}
            if rgb_malvar is not None:
                images["Malvar-He-Cutler"] = rgb_malvar

            create_demosaic_comparison_figure(
                images,
                os.path.join(output_dir, f"{basename}_comparison.png"),
                linear_to_srgb,
                title=f"Comparaison - {basename}",
            )

            # Figure de zoom
            edge_pos = find_edge_region(rgb_bilinear)
            center_pos = (rgb_bilinear.shape[0] // 2, rgb_bilinear.shape[1] // 2)
            create_demosaic_zoom_figure(
                images,
                edge_pos,
                center_pos,
                os.path.join(output_dir, f"{basename}_zoom.png"),
                linear_to_srgb,
                title=f"Zoom - {basename}",
            )

            results.append(result)

        except Exception as e:
            print(f"\nErreur lors du traitement de {tiff_path}: {e}")
            import traceback

            traceback.print_exc()

    if results:
        generate_report(results, output_dir)

    print(f"\n{'=' * 60}")
    print(f"Terminé! {len(results)} image(s) traitée(s) → {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TP1 Section 2: Dématriçage")
    parser.add_argument("--input-dir", "-i", default="images_intermediaires_sec1")
    parser.add_argument("--output-dir", "-o", default="images_intermediaires_sec2")
    parser.add_argument(
        "--no-malvar", action="store_true", help="Désactiver Malvar-He-Cutler"
    )
    parser.add_argument(
        "--learned",
        action="store_true",
        help="Activer le dématriçage appris (cycles supérieurs)",
    )

    args = parser.parse_args()
    process_mosaic_files(
        args.input_dir,
        args.output_dir,
        enable_malvar=not args.no_malvar,
        enable_learned=args.learned,
    )
