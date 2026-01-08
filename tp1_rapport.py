#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP1 Common HTML Report Generation

Shared utilities for generating HTML reports with consistent styling
and matplotlib figure generation.
"""

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# =============================================================================
# CSS Styles
# =============================================================================


def get_css_styles(accent_color="#4fc3f7"):
    """
    Get common CSS styles for reports.

    Args:
        accent_color: Primary accent color for headers/borders
    """
    return f"""
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&family=Fira+Code:wght@400;500&display=swap');
        
        * {{ box-sizing: border-box; }}
        
        body {{
            font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e8e8e8;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 20px;
        }}
        
        header {{
            text-align: center;
            padding: 40px 0;
            border-bottom: 2px solid rgba(255,255,255,0.1);
            margin-bottom: 40px;
        }}
        
        .student-name-field {{
            background: rgba(255,255,255,0.05);
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 8px;
            padding: 15px 25px;
            margin: 20px auto;
            max-width: 500px;
            text-align: left;
        }}
        
        .student-name-field label {{
            display: block;
            color: #a0a0a0;
            font-size: 0.9em;
            margin-bottom: 8px;
        }}
        
        .student-name-field input {{
            width: 100%;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 4px;
            padding: 10px 15px;
            color: #fff;
            font-size: 1em;
            font-family: 'Source Sans Pro', sans-serif;
        }}
        
        .student-name-field input:focus {{
            outline: none;
            border-color: {accent_color};
            box-shadow: 0 0 0 2px {accent_color}40;
        }}
        
        .student-name-field input::placeholder {{
            color: #666;
            font-style: italic;
        }}
        
        h1 {{
            font-size: 2.5em;
            font-weight: 700;
            color: #fff;
            margin: 0 0 15px 0;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        
        .subtitle {{
            font-size: 1.2em;
            color: #a0a0a0;
            margin: 0;
        }}
        
        .date-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.1);
            padding: 8px 20px;
            border-radius: 20px;
            margin-top: 20px;
            font-size: 0.9em;
            color: #b0b0b0;
        }}
        
        .image-section {{
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 40px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }}
        
        .image-section h2 {{
            color: {accent_color};
            font-size: 1.6em;
            margin: 0 0 25px 0;
            padding-bottom: 15px;
            border-bottom: 2px solid {accent_color}40;
        }}
        
        h3 {{
            color: #e0e1dd;
            font-size: 1.3em;
            margin: 30px 0 20px 0;
        }}
        
        h3::before {{
            content: '';
            display: inline-block;
            width: 4px;
            height: 24px;
            background: {accent_color};
            margin-right: 12px;
            border-radius: 2px;
            vertical-align: middle;
        }}
        
        .algorithm-box {{
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid {accent_color};
        }}
        
        .algorithm-box h4 {{
            color: {accent_color};
            margin: 0 0 10px 0;
        }}
        
        .algorithm-box p {{
            margin: 5px 0;
            line-height: 1.6;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .metrics-table th {{
            background: {accent_color}30;
            color: {accent_color};
            padding: 14px 16px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        .metrics-table td {{
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-family: 'Fira Code', monospace;
            font-size: 0.95em;
        }}
        
        .metrics-table tr:last-child td {{
            border-bottom: none;
        }}
        
        .metrics-table tr:hover {{
            background: rgba(255,255,255,0.03);
        }}
        
        .figure-container {{
            text-align: center;
            margin: 15px 0;
            padding: 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
        }}
        
        .figure-container img {{
            max-width: 60%;
            max-height: 400px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .figure-container img:hover {{
            transform: scale(1.02);
            box-shadow: 0 6px 30px rgba(0,0,0,0.5);
        }}
        
        .figure-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #a0a0a0;
            font-size: 0.9em;
        }}
        
        /* Lightbox/Modal styles */
        .lightbox {{
            display: none;
            position: fixed;
            z-index: 9999;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            animation: fadeIn 0.3s;
        }}
        
        .lightbox.active {{
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .lightbox-content {{
            max-width: 95vw;
            max-height: 95vh;
            margin: auto;
            animation: zoomIn 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            box-sizing: border-box;
        }}
        
        .lightbox-content img {{
            max-width: 100%;
            max-height: 95vh;
            width: auto;
            height: auto;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 8px 40px rgba(0,0,0,0.8);
        }}
        
        .lightbox-close {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: #fff;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.2s;
        }}
        
        .lightbox-close:hover {{
            color: #ffc107;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        @keyframes zoomIn {{
            from {{ transform: scale(0.8); opacity: 0; }}
            to {{ transform: scale(1); opacity: 1; }}
        }}
        
        /* Image grid styles */
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .image-grid-item {{
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            background: rgba(0,0,0,0.3);
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .image-grid-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.5);
        }}
        
        .image-grid-item img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        .image-grid-item .image-label {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, rgba(0,0,0,0.8), transparent);
            color: #fff;
            padding: 15px 10px 10px;
            font-size: 0.9em;
            text-align: center;
        }}
        
        /* Comparison grid styles */
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }}
        
        .comparison-pair {{
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .comparison-pair-title {{
            text-align: center;
            color: #fff;
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(255,255,255,0.2);
        }}
        
        .comparison-images {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}
        
        .comparison-image-item {{
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            background: rgba(0,0,0,0.2);
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .comparison-image-item:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.5);
        }}
        
        .comparison-image-item img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        .comparison-image-label {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, rgba(0,0,0,0.9), transparent);
            color: #fff;
            padding: 12px 8px 8px;
            font-size: 0.85em;
            text-align: center;
            font-weight: 500;
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metadata-card {{
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid {accent_color};
        }}
        
        .metadata-card h4 {{
            color: {accent_color};
            margin: 0 0 12px 0;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metadata-card .value {{
            font-family: 'Fira Code', monospace;
            font-size: 1.1em;
            color: #fff;
            word-break: break-all;
        }}
        
        .formula-box {{
            background: rgba(0,0,0,0.4);
            padding: 20px 25px;
            border-radius: 8px;
            font-family: 'Fira Code', monospace;
            font-size: 1.1em;
            text-align: center;
            margin: 20px 0;
            border: 1px solid rgba(255,255,255,0.1);
            color: #ffd54f;
        }}
        
        .matrix-display {{
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            font-family: 'Fira Code', monospace;
            font-size: 0.85em;
            overflow-x: auto;
            white-space: pre;
            color: #b0bec5;
        }}
        
        .bayer-grid {{
            display: grid;
            grid-template-columns: repeat(2, 40px);
            gap: 4px;
            margin-top: 10px;
        }}
        
        .bayer-cell {{
            width: 40px;
            height: 40px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2em;
            color: #fff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        }}
        
        .bayer-cell.R {{ background: linear-gradient(135deg, #f44336, #c62828); }}
        .bayer-cell.G {{ background: linear-gradient(135deg, #4caf50, #2e7d32); }}
        .bayer-cell.B {{ background: linear-gradient(135deg, #2196f3, #1565c0); }}
        
        .wb-values {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 10px;
        }}
        
        .wb-chip {{
            padding: 8px 16px;
            border-radius: 20px;
            font-family: 'Fira Code', monospace;
            font-weight: 500;
        }}
        
        .wb-chip.r {{ background: rgba(244, 67, 54, 0.3); color: #ef9a9a; }}
        .wb-chip.g {{ background: rgba(76, 175, 80, 0.3); color: #a5d6a7; }}
        .wb-chip.b {{ background: rgba(33, 150, 243, 0.3); color: #90caf9; }}
        
        code {{
            background: {accent_color}20;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Fira Code', monospace;
            font-size: 0.9em;
        }}
        
        footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9em;
        }}
    """


# =============================================================================
# HTML Document Structure
# =============================================================================


def html_document(title, subtitle, icon, content, accent_color="#4fc3f7"):
    """
    Create a complete HTML document with consistent styling.

    Args:
        title: Page title (h1)
        subtitle: Page subtitle
        icon: Emoji icon for title
        content: HTML content for body
        accent_color: Primary accent color

    Returns:
        Complete HTML string
    """
    date_str = datetime.now().strftime("%d %B %Y à %H:%M")
    css = get_css_styles(accent_color)

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{css}</style>
</head>
<body>
    <!-- Lightbox Modal -->
    <div id="lightbox" class="lightbox" onclick="closeLightbox(event)">
        <span class="lightbox-close">&times;</span>
        <div class="lightbox-content">
            <img id="lightbox-img" src="" alt="">
        </div>
    </div>
    
    <div class="container">
        <header>
            <h1>{icon} {title}</h1>
            <p class="subtitle">{subtitle}</p>
            <div class="date-badge">Généré le {date_str}</div>
        </header>
        
        {content}
        
        <footer>
            <p>GIF 4105/7105 &mdash; TP1 | Formation d'une image</p>
        </footer>
    </div>
    
    <script>
        function openLightbox(img) {{
            const lightbox = document.getElementById('lightbox');
            const lightboxImg = document.getElementById('lightbox-img');
            const fullSizeSrc = img.getAttribute('data-fullsize') || img.src;
            lightboxImg.src = fullSizeSrc;
            lightbox.classList.add('active');
            document.body.style.overflow = 'hidden';
        }}
        
        function closeLightbox(event) {{
            const lightbox = document.getElementById('lightbox');
            // Only close if clicking on the background or close button
            if (event.target === lightbox || event.target.classList.contains('lightbox-close')) {{
                lightbox.classList.remove('active');
                document.body.style.overflow = 'auto';
            }}
        }}
        
        // Close lightbox on Escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                const lightbox = document.getElementById('lightbox');
                lightbox.classList.remove('active');
                document.body.style.overflow = 'auto';
            }}
        }});
    </script>
</body>
</html>
"""


# =============================================================================
# Reusable HTML Components
# =============================================================================


def section(title, content, icon="📷"):
    """Create an image-section div with title."""
    return f"""
        <div class="image-section">
            <h2><span class="icon">{icon}</span> {title}</h2>
            {content}
        </div>
    """


def subsection(title, content=""):
    """Create an h3 subsection."""
    return f"<h3>{title}</h3>\n{content}"


def figure(img_src, caption="", alt="", clickable=True):
    """Create a figure container with image and caption. Images are clickable to view larger."""
    alt = alt or caption
    caption_html = f'<p class="figure-caption">{caption}</p>' if caption else ""
    click_class = "clickable-image" if clickable else ""
    return f"""
        <div class="figure-container">
            <img src="{img_src}" alt="{alt}" class="{click_class}" data-fullsize="{img_src}" onclick="openLightbox(this)">
            {caption_html}
        </div>
    """


def image_grid(images, title=""):
    """
    Create a grid of clickable images.
    
    Args:
        images: List of dicts with 'src' (image path), 'label' (caption), and optionally 'alt'
        title: Optional title for the grid section
    
    Returns:
        HTML string for the image grid
    """
    grid_items = ""
    for img_data in images:
        src = img_data.get("src", "")
        label = img_data.get("label", "")
        alt = img_data.get("alt", label)
        grid_items += f"""
            <div class="image-grid-item" onclick="openLightbox(this.querySelector('img'))">
                <img src="{src}" alt="{alt}" data-fullsize="{src}">
                <div class="image-label">{label}</div>
            </div>
        """
    
    title_html = f'<h3>{title}</h3>' if title else ""
    return f"""
        {title_html}
        <div class="image-grid">
            {grid_items}
        </div>
    """


def comparison_grid(comparisons, title=""):
    """
    Create a grid of comparison pairs (final image vs reference image).
    
    Args:
        comparisons: List of dicts with 'basename', 'final_src' (final image path), 
                     'reference_src' (reference image path), and optionally 'alt'
        title: Optional title for the grid section
    
    Returns:
        HTML string for the comparison grid
    """
    comparison_items = ""
    for comp_data in comparisons:
        basename = comp_data.get("basename", "")
        final_src = comp_data.get("final_src", "")
        reference_src = comp_data.get("reference_src", "")
        final_alt = comp_data.get("final_alt", f"Image finale - {basename}")
        reference_alt = comp_data.get("reference_alt", f"Référence sRGB - {basename}")
        
        comparison_items += f"""
            <div class="comparison-pair">
                <div class="comparison-pair-title">{basename}</div>
                <div class="comparison-images">
                    <div class="comparison-image-item" onclick="openLightbox(this.querySelector('img'))">
                        <img src="{final_src}" alt="{final_alt}" data-fullsize="{final_src}">
                        <div class="comparison-image-label">Votre résultat</div>
                    </div>
                    <div class="comparison-image-item" onclick="openLightbox(this.querySelector('img'))">
                        <img src="{reference_src}" alt="{reference_alt}" data-fullsize="{reference_src}">
                        <div class="comparison-image-label">Référence sRGB</div>
                    </div>
                </div>
            </div>
        """
    
    title_html = f'<h3>{title}</h3>' if title else ""
    return f"""
        {title_html}
        <div class="comparison-grid">
            {comparison_items}
        </div>
    """


def table(headers, rows):
    """
    Create a metrics table.

    Args:
        headers: List of column headers
        rows: List of row data (each row is a list of cell values)
    """
    header_html = "".join(f"<th>{h}</th>" for h in headers)

    rows_html = ""
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        rows_html += f"<tr>{cells}</tr>\n"

    return f"""
        <table class="metrics-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    """


def algorithm_box(title, description):
    """Create an algorithm description box."""
    return f"""
        <div class="algorithm-box">
            <h4>{title}</h4>
            {description}
        </div>
    """


def metadata_card(title, value, extra_html=""):
    """Create a metadata display card."""
    return f"""
        <div class="metadata-card">
            <h4>{title}</h4>
            <div class="value">{value}</div>
            {extra_html}
        </div>
    """


def metadata_grid(cards):
    """Wrap cards in a metadata grid."""
    return f'<div class="metadata-grid">{cards}</div>'


def bayer_grid_html(pattern_2x2):
    """Create HTML for Bayer pattern visualization."""
    html = '<div class="bayer-grid">'
    for row in pattern_2x2:
        for cell in row:
            html += f'<div class="bayer-cell {cell}">{cell}</div>'
    html += "</div>"
    return html


def wb_chips_html(wb_values):
    """Create HTML for white balance multiplier chips."""
    return f"""
        <div class="wb-values">
            <span class="wb-chip r">R: {wb_values[0]:.4f}</span>
            <span class="wb-chip g">G: {wb_values[1]:.4f}</span>
            <span class="wb-chip b">B: {wb_values[2]:.4f}</span>
        </div>
    """


def matrix_html(matrix):
    """Format a matrix for display."""
    lines = []
    for row in matrix:
        values = ", ".join(f"{v:8.5f}" for v in row)
        lines.append(f"  [{values}]")
    return '<div class="matrix-display">' + "\n".join(lines) + "</div>"


def formula_box(formula):
    """Create a formula display box."""
    return f'<div class="formula-box">{formula}</div>'


# =============================================================================
# Matplotlib Figure Generation - Section 1
# =============================================================================


def create_bayer_zoom_figure(
    raw_data, pattern_2x2, start_y, start_x, output_path, title=""
):
    """
    Create a figure showing a 16x16 region with Bayer pattern visualization.

    Args:
        raw_data: 2D normalized mosaic image
        pattern_2x2: 2x2 list of color characters (e.g., [['R','G'],['G','B']])
        start_y, start_x: Top-left corner of region
        output_path: Where to save the figure
        title: Optional figure title
    """
    region = raw_data[start_y : start_y + 16, start_x : start_x + 16]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Grayscale view
    axes[0].imshow(region, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(
        f"Mosaic brute (niveaux de gris)\n[{start_y}:{start_y+16}, {start_x}:{start_x+16}]"
    )
    for i in range(17):
        axes[0].axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.5)
        axes[0].axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.5)

    # Color-coded Bayer view
    color_map = {"R": [1, 0.2, 0.2], "G": [0.2, 0.8, 0.2], "B": [0.2, 0.2, 1]}
    rgb_display = np.zeros((16, 16, 3))

    for i in range(16):
        for j in range(16):
            color = pattern_2x2[(start_y + i) % 2][(start_x + j) % 2]
            rgb_display[i, j] = np.array(color_map[color]) * region[i, j]

    axes[1].imshow(rgb_display)
    pattern_str = "".join(pattern_2x2[i][j] for i in range(2) for j in range(2))
    axes[1].set_title(f"Motif de Bayer coloré\nPatron: {pattern_str}")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


def find_interesting_region(raw_data, size=16):
    """Find a region with high contrast for visualization."""
    H, W = raw_data.shape
    candidates = [
        (H // 4, W // 4),
        (H // 2, W // 2),
        (H // 4, W // 2),
        (H // 2, W // 4),
        (H // 3, W // 3),
    ]

    best_pos, best_std = (H // 4, W // 4), 0
    for y, x in candidates:
        y, x = (y // 2) * 2, (x // 2) * 2  # Align to Bayer pattern
        if y + size <= H and x + size <= W:
            std = np.std(raw_data[y : y + size, x : x + size])
            if std > best_std:
                best_std, best_pos = std, (y, x)

    return best_pos


# =============================================================================
# Matplotlib Figure Generation - Section 2
# =============================================================================


def create_demosaic_comparison_figure(
    images, output_path, linear_to_srgb_func, title=""
):
    """
    Create side-by-side comparison figure for demosaicing results.

    Args:
        images: Dict of {name: rgb_image} pairs
        output_path: Where to save the figure
        linear_to_srgb_func: Function to convert linear RGB to sRGB
        title: Optional figure title
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, images.items()):
        ax.imshow(linear_to_srgb_func(img))
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


def create_difference_figure(
    img1, img2, name1, name2, output_path, linear_to_srgb_func, title=""
):
    """
    Create difference visualization figure between two images.

    Args:
        img1, img2: RGB images to compare
        name1, name2: Names for the images
        output_path: Where to save the figure
        linear_to_srgb_func: Function to convert linear RGB to sRGB
        title: Optional figure title

    Returns:
        dict with 'mean' and 'max' difference statistics
    """
    diff = np.sum(np.abs(img1 - img2), axis=2)
    stats = {"mean": float(np.mean(diff)), "max": float(np.max(diff))}
    vmax = np.percentile(diff, 99) or 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(linear_to_srgb_func(img1))
    axes[0].set_title(name1, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(linear_to_srgb_func(img2))
    axes[1].set_title(name2, fontweight="bold")
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap="viridis", vmin=0, vmax=vmax)
    axes[2].set_title(f"Différence\nMoyenne: {stats['mean']:.4f}", fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")
    return stats


def find_edge_region(image, size=150):
    """Find a region with strong edges for visualization."""
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    gradient = np.sqrt(ndimage.sobel(gray, 1) ** 2 + ndimage.sobel(gray, 0) ** 2)

    H, W = gray.shape
    best_pos, best_score = (H // 2, W // 2), 0

    for y in range(size, H - size, size // 2):
        for x in range(size, W - size, size // 2):
            score = np.mean(gradient[y : y + size, x : x + size])
            if score > best_score:
                best_score, best_pos = score, (y + size // 2, x + size // 2)

    return best_pos


def create_demosaic_zoom_figure(
    images, edge_pos, center_pos, output_path, linear_to_srgb_func, size=150, title=""
):
    """
    Create zoomed comparison at edge and center regions.

    Args:
        images: Dict of {name: rgb_image} pairs
        edge_pos: (y, x) center of edge region
        center_pos: (y, x) center of image center region
        output_path: Where to save the figure
        linear_to_srgb_func: Function to convert linear RGB to sRGB
        size: Size of zoom region
        title: Optional figure title
    """
    method_names = list(images.keys())
    fig, axes = plt.subplots(2, len(method_names), figsize=(4 * len(method_names), 8))

    for col, name in enumerate(method_names):
        img = images[name]
        H, W = img.shape[:2]

        # Edge region
        y, x = edge_pos
        y1, y2 = max(0, y - size // 2), min(H, y + size // 2)
        x1, x2 = max(0, x - size // 2), min(W, x + size // 2)
        axes[0, col].imshow(
            linear_to_srgb_func(img[y1:y2, x1:x2]), interpolation="nearest"
        )
        if col == 0:
            axes[0, col].set_ylabel("Région avec contours", fontweight="bold")
        axes[0, col].set_title(name, fontweight="bold")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        # Center region
        y, x = center_pos
        y1, y2 = max(0, y - size // 2), min(H, y + size // 2)
        x1, x2 = max(0, x - size // 2), min(W, x + size // 2)
        axes[1, col].imshow(
            linear_to_srgb_func(img[y1:y2, x1:x2]), interpolation="nearest"
        )
        if col == 0:
            axes[1, col].set_ylabel("Centre de l'image", fontweight="bold")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


# =============================================================================
# Matplotlib Figure Generation - Section 3
# =============================================================================


def create_neutral_point_figure(
    image, click_pos, region_size, output_path, linear_to_srgb_func, title=""
):
    """
    Show selected neutral point on image for manual white balance.

    Args:
        image: RGB image
        click_pos: (y, x) position of neutral point
        region_size: Size of selection region
        output_path: Where to save the figure
        linear_to_srgb_func: Function to convert linear RGB to sRGB
        title: Optional figure title
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(linear_to_srgb_func(image))

    y, x = click_pos
    half = region_size // 2

    rect = plt.Rectangle(
        (x - half, y - half),
        region_size,
        region_size,
        fill=False,
        edgecolor="red",
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.axhline(y=y, color="yellow", linestyle="--", alpha=0.7, linewidth=1)
    ax.axvline(x=x, color="yellow", linestyle="--", alpha=0.7, linewidth=1)
    ax.plot(x, y, "r+", markersize=20, markeredgewidth=2)

    ax.set_title(f"{title}\nPoint neutre: ({x}, {y})", fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


def create_wb_comparison_figure(results, output_path, linear_to_srgb_func, title=""):
    """
    Side-by-side white balance comparison with multipliers.

    Args:
        results: Dict of {name: {'image': img, 'multipliers': (r,g,b) or None}}
        output_path: Where to save the figure
        linear_to_srgb_func: Function to convert linear RGB to sRGB
        title: Optional figure title
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        ax.imshow(linear_to_srgb_func(data["image"]))
        mult = data.get("multipliers")
        if mult:
            ax.set_title(
                f"{name}\nR={mult[0]:.2f}, G={mult[1]:.2f}, B={mult[2]:.2f}",
                fontsize=10,
                fontweight="bold",
            )
        else:
            ax.set_title(name, fontweight="bold")
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


def create_xyz_comparison_figure(
    results, output_path, linear_to_srgb_func, xyz_to_srgb_func, title=""
):
    """
    Show RGB and XYZ→sRGB side by side.

    Args:
        results: Dict of {name: {'rgb': rgb_img, 'xyz': xyz_img}}
        output_path: Where to save the figure
        linear_to_srgb_func: Function to convert linear RGB to sRGB
        xyz_to_srgb_func: Function to convert XYZ to sRGB
        title: Optional figure title
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))

    if n == 1:
        axes = axes.reshape(-1, 1)

    for col, (name, data) in enumerate(results.items()):
        axes[0, col].imshow(linear_to_srgb_func(data["rgb"]))
        axes[0, col].set_title(f"{name}\n(RGB après balance)", fontweight="bold")
        axes[0, col].axis("off")

        axes[1, col].imshow(xyz_to_srgb_func(data["xyz"]))
        axes[1, col].set_title(f"{name}\n(XYZ → sRGB)", fontweight="bold")
        axes[1, col].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


# =============================================================================
# Matplotlib Figure Generation - Section 4
# =============================================================================


def create_tonemapping_curves_figure(output_path):
    """Create figure showing tone mapping curves (Linear, Reinhard)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.linspace(0, 5, 500)

    # Linear (clipped)
    ax.plot(x, np.clip(x, 0, 1), "b-", lw=2, label="Linéaire (clippé)")

    # Reinhard
    ax.plot(x, x / (1 + x), "r-", lw=2, label="Reinhard: L/(1+L)")

    ax.axhline(y=1, color="gray", ls="--", alpha=0.5)
    ax.axvline(x=1, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Luminance entrée", fontsize=12)
    ax.set_ylabel("Luminance sortie", fontsize=12)
    ax.set_title("Courbes de mappage tonal", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


def create_tonemapping_comparison_figure(
    xyz_image,
    output_path,
    tonemap_funcs,
    xyz_to_linear_srgb_func,
    linear_to_srgb_func,
    title="",
):
    """
    Compare tone mapping operators.

    Args:
        xyz_image: Input XYZ image
        output_path: Where to save the figure
        tonemap_funcs: Dict of {name: tonemap_function}
        xyz_to_linear_srgb_func: Function to convert XYZ to linear sRGB
        linear_to_srgb_func: Function to convert linear RGB to sRGB
        title: Optional figure title

    Returns:
        Dict of {name: srgb_image} results
    """
    # Apply operators
    results = {name: func(xyz_image) for name, func in tonemap_funcs.items()}

    # Convert to sRGB for display
    srgb_results = {}
    for name, xyz in results.items():
        rgb_linear = xyz_to_linear_srgb_func(xyz)
        srgb_results[name] = linear_to_srgb_func(rgb_linear)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: images
    for col, (name, srgb) in enumerate(srgb_results.items()):
        axes[0, col].imshow(np.clip(srgb, 0, 1))
        axes[0, col].set_title(name, fontweight="bold")
        axes[0, col].axis("off")

    # Bottom row: histograms
    colors = ["steelblue", "coral", "seagreen"]
    for col, (name, srgb) in enumerate(srgb_results.items()):
        lum = 0.2126 * srgb[:, :, 0] + 0.7152 * srgb[:, :, 1] + 0.0722 * srgb[:, :, 2]
        axes[1, col].hist(
            lum.flatten(), bins=100, color=colors[col], alpha=0.7, density=True
        )
        axes[1, col].set_title("Histogramme luminance", fontweight="bold")
        axes[1, col].set_xlim(0, 1)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")

    return srgb_results


def create_oetf_comparison_figure(linear_rgb, srgb, output_path, title=""):
    """
    Before/after OETF comparison.

    Args:
        linear_rgb: Linear RGB image
        srgb: sRGB-encoded image
        output_path: Where to save the figure
        title: Optional figure title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(np.clip(linear_rgb, 0, 1))
    axes[0].set_title("Avant OETF (linéaire)", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(np.clip(srgb, 0, 1))
    axes[1].set_title("Après OETF (sRGB)", fontweight="bold")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


def create_jpeg_comparison_figure(img_8bit, jpeg_results, output_path, title=""):
    """
    JPEG compression artifacts comparison.

    Args:
        img_8bit: Original 8-bit image
        jpeg_results: Dict of {quality: {'compressed': img, 'artifacts_amp': img,
                                          'size_kb': float, 'psnr': float}}
        output_path: Where to save the figure
        title: Optional figure title
    """
    qualities = list(jpeg_results.keys())
    n = len(qualities) + 1

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    # PNG reference
    axes[0, 0].imshow(img_8bit)
    axes[0, 0].set_title("PNG (sans perte)", fontweight="bold")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(np.zeros_like(img_8bit))
    axes[1, 0].set_title("Artefacts: Aucun", fontweight="bold")
    axes[1, 0].axis("off")

    for i, q in enumerate(qualities):
        data = jpeg_results[q]
        axes[0, i + 1].imshow(data["compressed"])
        axes[0, i + 1].set_title(
            f"Q={q}\n{data['size_kb']:.1f}KB, PSNR={data['psnr']:.1f}dB",
            fontweight="bold",
        )
        axes[0, i + 1].axis("off")

        axes[1, i + 1].imshow(data["artifacts_amp"])
        axes[1, i + 1].set_title("Artefacts (×5)", fontweight="bold")
        axes[1, i + 1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


def create_filesize_quality_graph(jpeg_results, png_size, output_path, title=""):
    """
    File size vs quality graph.

    Args:
        jpeg_results: Dict of {quality: {'size_kb': float, 'psnr': float, ...}}
        png_size: PNG file size in bytes
        output_path: Where to save the figure
        title: Optional figure title
    """
    qualities = sorted(jpeg_results.keys(), reverse=True)
    sizes = [jpeg_results[q]["size_kb"] for q in qualities]
    psnrs = [jpeg_results[q]["psnr"] for q in qualities]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(qualities, sizes, "bo-", lw=2, ms=8, label="JPEG")
    axes[0].axhline(
        y=png_size / 1024,
        color="g",
        ls="--",
        lw=2,
        label=f"PNG ({png_size/1024:.1f}KB)",
    )
    axes[0].set_xlabel("Qualité JPEG")
    axes[0].set_ylabel("Taille (KB)")
    axes[0].set_title("Taille vs Qualité", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis()

    axes[1].plot(qualities, psnrs, "ro-", lw=2, ms=8)
    axes[1].set_xlabel("Qualité JPEG")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("PSNR vs Qualité", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


def create_dynamic_range_figure(linear_rgb, srgb, analysis, output_path, title=""):
    """
    Dynamic range visualization with clipping analysis.

    Args:
        linear_rgb: Linear RGB image
        srgb: sRGB-encoded image
        analysis: Dict with 'highlight_clipped_percent', 'shadow_crushed_percent',
                  'dynamic_range_stops', 'min_luminance', 'max_luminance'
        output_path: Where to save the figure
        title: Optional figure title
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    lum = (
        0.2126 * linear_rgb[:, :, 0]
        + 0.7152 * linear_rgb[:, :, 1]
        + 0.0722 * linear_rgb[:, :, 2]
    )
    highlight_mask = lum >= 0.95
    shadow_mask = lum <= 0.05

    axes[0, 0].imshow(np.clip(srgb, 0, 1))
    axes[0, 0].set_title("Image sRGB", fontweight="bold")
    axes[0, 0].axis("off")

    highlight_vis = np.clip(srgb.copy(), 0, 1)
    highlight_vis[highlight_mask] = [1, 0, 0]
    axes[0, 1].imshow(highlight_vis)
    axes[0, 1].set_title(
        f"Hautes lumières écrêtées\n({analysis['highlight_clipped_percent']:.2f}%)",
        fontweight="bold",
    )
    axes[0, 1].axis("off")

    shadow_vis = np.clip(srgb.copy(), 0, 1)
    shadow_vis[shadow_mask] = [0, 0, 1]
    axes[0, 2].imshow(shadow_vis)
    axes[0, 2].set_title(
        f"Ombres écrasées\n({analysis['shadow_crushed_percent']:.2f}%)",
        fontweight="bold",
    )
    axes[0, 2].axis("off")

    axes[1, 0].hist(lum.flatten(), bins=100, color="steelblue", alpha=0.7)
    axes[1, 0].axvline(x=0.01, color="blue", ls="--", label="Seuil ombres")
    axes[1, 0].axvline(x=0.99, color="red", ls="--", label="Seuil hautes lumières")
    axes[1, 0].set_title("Histogramme (linéaire)", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, 1)

    lum_srgb = 0.2126 * srgb[:, :, 0] + 0.7152 * srgb[:, :, 1] + 0.0722 * srgb[:, :, 2]
    axes[1, 1].hist(lum_srgb.flatten(), bins=100, color="coral", alpha=0.7)
    axes[1, 1].set_title("Histogramme (sRGB)", fontweight="bold")
    axes[1, 1].set_xlim(0, 1)

    stats = f"""Plage dynamique: {analysis['dynamic_range_stops']:.1f} stops
Luminance min: {analysis['min_luminance']:.4f}
Luminance max: {analysis['max_luminance']:.4f}
Hautes lumières: {analysis['highlight_clipped_percent']:.2f}%
Ombres: {analysis['shadow_crushed_percent']:.2f}%"""

    axes[1, 2].text(
        0.1,
        0.5,
        stats,
        transform=axes[1, 2].transAxes,
        fontsize=11,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )
    axes[1, 2].axis("off")
    axes[1, 2].set_title("Statistiques", fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {output_path}")


# =============================================================================
# Report File Writing
# =============================================================================


def save_report(html_content, filepath):
    """Save HTML report to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n{'='*60}")
    print(f"Rapport HTML généré: {filepath}")
    print("=" * 60)
