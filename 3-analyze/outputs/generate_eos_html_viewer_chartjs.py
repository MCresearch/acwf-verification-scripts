#!/usr/bin/env python
"""
Generate standalone HTML viewer for EOS comparison data using Chart.js with ACWF styling.

This script converts EOS comparison data from JSON files into a self-contained
interactive HTML file that mimics the design of the ACWF verification website.
Uses Chart.js instead of Plotly for lighter weight and better integration with ACWF styling.

Version 2: Fixed layout with configurations in columns, controls under periodic table.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

# Import existing utilities
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import quantities_for_comparison as qc

# ===== CONFIGURATION =====

CODE_COLORS = [
    '#25cff2',  # Cyan (ACWF primary chart color)
    '#3dd5f3',  # Light cyan
    '#0d6efd',  # Blue (Bootstrap primary)
    '#198754',  # Green
    '#ffc107',  # Yellow
    '#dc3545',  # Red
    '#6c757d',  # Gray
    '#0dcaf0',  # Info blue
    '#20c997',  # Teal
    '#fd7e14',  # Orange
    '#e83e8c',  # Pink
    '#6610f2'   # Indigo
]

EXPECTED_SCRIPT_VERSION = ['0.0.3', '0.0.4']

# Complete periodic table layout: element -> (row, column)
# 18 columns × 10 rows grid
PERIODIC_TABLE_LAYOUT = {
    # Period 1
    'H': (1, 1), 'He': (1, 18),
    # Period 2
    'Li': (2, 1), 'Be': (2, 2),
    'B': (2, 13), 'C': (2, 14), 'N': (2, 15), 'O': (2, 16), 'F': (2, 17), 'Ne': (2, 18),
    # Period 3
    'Na': (3, 1), 'Mg': (3, 2),
    'Al': (3, 13), 'Si': (3, 14), 'P': (3, 15), 'S': (3, 16), 'Cl': (3, 17), 'Ar': (3, 18),
    # Period 4
    'K': (4, 1), 'Ca': (4, 2),
    'Sc': (4, 3), 'Ti': (4, 4), 'V': (4, 5), 'Cr': (4, 6), 'Mn': (4, 7), 'Fe': (4, 8),
    'Co': (4, 9), 'Ni': (4, 10), 'Cu': (4, 11), 'Zn': (4, 12),
    'Ga': (4, 13), 'Ge': (4, 14), 'As': (4, 15), 'Se': (4, 16), 'Br': (4, 17), 'Kr': (4, 18),
    # Period 5
    'Rb': (5, 1), 'Sr': (5, 2),
    'Y': (5, 3), 'Zr': (5, 4), 'Nb': (5, 5), 'Mo': (5, 6), 'Tc': (5, 7), 'Ru': (5, 8),
    'Rh': (5, 9), 'Pd': (5, 10), 'Ag': (5, 11), 'Cd': (5, 12),
    'In': (5, 13), 'Sn': (5, 14), 'Sb': (5, 15), 'Te': (5, 16), 'I': (5, 17), 'Xe': (5, 18),
    # Period 6
    'Cs': (6, 1), 'Ba': (6, 2),
    'La': (6, 3),  # Lanthanide series marker
    'Hf': (6, 4), 'Ta': (6, 5), 'W': (6, 6), 'Re': (6, 7), 'Os': (6, 8), 'Ir': (6, 9),
    'Pt': (6, 10), 'Au': (6, 11), 'Hg': (6, 12),
    'Tl': (6, 13), 'Pb': (6, 14), 'Bi': (6, 15), 'Po': (6, 16), 'At': (6, 17), 'Rn': (6, 18),
    # Period 7
    'Fr': (7, 1), 'Ra': (7, 2),
    'Ac': (7, 3),  # Actinide series marker
    'Rf': (7, 4), 'Db': (7, 5), 'Sg': (7, 6), 'Bh': (7, 7), 'Hs': (7, 8), 'Mt': (7, 9),
    'Ds': (7, 10), 'Rg': (7, 11), 'Cn': (7, 12),
    'Nh': (7, 13), 'Fl': (7, 14), 'Mc': (7, 15), 'Lv': (7, 16), 'Ts': (7, 17), 'Og': (7, 18),
    # Lanthanides (row 9)
    'Ce': (9, 4), 'Pr': (9, 5), 'Nd': (9, 6), 'Pm': (9, 7), 'Sm': (9, 8), 'Eu': (9, 9),
    'Gd': (9, 10), 'Tb': (9, 11), 'Dy': (9, 12), 'Ho': (9, 13), 'Er': (9, 14), 'Tm': (9, 15),
    'Yb': (9, 16), 'Lu': (9, 17),
    # Actinides (row 10)
    'Th': (10, 4), 'Pa': (10, 5), 'U': (10, 6), 'Np': (10, 7), 'Pu': (10, 8), 'Am': (10, 9),
    'Cm': (10, 10), 'Bk': (10, 11), 'Cf': (10, 12), 'Es': (10, 13), 'Fm': (10, 14), 'Md': (10, 15),
    'No': (10, 16), 'Lr': (10, 17),
}

# Elements disabled in the notebook (Z > 96 or unstable)
DISABLED_ELEMENTS = [
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',  # Later actinides
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',  # Transactinides
    'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'  # Superheavy
]

# ===== ARGUMENT PARSING =====

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate standalone HTML viewer for EOS comparison (Chart.js + ACWF styling)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --dataset unaries-verification-PBE-v1 --codes abacus vasp
  %(prog)s --dataset oxides-verification-PBE-v1 --codes abacus_sg15 abacus_tzdp --max-colorbar 1.5
        '''
    )
    parser.add_argument('--dataset', required=True,
                       help='Dataset name (e.g., unaries-verification-PBE-v1)')
    parser.add_argument('--codes', nargs='+', required=True,
                       help='Code names to include (space-separated)')
    parser.add_argument('--output',
                       help='Output HTML filename (default: eos_viewer_chartjs_{dataset}.html)')
    parser.add_argument('--title', default='EOS Comparison Viewer',
                       help='Page title (default: EOS Comparison Viewer)')
    parser.add_argument('--max-colorbar', type=float,
                       help='Fixed maximum for heatmap colorbar (default: auto-scale)')
    parser.add_argument('--results-dir', default='.',
                       help='Directory containing results JSON files (default: current directory)')
    return parser.parse_args()

# ===== DATA LOADING =====

def load_results_data(dataset, codes, results_dir):
    """
    Load results data from JSON files.

    Args:
        dataset: Dataset name (e.g., 'unaries-verification-PBE-v1')
        codes: List of code names
        results_dir: Directory containing JSON files

    Returns:
        Dictionary mapping code names to their result data
    """
    results = {}
    for code in codes:
        filepath = Path(results_dir) / f'results-{dataset}-{code}.json'
        if not filepath.exists():
            print(f"ERROR: File not found: {filepath}", file=sys.stderr)
            sys.exit(1)

        with open(filepath) as f:
            data = json.load(f)

        if data.get('script_version') not in EXPECTED_SCRIPT_VERSION:
            print(f"WARNING: {code} has version {data.get('script_version')}, "
                  f"expected {EXPECTED_SCRIPT_VERSION}", file=sys.stderr)

        results[code] = data

    return results

# ===== CONFIGURATION DETECTION =====

def get_configurations(dataset):
    """
    Determine configurations based on dataset name.

    Args:
        dataset: Dataset name

    Returns:
        List of configuration strings
    """
    if 'unaries' in dataset.lower():
        return ['X/SC', 'X/BCC', 'X/FCC', 'X/Diamond']
    else:  # oxides
        return ['XO', 'XO2', 'XO3', 'X2O', 'X2O3', 'X2O5']

# ===== DATA TRANSFORMATION =====

def prepare_eos_data(results_dict, configurations):
    """
    Transform raw JSON data into Chart.js-ready format.
    """
    eos_data = {}

    # Get all unique element-config combinations
    all_keys = set()
    for code_data in results_dict.values():
        all_keys.update(code_data.get('BM_fit_data', {}).keys())

    for key in sorted(all_keys):
        element, configuration = key.split('-', 1)
        eos_data[key] = {}

        for code, code_data in results_dict.items():
            # Check if this code has data for this element-config
            if key not in code_data.get('BM_fit_data', {}):
                continue

            bm_fit = code_data['BM_fit_data'][key]
            eos_points = code_data.get('eos_data', {}).get(key)
            num_atoms = code_data.get('num_atoms_in_sim_cell', {}).get(key)

            if bm_fit is None or num_atoms is None:
                continue

            # Calculate scaling to formula unit
            try:
                scaling = qc.get_volume_scaling_to_formula_unit(
                    num_atoms, element, configuration
                )
            except (ValueError, AssertionError) as e:
                print(f"WARNING: Scaling error for {code} {key}: {e}", file=sys.stderr)
                continue

            code_entry = {}

            # Process EOS points if available
            scaled_volumes = None
            if eos_points:
                try:
                    volumes, energies = np.array(eos_points).T
                    scaled_volumes = volumes / scaling
                    code_entry['volumes'] = scaled_volumes.tolist()
                    code_entry['energies'] = (
                        (energies - bm_fit['E0']) / scaling
                    ).tolist()
                except Exception as e:
                    print(f"WARNING: Error processing EOS points for {code} {key}: {e}", file=sys.stderr)

            # Process fit data
            if bm_fit and all(k in bm_fit for k in ['min_volume', 'bulk_modulus_ev_ang3', 'bulk_deriv', 'E0']):
                try:
                    # Generate dense volume grid for smooth curves
                    V0 = bm_fit['min_volume'] / scaling

                    # Determine volume range
                    if scaled_volumes is not None and len(scaled_volumes) > 0:
                        vol_min = np.min(scaled_volumes)
                        vol_max = np.max(scaled_volumes)
                        vol_range = vol_max - vol_min
                        dense_vol_min = vol_min - 0.01 * vol_range
                        dense_vol_max = vol_max + 0.01 * vol_range
                    else:
                        dense_vol_min = V0 * 0.94
                        dense_vol_max = V0 * 1.06

                    dense_vols = np.linspace(dense_vol_min, dense_vol_max, 100)

                    # Evaluate BM function
                    dense_energies = qc.birch_murnaghan(
                        V=dense_vols * scaling,
                        E0=bm_fit['E0'],
                        V0=bm_fit['min_volume'],
                        B0=bm_fit['bulk_modulus_ev_ang3'],
                        B01=bm_fit['bulk_deriv']
                    )

                    code_entry['fit'] = {
                        'volumes_dense': dense_vols.tolist(),
                        'energies_dense': (
                            (dense_energies - bm_fit['E0']) / scaling
                        ).tolist(),
                        'V0': float(V0),
                        'B0': float(bm_fit['bulk_modulus_ev_ang3']),
                        'B01': float(bm_fit['bulk_deriv']),
                        'residuals': float(bm_fit.get('residuals', 0))
                    }
                except Exception as e:
                    print(f"WARNING: Error processing BM fit for {code} {key}: {e}", file=sys.stderr)

            if code_entry:
                eos_data[key][code] = code_entry

    return eos_data

# ===== COMPARISON METRICS =====

def compute_all_comparisons(results_dict, configurations, codes):
    """
    Pre-compute all comparison matrices (delta, epsilon, nu).
    """
    comparison_data = {}

    # Get all keys
    all_keys = set()
    for code_data in results_dict.values():
        all_keys.update(code_data.get('BM_fit_data', {}).keys())

    for key in sorted(all_keys):
        element, configuration = key.split('-', 1)

        # Collect fit parameters for all codes
        fits = {}
        for code in codes:
            if key not in results_dict[code].get('BM_fit_data', {}):
                continue

            bm_fit = results_dict[code]['BM_fit_data'][key]
            num_atoms = results_dict[code].get('num_atoms_in_sim_cell', {}).get(key)

            if bm_fit is None or num_atoms is None:
                continue

            if not all(k in bm_fit for k in ['min_volume', 'bulk_modulus_ev_ang3', 'bulk_deriv']):
                continue

            try:
                scaling = qc.get_volume_scaling_to_formula_unit(
                    num_atoms, element, configuration
                )

                fits[code] = {
                    'V0': bm_fit['min_volume'] / scaling,
                    'B0': bm_fit['bulk_modulus_ev_ang3'],
                    'B01': bm_fit['bulk_deriv']
                }
            except (ValueError, AssertionError) as e:
                print(f"WARNING: Scaling error for comparison {code} {key}: {e}", file=sys.stderr)
                continue

        if len(fits) < 2:
            continue  # Need at least 2 codes for comparison

        # Build comparison matrices
        code_list = list(fits.keys())
        n = len(code_list)

        delta_matrix = [[0.0] * n for _ in range(n)]
        epsilon_matrix = [[0.0] * n for _ in range(n)]
        nu_matrix = [[0.0] * n for _ in range(n)]

        for i, code1 in enumerate(code_list):
            for j, code2 in enumerate(code_list):
                if i == j:
                    continue

                f1 = fits[code1]
                f2 = fits[code2]

                try:
                    # Calculate delta
                    delta_val = qc.delta(
                        f1['V0'], f1['B0'], f1['B01'],
                        f2['V0'], f2['B0'], f2['B01'],
                        1.0, 0.0, 0.0
                    )
                    delta_matrix[i][j] = float(delta_val)

                    # Calculate epsilon
                    epsilon_val = qc.epsilon(
                        f1['V0'], f1['B0'], f1['B01'],
                        f2['V0'], f2['B0'], f2['B01'],
                        1.0, 0.0, 0.0
                    )
                    epsilon_matrix[i][j] = float(epsilon_val)

                    # Calculate nu
                    nu_val = qc.nu(
                        f1['V0'], f1['B0'], f1['B01'],
                        f2['V0'], f2['B0'], f2['B01'],
                        100.0, 1.0/20, 1.0/400
                    )
                    nu_matrix[i][j] = float(nu_val)
                except Exception as e:
                    print(f"WARNING: Error computing comparison for {code1} vs {code2} in {key}: {e}", file=sys.stderr)

        comparison_data[key] = {
            'codes': code_list,
            'delta': delta_matrix,
            'epsilon': epsilon_matrix,
            'nu': nu_matrix
        }

    return comparison_data

# ===== HTML GENERATION =====

def generate_periodic_table_html(available_elements):
    """Generate HTML for periodic table."""
    html_parts = []

    for element, (row, col) in sorted(PERIODIC_TABLE_LAYOUT.items(), key=lambda x: (x[1][0], x[1][1])):
        disabled = element in DISABLED_ELEMENTS or element not in available_elements
        disabled_class = ' disabled' if disabled else ''
        onclick = '' if disabled else f"onElementClick(this, '{element}')"

        html_parts.append(f'''        <div class="element{disabled_class}"
             style="grid-column: {col}; grid-row: {row};"
             onclick="{onclick}"
             title="{element}">
            {element}
        </div>''')

    return '\n'.join(html_parts)

def generate_html(eos_data, comparison_data, configs, codes, title, dataset, max_colorbar):
    """Generate complete HTML file with Chart.js and ACWF styling."""

    # Build code color mapping
    color_codes = CODE_COLORS[1:]
    code_color_map = {
        code: color_codes[i % len(color_codes)]
        for i, code in enumerate(codes)
    }

    # Generate checkboxes
    checkboxes_html = '\n'.join([
        f'                <label><input type="checkbox" value="{code}" '
        f'onchange="updatePlots()" checked> {code}</label>'
        for code in codes
    ])

    # Generate periodic table HTML
    available_elements = set()
    for key in eos_data.keys():
        element = key.split('-')[0]
        available_elements.add(element)

    periodic_table_html = generate_periodic_table_html(available_elements)

    # Prepare embedded data (minified JSON)
    embedded_eos_json = json.dumps(eos_data, separators=(',', ':'))
    embedded_comparison_json = json.dumps(comparison_data, separators=(',', ':'))
    config_list_json = json.dumps(configs)
    code_colors_json = json.dumps(code_color_map)
    max_colorbar_json = json.dumps(max_colorbar)

    # Complete HTML template with Chart.js and ACWF styling
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>

    <!-- Chart.js from CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

    <!-- MathJax for LaTeX rendering -->
    <script>
    MathJax = {{
        tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
        }},
        svg: {{
            fontCache: 'global'
        }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <style>
        /* ===== ACWF-INSPIRED STYLING ===== */

        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         Oxygen, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            margin: 0;
            padding: 0 20px 40px 20px;
            background: #f0f8ff;  /* Alice Blue - ACWF background */
            color: #212529;
        }}

        #header {{
            background: #fff;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}

        #header h1 {{
            margin: 0 0 10px 0;
            font-size: 24px;
            font-weight: 500;
            color: #212529;
        }}

        #header p {{
            margin: 5px 0;
            color: #455860;
            font-size: 14px;
        }}

        .help-button {{
            background: #0d6efd;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            margin-top: 10px;
            font-size: 14px;
            font-weight: 500;
            transition: background-color 0.2s;
        }}

        .help-button:hover {{
            background: #0b5ed7;
        }}

        #help {{
            background: #fff;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 6px;
            border-left: 4px solid #0d6efd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        #help h3 {{
            margin-top: 0;
            color: #212529;
            font-size: 18px;
            font-weight: 500;
        }}

        #help ol, #help ul {{
            color: #455860;
            line-height: 1.6;
        }}

        /* ===== PERIODIC TABLE (ACWF DESIGN) ===== */

        .ptable {{
            display: grid;
            grid-template-columns: repeat(18, 38px);
            grid-template-rows: repeat(10, 38px);
            gap: 1px;
            margin: 0 auto 20px auto;
            width: fit-content;
            justify-content: center;
        }}

        .element {{
            width: 38px;
            height: 38px;
            background: #a3d7f5;  /* Light sky blue - ACWF element color */
            border: 1px solid #999;
            border-radius: 10%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 19px;
            font-weight: 400;
            color: #000;
            user-select: none;
            transition: all 0.2s;
        }}

        .element:hover:not(.disabled) {{
            filter: brightness(85%);
            transform: scale(1.2);
            z-index: 10;
        }}

        .element.selected {{
            background: #2747fd;  /* Bright blue - ACWF selected */
            color: #fff;
        }}

        .element.disabled {{
            background: #eee;
            border-color: #b3b2b2;
            color: #b3b2b2;
            cursor: not-allowed;
            pointer-events: none;
        }}

        /* ===== CONTROLS (UNDER PERIODIC TABLE) ===== */

        #controls {{
            background: #fff;
            padding: 15px 20px;
            margin: 0 auto 20px auto;
            max-width: 900px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            justify-content: center;
            align-items: flex-start;
            gap: 60px;
        }}

        #controls h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            font-weight: 500;
            color: #212529;
        }}

        #code-selector {{
            display: flex;
            flex-direction: column;
        }}

        #code-selector label {{
            margin: 3px 0;
            cursor: pointer;
            color: #455860;
            font-size: 13px;
            display: flex;
            align-items: center;
        }}

        #code-selector input[type="checkbox"] {{
            margin-right: 8px;
        }}

        #metric-selector select {{
            padding: 6px 12px;
            font-size: 13px;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            background: white;
            color: #455860;
            cursor: pointer;
            min-width: 180px;
        }}

        /* ===== ELEMENT INFO SECTION ===== */

        #element-info {{
            background: #fff;
            padding: 20px;
            margin: 20px auto;
            max-width: 1800px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        #element-info h2 {{
            margin: 0 0 20px 0;
            text-align: center;
            color: #212529;
            font-size: 22px;
            font-weight: 500;
        }}

        /* ===== CONFIGURATION COLUMNS ===== */

        .config-columns {{
            display: flex;
            flex-direction: column;
            gap: 30px;
        }}

        .config-section {{
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            background: #fafafa;
        }}

        .config-header {{
            font-size: 16px;
            font-weight: 500;
            color: #212529;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #0d6efd;
        }}

        .plot-row {{
            display: flex;
            gap: 20px;
        }}

        .eos-plot, .heatmap-plot {{
            flex: 1;
            height: 500px;
            position: relative;
            background: #fff;
            border-radius: 6px;
            padding: 10px;
        }}

        .eos-canvas {{
            width: 100% !important;
            height: 100% !important;
        }}

        /* ===== HEATMAP TABLE ===== */

        .heatmap-container {{
            width: 100%;
            height: 100%;
            overflow: auto;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .heatmap-table {{
            border-collapse: collapse;
            font-size: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .heatmap-table th,
        .heatmap-table td {{
            padding: 8px 12px;
            text-align: center;
            border: 1px solid #dee2e6;
        }}

        .heatmap-table th {{
            background: #f8f9fa;
            font-weight: 500;
            color: #212529;
            position: sticky;
        }}

        .heatmap-table thead th {{
            top: 0;
            z-index: 10;
        }}

        .heatmap-table tbody th {{
            left: 0;
            z-index: 5;
            background: #f8f9fa;
        }}

        .heatmap-table td {{
            color: #000;
            font-weight: 500;
            min-width: 60px;
        }}

        .no-data-message {{
            text-align: center;
            color: #6c757d;
            padding: 100px 50px;
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{title}</h1>
        <p>Dataset: <strong>{dataset}</strong></p>
        <p>Codes: {', '.join(codes)}</p>
        <button class="help-button" onclick="toggleHelp()">Show/Hide Help</button>
    </div>

    <div id="help" style="display:none;">
        <h3>How to Use This Viewer</h3>
        <ol>
            <li><strong>Click Element:</strong> Click any enabled element in the periodic table to view its EOS curves and comparison heatmaps for all configurations.</li>
            <li><strong>Select Codes:</strong> Check/uncheck codes in the control panel to filter which codes are displayed and compared.</li>
            <li><strong>Choose Metric:</strong> Select a comparison metric (epsilon, delta, or nu) from the dropdown.</li>
        </ol>

        <h3 style="margin-top: 25px;">Visualization Guide</h3>
        <p><strong>EOS Plot (Left):</strong> Shows energy-volume curves. Points are calculated data, smooth curves are Birch-Murnaghan fits.</p>
        <p><strong>Heatmap (Right):</strong> Shows pairwise comparison between selected codes using the chosen metric. Lower values (lighter colors) indicate better agreement.</p>

        <h3 style="margin-top: 25px;">Theoretical Background</h3>

        <h4 style="margin-top: 15px; color: #0d6efd;">Equation of State (EOS)</h4>
        <p>The equation of state describes the relationship between energy and volume for a material. This tool fits the computed energy-volume data points to the <strong>Birch-Murnaghan equation of state</strong>:</p>
        <p style="text-align: center; background: #f8f9fa; padding: 15px; border-radius: 4px;">
        $$E(V) = E_0 + \\frac{{9}}{{16}} B_0 V_0 \\left[ \\left(\\frac{{V_0}}{{V}}\\right)^{{2/3}} - 1 \\right]^2 \\left[ 6 - 4\\left(\\frac{{V_0}}{{V}}\\right)^{{2/3}} + B'_0 \\left( \\left(\\frac{{V_0}}{{V}}\\right)^{{2/3}} - 1 \\right) \\right]$$
        </p>
        <p>Where:</p>
        <ul style="line-height: 1.8;">
            <li><strong>$E_0$</strong>: Equilibrium energy (minimum of the curve)</li>
            <li><strong>$V_0$</strong>: Equilibrium volume per formula unit at minimum energy</li>
            <li><strong>$B_0$</strong>: Bulk modulus (stiffness - resistance to compression)</li>
            <li><strong>$B'_0$ (B01)</strong>: Derivative of bulk modulus with respect to pressure</li>
        </ul>

        <h4 style="margin-top: 15px; color: #0d6efd;">Comparison Metrics</h4>
        <p>Three metrics quantify the agreement between different DFT implementations:</p>
        <ul style="line-height: 1.8;">
            <li><strong>$\\delta$ (Delta)</strong> - a metric that represents the area between the two EOS curves. This metric has the shortcoming of being too sensitive to the value of the bulk modulus of the material. In this website, the values are normalized by the number of atoms.</li>
            <li><strong>$\\varepsilon$ (Epsilon)</strong> - a metric that represents the area between the two EOS curves normalized by the average variance of the two curves.</li>
            <li><strong>$\\nu$ (Nu)</strong> - a metric that captures the relative difference of the Birch-Murnaghan fitting parameters with specified weights.</li>
        </ul>

        <h4 style="margin-top: 15px; color: #0d6efd;">Purpose of Verification</h4>
        <p>
        This tool visualizes results from the <strong>AiiDA Common Workflows (ACWF) verification project</strong>, which systematically compares DFT code implementations for chemical elements Z=1 to Z=96. The goal is to assess the <em>precision</em> of different DFT codes and pseudopotential/basis set combinations, not their accuracy compared to experiment.
        </p>
        <p>
        By comparing implementations that should give identical results (same functional, similar pseudopotentials), we can identify numerical precision issues, implementation differences, and establish confidence in computational predictions.
        </p>

        <h4 style="margin-top: 15px; color: #0d6efd;">References</h4>
        <ul style="line-height: 1.8;">
            <li>ACWF verification: <strong>Bosoni et al., Nature Reviews Physics 6, 45-58 (2024)</strong><br>
            <a href="https://doi.org/10.1038/s42254-023-00655-3" target="_blank" style="color: #0d6efd;">https://doi.org/10.1038/s42254-023-00655-3</a></li>
            <li>Original delta metric: <strong>Lejaeghere et al., Science 351, 1415-1418 (2016)</strong><br>
            <a href="https://doi.org/10.1126/science.aad3000" target="_blank" style="color: #0d6efd;">https://doi.org/10.1126/science.aad3000</a></li>
            <li>Birch-Murnaghan EOS: <strong>Birch, Physical Review 71, 809 (1947)</strong></li>
        </ul>
    </div>

    <div class="ptable">
{periodic_table_html}
    </div>

    <div id="controls">
        <div id="code-selector">
            <h3>Select Codes:</h3>
{checkboxes_html}
        </div>

        <div id="metric-selector">
            <h3>Comparison Metric:</h3>
            <select id="metric" onchange="updatePlots()">
                <option value="epsilon" selected>Epsilon (ε)</option>
                <option value="delta">Delta (δ) [meV]</option>
                <option value="nu">Nu (ν) [%]</option>
            </select>
        </div>
    </div>

    <div id="element-info" style="display:none;">
        <h2 id="element-name"></h2>
        <div id="config-columns" class="config-columns"></div>
    </div>

    <script>
        // ===== EMBEDDED DATA =====
        const EOS_DATA = {embedded_eos_json};
        const COMPARISON_DATA = {embedded_comparison_json};
        const CONFIGURATIONS = {config_list_json};
        const CODE_COLORS = {code_colors_json};
        const MAX_COLORBAR = {max_colorbar_json};

        // ===== STATE MANAGEMENT =====
        let currentElement = null;
        let eosChartInstances = {{}};  // Store chart instances by config

        // ===== EVENT HANDLERS =====
        function onElementClick(targetElement, element) {{
            currentElement = element;
            document.getElementById('element-name').textContent = element;
            document.getElementById('element-info').style.display = 'block';

            // Highlight selected element
            document.querySelectorAll('.element').forEach(el =>
                el.classList.remove('selected'));
            targetElement.classList.add('selected');

            // Render all configurations
            updatePlots();
        }}

        function updatePlots() {{
            if (!currentElement) return;

            const selectedCodes = getSelectedCodes();
            const metric = document.getElementById('metric').value;

            renderAllConfigurations(currentElement, selectedCodes, metric);
        }}

        function getSelectedCodes() {{
            const checkboxes = document.querySelectorAll('#code-selector input[type="checkbox"]:checked');
            return Array.from(checkboxes).map(cb => cb.value);
        }}

        function toggleHelp() {{
            const help = document.getElementById('help');
            help.style.display = help.style.display === 'none' ? 'block' : 'none';
        }}

        // ===== RENDER ALL CONFIGURATIONS =====

        function renderAllConfigurations(element, selectedCodes, metric) {{
            const container = document.getElementById('config-columns');
            container.innerHTML = '';

            // Destroy all old chart instances
            Object.values(eosChartInstances).forEach(chart => chart.destroy());
            eosChartInstances = {{}};

            CONFIGURATIONS.forEach(config => {{
                const key = `${{element}}-${{config}}`;
                const data = EOS_DATA[key];

                // Create section for this configuration
                const section = document.createElement('div');
                section.className = 'config-section';

                const configNice = config.replace('X/', '').replace(/(\\d)/g, '₃$1');
                const header = document.createElement('div');
                header.className = 'config-header';
                header.textContent = `${{element}} (${{configNice}})`;
                section.appendChild(header);

                const plotRow = document.createElement('div');
                plotRow.className = 'plot-row';

                // EOS plot container
                const eosDiv = document.createElement('div');
                eosDiv.className = 'eos-plot';
                const eosCanvas = document.createElement('canvas');
                eosCanvas.className = 'eos-canvas';
                eosCanvas.id = `eos-canvas-${{config.replace('/', '-')}}`;
                eosDiv.appendChild(eosCanvas);
                plotRow.appendChild(eosDiv);

                // Heatmap container
                const heatmapDiv = document.createElement('div');
                heatmapDiv.className = 'heatmap-plot';
                heatmapDiv.id = `heatmap-${{config.replace('/', '-')}}`;
                plotRow.appendChild(heatmapDiv);

                section.appendChild(plotRow);
                container.appendChild(section);

                // Render plots
                renderEOSPlot(element, config, selectedCodes, eosCanvas.id);
                renderHeatmap(element, config, metric, selectedCodes, heatmapDiv.id);
            }});
        }}

        // ===== CHART.JS PLOTTING =====

        function renderEOSPlot(element, config, selectedCodes, canvasId) {{
            const key = `${{element}}-${{config}}`;
            const data = EOS_DATA[key];

            const canvas = document.getElementById(canvasId);
            if (!canvas) return;

            if (!data || Object.keys(data).length === 0) {{
                canvas.parentElement.innerHTML =
                    '<div class="no-data-message">No EOS data available</div>';
                return;
            }}

            const datasets = [];

            selectedCodes.forEach(code => {{
                if (!data[code]) return;

                const color = CODE_COLORS[code] || '#000000';

                // EOS points (scatter)
                if (data[code].volumes && data[code].energies) {{
                    datasets.push({{
                        label: `${{code}}`,
                        data: data[code].volumes.map((v, i) => ({{
                            x: v,
                            y: data[code].energies[i]
                        }})),
                        backgroundColor: color,
                        borderColor: color,
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        showLine: false
                    }});
                }}

                // BM fit curve (line) - hidden from legend
                if (data[code].fit && data[code].fit.volumes_dense) {{
                    datasets.push({{
                        label: `${{code}}_fit_hidden`,
                        data: data[code].fit.volumes_dense.map((v, i) => ({{
                            x: v,
                            y: data[code].fit.energies_dense[i]
                        }})),
                        borderColor: color,
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 0,
                        showLine: true,
                        fill: false
                    }});
                }}
            }});

            if (datasets.length === 0) {{
                canvas.parentElement.innerHTML =
                    '<div class="no-data-message">No data for selected codes</div>';
                return;
            }}

            const ctx = canvas.getContext('2d');
            const chart = new Chart(ctx, {{
                type: 'scatter',
                data: {{ datasets: datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'right',
                            labels: {{
                                font: {{
                                    family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto',
                                    size: 11
                                }},
                                color: '#212529',
                                padding: 6,
                                usePointStyle: true,
                                boxWidth: 6,
                                filter: function(item, chart) {{
                                    // Hide fit lines from legend (only show data points)
                                    return !item.text.includes('_fit_hidden');
                                }}
                            }}
                        }},
                        title: {{
                            display: true,
                            text: 'EOS Curves',
                            font: {{
                                size: 14,
                                weight: '500'
                            }},
                            color: '#212529'
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'linear',
                            title: {{
                                display: true,
                                text: 'Volume per formula unit (Å³)',
                                color: '#455860',
                                font: {{ size: 11, weight: '500' }}
                            }},
                            grid: {{ color: '#dee2e6' }},
                            ticks: {{
                                color: '#455860',
                                font: {{ size: 10 }}
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'E-TS per formula unit (eV)',
                                color: '#455860',
                                font: {{ size: 11, weight: '500' }}
                            }},
                            grid: {{ color: '#dee2e6' }},
                            ticks: {{
                                color: '#455860',
                                font: {{ size: 10 }}
                            }}
                        }}
                    }}
                }}
            }});

            eosChartInstances[config] = chart;
        }}

        function renderHeatmap(element, config, metric, selectedCodes, divId) {{
            const key = `${{element}}-${{config}}`;
            const compData = COMPARISON_DATA[key];
            const div = document.getElementById(divId);

            if (!div) return;

            if (!compData) {{
                div.innerHTML = '<div class="no-data-message">No comparison data available</div>';
                return;
            }}

            // Filter matrix for selected codes
            const allCodes = compData.codes;
            const indices = selectedCodes.map(c => allCodes.indexOf(c)).filter(i => i >= 0);

            if (indices.length < 2) {{
                div.innerHTML = '<div class="no-data-message">Select at least 2 codes for comparison</div>';
                return;
            }}

            const matrix = indices.map(i =>
                indices.map(j => compData[metric][i][j])
            );
            const labels = indices.map(i => allCodes[i]);

            // Determine color scale maximum
            const flatValues = matrix.flat().filter(v => v > 0);
            const maxVal = MAX_COLORBAR || (flatValues.length > 0 ? Math.max(...flatValues) : 1.0);

            // Build HTML table
            const metricName = metric === 'epsilon' ? 'ε' : (metric === 'delta' ? 'δ [meV]' : 'ν [%]');

            let html = '<div class="heatmap-container">';
            html += '<div>';
            html += `<h4 style="text-align:center;color:#212529;font-size:14px;font-weight:500;margin:0 0 10px 0;">${{metricName}} Comparison</h4>`;
            html += '<table class="heatmap-table">';
            html += '<thead><tr><th></th>';
            labels.forEach(code => {{
                html += `<th>${{code}}</th>`;
            }});
            html += '</tr></thead><tbody>';

            labels.forEach((rowCode, i) => {{
                html += `<tr><th>${{rowCode}}</th>`;
                labels.forEach((colCode, j) => {{
                    const value = matrix[i][j];
                    const color = getHeatmapColor(value, maxVal);
                    const displayValue = value.toFixed(2);  // Always show value including 0.00
                    html += `<td style="background-color: ${{color}}">${{displayValue}}</td>`;
                }});
                html += '</tr>';
            }});
            html += '</tbody></table>';
            html += '</div></div>';

            div.innerHTML = html;
        }}

        function getHeatmapColor(value, max) {{
            if (value === 0) return '#fff';  // White for 0.0

            // White to red gradient
            const intensity = Math.min(Math.abs(value) / max, 1.0);
            const r = 255;
            const g = Math.floor(255 * (1 - intensity * 0.8));
            const b = Math.floor(255 * (1 - intensity * 0.8));
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}

        // ===== INITIALIZATION =====
        console.log('EOS Viewer (Chart.js + ACWF) loaded successfully');
        console.log(`Available element-configurations: ${{Object.keys(EOS_DATA).length}}`);
        console.log(`Codes: ${{Object.keys(CODE_COLORS).join(', ')}}`);
    </script>
</body>
</html>
''';

    return html

# ===== MAIN =====

def main():
    """Main execution function."""
    args = parse_arguments()

    print(f"Loading data for dataset: {args.dataset}")
    print(f"Codes: {', '.join(args.codes)}")

    # Load data
    results = load_results_data(args.dataset, args.codes, args.results_dir)
    configs = get_configurations(args.dataset)

    print(f"Processing {len(configs)} configurations...")

    # Process data
    eos_data = prepare_eos_data(results, configs)
    comparison_data = compute_all_comparisons(results, configs, args.codes)

    print(f"Found data for {len(eos_data)} element-configuration combinations")
    print(f"Computed comparisons for {len(comparison_data)} combinations")
    print(f"Generating HTML...")

    # Generate HTML
    html_content = generate_html(
        eos_data, comparison_data, configs, args.codes,
        args.title, args.dataset, args.max_colorbar
    )

    # Determine output filename
    output_file = args.output or f"eos_viewer_chartjs_{args.dataset}.html"

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    file_size_kb = os.path.getsize(output_file) / 1024

    print(f"\n✓ Success!")
    print(f"  Output: {output_file}")
    print(f"  Size: {file_size_kb:.1f} KB")
    print(f"  Elements: {len(set(k.split('-')[0] for k in eos_data.keys()))}")
    print(f"\nOpen in browser:")
    print(f"  file://{os.path.abspath(output_file)}")

if __name__ == '__main__':
    main()
