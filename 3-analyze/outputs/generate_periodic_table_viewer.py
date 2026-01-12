#!/usr/bin/env python
"""
Generate interactive HTML periodic table comparison viewer.

This script generates a self-contained HTML file that displays an interactive
periodic table for comparing DFT code implementations. Users can select multiple
codes and metrics to visualize pairwise comparison differences color-coded on
element cells.

Usage:
  ./generate_periodic_table_viewer.py \\
    --oxides-codes abacus vasp qe \\
    --unaries-codes abacus vasp qe fleur \\
    --output periodic_table_comparison.html \\
    --results-dir .
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Import existing utilities
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import quantities_for_comparison as qc

# ===== CONFIGURATION =====

__version__ = "1.0.0"

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

# Configuration mappings
OXIDE_CONFIGS = ['X2O3', 'X2O5', 'X2O', 'XO', 'XO2', 'XO3']
UNARY_CONFIGS = ['X/SC', 'X/BCC', 'X/Diamond', 'X/FCC']

# Metric parameters
METRIC_PARAMS = {
    'epsilon': {'prefact': 1.0, 'weight_b0': 1/20, 'weight_b1': 1/400},
    'nu': {'prefact': 100.0, 'weight_b0': 1/20, 'weight_b1': 1/400},
    'delta': {'prefact': 1.0, 'weight_b0': 0, 'weight_b1': 0},
    'V0_rel_diff': {'prefact': 100.0, 'weight_b0': 0, 'weight_b1': 0},
    'B0_rel_diff': {'prefact': 100.0, 'weight_b0': 0, 'weight_b1': 0},
    'B1_rel_diff': {'prefact': 100.0, 'weight_b0': 0, 'weight_b1': 0}
}

METRIC_FUNCTIONS = {
    'epsilon': qc.epsilon,
    'nu': qc.nu,
    'delta': qc.delta,
    'V0_rel_diff': qc.V0_rel_diff,
    'B0_rel_diff': qc.B0_rel_diff,
    'B1_rel_diff': qc.B1_rel_diff
}

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
            print(f"WARNING: File not found: {filepath}, skipping code '{code}'", file=sys.stderr)
            continue

        with open(filepath) as f:
            data = json.load(f)

        if data.get('script_version') not in EXPECTED_SCRIPT_VERSION:
            print(f"WARNING: {code} has version {data.get('script_version')}, "
                  f"expected {EXPECTED_SCRIPT_VERSION}", file=sys.stderr)

        results[code] = data

    return results

def get_scaled_params(results_data, element, config):
    """
    Extract and scale BM fit parameters to formula unit.

    Args:
        results_data: Results dictionary for a code
        element: Element symbol (e.g., 'Al', 'Fe')
        config: Configuration (e.g., 'XO2', 'X/FCC')

    Returns:
        Tuple of (v0, b0, b1) scaled to formula unit, or None if data missing
    """
    # All keys use "Element-Config" format
    key = f"{element}-{config}"

    bm_data = results_data.get('BM_fit_data', {}).get(key)
    if bm_data is None:
        return None

    num_atoms_cell = results_data.get('num_atoms_in_sim_cell', {}).get(key)
    if num_atoms_cell is None:
        return None

    # Get formula unit scaling
    scaling = qc.get_volume_scaling_to_formula_unit(num_atoms_cell, element, config)

    # Extract and scale parameters
    v0 = bm_data['min_volume'] / scaling
    b0 = bm_data['bulk_modulus_ev_ang3']
    b1 = bm_data['bulk_deriv']

    return (v0, b0, b1)

def compute_pairwise_comparison(params1, params2, metric_name):
    """
    Compute a single metric comparison between two parameter sets.

    Args:
        params1: Tuple of (v0, b0, b1) for code 1
        params2: Tuple of (v0, b0, b1) for code 2
        metric_name: Name of metric to compute

    Returns:
        Comparison value, or None if computation fails
    """
    if params1 is None or params2 is None:
        return None

    v0_1, b0_1, b1_1 = params1
    v0_2, b0_2, b1_2 = params2

    metric_func = METRIC_FUNCTIONS[metric_name]
    metric_params = METRIC_PARAMS[metric_name]

    try:
        value = metric_func(
            v0_1, b0_1, b1_1,
            v0_2, b0_2, b1_2,
            metric_params['prefact'],
            metric_params['weight_b0'],
            metric_params['weight_b1']
        )
        # Handle numpy scalars
        if isinstance(value, np.ndarray):
            value = float(value)
        return value
    except Exception as e:
        print(f"WARNING: Failed to compute {metric_name}: {e}", file=sys.stderr)
        return None

def compute_all_comparisons(results_dict, element, config, codes):
    """
    Compute all 6 metrics for all code pairs for a given element-config.

    Args:
        results_dict: Dictionary mapping code names to their results data
        element: Element symbol
        config: Configuration string
        codes: List of code names

    Returns:
        Dictionary with structure:
        {
            'codes': list of code names,
            'comparisons': {
                'epsilon': [[...], ...],  # n×n matrix
                'nu': [[...], ...],
                ...
            }
        }
    """
    n = len(codes)

    # Initialize matrices
    comparisons = {
        metric: [[0.0] * n for _ in range(n)]
        for metric in METRIC_FUNCTIONS.keys()
    }

    # Get parameters for all codes
    params = {}
    for code in codes:
        if code in results_dict:
            params[code] = get_scaled_params(results_dict[code], element, config)
        else:
            params[code] = None

    # Compute all pairwise comparisons
    for i, code1 in enumerate(codes):
        for j, code2 in enumerate(codes):
            if i == j:
                # Diagonal is always 0
                continue

            for metric_name in METRIC_FUNCTIONS.keys():
                value = compute_pairwise_comparison(params[code1], params[code2], metric_name)
                if value is not None:
                    comparisons[metric_name][i][j] = value
                else:
                    # Mark as NaN for missing data
                    comparisons[metric_name][i][j] = float('nan')

    return {
        'codes': codes,
        'comparisons': comparisons
    }

def prepare_eos_data(results_dicts, codes_dict):
    """
    Prepare EOS curve data for JavaScript visualization.

    Args:
        results_dicts: {'oxides': {...}, 'unaries': {...}}
        codes_dict: {'oxides': [...], 'unaries': [...]}

    Returns:
        dict with structure:
        {
          'oxides': {element-config: {code: {volumes, energies, fit}}},
          'unaries': {element-config: {code: {volumes, energies, fit}}}
        }
    """
    eos_data = {'oxides': {}, 'unaries': {}}

    # Process each dataset
    for dataset_type in ['oxides', 'unaries']:
        if dataset_type not in results_dicts or not codes_dict.get(dataset_type):
            continue

        configs = OXIDE_CONFIGS if dataset_type == 'oxides' else UNARY_CONFIGS
        codes = codes_dict[dataset_type]

        for element in PERIODIC_TABLE_LAYOUT.keys():
            if element in DISABLED_ELEMENTS:
                continue

            for config in configs:
                # Both oxides and unaries use "Element-Config" format
                # Examples: "Al-X/FCC" (unaries), "Al-XO2" (oxides)
                key = f"{element}-{config}"

                eos_data[dataset_type][key] = {}

                for code in codes:
                    if code not in results_dicts[dataset_type]:
                        continue

                    results = results_dicts[dataset_type][code]

                    # Extract raw EOS data
                    eos_raw = results.get('eos_data', {}).get(key)
                    bm_fit = results.get('BM_fit_data', {}).get(key)
                    num_atoms = results.get('num_atoms_in_sim_cell', {}).get(key)

                    if not eos_raw or not bm_fit or not num_atoms:
                        continue

                    # Scale to formula unit
                    scaling = qc.get_volume_scaling_to_formula_unit(num_atoms, element, config)

                    # Extract and scale raw data points
                    volumes = np.array([v for v, e in eos_raw]) / scaling
                    energies = (np.array([e for v, e in eos_raw]) - bm_fit['E0']) / scaling

                    # Generate dense grid for fitted curve (100 points)
                    vol_min, vol_max = np.min(volumes), np.max(volumes)
                    vol_range = vol_max - vol_min
                    dense_vols = np.linspace(vol_min - 0.01*vol_range, vol_max + 0.01*vol_range, 100)

                    # Evaluate Birch-Murnaghan on dense grid
                    dense_energies = qc.birch_murnaghan(
                        V=dense_vols * scaling,
                        E0=bm_fit['E0'],
                        V0=bm_fit['min_volume'],
                        B0=bm_fit['bulk_modulus_ev_ang3'],
                        B01=bm_fit['bulk_deriv']
                    )
                    dense_energies = (dense_energies - bm_fit['E0']) / scaling

                    eos_data[dataset_type][key][code] = {
                        'volumes': volumes.tolist(),
                        'energies': energies.tolist(),
                        'fit': {
                            'volumes_dense': dense_vols.tolist(),
                            'energies_dense': dense_energies.tolist(),
                            'V0': float(bm_fit['min_volume'] / scaling),
                            'B0': float(bm_fit['bulk_modulus_ev_ang3']),
                            'B01': float(bm_fit['bulk_deriv']),
                            'residuals': float(bm_fit.get('residuals', 0.0))
                        }
                    }

    return eos_data

def prepare_comparison_data(results_dicts, codes_dict):
    """
    Prepare complete comparison data structure for JavaScript.

    Args:
        results_dicts: {'oxides': {...}, 'unaries': {...}}
        codes_dict: {'oxides': [...], 'unaries': [...]}

    Returns:
        Nested dict structure ready for JSON embedding
    """
    data = {
        'oxides': {},
        'unaries': {},
        'eos_data': {},
        'available_codes': codes_dict,
        'metadata': {
            'script_version': __version__,
            'generation_date': datetime.now().strftime('%Y-%m-%d')
        }
    }

    # Process oxides
    print("Processing oxides...")
    if 'oxides' in results_dicts and codes_dict.get('oxides'):
        for element in PERIODIC_TABLE_LAYOUT.keys():
            if element in DISABLED_ELEMENTS:
                continue

            for config in OXIDE_CONFIGS:
                key = f"{element}-{config}"
                comparison = compute_all_comparisons(
                    results_dicts['oxides'],
                    element,
                    config,
                    codes_dict['oxides']
                )
                data['oxides'][key] = comparison

    # Process unaries
    print("Processing unaries...")
    if 'unaries' in results_dicts and codes_dict.get('unaries'):
        for element in PERIODIC_TABLE_LAYOUT.keys():
            if element in DISABLED_ELEMENTS:
                continue

            for config in UNARY_CONFIGS:
                # Use consistent key format: "Element-Config"
                key = f"{element}-{config}"
                comparison = compute_all_comparisons(
                    results_dicts['unaries'],
                    element,
                    config,
                    codes_dict['unaries']
                )
                data['unaries'][key] = comparison

    # Prepare EOS data
    print("Processing EOS data...")
    data['eos_data'] = prepare_eos_data(results_dicts, codes_dict)

    return data

# ===== HTML GENERATION =====

def generate_periodic_table_html(dataset_type):
    """Generate HTML for the periodic table grid."""

    html = f'<div class="periodic-table" id="periodic-table-{dataset_type}" style="display: none;">\n'

    for element, (row, col) in PERIODIC_TABLE_LAYOUT.items():
        if element in DISABLED_ELEMENTS:
            disabled_class = ' disabled'
        else:
            disabled_class = ''

        html += f'  <div class="element-cell{disabled_class}" '
        html += f'data-element="{element}" '
        html += f'style="grid-column: {col}; grid-row: {row};" '
        html += f'onclick="onElementClick(\'{element}\')">\n'
        html += f'    <span class="element-symbol">{element}</span>\n'
        html += f'    <div class="element-cell-inner {dataset_type}">\n'

        # Generate sub-cells for configurations
        configs = OXIDE_CONFIGS if dataset_type == 'oxides' else UNARY_CONFIGS
        for idx, config in enumerate(configs):
            html += f'      <div class="config-subcell" data-config="{config}" data-index="{idx}"></div>\n'

        html += '    </div>\n'
        html += '  </div>\n'

    html += '</div>\n'

    return html

def generate_html(comparison_data, output_file):
    """Generate complete self-contained HTML file."""

    # Embed data as JavaScript variable
    embedded_data = json.dumps(comparison_data, separators=(',', ':'))

    # Get code lists for checkboxes
    oxides_codes = comparison_data['available_codes'].get('oxides', [])
    unaries_codes = comparison_data['available_codes'].get('unaries', [])
    all_codes = sorted(list(set(oxides_codes + unaries_codes)))

    # Generate periodic tables for both datasets
    periodic_table_unaries = generate_periodic_table_html('unaries')
    periodic_table_oxides = generate_periodic_table_html('oxides')

    # Build HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DFT Code Comparison - Interactive Periodic Table</title>

    <style>
{get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>DFT Code Comparison - Interactive Periodic Table</h1>
            <p class="subtitle">Compare DFT code implementations across the periodic table</p>
        </header>

        <div class="control-panel">
            <div class="control-section">
                <h3>Dataset</h3>
                <div class="dataset-selector">
                    <label>
                        <input type="radio" name="dataset" value="unaries" checked>
                        Unaries (4 configurations)
                    </label>
                    <label>
                        <input type="radio" name="dataset" value="oxides">
                        Oxides (6 configurations)
                    </label>
                </div>
            </div>

            <div class="control-section">
                <h3>Metric</h3>
                <select id="metric-selector">
                    <option value="epsilon">Epsilon (ε) - Normalized energy difference</option>
                    <option value="nu">Nu (ν) - Combined parameter metric [%]</option>
                    <option value="delta">Delta (Δ) - DeltaTest [meV/atom]</option>
                    <option value="V0_rel_diff">V₀ relative difference [%]</option>
                    <option value="B0_rel_diff">B₀ relative difference [%]</option>
                    <option value="B1_rel_diff">B₁' relative difference [%]</option>
                </select>
            </div>

            <div class="control-section">
                <h3>Select Codes to Compare</h3>
                <div class="code-selector-grid">
                    <div class="code-dropdown-group">
                        <label for="code-a-selector">Code A:</label>
                        <select id="code-a-selector" class="code-dropdown">
{generate_code_options_html(all_codes, 0)}
                        </select>
                    </div>
                    <div class="vs-separator">vs</div>
                    <div class="code-dropdown-group">
                        <label for="code-b-selector">Code B:</label>
                        <select id="code-b-selector" class="code-dropdown">
{generate_code_options_html(all_codes, 1)}
                        </select>
                    </div>
                </div>
            </div>
        </div>

        <div class="periodic-table-container">
{periodic_table_unaries}
{periodic_table_oxides}
        </div>

        <div class="colorbar-container" id="colorbar-container">
            <!-- Colorbar will be dynamically generated -->
        </div>

        <div class="eos-panel-container" id="eos-panel-container" style="display: none;">
            <div class="eos-panel-header">
                <h3 id="eos-panel-title">Element: <span id="eos-element-name"></span> - EOS Curves</h3>
                <button onclick="closeEOSPanel()" class="close-button">Close</button>
            </div>
            <div class="eos-config-grid" id="eos-config-grid">
                <!-- Dynamically populated with EOS plots for each configuration -->
            </div>
        </div>

        <footer>
            <p>Generated by generate_periodic_table_viewer.py v{__version__}</p>
            <p>Data from ACWF verification project</p>
        </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@2.0.1/dist/chartjs-chart-matrix.min.js"></script>

    <script>
{get_javascript_code(embedded_data)}
    </script>
</body>
</html>
'''

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Generated: {output_file}")

def generate_code_options_html(codes, default_index):
    """Generate HTML for code selection dropdown options."""
    html = ''
    for i, code in enumerate(codes):
        selected = ' selected' if i == default_index else ''
        html += f'                            <option value="{code}"{selected}>{code}</option>\n'
    return html

def get_css_styles():
    """Return CSS styles for the HTML."""
    return '''
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f0f8ff;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        header h1 {
            color: #0d6efd;
            font-size: 2em;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #666;
            font-size: 1.1em;
        }

        .control-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .control-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .control-section h3 {
            margin-bottom: 15px;
            color: #0d6efd;
            font-size: 1.2em;
        }

        .dataset-selector label {
            display: block;
            margin-bottom: 10px;
            cursor: pointer;
        }

        .dataset-selector input[type="radio"] {
            margin-right: 8px;
        }

        #metric-selector {
            width: 100%;
            padding: 10px;
            font-size: 1em;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
        }

        .code-selector-grid {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 15px;
            align-items: center;
        }

        .code-dropdown-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .code-dropdown-group label {
            font-weight: 600;
            color: #666;
            font-size: 0.95em;
        }

        .code-dropdown {
            width: 100%;
            padding: 10px;
            font-size: 1em;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            cursor: pointer;
        }

        .code-dropdown:hover {
            border-color: #0d6efd;
        }

        .code-dropdown:focus {
            outline: none;
            border-color: #0d6efd;
            box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.1);
        }

        .vs-separator {
            font-size: 1.2em;
            font-weight: bold;
            color: #0d6efd;
            text-align: center;
        }

        .periodic-table-container {
            position: relative;
            margin: 30px 0;
            overflow-x: auto;
        }

        .periodic-table {
            display: grid;
            grid-template-columns: repeat(18, 70px);
            grid-template-rows: repeat(7, 65px) 15px repeat(2, 65px);
            gap: 2px;
            margin: 0 auto;
            width: fit-content;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .element-cell {
            position: relative;
            border: 1px solid #999;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            user-select: none;
            transition: all 0.2s;
            height: 65px;
        }

        .element-cell:hover:not(.disabled) {
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            z-index: 10;
        }

        .element-cell.disabled {
            background: #e0e0e0;
            cursor: not-allowed;
            opacity: 0.5;
        }

        .element-symbol {
            position: absolute;
            top: 1px;
            left: 3px;
            font-size: 10px;
            font-weight: bold;
            color: #333;
            z-index: 1;
            pointer-events: none;
        }

        .element-cell-inner {
            display: grid;
            width: 100%;
            height: 100%;
            padding-top: 12px;
        }

        .element-cell-inner.unaries {
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
        }

        .element-cell-inner.oxides {
            grid-template-columns: 1fr 1fr 1fr;
            grid-template-rows: 1fr 1fr;
        }

        .config-subcell {
            border: 0.5px solid #ddd;
            background: #fafafa;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 700;
            color: #000;
            text-shadow: 0 0 3px rgba(255,255,255,0.9);
            overflow: hidden;
        }

        .config-subcell.no-data {
            background: #f5f5f5;
            color: #999;
        }

        .colorbar-container {
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .colorbar-legend {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }

        .colorbar-gradient {
            width: 400px;
            height: 30px;
            border-radius: 4px;
        }

        .colorbar-labels {
            display: flex;
            justify-content: space-between;
            width: 400px;
            font-size: 0.9em;
            color: #666;
        }

        .matrix-container,
        .element-detail-container {
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .matrix-container h3,
        .element-detail-container h3 {
            margin-bottom: 20px;
            color: #0d6efd;
        }

        .close-button {
            float: right;
            padding: 8px 16px;
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }

        .close-button:hover {
            background: #bb2d3b;
        }

        .eos-panel-container {
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .eos-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #0d6efd;
        }

        .eos-panel-header h3 {
            margin: 0;
            color: #0d6efd;
        }

        .eos-config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 25px;
        }

        .eos-config-section {
            background: #f8f9fa;
            border-radius: 6px;
            padding: 15px;
            border: 1px solid #dee2e6;
        }

        .eos-config-header {
            font-weight: 600;
            color: #0d6efd;
            margin-bottom: 12px;
            text-align: center;
            font-size: 1.1em;
        }

        .eos-plot-canvas {
            width: 100%;
        }

        footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }

        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    '''

def get_javascript_code(embedded_data):
    """Return JavaScript code for the HTML."""
    return f'''
        // Embedded comparison data
        const COMPARISON_DATA = {embedded_data};

        // Code colors
        const CODE_COLORS = {json.dumps(CODE_COLORS)};

        // Application state
        const appState = {{
            dataset: 'unaries',
            metric: 'epsilon',
            codeA: null,
            codeB: null,
            selectedElement: null,
            chartInstances: {{}},
            eosChartInstances: {{}}  // Store EOS chart instances
        }};

        // Configuration mappings
        const CONFIGS = {{
            unaries: ['X/SC', 'X/BCC', 'X/Diamond', 'X/FCC'],
            oxides: ['X2O3', 'X2O5', 'X2O', 'XO', 'XO2', 'XO3']
        }};

        // Metric thresholds for color mapping
        const METRIC_THRESHOLDS = {{
            epsilon: {{ excellent: 0.06, good: 0.20, outlier: 1.0 }},
            nu: {{ excellent: 0.10, good: 0.33, outlier: 1.65 }},
            delta: {{ max: 5.0 }},
            V0_rel_diff: {{ range: 10.0 }},
            B0_rel_diff: {{ range: 10.0 }},
            B1_rel_diff: {{ range: 10.0 }}
        }};

        // Color mapping functions
        function hexToRgb(hex) {{
            const result = /^#?([a-f\\d]{{2}})([a-f\\d]{{2}})([a-f\\d]{{2}})$/i.exec(hex);
            return result ? {{
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16)
            }} : null;
        }}

        function rgbToHex(r, g, b) {{
            return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
        }}

        function interpolateColor(color1, color2, factor) {{
            const c1 = hexToRgb(color1);
            const c2 = hexToRgb(color2);
            const r = Math.round(c1.r + (c2.r - c1.r) * factor);
            const g = Math.round(c1.g + (c2.g - c1.g) * factor);
            const b = Math.round(c1.b + (c2.b - c1.b) * factor);
            return rgbToHex(r, g, b);
        }}

        function getComparisonColor(value, metric) {{
            if (isNaN(value) || value === null) {{
                return '#fafafa';  // No data
            }}

            value = Math.abs(value);  // Use absolute value for coloring

            const thresholds = METRIC_THRESHOLDS[metric];

            if (metric === 'epsilon' || metric === 'nu') {{
                // Quality colormap
                if (value <= thresholds.excellent) {{
                    return '#555998';  // Excellent - dark blue
                }} else if (value <= thresholds.good) {{
                    // Excellent to good: blue to yellow
                    const factor = (value - thresholds.excellent) / (thresholds.good - thresholds.excellent);
                    return interpolateColor('#6B71AD', '#EEE992', factor);
                }} else if (value <= thresholds.outlier) {{
                    // Good to outlier: yellow to red
                    const factor = (value - thresholds.good) / (thresholds.outlier - thresholds.good);
                    return interpolateColor('#EEE992', '#f53216', factor);
                }} else {{
                    return '#bf0000';  // Outlier - dark red
                }}
            }} else if (metric === 'delta') {{
                // White to red gradient
                const intensity = Math.min(value / thresholds.max, 1.0);
                const r = 255;
                const g = Math.floor(255 * (1 - intensity));
                const b = Math.floor(255 * (1 - intensity));
                return rgbToHex(r, g, b);
            }} else {{
                // Diverging colormap for rel_diff metrics
                const normalized = value / thresholds.range;  // 0 to 1 (for ±range%)
                if (normalized > 1) {{
                    return '#b2182b';  // Dark red for > range
                }}
                return interpolateColor('#ffffff', '#b2182b', normalized);
            }}
        }}

        // Data access functions
        function getComparisonValue(element, config, codeA, codeB, metric) {{
            if (!codeA || !codeB) return null;

            // Both unaries and oxides use "Element-Config" format
            const key = `${{element}}-${{config}}`;

            // Check if data exists
            if (!COMPARISON_DATA[appState.dataset]) {{
                console.error('No data for dataset:', appState.dataset);
                return null;
            }}

            const data = COMPARISON_DATA[appState.dataset][key];

            if (!data) return null;

            // Get indices of the two codes
            const indexA = data.codes.indexOf(codeA);
            const indexB = data.codes.indexOf(codeB);

            if (indexA < 0 || indexB < 0) return null;

            // Get the pairwise comparison value
            const value = data.comparisons[metric][indexA][indexB];

            return isNaN(value) ? null : Math.abs(value);
        }}

        // Rendering functions
        function updatePeriodicTable() {{
            const configs = CONFIGS[appState.dataset];
            const cells = document.querySelectorAll('.element-cell:not(.disabled)');

            cells.forEach(cell => {{
                const element = cell.dataset.element;
                const subcells = cell.querySelectorAll('.config-subcell');

                subcells.forEach((subcell, idx) => {{
                    if (idx >= configs.length) return;

                    const config = configs[idx];
                    const value = getComparisonValue(element, config, appState.codeA, appState.codeB, appState.metric);

                    if (value === null) {{
                        subcell.style.backgroundColor = '#fafafa';
                        subcell.classList.add('no-data');
                        subcell.textContent = '—';
                    }} else {{
                        subcell.style.backgroundColor = getComparisonColor(value, appState.metric);
                        subcell.classList.remove('no-data');

                        // Format value based on metric
                        let displayValue;
                        if (appState.metric === 'delta') {{
                            displayValue = value.toFixed(1);  // meV/atom
                        }} else if (appState.metric === 'epsilon') {{
                            displayValue = value.toFixed(2);  // unitless
                        }} else {{
                            displayValue = value.toFixed(1);  // percentage or nu
                        }}
                        subcell.textContent = displayValue;
                    }}
                }});
            }});
        }}

        function updateColorbar() {{
            const container = document.getElementById('colorbar-container');
            const metric = appState.metric;
            const thresholds = METRIC_THRESHOLDS[metric];

            let html = '<h3 style="margin-bottom: 15px; color: #0d6efd;">Comparing: ';
            html += `<span style="color: #198754;">${{appState.codeA}}</span> vs `;
            html += `<span style="color: #198754;">${{appState.codeB}}</span></h3>`;
            html += '<div class="colorbar-legend">';

            if (metric === 'epsilon' || metric === 'nu') {{
                // Quality colormap
                html += '<div class="colorbar-gradient" style="background: linear-gradient(to right, #555998, #6B71AD, #EEE992, #f53216, #bf0000);"></div>';
                html += '<div class="colorbar-labels">';
                html += `<span>Excellent (≤${{thresholds.excellent}})</span>`;
                html += `<span>Good (≤${{thresholds.good}})</span>`;
                html += `<span>Outlier (>${{thresholds.outlier}})</span>`;
                html += '</div>';
            }} else if (metric === 'delta') {{
                html += '<div class="colorbar-gradient" style="background: linear-gradient(to right, #ffffff, #ff0000);"></div>';
                html += '<div class="colorbar-labels">';
                html += `<span>0</span>`;
                html += `<span>${{thresholds.max}} meV/atom</span>`;
                html += '</div>';
            }} else {{
                html += '<div class="colorbar-gradient" style="background: linear-gradient(to right, #ffffff, #b2182b);"></div>';
                html += '<div class="colorbar-labels">';
                html += `<span>0%</span>`;
                html += `<span>±${{thresholds.range}}%</span>`;
                html += '</div>';
            }}

            html += '</div>';
            container.innerHTML = html;
        }}

        function renderAll() {{
            updatePeriodicTable();
            updateColorbar();
        }}

        // Event handlers
        function onDatasetChange(dataset) {{
            appState.dataset = dataset;
            document.getElementById('periodic-table-unaries').style.display = dataset === 'unaries' ? 'grid' : 'none';
            document.getElementById('periodic-table-oxides').style.display = dataset === 'oxides' ? 'grid' : 'none';
            renderAll();
        }}

        function onMetricChange(metric) {{
            appState.metric = metric;
            renderAll();
        }}

        function onCodeAChange(code) {{
            appState.codeA = code;
            renderAll();
        }}

        function onCodeBChange(code) {{
            appState.codeB = code;
            renderAll();
        }}

        function onElementClick(element) {{
            appState.selectedElement = element;
            showEOSPanel(element);
        }}

        function showEOSPanel(element) {{
            const panel = document.getElementById('eos-panel-container');
            const elementName = document.getElementById('eos-element-name');
            const configGrid = document.getElementById('eos-config-grid');

            // Check if both codes are selected
            if (!appState.codeA || !appState.codeB) {{
                console.warn('No codes selected for comparison');
                return;
            }}

            // Update title
            elementName.textContent = element;

            // Destroy previous charts
            Object.values(appState.eosChartInstances).forEach(chart => chart.destroy());
            appState.eosChartInstances = {{}};

            // Get configurations for current dataset
            const configs = CONFIGS[appState.dataset];

            // Clear grid and populate with EOS plots
            configGrid.innerHTML = '';

            // Check if EOS data exists for this dataset
            if (!COMPARISON_DATA.eos_data || !COMPARISON_DATA.eos_data[appState.dataset]) {{
                console.error('No EOS data available for dataset:', appState.dataset);
                return;
            }}

            let plotCount = 0;
            configs.forEach(config => {{
                // Both unaries and oxides use "Element-Config" format
                const key = `${{element}}-${{config}}`;
                const eosData = COMPARISON_DATA.eos_data[appState.dataset][key];

                if (!eosData || (!eosData[appState.codeA] && !eosData[appState.codeB])) {{
                    return;  // Skip if no data for either code
                }}

                // Create section for this configuration
                const section = document.createElement('div');
                section.className = 'eos-config-section';

                const header = document.createElement('div');
                header.className = 'eos-config-header';
                header.textContent = `${{element}} (${{config}})`;
                section.appendChild(header);

                const canvas = document.createElement('canvas');
                canvas.className = 'eos-plot-canvas';
                canvas.id = `eos-canvas-${{plotCount}}`;
                section.appendChild(canvas);

                configGrid.appendChild(section);

                // Render EOS plot
                renderEOSPlot(canvas.id, eosData, appState.codeA, appState.codeB);
                plotCount++;
            }});

            // Show panel if we have at least one plot
            if (plotCount > 0) {{
                panel.style.display = 'block';
                // Scroll to panel smoothly
                setTimeout(() => {{
                    panel.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}, 100);
            }} else {{
                console.warn('No EOS plots to display for element:', element);
            }}
        }}

        function renderEOSPlot(canvasId, eosData, codeA, codeB) {{
            const ctx = document.getElementById(canvasId);
            if (!ctx) return;

            const datasets = [];

            // Add datasets for Code A
            if (eosData[codeA]) {{
                const colorA = CODE_COLORS[0];  // First color for Code A

                // Raw data points
                datasets.push({{
                    label: codeA,
                    data: eosData[codeA].volumes.map((v, i) => ({{
                        x: v,
                        y: eosData[codeA].energies[i]
                    }})),
                    backgroundColor: colorA,
                    borderColor: colorA,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    showLine: false
                }});

                // Fitted curve
                datasets.push({{
                    label: `${{codeA}}_fit_hidden`,
                    data: eosData[codeA].fit.volumes_dense.map((v, i) => ({{
                        x: v,
                        y: eosData[codeA].fit.energies_dense[i]
                    }})),
                    borderColor: colorA,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    showLine: true,
                    fill: false
                }});
            }}

            // Add datasets for Code B
            if (eosData[codeB]) {{
                const colorB = CODE_COLORS[2];  // Different color for Code B

                // Raw data points
                datasets.push({{
                    label: codeB,
                    data: eosData[codeB].volumes.map((v, i) => ({{
                        x: v,
                        y: eosData[codeB].energies[i]
                    }})),
                    backgroundColor: colorB,
                    borderColor: colorB,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    showLine: false
                }});

                // Fitted curve
                datasets.push({{
                    label: `${{codeB}}_fit_hidden`,
                    data: eosData[codeB].fit.volumes_dense.map((v, i) => ({{
                        x: v,
                        y: eosData[codeB].fit.energies_dense[i]
                    }})),
                    borderColor: colorB,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    showLine: true,
                    fill: false
                }});
            }}

            // Create Chart.js chart
            const chart = new Chart(ctx, {{
                type: 'scatter',
                data: {{ datasets: datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 1.5,  // Width:Height = 1.5:1
                    plugins: {{
                        legend: {{
                            position: 'top',
                            labels: {{
                                filter: function(item) {{
                                    return !item.text.includes('_fit_hidden');
                                }}
                            }}
                        }},
                        title: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'linear',
                            title: {{
                                display: true,
                                text: 'Volume per formula unit (Å³)'
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'E-E₀ per formula unit (eV)'
                            }}
                        }}
                    }}
                }}
            }});

            appState.eosChartInstances[canvasId] = chart;
        }}

        function closeEOSPanel() {{
            document.getElementById('eos-panel-container').style.display = 'none';

            // Destroy all charts to free memory
            Object.values(appState.eosChartInstances).forEach(chart => chart.destroy());
            appState.eosChartInstances = {{}};
        }}

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', () => {{
            // Initialize selected codes from dropdowns
            const codeASelector = document.getElementById('code-a-selector');
            const codeBSelector = document.getElementById('code-b-selector');

            appState.codeA = codeASelector.value;
            appState.codeB = codeBSelector.value;

            // Attach event listeners
            document.querySelectorAll('input[name="dataset"]').forEach(radio => {{
                radio.addEventListener('change', (e) => onDatasetChange(e.target.value));
            }});

            document.getElementById('metric-selector').addEventListener('change', (e) => {{
                onMetricChange(e.target.value);
            }});

            codeASelector.addEventListener('change', (e) => {{
                onCodeAChange(e.target.value);
            }});

            codeBSelector.addEventListener('change', (e) => {{
                onCodeBChange(e.target.value);
            }});

            // Initial render
            onDatasetChange('unaries');
        }});
    '''

# ===== ARGUMENT PARSING =====

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML periodic table comparison viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --oxides-codes abacus vasp --unaries-codes abacus vasp fleur
  %(prog)s --oxides-codes abacus_sg15 vasp --unaries-codes abacus_sg15 vasp --output comparison.html
        '''
    )
    parser.add_argument('--oxides-codes', nargs='+', default=[],
                       help='Code names for oxides dataset (space-separated)')
    parser.add_argument('--unaries-codes', nargs='+', default=[],
                       help='Code names for unaries dataset (space-separated)')
    parser.add_argument('--output', default='periodic_table_comparison.html',
                       help='Output HTML filename (default: periodic_table_comparison.html)')
    parser.add_argument('--results-dir', default='.',
                       help='Directory containing results JSON files (default: current directory)')
    return parser.parse_args()

# ===== MAIN =====

def main():
    """Main entry point."""
    args = parse_arguments()

    if not args.oxides_codes and not args.unaries_codes:
        print("ERROR: Must specify at least --oxides-codes or --unaries-codes", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from: {args.results_dir}")

    # Load results data
    results_dicts = {}
    codes_dict = {}

    if args.oxides_codes:
        print(f"Loading oxides data for codes: {', '.join(args.oxides_codes)}")
        results_dicts['oxides'] = load_results_data('oxides-verification-PBE-v1', args.oxides_codes, args.results_dir)
        codes_dict['oxides'] = list(results_dicts['oxides'].keys())

    if args.unaries_codes:
        print(f"Loading unaries data for codes: {', '.join(args.unaries_codes)}")
        results_dicts['unaries'] = load_results_data('unaries-verification-PBE-v1', args.unaries_codes, args.results_dir)
        codes_dict['unaries'] = list(results_dicts['unaries'].keys())

    # Prepare comparison data
    print("Computing pairwise comparisons...")
    comparison_data = prepare_comparison_data(results_dicts, codes_dict)

    # Generate HTML
    print(f"Generating HTML: {args.output}")
    generate_html(comparison_data, args.output)

    print("Done!")

if __name__ == '__main__':
    main()
