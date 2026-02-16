#!/usr/bin/env python3
"""
ChemDB Search - Chemical database similarity search tool.

Search Enamine REAL Space database for structurally similar molecules
using Tanimoto similarity coefficients.
"""

import click
import logging
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from datetime import datetime

from rdkit.Chem import Descriptors

from config import settings
from molecule_io import MoleculeReader, MoleculeWriter
from api_clients.smallworld_client import SmallWorldClient
from fingerprint_engine import FingerprintEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """ChemDB Search - Chemical database similarity search tool."""
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--threshold', '-t',
    type=click.FloatRange(0.0, 1.0),
    default=settings.default_similarity_threshold,
    help='Tanimoto similarity threshold (0.0-1.0)'
)
@click.option(
    '--max-results', '-n',
    type=int,
    default=settings.default_max_results,
    help='Maximum number of results to return'
)
@click.option(
    '--database', '-d',
    type=str,
    default=settings.default_database,
    help='Enamine REAL database to search'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default=settings.output_dir,
    help='Output directory for results'
)
@click.option(
    '--output-format',
    type=click.Choice(['sdf', 'json', 'both']),
    default='both',
    help='Output format(s)'
)
@click.option(
    '--fingerprint-type',
    type=click.Choice(['morgan', 'rdkit', 'maccs', 'atompair']),
    default='morgan',
    help='Fingerprint type for local validation (info only)'
)
@click.option(
    '--substructure-match', '-sub',
    is_flag=True,
    help='Only return results whose heavy-atom bond graph contains the query as a substructure (ignores atom types and bond orders)'
)
@click.option(
    '--atom-substructure-match', '-asub',
    is_flag=True,
    help='Only return results that match the query atom types as a substructure (ignores bond orders)'
)
@click.option(
    '--exact-substructure-match', '-esub',
    is_flag=True,
    help='Only return results that contain the query as an exact substructure (preserves atom types and bond orders)'
)
@click.option(
    '--substructure-file', '-sf',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Separate SDF/PDB/SMI file to use as the substructure filter (instead of the query molecule)'
)
@click.option(
    '--png',
    is_flag=True,
    help='Write 2D depiction PNGs into a subfolder'
)
@click.option(
    '--svg',
    is_flag=True,
    help='Write 2D depiction SVGs into a subfolder'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Verbose output'
)
def search(
    input_file: Path,
    threshold: float,
    max_results: int,
    database: str,
    output_dir: Path,
    output_format: str,
    fingerprint_type: str,
    substructure_match: bool,
    atom_substructure_match: bool,
    exact_substructure_match: bool,
    substructure_file: Optional[Path],
    png: bool,
    svg: bool,
    verbose: bool
):
    """
    Search Enamine REAL Space database for structurally similar molecules.

    INPUT_FILE: Path to SDF, PDB, or SMI file containing one or more query molecules.
    Multi-molecule SDF and SMI files are supported for batch processing.

    Example:
        python search.py search aspirin.sdf -t 0.8 -n 50
        python search.py search queries.smi -v
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo(f"ChemDB Search - Enamine REAL Space Similarity Search")
    click.echo(f"{'='*60}")

    # 1. Read input molecules
    click.echo(f"\nReading query molecules from: {input_file}")
    try:
        queries = MoleculeReader.read_molecules(input_file)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not queries:
        click.echo("Error: No valid molecules found in input file", err=True)
        sys.exit(1)

    click.echo(f"Found {len(queries)} query molecule(s)")

    # 1b. Load separate substructure filter molecule if provided
    sub_mol = None
    if substructure_file and (substructure_match or atom_substructure_match or exact_substructure_match):
        try:
            sub_mol, sub_smiles = MoleculeReader.read_molecule(substructure_file)
        except ValueError as e:
            click.echo(f"Error reading substructure file: {e}", err=True)
            sys.exit(1)
        if sub_mol is None:
            click.echo("Error: Failed to read molecule from substructure file", err=True)
            sys.exit(1)
        click.echo(f"Substructure filter molecule: {sub_smiles}")

    # 2. Initialize API client
    client = SmallWorldClient()

    if not client.is_available():
        click.echo("Warning: Cannot connect to SmallWorld API", err=True)
        click.echo("Continuing anyway...", err=True)

    click.echo(f"\nSearching database: {database}")
    click.echo(f"Similarity threshold: {threshold}")
    click.echo(f"Maximum results per query: {max_results}")

    # 3. Search for each query molecule (parallel API calls)
    all_results = []
    query_summaries = [None] * len(queries)
    per_query_results = [None] * len(queries)

    def _search_one(qi, smiles, retries=3, delay=5):
        for attempt in range(retries):
            try:
                return qi, client.similarity_search(
                    smiles=smiles,
                    threshold=threshold,
                    max_results=max_results,
                    database=database
                )
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"Query {qi} failed (attempt {attempt + 1}/{retries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Query {qi} failed after {retries} attempts: {e}")
                    raise

    workers = min(len(queries), 24)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_search_one, qi, smiles): (qi, mol, smiles)
            for qi, (mol, smiles) in enumerate(queries)
        }
        for future in as_completed(futures):
            qi, mol, smiles = futures[future]
            try:
                results = future.result()[1]
            except Exception as e:
                click.echo(f"\n--- Query {qi + 1}/{len(queries)}: {smiles} ---")
                click.echo(f"  Error: {e}", err=True)
                query_summaries[qi] = {'index': qi, 'smiles': smiles, 'num_results': 0, 'error': str(e)}
                per_query_results[qi] = []
                continue

            click.echo(f"\n--- Query {qi + 1}/{len(queries)}: {smiles} ---")

            if not results:
                click.echo(f"  No similar molecules found")
                query_summaries[qi] = {'index': qi, 'smiles': smiles, 'num_results': 0}
                per_query_results[qi] = []
                continue

            # Filter by substructure match if requested
            filter_mol = sub_mol if sub_mol is not None else mol
            if exact_substructure_match:
                pre = len(results)
                results = [
                    r for r in results
                    if FingerprintEngine.has_exact_substructure_match(filter_mol, r['smiles'])
                ]
                click.echo(f"  Exact substructure filter: {len(results)}/{pre}")
            elif atom_substructure_match:
                pre = len(results)
                results = [
                    r for r in results
                    if FingerprintEngine.has_atom_substructure_match(filter_mol, r['smiles'])
                ]
                click.echo(f"  Atom-type substructure filter: {len(results)}/{pre}")
            elif substructure_match:
                pre = len(results)
                results = [
                    r for r in results
                    if FingerprintEngine.has_generic_substructure_match(filter_mol, r['smiles'])
                ]
                click.echo(f"  Generic substructure filter: {len(results)}/{pre}")

            for r in results:
                r['query_index'] = qi
                r['query_smiles'] = smiles

            click.echo(f"  Found {len(results)} results")
            query_summaries[qi] = {'index': qi, 'smiles': smiles, 'num_results': len(results)}
            per_query_results[qi] = results

    # Combine results in query order and deduplicate by SMILES
    for results in per_query_results:
        if results:
            all_results.extend(results)

    if not all_results:
        click.echo("\nNo results found for any query.")
        sys.exit(0)

    # Deduplicate: keep the entry with the highest similarity for each SMILES
    seen = {}
    for r in all_results:
        smi = r['smiles']
        if smi not in seen or r.get('similarity', 0) > seen[smi].get('similarity', 0):
            seen[smi] = r
    pre_dedup = len(all_results)
    all_results = sorted(seen.values(), key=lambda r: r.get('similarity', 0), reverse=True)

    click.echo(f"\n{'='*60}")
    click.echo(f"Total: {len(all_results)} unique results from {len(queries)} query molecule(s)")
    if pre_dedup > len(all_results):
        click.echo(f"  ({pre_dedup - len(all_results)} duplicates removed)")

    # 4. Display top results
    click.echo("\nTop 5 results:")
    for i, result in enumerate(all_results[:5], 1):
        sim = result.get('similarity', 0)
        smiles_str = result.get('smiles', '')[:60]
        mol_id = result.get('id', 'N/A')
        qi = result.get('query_index', 0)
        click.echo(f"  {i}. Similarity: {sim:.3f} | ID: {mol_id} | Query: {qi}")
        click.echo(f"     SMILES: {smiles_str}")

    # 5. Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"search_results_{timestamp}"

    if output_format in ['sdf', 'both']:
        sdf_path = output_dir / f"{base_name}.sdf"
        MoleculeWriter.write_sdf(all_results, sdf_path)
        click.echo(f"\nSDF output: {sdf_path}")

    if output_format in ['json', 'both']:
        json_path = output_dir / f"{base_name}.json"

        json_output = {
            'queries': query_summaries,
            'search_params': {
                'threshold': threshold,
                'max_results': max_results,
                'database': database,
                'fingerprint_type': fingerprint_type
            },
            'input_file': str(input_file),
            'timestamp': timestamp,
            'num_queries': len(queries),
            'total_results': len(all_results),
            'results': all_results
        }

        MoleculeWriter.write_json(json_output, json_path)
        click.echo(f"JSON output: {json_path}")

    if png:
        png_dir = output_dir / f"{base_name}_png"
        MoleculeWriter.write_png(all_results, png_dir)
        click.echo(f"PNG output: {png_dir}")

    if svg:
        svg_dir = output_dir / f"{base_name}_svg"
        MoleculeWriter.write_svg(all_results, svg_dir)
        click.echo(f"SVG output: {svg_dir}")

    click.echo(f"\nSearch completed successfully!")


@cli.command()
@click.argument('results_file', type=click.Path(exists=True, path_type=Path))
@click.argument('substructure_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--substructure-match', '-sub',
    is_flag=True,
    help='Filter by generic substructure (ignores atom types and bond orders)'
)
@click.option(
    '--atom-substructure-match', '-asub',
    is_flag=True,
    help='Filter by atom-type substructure (matches atom types, ignores bond orders)'
)
@click.option(
    '--exact-substructure-match', '-esub',
    is_flag=True,
    help='Filter by exact substructure (preserves atom types and bond orders)'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default=settings.output_dir,
    help='Output directory for results'
)
@click.option(
    '--output-format',
    type=click.Choice(['sdf', 'json', 'both']),
    default='both',
    help='Output format(s)'
)
@click.option(
    '--png',
    is_flag=True,
    help='Write 2D depiction PNGs into a subfolder'
)
@click.option(
    '--svg',
    is_flag=True,
    help='Write 2D depiction SVGs into a subfolder'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Verbose output'
)
def filter(
    results_file: Path,
    substructure_file: Path,
    substructure_match: bool,
    atom_substructure_match: bool,
    exact_substructure_match: bool,
    output_dir: Path,
    output_format: str,
    png: bool,
    svg: bool,
    verbose: bool
):
    """
    Filter an existing molecule file by substructure match.

    RESULTS_FILE: SDF/SMI file containing molecules to filter.
    SUBSTRUCTURE_FILE: SDF/PDB/SMI file with the substructure query molecule.

    Requires one of -sub, -asub, or -esub.

    Example:
        python search.py filter results.sdf fragment.sdf -sub
        python search.py filter results.smi query.sdf -asub
        python search.py filter results.sdf query.sdf -esub -o ./filtered
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not (substructure_match or atom_substructure_match or exact_substructure_match):
        click.echo("Error: Specify a filter mode: -sub, -asub, or -esub", err=True)
        sys.exit(1)

    click.echo(f"ChemDB Search - Substructure Filter")
    click.echo(f"{'='*60}")

    # Load substructure query molecule
    try:
        sub_mol, sub_smiles = MoleculeReader.read_molecule(substructure_file)
    except ValueError as e:
        click.echo(f"Error reading substructure file: {e}", err=True)
        sys.exit(1)
    if sub_mol is None:
        click.echo("Error: Failed to read molecule from substructure file", err=True)
        sys.exit(1)
    click.echo(f"Substructure query: {sub_smiles}")

    # Select filter function
    if exact_substructure_match:
        filter_func = FingerprintEngine.has_exact_substructure_match
        filter_name = "Exact substructure"
    elif atom_substructure_match:
        filter_func = FingerprintEngine.has_atom_substructure_match
        filter_name = "Atom-type substructure"
    else:
        filter_func = FingerprintEngine.has_generic_substructure_match
        filter_name = "Generic substructure"

    # Stream molecules: deduplicate and filter in one pass (memory-efficient)
    click.echo(f"Streaming molecules from: {results_file}")
    seen = set()
    passed = []
    total = 0
    duplicates = 0

    for _, smiles in MoleculeReader.iter_molecules(results_file):
        total += 1
        if total % 1_000_000 == 0:
            click.echo(f"  processed {total:,} molecules ({len(passed):,} passed so far)...")
        if smiles in seen:
            duplicates += 1
            continue
        seen.add(smiles)
        if filter_func(sub_mol, smiles):
            passed.append({'smiles': smiles})

    if total == 0:
        click.echo("Error: No valid molecules found in results file", err=True)
        sys.exit(1)

    unique = total - duplicates
    click.echo(f"\nProcessed {total:,} molecules ({unique:,} unique)")
    if duplicates:
        click.echo(f"  ({duplicates:,} duplicates removed)")
    click.echo(f"{filter_name} filter: {len(passed):,}/{unique:,} passed")

    if not passed:
        click.echo("No molecules passed the filter.")
        sys.exit(0)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"filtered_{timestamp}"

    if output_format in ['sdf', 'both']:
        sdf_path = output_dir / f"{base_name}.sdf"
        MoleculeWriter.write_sdf(passed, sdf_path)
        click.echo(f"SDF output: {sdf_path}")

    if output_format in ['json', 'both']:
        json_path = output_dir / f"{base_name}.json"
        json_output = {
            'substructure_query': sub_smiles,
            'filter_mode': filter_name,
            'input_file': str(results_file),
            'timestamp': timestamp,
            'total_input': unique,
            'total_passed': len(passed),
            'results': passed
        }
        MoleculeWriter.write_json(json_output, json_path)
        click.echo(f"JSON output: {json_path}")

    if png:
        png_dir = output_dir / f"{base_name}_png"
        MoleculeWriter.write_png(passed, png_dir)
        click.echo(f"PNG output: {png_dir}")

    if svg:
        svg_dir = output_dir / f"{base_name}_svg"
        MoleculeWriter.write_svg(passed, svg_dir)
        click.echo(f"SVG output: {svg_dir}")

    click.echo(f"\nFiltering completed successfully!")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--n-clusters', '-n',
    type=int,
    default=100,
    help='Number of clusters (default: 100)'
)
@click.option(
    '--substructure-match', '-sub',
    is_flag=True,
    help='Only cluster molecules whose heavy-atom bond graph contains the substructure query'
)
@click.option(
    '--atom-substructure-match', '-asub',
    is_flag=True,
    help='Only cluster molecules that match the substructure query atom types'
)
@click.option(
    '--exact-substructure-match', '-esub',
    is_flag=True,
    help='Only cluster molecules that contain the query as an exact substructure'
)
@click.option(
    '--substructure-file', '-sf',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='SDF/PDB/SMI file with the substructure query molecule (required with -sub/-asub/-esub)'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default=settings.output_dir,
    help='Output directory for results'
)
@click.option(
    '--output-format',
    type=click.Choice(['sdf', 'json', 'both']),
    default='both',
    help='Output format(s)'
)
@click.option(
    '--png',
    is_flag=True,
    help='Write 2D depiction PNGs into a subfolder'
)
@click.option(
    '--svg',
    is_flag=True,
    help='Write 2D depiction SVGs into a subfolder'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Verbose output'
)
def cluster(
    input_file: Path,
    n_clusters: int,
    substructure_match: bool,
    atom_substructure_match: bool,
    exact_substructure_match: bool,
    substructure_file: Optional[Path],
    output_dir: Path,
    output_format: str,
    png: bool,
    svg: bool,
    verbose: bool
):
    """
    Cluster molecules by ECFP4 Tanimoto similarity and output medoids.

    INPUT_FILE: SDF or SMI file containing molecules to cluster.

    Optionally pre-filter molecules by substructure before clustering
    using -sub/-asub/-esub with -sf.

    Uses ECFP4 fingerprints (Morgan radius=2, 2048 bits) with MaxMin
    diversity picking to select N maximally diverse representatives
    using Tanimoto distance, then assigns remaining molecules to
    the nearest medoid to form clusters.

    Example:
        python search.py cluster results.sdf -n 50
        python search.py cluster molecules.smi -n 200 -esub -sf fragment.sdf
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo(f"ChemDB Search - Molecular Clustering")
    click.echo(f"{'='*60}")

    # Load substructure filter if requested
    use_filter = substructure_match or atom_substructure_match or exact_substructure_match
    if use_filter:
        if not substructure_file:
            click.echo("Error: -sf is required when using -sub, -asub, or -esub", err=True)
            sys.exit(1)
        try:
            sub_mol, sub_smiles = MoleculeReader.read_molecule(substructure_file)
        except ValueError as e:
            click.echo(f"Error reading substructure file: {e}", err=True)
            sys.exit(1)
        if sub_mol is None:
            click.echo("Error: Failed to read molecule from substructure file", err=True)
            sys.exit(1)
        click.echo(f"Substructure query: {sub_smiles}")

        if exact_substructure_match:
            filter_func = FingerprintEngine.has_exact_substructure_match
            filter_name = "Exact substructure"
        elif atom_substructure_match:
            filter_func = FingerprintEngine.has_atom_substructure_match
            filter_name = "Atom-type substructure"
        else:
            filter_func = FingerprintEngine.has_generic_substructure_match
            filter_name = "Generic substructure"

    # Stream, deduplicate, and optionally filter molecules
    click.echo(f"Loading molecules from: {input_file}")
    seen = set()
    molecules = []
    total = 0
    duplicates = 0

    for mol, smiles in MoleculeReader.iter_molecules(input_file):
        total += 1
        if total % 1_000_000 == 0:
            click.echo(f"  processed {total:,} molecules ({len(molecules):,} kept so far)...")
        if smiles in seen:
            duplicates += 1
            continue
        seen.add(smiles)
        if use_filter and not filter_func(sub_mol, smiles):
            continue
        molecules.append((mol, smiles))

    if not molecules:
        click.echo("Error: No valid molecules found (after filtering)", err=True)
        sys.exit(1)

    unique = total - duplicates
    click.echo(f"Processed {total:,} molecules ({unique:,} unique)")
    if duplicates:
        click.echo(f"  ({duplicates:,} duplicates removed)")
    if use_filter:
        click.echo(f"  {filter_name} filter: {len(molecules):,}/{unique:,} passed")

    n_mols = len(molecules)
    if n_clusters >= n_mols:
        click.echo(f"Warning: n_clusters ({n_clusters}) >= molecules ({n_mols}), outputting all molecules")
        n_clusters = n_mols

    click.echo(f"Molecules: {n_mols}")
    click.echo(f"Target clusters: {n_clusters}")
    click.echo(f"Fingerprint: ECFP4 (Morgan radius=2, 2048 bits)")

    # Generate ECFP4 fingerprints
    click.echo("\nGenerating fingerprints...")
    from rdkit import DataStructs
    from rdkit.SimDivFilters import rdSimDivPickers
    fps = []
    for mol, smiles in molecules:
        fp = FingerprintEngine.generate_fingerprint(mol, 'morgan', radius=2, n_bits=2048)
        fps.append(fp)

    # MaxMin diversity picking (maximally diverse medoids using Tanimoto distance)
    click.echo("Selecting diverse medoids (MaxMin picking)...")
    picker = rdSimDivPickers.MaxMinPicker()
    pick_indices = list(picker.LazyBitVectorPick(fps, n_mols, n_clusters))

    # Assign each molecule to its nearest medoid by Tanimoto similarity
    click.echo("Assigning molecules to clusters...")
    picked_fps = [fps[i] for i in pick_indices]
    cluster_sizes = [0] * n_clusters
    for i in range(n_mols):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], picked_fps)
        cluster_sizes[int(np.argmax(sims))] += 1

    medoids = []
    for c, idx in enumerate(pick_indices):
        mol, smiles = molecules[idx]
        mw = Descriptors.ExactMolWt(mol)
        medoids.append({
            'smiles': smiles,
            'cluster': int(c),
            'cluster_size': int(cluster_sizes[c]),
            'molecular_weight': round(mw, 2),
        })

    # Sort by molecular weight ascending
    medoids.sort(key=lambda x: x['molecular_weight'])

    click.echo(f"\n{'='*60}")
    click.echo(f"Clustered {n_mols} molecules into {n_clusters} clusters")
    sizes = [m['cluster_size'] for m in medoids]
    click.echo(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, median={sorted(sizes)[len(sizes)//2]}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"cluster_medoids_{timestamp}"

    if output_format in ['sdf', 'both']:
        sdf_path = output_dir / f"{base_name}.sdf"
        MoleculeWriter.write_sdf(medoids, sdf_path)
        click.echo(f"SDF output: {sdf_path}")

    if output_format in ['json', 'both']:
        json_path = output_dir / f"{base_name}.json"
        json_output = {
            'input_file': str(input_file),
            'timestamp': timestamp,
            'n_clusters': n_clusters,
            'n_molecules': n_mols,
            'fingerprint_type': 'ECFP4 (Morgan radius=2, 2048 bits)',
        }
        if use_filter:
            json_output['substructure_query'] = sub_smiles
            json_output['filter_mode'] = filter_name
        json_output['medoids'] = medoids
        MoleculeWriter.write_json(json_output, json_path)
        click.echo(f"JSON output: {json_path}")

    if png:
        png_dir = output_dir / f"{base_name}_png"
        MoleculeWriter.write_png(medoids, png_dir)
        click.echo(f"PNG output: {png_dir}")

    if svg:
        svg_dir = output_dir / f"{base_name}_svg"
        MoleculeWriter.write_svg(medoids, svg_dir)
        click.echo(f"SVG output: {svg_dir}")

    click.echo(f"\nClustering completed successfully!")


@cli.command(name='list-databases')
def list_databases():
    """List available databases."""
    client = SmallWorldClient()
    databases = client.get_available_databases()

    click.echo("Available databases:")
    for db in databases:
        click.echo(f"  - {db}")


@cli.command(name='validate-smiles')
@click.argument('smiles_string')
def validate_smiles(smiles_string: str):
    """Validate a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        click.echo(f"Invalid SMILES: {smiles_string}", err=True)
        sys.exit(1)

    click.echo(f"Valid SMILES: {smiles_string}")
    click.echo(f"Canonical SMILES: {Chem.MolToSmiles(mol)}")
    click.echo(f"Isomeric SMILES: {Chem.MolToSmiles(mol, isomericSmiles=True)}")
    click.echo(f"Molecular Formula: {rdMolDescriptors.CalcMolFormula(mol)}")
    click.echo(f"Molecular Weight: {rdMolDescriptors.CalcExactMolWt(mol):.2f}")
    click.echo(f"Number of Atoms: {mol.GetNumAtoms()}")
    click.echo(f"Number of Bonds: {mol.GetNumBonds()}")


if __name__ == '__main__':
    cli()
