#!/usr/bin/env python3
"""
ChemDB Search - Chemical database similarity search tool.

Search Enamine REAL Space database for structurally similar molecules
using Tanimoto similarity coefficients.
"""

import click
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

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
    '--exact-substructure-match', '-esub',
    is_flag=True,
    help='Only return results that contain the query as an exact substructure (preserves atom types and bond orders)'
)
@click.option(
    '--substructure-file', '-sf',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Separate SDF/PDB file to use as the substructure filter (instead of the query molecule)'
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
    exact_substructure_match: bool,
    substructure_file: Optional[Path],
    verbose: bool
):
    """
    Search Enamine REAL Space database for structurally similar molecules.

    INPUT_FILE: Path to SDF or PDB file containing one or more query molecules.
    Multi-molecule SDF files are supported for batch processing.

    Example:
        python search.py search aspirin.sdf -t 0.8 -n 50
        python search.py search multi_query.sdf -v
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
    if substructure_file and (substructure_match or exact_substructure_match):
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

    # 3. Search for each query molecule
    all_results = []
    query_summaries = []

    for qi, (mol, smiles) in enumerate(queries):
        click.echo(f"\n--- Query {qi + 1}/{len(queries)}: {smiles} ---")

        results = client.similarity_search(
            smiles=smiles,
            threshold=threshold,
            max_results=max_results,
            database=database
        )

        if not results:
            click.echo(f"  No similar molecules found")
            query_summaries.append({'index': qi, 'smiles': smiles, 'num_results': 0})
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
        elif substructure_match:
            pre = len(results)
            results = [
                r for r in results
                if FingerprintEngine.has_generic_substructure_match(filter_mol, r['smiles'])
            ]
            click.echo(f"  Generic substructure filter: {len(results)}/{pre}")

        # Tag results with query info
        for r in results:
            r['query_index'] = qi
            r['query_smiles'] = smiles

        click.echo(f"  Found {len(results)} results")
        query_summaries.append({'index': qi, 'smiles': smiles, 'num_results': len(results)})
        all_results.extend(results)

    if not all_results:
        click.echo("\nNo results found for any query.")
        sys.exit(0)

    click.echo(f"\n{'='*60}")
    click.echo(f"Total: {len(all_results)} results from {len(queries)} query molecule(s)")

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

    click.echo(f"\nSearch completed successfully!")


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
