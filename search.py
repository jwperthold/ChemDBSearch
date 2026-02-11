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
    verbose: bool
):
    """
    Search Enamine REAL Space database for structurally similar molecules.

    INPUT_FILE: Path to SDF or PDB file containing query molecule

    Example:
        python search.py aspirin.sdf -t 0.8 -n 50
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo(f"ChemDB Search - Enamine REAL Space Similarity Search")
    click.echo(f"{'='*60}")

    # 1. Read input molecule
    click.echo(f"\nReading query molecule from: {input_file}")
    try:
        mol, smiles = MoleculeReader.read_molecule(input_file)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if mol is None:
        click.echo("Error: Failed to read molecule", err=True)
        sys.exit(1)

    click.echo(f"Query SMILES: {smiles}")

    # 2. Initialize API client
    client = SmallWorldClient()

    # Check API availability
    if not client.is_available():
        click.echo("Warning: Cannot connect to SmallWorld API", err=True)
        click.echo("Continuing anyway...", err=True)

    # 3. Perform similarity search
    click.echo(f"\nSearching database: {database}")
    click.echo(f"Similarity threshold: {threshold}")
    click.echo(f"Maximum results: {max_results}")

    with click.progressbar(length=1, label='Searching', show_eta=False) as bar:
        results = client.similarity_search(
            smiles=smiles,
            threshold=threshold,
            max_results=max_results,
            database=database
        )
        bar.update(1)

    if not results:
        click.echo("\nNo similar molecules found")
        click.echo("Try lowering the threshold or checking your molecule structure.")
        sys.exit(0)

    click.echo(f"\nFound {len(results)} similar molecules")

    # 4. Display top results
    click.echo("\nTop 5 results:")
    for i, result in enumerate(results[:5], 1):
        sim = result.get('similarity', 0)
        smiles_str = result.get('smiles', '')[:60]
        mol_id = result.get('id', 'N/A')
        click.echo(f"  {i}. Similarity: {sim:.3f} | ID: {mol_id}")
        click.echo(f"     SMILES: {smiles_str}")

    # 5. Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"search_results_{timestamp}"

    if output_format in ['sdf', 'both']:
        sdf_path = output_dir / f"{base_name}.sdf"
        MoleculeWriter.write_sdf(results, sdf_path)
        click.echo(f"\nSDF output: {sdf_path}")

    if output_format in ['json', 'both']:
        json_path = output_dir / f"{base_name}.json"

        # Prepare JSON output with metadata
        json_output = {
            'query': {
                'smiles': smiles,
                'input_file': str(input_file),
                'timestamp': timestamp
            },
            'search_params': {
                'threshold': threshold,
                'max_results': max_results,
                'database': database,
                'fingerprint_type': fingerprint_type
            },
            'results': results,
            'num_results': len(results)
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
