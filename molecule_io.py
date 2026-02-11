from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import cairosvg
from pathlib import Path
from typing import Optional, List, Tuple
import logging
import json

logger = logging.getLogger(__name__)


class MoleculeReader:
    """Handles reading molecules from SDF and PDB files."""

    @staticmethod
    def read_molecule(file_path: Path) -> Tuple[Optional[Chem.Mol], str]:
        """
        Read a molecule from SDF or PDB file.

        Returns:
            Tuple of (molecule object, isomeric SMILES string)
            Returns (None, "") if reading fails
        """
        suffix = file_path.suffix.lower()

        if suffix == '.sdf':
            return MoleculeReader._read_sdf(file_path)
        elif suffix == '.pdb':
            return MoleculeReader._read_pdb(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Only .sdf and .pdb are supported.")

    @staticmethod
    def read_molecules(file_path: Path) -> List[Tuple[Chem.Mol, str]]:
        """
        Read all molecules from an SDF or PDB file.

        Returns:
            List of (molecule object, isomeric SMILES string) tuples
        """
        suffix = file_path.suffix.lower()

        if suffix == '.sdf':
            supplier = Chem.SDMolSupplier(str(file_path))
            molecules = []
            for i, mol in enumerate(supplier):
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    logger.info(f"Read molecule {i}: {smiles}")
                    molecules.append((mol, smiles))
                else:
                    logger.warning(f"Skipping invalid molecule at index {i}")
            return molecules
        elif suffix == '.pdb':
            mol, smiles = MoleculeReader._read_pdb(file_path)
            if mol is not None:
                return [(mol, smiles)]
            return []
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Only .sdf and .pdb are supported.")

    @staticmethod
    def _read_sdf(file_path: Path) -> Tuple[Optional[Chem.Mol], str]:
        """
        Read first molecule from SDF file.

        IMPORTANT: Preserves stereochemistry using isomericSmiles=True
        """
        supplier = Chem.SDMolSupplier(str(file_path))

        for mol in supplier:
            if mol is not None:
                # Use isomeric SMILES to preserve stereochemistry from 3D coordinates
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                logger.info(f"Read molecule from SDF: {smiles}")
                logger.debug(f"Number of atoms: {mol.GetNumAtoms()}")
                return mol, smiles

        logger.error(f"No valid molecules found in {file_path}")
        return None, ""

    @staticmethod
    def _read_pdb(file_path: Path) -> Tuple[Optional[Chem.Mol], str]:
        """
        Read molecule from PDB file.

        Note: PDB files can be tricky with bond order assignment.
        IMPORTANT: Preserves stereochemistry using isomericSmiles=True
        """
        mol = Chem.MolFromPDBFile(str(file_path), sanitize=True, removeHs=False)

        if mol is None:
            logger.error(f"Failed to read PDB file: {file_path}")
            return None, ""

        # Try to assign proper bond orders and sanitize
        # This is important for accurate fingerprinting
        try:
            # Sanitize molecule
            Chem.SanitizeMol(mol)
            # Use isomeric SMILES to preserve stereochemistry
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            logger.info(f"Read molecule from PDB: {smiles}")
            logger.debug(f"Number of atoms: {mol.GetNumAtoms()}")
            return mol, smiles
        except Exception as e:
            logger.error(f"Error processing PDB molecule: {e}")
            # Try without sanitization
            try:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                logger.warning(f"Molecule read without full sanitization: {smiles}")
                return mol, smiles
            except:
                return None, ""


class MoleculeWriter:
    """Handles writing search results to various formats."""

    @staticmethod
    def write_sdf(molecules: List[dict], output_path: Path):
        """
        Write molecules to SDF file.

        Args:
            molecules: List of dicts with 'smiles' and metadata
            output_path: Output SDF file path
        """
        writer = Chem.SDWriter(str(output_path))
        written_count = 0

        for i, mol_data in enumerate(molecules):
            smiles = mol_data.get('smiles')
            if not smiles:
                continue

            # Convert SMILES to molecule object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES at index {i}: {smiles}")
                continue

            # Set molecule name (first line in SDF block)
            name = mol_data.get('id') or mol_data.get('name') or smiles
            mol.SetProp("_Name", str(name))

            # Add explicit hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D coordinates
            params = AllChem.ETKDGv3()
            embed_result = AllChem.EmbedMolecule(mol, params)
            if embed_result == -1:
                logger.warning(f"3D embedding failed for index {i}, trying random coords")
                params.useRandomCoords = True
                AllChem.EmbedMolecule(mol, params)

            # Optimize geometry with MMFF94
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                logger.warning(f"MMFF optimization failed for index {i}, trying UFF")
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                except Exception:
                    pass  # Keep unoptimized coords

            # Add metadata as properties
            for key, value in mol_data.items():
                if key != 'smiles':
                    mol.SetProp(key, str(value))

            # Write molecule
            writer.write(mol)
            written_count += 1

        writer.close()
        logger.info(f"Wrote {written_count} molecules to {output_path}")

    @staticmethod
    def write_png(molecules: List[dict], output_dir: Path, size: int = 300):
        """
        Write 2D depiction PNGs for each molecule.

        Args:
            molecules: List of dicts with 'smiles' and metadata
            output_dir: Directory to write PNG files into
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        written_count = 0

        for i, mol_data in enumerate(molecules):
            smiles = mol_data.get('smiles')
            if not smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES at index {i}: {smiles}")
                continue

            AllChem.Compute2DCoords(mol)
            name = mol_data.get('id') or mol_data.get('name') or f"mol_{i}"
            # Sanitize filename
            safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in str(name))
            png_path = output_dir / f"{safe_name}.png"

            drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path),
                             output_width=size, output_height=size)
            written_count += 1

        logger.info(f"Wrote {written_count} PNG images to {output_dir}")

    @staticmethod
    def write_svg(molecules: List[dict], output_dir: Path, size: int = 300):
        """
        Write 2D depiction SVGs for each molecule.

        Args:
            molecules: List of dicts with 'smiles' and metadata
            output_dir: Directory to write SVG files into
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        written_count = 0

        for i, mol_data in enumerate(molecules):
            smiles = mol_data.get('smiles')
            if not smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES at index {i}: {smiles}")
                continue

            AllChem.Compute2DCoords(mol)
            name = mol_data.get('id') or mol_data.get('name') or f"mol_{i}"
            safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in str(name))
            svg_path = output_dir / f"{safe_name}.svg"

            drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            svg_path.write_text(svg)
            written_count += 1

        logger.info(f"Wrote {written_count} SVG images to {output_dir}")

    @staticmethod
    def write_json(results: dict, output_path: Path):
        """Write results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Wrote JSON results to {output_path}")
