from rdkit import Chem
from rdkit.Chem import AllChem
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

            # Add explicit hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D coordinates
            embed_result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if embed_result == -1:
                logger.warning(f"3D embedding failed for index {i}, trying random coords")
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), useRandomCoords=True)

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
    def write_json(results: dict, output_path: Path):
        """Write results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Wrote JSON results to {output_path}")
