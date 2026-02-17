from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import cairosvg
from pathlib import Path
from typing import Iterator, Optional, List, Tuple
import gzip
import logging
import json

logger = logging.getLogger(__name__)


def _canonicalize_smiles(raw_smiles: str):
    """Parse and canonicalize a SMILES string. Module-level for multiprocessing."""
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


class MoleculeReader:
    """Handles reading molecules from SDF, PDB, and SMI files."""

    @staticmethod
    def _resolve_format(file_path: Path) -> Tuple[str, bool]:
        """Return (format_suffix, is_gzipped) from file path."""
        if file_path.suffix.lower() == '.gz':
            return Path(file_path.stem).suffix.lower(), True
        return file_path.suffix.lower(), False

    @staticmethod
    def read_molecule(file_path: Path) -> Tuple[Optional[Chem.Mol], str]:
        """
        Read a molecule from SDF, PDB, or SMI file (optionally gzip-compressed).

        Returns:
            Tuple of (molecule object, isomeric SMILES string)
            Returns (None, "") if reading fails
        """
        fmt, is_gz = MoleculeReader._resolve_format(file_path)

        if fmt == '.sdf':
            return MoleculeReader._read_sdf(file_path, gz=is_gz)
        elif fmt == '.pdb':
            return MoleculeReader._read_pdb(file_path, gz=is_gz)
        elif fmt == '.smi':
            return MoleculeReader._read_smi_single(file_path, gz=is_gz)
        else:
            raise ValueError(f"Unsupported file format: {fmt}. Supported: .sdf, .pdb, .smi (optionally .gz)")

    @staticmethod
    def read_molecules(file_path: Path) -> List[Tuple[Chem.Mol, str]]:
        """
        Read all molecules from an SDF, PDB, or SMI file (optionally gzip-compressed).

        Returns:
            List of (molecule object, isomeric SMILES string) tuples
        """
        fmt, is_gz = MoleculeReader._resolve_format(file_path)

        if fmt == '.sdf':
            if is_gz:
                fh = gzip.open(file_path, 'rb')
                supplier = Chem.ForwardSDMolSupplier(fh)
            else:
                supplier = Chem.SDMolSupplier(str(file_path))
                fh = None
            molecules = []
            for i, mol in enumerate(supplier):
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    logger.info(f"Read molecule {i}: {smiles}")
                    molecules.append((mol, smiles))
                else:
                    logger.warning(f"Skipping invalid molecule at index {i}")
            if fh:
                fh.close()
            return molecules
        elif fmt == '.pdb':
            mol, smiles = MoleculeReader._read_pdb(file_path, gz=is_gz)
            if mol is not None:
                return [(mol, smiles)]
            return []
        elif fmt == '.smi':
            return MoleculeReader._read_smi(file_path, gz=is_gz)
        else:
            raise ValueError(f"Unsupported file format: {fmt}. Supported: .sdf, .pdb, .smi (optionally .gz)")

    @staticmethod
    def iter_molecules(file_path: Path) -> Iterator[Tuple[Chem.Mol, str]]:
        """
        Yield (mol, smiles) tuples one at a time. Memory-efficient for large files.

        Supports SDF, PDB, SMI, and their .gz-compressed variants.
        """
        fmt, is_gz = MoleculeReader._resolve_format(file_path)

        if fmt == '.sdf':
            if is_gz:
                fh = gzip.open(file_path, 'rb')
                supplier = Chem.ForwardSDMolSupplier(fh)
            else:
                supplier = Chem.SDMolSupplier(str(file_path))
                fh = None
            for i, mol in enumerate(supplier):
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    yield mol, smiles
                else:
                    logger.warning(f"Skipping invalid molecule at index {i}")
            if fh:
                fh.close()
        elif fmt == '.pdb':
            mol, smiles = MoleculeReader._read_pdb(file_path, gz=is_gz)
            if mol is not None:
                yield mol, smiles
        elif fmt == '.smi':
            opener = gzip.open if is_gz else open
            with opener(file_path, 'rt') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(None, 1)
                    raw_smiles = parts[0]
                    mol = Chem.MolFromSmiles(raw_smiles)
                    if mol is None:
                        logger.warning(f"Skipping invalid SMILES at line {i + 1}: {raw_smiles}")
                        continue
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    yield mol, smiles
        else:
            raise ValueError(f"Unsupported file format: {fmt}. Supported: .sdf, .pdb, .smi (optionally .gz)")

    @staticmethod
    def iter_smiles_parallel(file_path: Path, n_workers: int = 0,
                             chunk_size: int = 10000) -> Iterator[str]:
        """
        Yield canonical SMILES from a file using multiprocessing for .smi files.

        For .sdf/.pdb files or n_workers < 2, falls back to single-threaded
        iter_molecules. For .smi files, distributes SMILES parsing across
        CPU cores for ~N× speedup.
        """
        fmt, is_gz = MoleculeReader._resolve_format(file_path)

        if fmt != '.smi' or n_workers < 2:
            for _, smiles in MoleculeReader.iter_molecules(file_path):
                yield smiles
            return

        from multiprocessing import Pool

        opener = gzip.open if is_gz else open
        with opener(file_path, 'rt') as f:
            def _raw_smiles():
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    yield line.split(None, 1)[0]

            with Pool(n_workers) as pool:
                for smiles in pool.imap(
                    _canonicalize_smiles, _raw_smiles(), chunksize=chunk_size
                ):
                    if smiles is not None:
                        yield smiles

    @staticmethod
    def _read_sdf(file_path: Path, gz: bool = False) -> Tuple[Optional[Chem.Mol], str]:
        """
        Read first molecule from SDF file.

        IMPORTANT: Preserves stereochemistry using isomericSmiles=True
        """
        if gz:
            fh = gzip.open(file_path, 'rb')
            supplier = Chem.ForwardSDMolSupplier(fh)
        else:
            supplier = Chem.SDMolSupplier(str(file_path))
            fh = None

        for mol in supplier:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                logger.info(f"Read molecule from SDF: {smiles}")
                logger.debug(f"Number of atoms: {mol.GetNumAtoms()}")
                if fh:
                    fh.close()
                return mol, smiles

        if fh:
            fh.close()
        logger.error(f"No valid molecules found in {file_path}")
        return None, ""

    @staticmethod
    def _read_pdb(file_path: Path, gz: bool = False) -> Tuple[Optional[Chem.Mol], str]:
        """
        Read molecule from PDB file.

        Note: PDB files can be tricky with bond order assignment.
        IMPORTANT: Preserves stereochemistry using isomericSmiles=True
        """
        if gz:
            with gzip.open(file_path, 'rt') as f:
                mol = Chem.MolFromPDBBlock(f.read(), sanitize=True, removeHs=False)
        else:
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


    @staticmethod
    def _read_smi(file_path: Path, gz: bool = False) -> List[Tuple[Chem.Mol, str]]:
        """
        Read all molecules from a SMILES (.smi) file.

        Each line: SMILES [whitespace name]. Blank lines and lines
        starting with # are skipped.
        """
        opener = gzip.open if gz else open
        molecules = []
        with opener(file_path, 'rt') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(None, 1)
                raw_smiles = parts[0]
                mol = Chem.MolFromSmiles(raw_smiles)
                if mol is None:
                    logger.warning(f"Skipping invalid SMILES at line {i + 1}: {raw_smiles}")
                    continue
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                logger.info(f"Read molecule {len(molecules)}: {smiles}")
                molecules.append((mol, smiles))
        return molecules

    @staticmethod
    def _read_smi_single(file_path: Path, gz: bool = False) -> Tuple[Optional[Chem.Mol], str]:
        """Read the first valid molecule from a SMILES (.smi) file."""
        molecules = MoleculeReader._read_smi(file_path, gz=gz)
        if molecules:
            return molecules[0]
        logger.error(f"No valid molecules found in {file_path}")
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
