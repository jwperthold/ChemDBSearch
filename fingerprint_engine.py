from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, AdjustQueryParameters, AdjustQueryProperties
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FingerprintEngine:
    """
    Local fingerprint generation and Tanimoto similarity calculation.

    Useful for:
    - Pre-filtering before API search
    - Validation of results
    - Offline testing
    """

    FINGERPRINT_TYPES = {
        'morgan': 'Morgan (ECFP-like)',
        'rdkit': 'RDKit Topological',
        'maccs': 'MACCS Keys',
        'atompair': 'Atom Pairs'
    }

    @staticmethod
    def generate_fingerprint(
        mol: Chem.Mol,
        fp_type: str = 'morgan',
        radius: int = 2,
        n_bits: int = 2048
    ):
        """
        Generate molecular fingerprint.

        Args:
            mol: RDKit molecule object
            fp_type: Type of fingerprint ('morgan', 'rdkit', 'maccs', 'atompair')
            radius: Radius for Morgan fingerprints (default 2 = ECFP4)
            n_bits: Number of bits in fingerprint

        Returns:
            Fingerprint object
        """
        if fp_type == 'morgan':
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == 'rdkit':
            return Chem.RDKFingerprint(mol, fpSize=n_bits)
        elif fp_type == 'maccs':
            return AllChem.GetMACCSKeysFingerprint(mol)
        elif fp_type == 'atompair':
            return AllChem.GetAtomPairFingerprint(mol)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}. Valid types: {list(FingerprintEngine.FINGERPRINT_TYPES.keys())}")

    @staticmethod
    def calculate_tanimoto(fp1, fp2) -> float:
        """
        Calculate Tanimoto similarity between two fingerprints.

        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    @staticmethod
    def bulk_similarity(
        query_mol: Chem.Mol,
        target_mols: List[Chem.Mol],
        threshold: float = 0.7,
        fp_type: str = 'morgan'
    ) -> List[Tuple[int, float]]:
        """
        Calculate similarities between query and multiple targets.

        Args:
            query_mol: Query molecule
            target_mols: List of target molecules
            threshold: Minimum similarity threshold
            fp_type: Fingerprint type to use

        Returns:
            List of (index, similarity) tuples for molecules above threshold,
            sorted by similarity descending
        """
        query_fp = FingerprintEngine.generate_fingerprint(query_mol, fp_type)

        results = []
        for i, target_mol in enumerate(target_mols):
            if target_mol is None:
                continue

            target_fp = FingerprintEngine.generate_fingerprint(target_mol, fp_type)
            similarity = FingerprintEngine.calculate_tanimoto(query_fp, target_fp)

            if similarity >= threshold:
                results.append((i, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def make_generic_query(mol: Chem.Mol) -> Chem.Mol:
        """
        Convert a molecule to a generic query that matches only the
        heavy-atom bond graph (ignoring atom types and bond orders).

        Args:
            mol: RDKit molecule object

        Returns:
            Generic query molecule for substructure matching
        """
        mol = Chem.RemoveHs(mol)
        params = AdjustQueryParameters.NoAdjustments()
        params.makeAtomsGeneric = True
        params.makeBondsGeneric = True
        return AdjustQueryProperties(mol, params)

    @staticmethod
    def has_generic_substructure_match(
        query_mol: Chem.Mol,
        target_smiles: str
    ) -> bool:
        """
        Check if target contains the query's heavy-atom bond graph
        as a substructure (ignoring atom types and bond orders).

        Args:
            query_mol: Query molecule (will be converted to generic query)
            target_smiles: SMILES of the target molecule

        Returns:
            True if the target contains the query's connectivity pattern
        """
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            return False
        target_mol = Chem.AddHs(target_mol)
        generic_query = FingerprintEngine.make_generic_query(query_mol)
        return target_mol.HasSubstructMatch(generic_query)
