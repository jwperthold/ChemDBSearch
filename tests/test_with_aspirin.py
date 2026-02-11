import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from molecule_io import MoleculeReader, MoleculeWriter
from fingerprint_engine import FingerprintEngine

TEST_FILE = Path(__file__).parent.parent / "aspirin.sdf"


def test_read_aspirin():
    """Test reading aspirin from SDF file."""
    mol, smiles = MoleculeReader.read_molecule(TEST_FILE)

    assert mol is not None, "Failed to read molecule"
    assert smiles != "", "Failed to generate SMILES"

    # Aspirin: C9H8O4
    formula = rdMolDescriptors.CalcMolFormula(mol)
    assert "C9" in formula and "O4" in formula, f"Expected C9H8O4, got {formula}"


def test_stereochemistry_preservation():
    """Test that isomeric SMILES is used."""
    mol, smiles = MoleculeReader.read_molecule(TEST_FILE)
    isomeric = Chem.MolToSmiles(mol, isomericSmiles=True)
    assert smiles == isomeric, "MoleculeReader should return isomeric SMILES"


def test_fingerprint_generation():
    """Test fingerprint generation for all supported types."""
    mol, _ = MoleculeReader.read_molecule(TEST_FILE)

    for fp_type in ['morgan', 'rdkit', 'maccs', 'atompair']:
        fp = FingerprintEngine.generate_fingerprint(mol, fp_type)
        assert fp is not None, f"Failed to generate {fp_type} fingerprint"


def test_self_similarity():
    """Test that aspirin has 1.0 similarity with itself."""
    mol, _ = MoleculeReader.read_molecule(TEST_FILE)
    fp1 = FingerprintEngine.generate_fingerprint(mol)
    fp2 = FingerprintEngine.generate_fingerprint(mol)
    similarity = FingerprintEngine.calculate_tanimoto(fp1, fp2)
    assert similarity == 1.0, f"Self-similarity should be 1.0, got {similarity}"


def test_smiles_roundtrip():
    """Test SMILES round-trip preserves fingerprint."""
    mol, smiles = MoleculeReader.read_molecule(TEST_FILE)
    mol2 = Chem.MolFromSmiles(smiles)
    assert mol2 is not None, "Failed to convert SMILES back to molecule"

    fp1 = FingerprintEngine.generate_fingerprint(mol)
    fp2 = FingerprintEngine.generate_fingerprint(mol2)
    similarity = FingerprintEngine.calculate_tanimoto(fp1, fp2)
    assert similarity >= 0.99, f"Round-trip similarity should be ~1.0, got {similarity}"


def test_sdf_output_has_hydrogens_and_3d(tmp_path):
    """Test that output SDF contains explicit hydrogens and 3D coordinates."""
    molecules = [
        {'smiles': 'CC(=O)Oc1ccccc1C(=O)O', 'id': 'aspirin', 'similarity': '1.0'}
    ]
    out_sdf = tmp_path / "test_output.sdf"
    MoleculeWriter.write_sdf(molecules, out_sdf)

    suppl = Chem.SDMolSupplier(str(out_sdf), removeHs=False)
    mol = next(suppl)

    # Aspirin has 13 heavy atoms; with H should be 21
    assert mol.GetNumAtoms() > 13, f"Expected hydrogens, got {mol.GetNumAtoms()} atoms"

    # Check 3D coordinates
    conf = mol.GetConformer()
    assert conf.Is3D(), "Output SDF should have 3D coordinates"

    # Verify coords are non-trivial (not all zeros)
    pos = conf.GetAtomPosition(0)
    coords_sum = abs(pos.x) + abs(pos.y) + abs(pos.z)
    assert coords_sum > 0.01, "3D coordinates should be non-zero"


def test_sdf_output_metadata(tmp_path):
    """Test that metadata properties are written to SDF."""
    molecules = [
        {'smiles': 'CC(=O)Oc1ccccc1C(=O)O', 'id': 'test-id', 'similarity': '0.95'}
    ]
    out_sdf = tmp_path / "test_meta.sdf"
    MoleculeWriter.write_sdf(molecules, out_sdf)

    suppl = Chem.SDMolSupplier(str(out_sdf), removeHs=False)
    mol = next(suppl)
    assert mol.GetProp('id') == 'test-id'
    assert mol.GetProp('similarity') == '0.95'


@pytest.mark.integration
def test_api_search_aspirin():
    """Integration test: search for aspirin with default settings (threshold=0.7)."""
    from api_clients.smallworld_client import SmallWorldClient

    mol, smiles = MoleculeReader.read_molecule(TEST_FILE)
    client = SmallWorldClient()

    if not client.is_available():
        pytest.skip("SmallWorld API not available")

    # Use default settings: threshold=0.7, max_results=100
    results = client.similarity_search(
        smiles=smiles,
        threshold=0.7,
        max_results=100,
    )

    assert len(results) >= 1, "Aspirin should have hits at 0.7 threshold"
    assert results[0]['similarity'] >= 0.7
    assert results[0]['smiles'] != ''
    assert results[0]['id'] != ''


@pytest.mark.integration
def test_full_pipeline_aspirin(tmp_path):
    """Integration test: full pipeline from SDF read to output with default settings."""
    from api_clients.smallworld_client import SmallWorldClient

    # 1. Read input
    mol, smiles = MoleculeReader.read_molecule(TEST_FILE)
    assert mol is not None

    # 2. Search with defaults (threshold=0.7, max_results=100)
    client = SmallWorldClient()
    if not client.is_available():
        pytest.skip("SmallWorld API not available")

    results = client.similarity_search(
        smiles=smiles,
        threshold=0.7,
        max_results=100,
    )
    assert len(results) >= 1

    # 3. Write SDF output
    sdf_path = tmp_path / "aspirin_results.sdf"
    MoleculeWriter.write_sdf(results, sdf_path)
    assert sdf_path.exists()

    # 4. Verify SDF: hydrogens, 3D coords, metadata
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    first_mol = next(suppl)
    assert first_mol.GetNumAtoms() > first_mol.GetNumHeavyAtoms(), "Should have explicit H"
    assert first_mol.GetConformer().Is3D(), "Should have 3D coords"
    assert first_mol.HasProp('similarity'), "Should have similarity metadata"

    # 5. Write JSON output
    json_path = tmp_path / "aspirin_results.json"
    json_output = {
        'query': {'smiles': smiles, 'input_file': str(TEST_FILE)},
        'search_params': {'threshold': 0.7, 'max_results': 100},
        'results': results,
        'num_results': len(results),
    }
    MoleculeWriter.write_json(json_output, json_path)
    assert json_path.exists()

    import json
    with open(json_path) as f:
        data = json.load(f)
    assert data['num_results'] == len(results)
    assert data['query']['smiles'] == smiles


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
