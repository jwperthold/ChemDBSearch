import re
import requests
from typing import List, Dict
import logging
from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class SmallWorldClient(BaseAPIClient):
    """
    Client for SmallWorld API (sw.docking.org).

    Provides access to Enamine REAL Space and other databases without authentication.
    Returns ECFP4 Tanimoto similarity scores for hits.

    API docs: https://wiki.docking.org/index.php/How_to_use_SmallWorld_API
    """

    def __init__(self, base_url: str = "https://sw.docking.org"):
        self.base_url = base_url
        self.session = requests.Session()

    def similarity_search(
        self,
        smiles: str,
        threshold: float = 0.7,
        max_results: int = 100,
        database: str = "REAL-Database-22Q1.smi.anon"
    ) -> List[Dict]:
        """
        Search for similar molecules using SmallWorld API.

        SmallWorld uses graph-edit distance (dist) for initial candidate retrieval,
        then scores results with ECFP4 Tanimoto. We post-filter results by the
        user's Tanimoto threshold.

        Args:
            smiles: Query SMILES string
            threshold: Tanimoto similarity threshold (0.0-1.0), applied to ECFP4 scores
            max_results: Maximum number of results to return
            database: Database name to search (use get_available_databases() for valid names)

        Returns:
            List of dicts with smiles, similarity, id, and metadata
        """
        logger.info(f"Searching {database} with SMILES: {smiles}")
        logger.info(f"Tanimoto threshold: {threshold}, max results: {max_results}")

        try:
            raw_results = self._make_request(
                smiles=smiles,
                database=database,
                length=max_results
            )

            results = self._parse_results(raw_results, threshold, max_results)
            logger.info(f"Found {len(results)} molecules above threshold {threshold}")
            return results

        except requests.HTTPError as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 429:
                    logger.error("Rate limit exceeded - try again later")
                elif e.response.status_code == 503:
                    logger.error("Service temporarily unavailable")
            return []
        except requests.ConnectionError:
            logger.error("Network connection failed - check internet connectivity")
            return []
        except requests.Timeout:
            logger.error("Request timed out (120s limit)")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []

    def _make_request(self, smiles: str, database: str, length: int = 100) -> dict:
        """
        Make API request to SmallWorld /search/view endpoint.

        The endpoint returns a DataTables-compatible JSON response.
        """
        url = f"{self.base_url}/search/view"

        params = {
            'smi': smiles,
            'db': database,
            'dist': 8,    # graph-edit distance (0-16, higher = more diverse results)
            'sdist': 12,  # scored distance bound
            'length': length,
            'start': 0,
            'scores': 'Atom Alignment,ECFP4,Daylight',
        }

        logger.debug(f"Request URL: {url}")
        logger.debug(f"Request params: {params}")

        response = self.session.get(url, params=params, timeout=120)
        response.raise_for_status()

        return response.json()

    def _parse_results(self, response: dict, threshold: float, max_results: int) -> List[Dict]:
        """
        Parse SmallWorld DataTables JSON response.

        Response format:
        {
            "recordsTotal": int,
            "recordsFiltered": int,
            "data": [
                [
                    {alignment_obj},  # col 0: hitSmiles, qrySmiles, mw, mf, etc.
                    dist,             # col 1: anonymous graph-edit distance
                    ecfp4,            # col 2: ECFP4 Tanimoto score
                    daylight,         # col 3: Daylight Tanimoto score
                    topodist,         # col 4: topological distance
                    ...               # additional columns
                ], ...
            ]
        }
        """
        results = []
        data = response.get('data', [])

        for row in data:
            if len(row) < 4:
                continue

            alignment = row[0] if isinstance(row[0], dict) else {}
            ecfp4_tanimoto = float(row[2]) if len(row) > 2 else 0.0

            # Filter by Tanimoto threshold
            if ecfp4_tanimoto < threshold:
                continue

            # Extract hit SMILES and ID from alignment object
            hit_smiles_raw = alignment.get('hitSmiles', '')
            # hitSmiles format: "SMILES IDENTIFIER" (space-separated)
            parts = hit_smiles_raw.split(' ', 1)
            hit_smiles = parts[0] if parts else ''
            hit_id = parts[1] if len(parts) > 1 else ''

            # Clean HTML tags from molecular formula (e.g. C<sub>5</sub>H<sub>8</sub>O -> C5H8O)
            mf_raw = alignment.get('mf', '')
            mf_clean = re.sub(r'<[^>]+>', '', mf_raw)

            results.append({
                'smiles': hit_smiles,
                'similarity': ecfp4_tanimoto,
                'atom_alignment_score': float(row[1]) if len(row) > 1 else 0.0,
                'id': hit_id,
                'molecular_weight': alignment.get('mw', ''),
                'molecular_formula': mf_clean,
            })

            if len(results) >= max_results:
                break

        # Sort by ECFP4 Tanimoto similarity descending
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def is_available(self) -> bool:
        """Check if SmallWorld API is accessible."""
        try:
            response = self.session.get(
                f"{self.base_url}/search/maps",
                timeout=10,
                allow_redirects=True
            )
            return response.status_code == 200
        except:
            return False

    def get_available_databases(self) -> List[str]:
        """
        Retrieve list of available databases from /search/maps endpoint.

        Returns database keys that can be used as the 'db' parameter.
        """
        try:
            url = f"{self.base_url}/search/maps"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            maps_data = response.json()
            databases = []
            for key, info in maps_data.items():
                name = info.get('name', key) if isinstance(info, dict) else key
                databases.append(f"{key}  ({name})")
            return databases

        except Exception as e:
            logger.error(f"Failed to retrieve databases: {e}")
            # Fallback static list
            return [
                "REAL-Database-22Q1.smi.anon  (REAL-Database-22Q1-4.5B)",
                "REALDB-2025-07.smi.anon  (REALDB-25Q3-9.4B)",
                "ZINC20-All-25Q2.smi.anon  (ZINC20-All-25Q2-1.9B)",
            ]
