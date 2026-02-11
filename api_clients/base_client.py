from abc import ABC, abstractmethod
from typing import List, Dict


class BaseAPIClient(ABC):
    """Abstract base class for chemical database API clients."""

    @abstractmethod
    def similarity_search(
        self,
        smiles: str,
        threshold: float,
        max_results: int
    ) -> List[Dict]:
        """
        Perform similarity search.

        Args:
            smiles: Query molecule SMILES
            threshold: Similarity threshold (0-1 for Tanimoto)
            max_results: Maximum number of results

        Returns:
            List of dicts with molecule data and similarity scores
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the API is accessible."""
        pass
