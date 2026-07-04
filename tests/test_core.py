import unittest
import numpy as np
from ContextVector.core import ContextVector

class TestContextVector(unittest.TestCase):
    """
    Unit testing suite for the core ContextVector data primitive.
    Validates immutability, mathematical operations, and serialization pipelines.
    """

    def setUp(self):
        """
        Initializes baseline mock vectors before each test execution.
        """
        self.mock_embedding_1 = [0.1, 0.2, 0.3]
        self.mock_embedding_2 = [0.1, 0.2, 0.3]
        self.mock_embedding_3 = [-0.1, -0.2, -0.3]  # Orthogonal/opposite direction
        self.mock_payload = "Unit test vector payload tracking."
        self.mock_metadata = {"env": "test", "id": 101}

        self.cv1 = ContextVector(
            embedding=self.mock_embedding_1,
            payload=self.mock_payload,
            modality="text",
            metadata=self.mock_metadata
        )

    def test_initialization_and_properties(self):
        """Verifies properties are correctly assigned and accessible."""
        self.assertEqual(self.cv1.payload, self.mock_payload)
        self.assertEqual(self.cv1.modality, "text")
        self.assertEqual(self.cv1.dimensions, 3)
        self.assertEqual(self.cv1.metadata["env"], "test")
        self.assertIsInstance(self.cv1.embedding, np.ndarray)

    def test_vector_immutability(self):
        """Ensures the underlying array cannot be modified post-initialization."""
        with self.assertRaises(ValueError):
            # Attempting to mutate an element must trigger a NumPy ValueError
            self.cv1.embedding[0] = 99.9

    def test_structural_equality(self):
        """Validates that operator overloading (==) checks structural parity."""
        cv2 = ContextVector(
            embedding=self.mock_embedding_2,
            payload=self.mock_payload,
            modality="text",
            metadata=self.mock_metadata
        )
        # Separate memory references, identical contents
        self.assertIsNot(self.cv1, cv2)
        self.assertEqual(self.cv1, cv2)

    def test_native_similarity(self):
        """Validates the mathematical precision of the cosine similarity calculations."""
        # Identical vectors must yield a perfect similarity score of 1.0
        cv2 = ContextVector(self.mock_embedding_2, self.mock_payload)
        self.assertAlmostEqual(self.cv1.similarity(cv2), 1.0, places=5)

        # Dimension mismatches must raise a ValueError
        cv_invalid_dim = ContextVector([0.1, 0.2], "Bad dimension text")
        with self.assertRaises(ValueError):
            self.cv1.similarity(cv_invalid_dim)

    def test_serialization_cycle(self):
        """Verifies data structure parsing via to_dict and from_dict channels."""
        serialized_map = self.cv1.to_dict()
        
        # Verify JSON compatibility of the dictionary structure
        self.assertIsInstance(serialized_map["embedding"], list)
        self.assertEqual(serialized_map["payload"], self.mock_payload)

        # Reconstitute and verify parity
        cv_reconstituted = ContextVector.from_dict(serialized_map)
        self.assertEqual(self.cv1, cv_reconstituted)

if __name__ == "__main__":
    unittest.main()