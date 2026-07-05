import unittest
import numpy as np
from ContextVector.core import ContextVector

class TestContextVector(unittest.TestCase):
    """
    Unit testing suite for the core ContextVector data primitive.
    """

    def setUp(self):
        self.mock_embedding_1 = [0.1, 0.2, 0.3]
        self.mock_embedding_2 = [0.1, 0.2, 0.3]
        self.mock_payload = "Unit test vector payload tracking."
        self.mock_metadata = {"env": "test", "id": 101}

        self.cv1 = ContextVector(
            embedding=self.mock_embedding_1,
            payload=self.mock_payload,
            modality="text",
            metadata=self.mock_metadata
        )

    def test_initialization_and_properties(self):
        self.assertEqual(self.cv1.payload, self.mock_payload)
        self.assertEqual(self.cv1.modality, "text")
        self.assertEqual(self.cv1.dimensions, 3)
        self.assertEqual(self.cv1.metadata["env"], "test")
        self.assertIsInstance(self.cv1.embedding, np.ndarray)

    def test_invalid_initialization(self):
        """Verifies dimension boundaries and bad parsing inputs."""
        with self.assertRaises(TypeError):
            ContextVector(embedding="not-an-iterable", payload="test")
        with self.assertRaises(ValueError):
            ContextVector(embedding=[[0.1, 0.2], [0.3, 0.4]], payload="2D matrix error")
        with self.assertRaises(ValueError):
            ContextVector(embedding=[], payload="Empty error")

    def test_vector_immutability(self):
        with self.assertRaises(ValueError):
            self.cv1.embedding[0] = 99.9

    def test_structural_equality(self):
        cv2 = ContextVector(self.mock_embedding_2, self.mock_payload, "text", self.mock_metadata)
        self.assertIsNot(self.cv1, cv2)
        self.assertEqual(self.cv1, cv2)

    def test_native_similarity(self):
        cv2 = ContextVector(self.mock_embedding_2, self.mock_payload)
        self.assertAlmostEqual(self.cv1.similarity(cv2), 1.0, places=5)

    def test_serialization_cycle(self):
        serialized_map = self.cv1.to_dict()
        self.assertIsInstance(serialized_map["embedding"], list)
        cv_reconstituted = ContextVector.from_dict(serialized_map)
        self.assertEqual(self.cv1, cv_reconstituted)

    def test_container_magic_methods(self):
        """Verifies sequence protocols added to core primitive."""
        self.assertEqual(len(self.cv1), 3)
        self.assertAlmostEqual(self.cv1[0], 0.1, places=6)
        self.assertTrue(0.2 in self.cv1)
        self.assertFalse(99.9 in self.cv1)
        
        # Test iterator conversions
        extracted_list = list(self.cv1)
        self.assertEqual(extracted_list, self.mock_embedding_1)

    def test_representation_output(self):
        """Verifies clean human-readable output layouts."""
        expected_repr = "ContextVector(modality='text', dimensions=3, payload_type=str)"
        self.assertEqual(repr(self.cv1), expected_repr)

if __name__ == "__main__":
    unittest.main()