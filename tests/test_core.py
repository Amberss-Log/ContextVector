import unittest
import numpy as np
from ContextVector.core import ContextVector

class TestContextVectorRegression(unittest.TestCase):

    def setUp(self):
        self.mock_embedding_1 = [0.1, 0.2, 0.3]
        self.mock_payload = "Data integrity checks."
        self.cv1 = ContextVector(
            embedding=self.mock_embedding_1,
            payload=self.mock_payload,
            modality="text",
            provider_name="TestEngine",
            model_name="Model-V1"
        )

    def test_deserialization_tampering_guard(self):
        """High Priority Check: Verifies from_dict catches malformed/truncated arrays."""
        serialized_map = self.cv1.to_dict()
        
        # Simulate an adversarial database corruption (truncating the array)
        serialized_map["embedding"] = [0.1, 0.2]
        
        with self.assertRaises(ValueError):
            ContextVector.from_dict(serialized_map)

    def test_provenance_similarity_lockout(self):
        """Medium Priority Check: Verifies similarity blocks matching dims from different spaces."""
        # Same dimension layout (3), different underlying model space
        cv_alien_space = ContextVector(
            embedding=[0.1, 0.2, 0.3],
            payload=self.mock_payload,
            modality="text",
            provider_name="DifferentEngine",
            model_name="Model-X"
        )
        
        with self.assertRaises(ValueError):
            self.cv1.similarity(cv_alien_space)

    def test_invalid_numeric_values(self):
        """Fuzzing Test: Ensures NaN or Inf values are rejected instantly."""
        with self.assertRaises(ValueError):
            ContextVector(embedding=[0.1, float('nan'), 0.3], payload="bad math")
        with self.assertRaises(ValueError):
            ContextVector(embedding=[0.1, float('inf'), 0.3], payload="infinity loop")

    def test_empty_string_factory_guard(self):
        """Edge Case Check: Rejects empty or whitespace strings in from_text."""
        class MockProvider:
            def embed_text(self, text, model_name): return [0.1]
            
        with self.assertRaises(ValueError):
            ContextVector.from_text("", MockProvider(), "mock-model")
        with self.assertRaises(ValueError):
            ContextVector.from_text("   ", MockProvider(), "mock-model")

    def test_zero_vector_similarity_fallback(self):
        """Low Priority Check: Zero vectors yield 0.0 instead of crashing."""
        cv_zero = ContextVector(
            embedding=[0.0, 0.0, 0.0], 
            payload="empty space", 
            provider_name="TestEngine", 
            model_name="Model-V1"
        )
        self.assertEqual(self.cv1.similarity(cv_zero), 0.0)

if __name__ == "__main__":
    unittest.main()