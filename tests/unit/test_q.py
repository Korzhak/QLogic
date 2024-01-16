import unittest
import numpy as np
from src.QLogic import Quaternion


class TestQuaternion(unittest.TestCase):
    def setUp(self) -> None:
        self.q = Quaternion()

    def test_create_q_blank(self):
        q = Quaternion()
        pure_vector = np.array([1, 0, 0, 0], dtype=np.float64)
        self.assertTrue(np.array_equal(q.to_numpy(), pure_vector))

    def test_create_q_using_q(self):
        self.q = Quaternion(np.array([0.7071, 0, 0, 0.7071]))
        q = np.array([0.7071, 0, 0, 0.7071], dtype=np.float64)
        self.assertTrue(np.array_equal(self.q.to_numpy(), q))

    def test_create_q_using_rotation_vector(self):
        ...

    def test_create_q_using_euler_angles(self):
        ...

    def test_q_properties(self):
        self.q = Quaternion(np.array([0.7071, 0, 0, 0.7071]))
        self.assertEqual(self.q.w, np.float64(0.7071))
        self.assertEqual(self.q.x, np.float64(0))
        self.assertEqual(self.q.y, np.float64(0))
        self.assertEqual(self.q.z, np.float64(0.7071))

    def test_q_multiply(self):
        q1 = Quaternion(np.array([0.7071, 0, 0, 0.7071], dtype=np.float64))
        q2 = Quaternion(np.array([0.7071, 0.7071, 0.7071, 0.7071], dtype=np.float64))

        print(q1 * q2)

if __name__ == '__main__':
    unittest.main()
