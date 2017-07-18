import unittest
import numpy as np

import quaternions


class QuaternionTest(unittest.TestCase):
    # Schaub, Chapter 3
    schaub_example_dcm = np.array([[.892539, .157379, -.422618],
                                   [-.275451, .932257, -.23457],
                                   [.357073, .325773, .875426]])
    schaub_result = np.array([.961798, -.14565, .202665, .112505])

    def test_matrix_respects_product(self):
        q1 = quaternions.Quaternion.exp(quaternions.Quaternion(0, .1, .02, -.3))
        q2 = quaternions.Quaternion.exp(quaternions.Quaternion(0, -.2, .21, .083))
        np.testing.assert_allclose((q1 * q2).matrix, q1.matrix.dot(q2.matrix))

    def test_from_matrix(self):
        q = quaternions.Quaternion.from_matrix(QuaternionTest.schaub_example_dcm)
        np.testing.assert_allclose(QuaternionTest.schaub_result, q.coordinates, atol=1e-5, rtol=0)

    def test_from_matrix_twisted(self):
        q = quaternions.Quaternion.from_matrix(QuaternionTest.schaub_example_dcm * [-1, -1, 1])
        e1 = quaternions.Quaternion(*QuaternionTest.schaub_result)
        expected = e1 * quaternions.Quaternion(0, 0, 0, 1)
        np.testing.assert_allclose(expected.coordinates, q.coordinates, atol=1e-5, rtol=0)

    def test_from_rotation_vector_to_matrix(self):
        phi = np.array([-.295067, .410571, .227921])
        expected = np.array([
            [.892539, .157379, -.422618],
            [-.275451, .932257, -.23457],
            [.357073, .325773, .875426]])
        q = quaternions.Quaternion.from_rotation_vector(phi)
        np.testing.assert_allclose(expected, q.matrix, atol=1e-5, rtol=0)

    def test_qmethod(self):
        frame_1 = np.array([[2 / 3, 2 / 3, 1 / 3], [2 / 3, -1 / 3, -2 / 3]])
        frame_2 = np.array([[0.8, 0.6, 0], [-0.6, 0.8, 0]])
        q = quaternions.Quaternion.from_qmethod(frame_1.T, frame_2.T, np.ones(2))

        for a1 in np.arange(0, 1, .1):
            for a2 in np.arange(0, 1, .1):
                v1 = a1 * frame_1[0] + a2 * frame_1[1]
                v2 = a1 * frame_2[0] + a2 * frame_2[1]
                np.testing.assert_allclose(q.matrix.dot(v1), v2, atol=1e-10)

    def test_ra_dec_roll(self):
        for ra in np.linspace(-170, 180, 8):
            for dec in np.linspace(-90, 90, 8):
                for roll in np.linspace(10, 360, 8):

                    xyz = np.deg2rad(np.array([ra, dec, roll]))
                    c3, c2, c1 = np.cos(xyz)
                    s3, s2, s1 = np.sin(xyz)
                    expected = np.array([
                        [c2 * c3,               -c2 * s3,                 s2],       # noqa
                        [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],  # noqa
                        [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3,  c1 * c2]   # noqa
                    ])

                    obtained = quaternions.Quaternion.from_ra_dec_roll(ra, dec, roll)

                    np.testing.assert_allclose(expected, obtained.matrix, atol=1e-15)

    def test_to_rdr(self):
        for ra in np.linspace(-170, 170, 8):
            for dec in np.linspace(-88, 88, 8):
                for roll in np.linspace(-170, 170, 8):
                    q = quaternions.Quaternion.from_ra_dec_roll(ra, dec, roll)

                    np.testing.assert_allclose([ra, dec, roll], q.ra_dec_roll)

    def test_average_easy(self):
        q1 = quaternions.Quaternion(1, 0, 0, 0)
        q2 = quaternions.Quaternion(-1, 0, 0, 0)
        avg = quaternions.Quaternion.average(q1, q2)

        np.testing.assert_allclose(q1.coordinates, avg.coordinates)

    def test_average_mild(self):
        q1 = quaternions.Quaternion.exp(quaternions.Quaternion(0, .1, .3, .7))
        quats_l = []
        quats_r = []
        for i in np.arange(-.1, .11, .05):
            for j in np.arange(-.1, .11, .05):
                for k in np.arange(-.1, .11, .05):
                    q = quaternions.Quaternion.exp(quaternions.Quaternion(0, i, j, k))
                    quats_l.append(q1 * q)
                    quats_r.append(q * q1)

        avg_l = quaternions.Quaternion.average(*quats_l)
        avg_r = quaternions.Quaternion.average(*quats_r)
        np.testing.assert_allclose(q1.coordinates, avg_l.coordinates)
        np.testing.assert_allclose(q1.coordinates, avg_r.coordinates)

    def test_optical_axis_first(self):
        v1 = np.array([.02, .01, .99])
        v2 = np.array([-.01, .02, .99])
        oaf = quaternions.Quaternion.OpticalAxisFirst()
        np.testing.assert_allclose([.99, -.02, -.01], oaf.matrix.dot(v1))
        np.testing.assert_allclose([.99, .01, -.02], oaf.matrix.dot(v2))
