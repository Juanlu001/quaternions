from hypothesis import given
from hypothesis.strategies import floats

import numpy as np

from quaternions import Quaternion

τ = 2 * np.pi

@given(floats(min_value=-180, max_value=180),
       floats(min_value=-90, max_value=90),
       floats(min_value=0, max_value=360))
def test_quat_ra_dec_roll(ra, dec, roll):
    q = Quaternion.from_ra_dec_roll(ra, dec, 2.)
    ob_ra, ob_dec, ob_roll = q.ra_dec_roll
    assert abs(ob_dec - dec) < 1e-8
    assert abs(ob_ra - ra) < 1e-8
    

@given(floats(min_value=-2, max_value=2),
       floats(min_value=-2, max_value=2),
       floats(min_value=-2, max_value=2))
def test_quat_rotation_vector(rx, ry, rz):
    rot = np.array([rx, ry, rz])
    q = Quaternion.from_rotation_vector(rot)
    distance = np.linalg.norm(rot - q.rotation_vector)

    assert (distance % τ) < 1e-8
