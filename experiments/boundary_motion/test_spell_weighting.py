import numpy as np
from spell_weighting import complete_live_spells,fit_reference_seconds,spell_weights


def test_training_median_excludes_censored_spells():
    y=np.array([2,0,0,2,1,1,1,1,2,-100,0,0,2])
    periods=((0,9),(9,13))
    assert list(complete_live_spells(y,periods))==[(1,3),(4,8)]
    assert fit_reference_seconds([(y,periods,1)])==3.
    w=spell_weights(y,periods,1,3.,4.)
    np.testing.assert_allclose(w[1:3],np.sqrt(1.5))
    assert np.all(w[4:8]==1) and w[9]==0 and np.all(w[10:12]==1)
    assert np.all(w[y==2]==1)


def test_weights_cover_full_spell_across_chunk_boundary_and_are_capped():
    y=np.full(1100,2);y[450:550]=0;y[700:701]=1
    original=y.copy();w=spell_weights(y,((0,1100),),25,16.,4.)
    assert np.all(w[450:512]==2.) and np.all(w[512:550]==2.)
    assert w[700]==4. and np.all(w[y==2]==1)
    np.testing.assert_array_equal(y,original)


def test_half_boundaries_and_outside_periods_do_not_create_short_spells():
    y=np.array([2,0,0,0,0,2,-100,2])
    periods=((0,3),(3,6))
    assert list(complete_live_spells(y,periods))==[]
    np.testing.assert_array_equal(spell_weights(y,periods,25,10.),[1,1,1,1,1,1,0,0])
