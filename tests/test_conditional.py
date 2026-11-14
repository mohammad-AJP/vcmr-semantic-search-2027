import numpy as np
from vcmr.conditional import local_null_signals,flank_contrast

def test_pure_holistic_component_has_zero_conditional_excess():
    h=np.array([0.,1.,3.,2.,0.,-1.,-2.,1.])
    s=np.stack([2*h+4,3*h-2]); shifts=np.array([[0,2],[3,0],[5,1],[7,4]])
    r=local_null_signals(s,h,shifts)
    np.testing.assert_allclose(r['conditional'],0,atol=1e-10)
    assert np.max(np.abs(r['unconditional']))>.1

def test_zero_shifts_give_zero_excess():
    rng=np.random.default_rng(2);s=rng.normal(size=(3,30));h=rng.normal(size=30)
    r=local_null_signals(s,h,np.zeros((2,3),int))
    np.testing.assert_allclose(r['conditional'],0,atol=1e-12)
    np.testing.assert_allclose(r['unconditional'],0,atol=1e-12)

def test_regression_residual_orthogonal_and_local_excess_varies():
    rng=np.random.default_rng(5);h=rng.normal(size=50);s=rng.normal(size=(3,50))+h
    r=local_null_signals(s,h,rng.integers(0,50,size=(16,3)))
    np.testing.assert_allclose(r['residual']@r['holistic_z'],0,atol=1e-10)
    assert np.std(r['conditional'])>.05

def test_flanks_use_only_available_observations():
    s=np.array([0.,0.,1.,1.,0.,0.])
    np.testing.assert_allclose(flank_contrast(s,[2,0],[4,2]),[1,-1])
