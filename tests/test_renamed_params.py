"""n_workers + executor -> max_threads / max_processes (after 0.0.6): the old names still
work, with a FutureWarning; giving a thread and a process pool at once is an error."""
import numpy as np
import pytest
import scipy.ndimage as ndi

from tilewise_ccl import label_array


@pytest.fixture
def vol():
    rng = np.random.default_rng(3)
    return ndi.gaussian_filter(rng.random((24, 24, 24)), 1.5) > 0.5


def _labels(out):
    return np.asarray(out)


@pytest.mark.parametrize("kw", [dict(max_threads=3), dict(max_processes=2)])
def test_new_names_label_the_same(vol, kw):
    ref = _labels(label_array(vol, tile_shape=(8, 8, 8)))
    np.testing.assert_array_equal(_labels(label_array(vol, tile_shape=(8, 8, 8), **kw)), ref)


def test_process_pool_is_used(vol, capsys):
    import dask.array as da
    label_array(da.from_array(vol, chunks=8), tile_shape=(8, 8, 8), max_processes=2,
                verbose=True)
    assert "process pool x2" in capsys.readouterr().out


@pytest.mark.parametrize("kw, pool", [(dict(n_workers=2), "thread pool x2"),
                                      (dict(n_workers=2, executor="process"), "process pool x2")])
def test_old_names_warn_and_keep_their_meaning(vol, kw, pool, capsys):
    import dask.array as da
    with pytest.warns(FutureWarning):
        label_array(da.from_array(vol, chunks=8), tile_shape=(8, 8, 8), verbose=True, **kw)
    assert pool in capsys.readouterr().out


@pytest.mark.parametrize("kw", [dict(max_threads=2, max_processes=2),
                                dict(max_threads=2, n_workers=2),
                                dict(max_processes=2, executor="process")])
def test_conflicting_pools_are_an_error(vol, kw):
    with pytest.raises(TypeError, match="pass max_"):
        label_array(vol, tile_shape=(8, 8, 8), **kw)
