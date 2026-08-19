from types import SimpleNamespace

import pytest

from starVLA.dataloader import _vla_dataloader_kwargs


def test_vla_dataloader_workers_disabled_omit_worker_only_options():
    options = _vla_dataloader_kwargs(SimpleNamespace(num_workers=0))

    assert options == {"num_workers": 0, "pin_memory": False}


def test_vla_dataloader_workers_enabled_include_prefetch_options():
    options = _vla_dataloader_kwargs(
        SimpleNamespace(
            num_workers=4,
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=False,
        )
    )

    assert options == {
        "num_workers": 4,
        "pin_memory": False,
        "prefetch_factor": 3,
        "persistent_workers": True,
    }


@pytest.mark.parametrize(
    "config",
    (
        SimpleNamespace(num_workers=-1),
        SimpleNamespace(num_workers=1, prefetch_factor=0),
    ),
)
def test_vla_dataloader_rejects_invalid_worker_options(config):
    with pytest.raises(ValueError):
        _vla_dataloader_kwargs(config)
