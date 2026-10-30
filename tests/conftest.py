import warnings

import pytest

from spherex.config import load_config

warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture
def small_cfg():
    """Configuração reduzida para os testes rodarem em segundos."""
    return load_config(overrides={
        "synthetic": {"n_fields": 3, "shape": [96, 96], "n_stars": 60, "n_movers": 4},
        "injection": {"n_per_field": 8},
        "model": {"epochs": 3, "batch_size": 32},
    })
