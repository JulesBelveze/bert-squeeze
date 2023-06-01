import pytest

from tests.fixtures.dummy_models import Lr


@pytest.fixture(scope="module")
def lr_model():
    return Lr()
