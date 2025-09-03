import pytest

from hotbaud._utils import oom_self_reaper


@pytest.fixture(scope='session', autouse=True)
def cap_memory():
    with oom_self_reaper(kill_at_pct=.7):
        yield
