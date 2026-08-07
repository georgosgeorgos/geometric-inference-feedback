from gift.core.loop import GIFTLoop
from gift.config import GIFTConfig

def test_loop_init():
    cfg = GIFTConfig(max_iterations=2)
    loop = GIFTLoop(cfg)
    assert loop.config.max_iterations == 2
    assert loop.iteration == 0
