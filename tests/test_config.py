from gift.config import GIFTConfig

def test_defaults():
    cfg = GIFTConfig()
    assert cfg.tau_accept == 0.9
    assert cfg.tau_reject == 0.5
    assert cfg.candidates_per_image == 16
    assert cfg.max_iterations == 5

def test_override():
    cfg = GIFTConfig(tau_accept=0.8, candidates_per_image=8)
    assert cfg.tau_accept == 0.8
    assert cfg.candidates_per_image == 8
