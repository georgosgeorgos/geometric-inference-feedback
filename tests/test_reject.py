from gift.core.reject import classify_candidates, build_reject_pairs

def test_classify_basic():
    ious = [0.95, 0.7, 0.3, 0.91, 0.45]
    accepted, fda, discarded = classify_candidates(ious, tau_accept=0.9, tau_reject=0.5)
    assert accepted == [0, 3]       # indices with IoU >= 0.9
    assert fda == [1]               # indices with 0.5 < IoU < 0.9
    assert discarded == [2, 4]      # indices with IoU <= 0.5

def test_classify_boundary():
    ious = [0.9, 0.5]
    accepted, fda, discarded = classify_candidates(ious, tau_accept=0.9, tau_reject=0.5)
    assert accepted == [0]          # IoU >= tau_accept
    assert fda == []
    assert discarded == [1]         # IoU <= tau_reject

def test_build_reject_pairs():
    accepted_indices = [0, 2]
    images = ["img_a", "img_b", "img_c"]
    codes = ["code_a", "code_b", "code_c"]
    pairs = build_reject_pairs(accepted_indices, images, codes)
    assert pairs == [("img_a", "code_a"), ("img_c", "code_c")]
