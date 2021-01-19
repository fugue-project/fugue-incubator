from fugue_tune.trial import TrialCallback, TrialsTracker


def test_callback():
    tt = TrialsTracker()
    tb = TrialCallback(tt)
    tb.log_trial(a=1, b=2)
    tb.log_trial(a=2, b=3)
    key = tb.trial_id
    assert [{"a": 1, "b": 2}, {"a": 2, "b": 3}] == tt.get_raw_data()[key]
    assert not tb.prune()
