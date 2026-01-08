import pandas as pd

from astree_eta.features import clean_features


def test_clean_features_prefers_kloc_and_num_tu():
    df = pd.DataFrame(
        {
            "loc": [None],
            "kloc": [2.0],
            "num_files": [None],
            "num_tu": [5],
            "cpu_avg": [None],
            "server_load_avg": [0.7],
        }
    )
    cleaned = clean_features(df)
    assert cleaned.loc[0, "loc"] == 2000
    assert cleaned.loc[0, "num_files"] == 5
    assert cleaned.loc[0, "cpu_avg"] == 0.7
