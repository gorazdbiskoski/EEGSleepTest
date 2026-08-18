import gc

from data.data_loader_haanglanden import load_data_haaglanden
from data.data_loader_sleep_edfx import load_data_sleep_edfx
from data.subsample import subsample_stratified

HPO_SEARCH_SAMPLE_SIZE = 8000
HPO_SAMPLE_RANDOM_STATE = 42


def get_data_all_datasets(subsample=True):
    X_h, y_h = load_data_haaglanden()
    if subsample:
        X_h, y_h = subsample_stratified(
            X_h, y_h, HPO_SEARCH_SAMPLE_SIZE, HPO_SAMPLE_RANDOM_STATE
        )
    haaglanden = (X_h, y_h)
    del X_h, y_h
    gc.collect()

    X_s, y_s = load_data_sleep_edfx()
    if subsample:
        X_s, y_s = subsample_stratified(
            X_s, y_s, HPO_SEARCH_SAMPLE_SIZE, HPO_SAMPLE_RANDOM_STATE
        )
    sleep_edfx = (X_s, y_s)
    del X_s, y_s
    gc.collect()

    return haaglanden, sleep_edfx