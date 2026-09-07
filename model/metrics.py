"""The classification metrics ``evaluate_model`` returns and every optimizer logs.

Lives beside ``model_builder`` because it is that function's output contract:
adding a metric means editing these two files side by side.

Order matters: all ten optimizers append to the same ``best_results.csv`` with
the header written only once, so every one of them must emit these columns in
the same order or the appended rows silently misalign.
"""

METRIC_KEYS = ("val_accuracy", "precision", "recall", "f1")


def metrics_from(result):
    """The metric fields of an ``evaluate_model()`` result, for a CSV row."""
    return {key: result[key] for key in METRIC_KEYS}
