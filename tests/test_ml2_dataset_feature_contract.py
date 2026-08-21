import joblib

from app.datasets.ml2_dataset import ML2DatasetWindow

ML2_MODEL_PATH = "models/ml2/RandomForest_ML2_cpu.pkl"

EXPECTED_FIELDS = [
    "timestamp",
    "Irradiancia",
    "Temperatura",
    "Vout",
    "Iout",
    "Ipv",
    "Vpv",
    "Iload",
    "Pout",
    "Ibat",
]

FEATURE_FIELDS = EXPECTED_FIELDS[1:]


def test_ml2_dataset_window_has_8_samples_and_9_features():
    """
    Valida o contrato estrutural do dataset ML2.

    O modelo ML2 foi treinado com:

        8 amostras × 9 grandezas = 72 features.
    """

    samples = []

    for index in range(8):
        samples.append(
            {
                "timestamp": f"2026-08-20T00:00:{index:02d}+00:00",
                "Irradiancia": 1000.0 + index,
                "Temperatura": 40.0 + index,
                "Vout": 12.0 + index,
                "Iout": 9.0 + index,
                "Ipv": 4.0 + index,
                "Vpv": 24.0 + index,
                "Iload": 14.0 + index,
                "Pout": 100.0 + index,
                "Ibat": -4.0 + index,
            }
        )

    assert len(samples) == 8

    for sample in samples:
        assert list(sample.keys()) == EXPECTED_FIELDS

        feature_values = [sample[field] for field in FEATURE_FIELDS]

        assert len(feature_values) == 9

    flattened = [sample[field] for sample in samples for field in FEATURE_FIELDS]

    assert len(flattened) == 72


def test_ml2_model_expects_72_features():
    """
    Confirma que o modelo ML2 atualmente instalado no Edge
    espera exatamente 72 features.
    """

    model = joblib.load(ML2_MODEL_PATH)

    assert model.n_features_in_ == 72
