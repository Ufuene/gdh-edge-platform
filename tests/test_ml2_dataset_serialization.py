from datetime import datetime, timezone

from app.data_manager.sample import Sample
from app.datasets.ml2_dataset import ML2DatasetWindow

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


def create_sample(index: int) -> Sample:
    """
    Cria uma amostra representativa do sistema PV.
    """

    return Sample(
        timestamp=datetime(
            2026,
            8,
            20,
            1,
            0,
            index,
            tzinfo=timezone.utc,
        ),
        irradiance=1040.0 + index,
        temperature=47.0 + index,
        v_pv=24.0 + index,
        i_pv=4.0 + index,
        v_out=12.0 + index,
        i_out=9.0 + index,
        i_bat=-4.0 + index,
        i_load=14.0 + index,
        p_out=100.0 + index,
    )


def test_ml2_dataset_window_serialization_preserves_9_features():
    """
    Garante que o ML2DatasetWindow real preserve as nove
    grandezas necessárias ao modelo.
    """

    samples = [create_sample(index) for index in range(8)]

    window = ML2DatasetWindow.from_window(
        window_id="test-000001",
        window_start=0,
        window_end=7,
        samples=samples,
        label=1,
    )

    data = window.to_dict()

    assert data["window_id"] == "test-000001"
    assert data["window_start"] == 0
    assert data["window_end"] == 7
    assert data["label"] == 1

    serialized_samples = data["samples"]

    assert len(serialized_samples) == 8

    for sample in serialized_samples:

        assert list(sample.keys()) == EXPECTED_FIELDS

        features = [sample[field] for field in FEATURE_FIELDS]

        assert len(features) == 9


def test_ml2_dataset_window_serialization_produces_72_features():
    """
    Garante que a janela serializada produz exatamente
    as 72 features esperadas pelo ML2.
    """

    samples = [create_sample(index) for index in range(8)]

    window = ML2DatasetWindow.from_window(
        window_id="test-000002",
        window_start=10,
        window_end=17,
        samples=samples,
        label=0,
    )

    data = window.to_dict()

    flattened = [
        sample[field] for sample in data["samples"] for field in FEATURE_FIELDS
    ]

    assert len(flattened) == 72
