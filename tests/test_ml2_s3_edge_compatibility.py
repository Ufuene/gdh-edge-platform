"""
test_ml2_s3_edge_compatibility.py

Valida a compatibilidade entre:

    representação de uma janela ML2 no S3
                    e
    representação utilizada pelo ML2Service no Edge.

Contrato ML2:

    8 amostras
    9 features por amostra
    72 features totais

A ordem deve ser exatamente:

    1. Irradiancia
    2. Temperatura
    3. Vout
    4. Iout
    5. Ipv
    6. Vpv
    7. Iload
    8. Pout
    9. Ibat
"""

from datetime import datetime, timedelta

import numpy as np

from app.data_manager.sample import Sample
from app.inference.ml2_service import ML2Service


class FakeModel:
    """
    Modelo mínimo utilizado somente para construir
    o ML2Service durante o teste.
    """

    def predict(self, features):
        return np.array([0])

    def predict_proba(self, features):
        return np.array([[1.0, 0.0]])


def create_samples() -> list[Sample]:
    """
    Cria exatamente oito amostras determinísticas.

    Os valores são deliberadamente diferentes entre si
    para detectar qualquer alteração de ordem.
    """

    samples = []

    for index in range(8):

        samples.append(
            Sample(
                timestamp=datetime(2026, 8, 21) + timedelta(seconds=index * 5),
                irradiance=1000.0 + index,
                temperature=30.0 + index,
                v_pv=20.0 + index,
                i_pv=3.0 + index,
                v_out=12.0 + index,
                i_out=4.0 + index,
                i_bat=-5.0 + index,
                i_load=10.0 + index,
                p_out=50.0 + index,
            )
        )

    return samples


def build_s3_payload(samples: list[Sample]) -> dict:
    """
    Simula exatamente a representação ML2 enviada
    pelo Edge e armazenada no S3.
    """

    return {
        "schema_version": "gdh-edge-1.0",
        "metadata": {
            "device_id": "PV01",
            "timestamp": samples[-1].timestamp.isoformat(),
            "sequence": 8,
        },
        "window": {
            "window_id": "psc-000001",
            "window_start": 1,
            "window_end": 8,
            "label": 1,
            "samples": [
                {
                    "timestamp": sample.timestamp.isoformat(),
                    "Irradiancia": sample.irradiance,
                    "Temperatura": sample.temperature,
                    "Vout": sample.v_out,
                    "Iout": sample.i_out,
                    "Ipv": sample.i_pv,
                    "Vpv": sample.v_pv,
                    "Iload": sample.i_load,
                    "Pout": sample.p_out,
                    "Ibat": sample.i_bat,
                }
                for sample in samples
            ],
        },
    }


def build_features_from_s3(payload: dict) -> np.ndarray:
    """
    Reconstrói as 72 features a partir da representação
    armazenada no S3.

    A ordem é o contrato do ML2.
    """

    samples = payload["window"]["samples"]

    assert len(samples) == 8

    features = []

    for sample in samples:

        features.extend(
            [
                sample["Irradiancia"],
                sample["Temperatura"],
                sample["Vout"],
                sample["Iout"],
                sample["Ipv"],
                sample["Vpv"],
                sample["Iload"],
                sample["Pout"],
                sample["Ibat"],
            ]
        )

    assert len(features) == 72

    return np.asarray(
        features,
        dtype=np.float32,
    ).reshape(1, 72)


def test_s3_representation_matches_ml2_edge_representation():
    """
    Prova principal:

        representação S3
                ==
        representação utilizada pelo ML2 Edge

    Os 72 valores devem ser exatamente iguais.
    """

    samples = create_samples()

    payload = build_s3_payload(samples)

    s3_features = build_features_from_s3(payload)

    ml2_service = ML2Service(FakeModel())

    edge_features = ml2_service._build_features(samples)

    assert s3_features.shape == (1, 72)

    assert edge_features.shape == (1, 72)

    np.testing.assert_array_equal(
        s3_features,
        edge_features,
    )


def test_s3_ml2_contract_has_eight_samples_and_nine_features():
    """
    Verifica explicitamente o contrato estrutural:

        8 samples
        ×
        9 features
        =
        72 features
    """

    samples = create_samples()

    payload = build_s3_payload(samples)

    window = payload["window"]

    assert len(window["samples"]) == 8

    expected_fields = [
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

    for sample in window["samples"]:
        assert list(sample.keys()) == expected_fields

    features = build_features_from_s3(payload)

    assert features.shape == (1, 72)


def test_real_ml2_model_accepts_s3_derived_features():
    """
    Verifica que a representação reconstruída a partir do
    contrato S3 possui exatamente a dimensão esperada pelo
    modelo ML2 real.

    O modelo não é utilizado para validar a classificação.
    A finalidade deste teste é validar compatibilidade
    dimensional e estrutural.
    """

    import joblib

    model_path = "models/ml2/RandomForest_ML2_cpu.pkl"

    model = joblib.load(model_path)

    assert model.n_features_in_ == 72

    samples = create_samples()

    payload = build_s3_payload(samples)

    features = build_features_from_s3(payload)

    assert features.shape == (1, 72)

    prediction = model.predict(features)

    assert len(prediction) == 1
