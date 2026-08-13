import json
from datetime import datetime
from math import isclose, isfinite
from pathlib import Path

import joblib
import pytest

from app.data_manager.sample import Sample
from app.inference.ml2_service import ML2Service


class MockML2Model:
    """
    Modelo ML2 simulado utilizado nos testes.
    """

    def __init__(
        self,
        diagnosis=1,
        probability_normal=0.2,
        probability_psc=0.8,
    ):
        self.diagnosis = diagnosis
        self.probability_normal = probability_normal
        self.probability_psc = probability_psc
        self.received_features = None

    def predict(self, features):
        """
        Registra as features recebidas e retorna
        um diagnóstico simulado.
        """

        self.received_features = features

        return [self.diagnosis]

    def predict_proba(self, features):
        """
        Registra as features recebidas e retorna
        probabilidades simuladas.
        """

        self.received_features = features

        return [
            [
                self.probability_normal,
                self.probability_psc,
            ]
        ]


def create_sample(index: int) -> Sample:
    """
    Cria uma amostra identificável pelo índice.
    """

    return Sample(
        timestamp=datetime.now(),
        irradiance=800.0 + index,
        temperature=25.0 + index,
        v_pv=30.0 + index,
        i_pv=7.0 + index,
        v_out=14.5 + index,
        i_out=8.0 + index,
        i_bat=3.0 + index,
        i_load=5.0 + index,
        p_out=100.0 + index,
    )


def create_window():
    """
    Cria uma janela completa de 8 amostras.
    """

    return [create_sample(i) for i in range(8)]


def test_ml2_service_creation():
    """
    Verifica a criação correta do ML2Service.
    """

    model = MockML2Model()

    service = ML2Service(model)

    assert service._model == model


def test_predict_returns_diagnosis():
    """
    Verifica se o ML2Service retorna o diagnóstico
    produzido pelo modelo.
    """

    model = MockML2Model(diagnosis=1)

    service = ML2Service(model)

    window = create_window()

    diagnosis = service.predict(window)

    assert diagnosis == 1


def test_predict_returns_normal_diagnosis():
    """
    Verifica o diagnóstico correspondente à classe normal.
    """

    model = MockML2Model(diagnosis=0)

    service = ML2Service(model)

    window = create_window()

    diagnosis = service.predict(window)

    assert diagnosis == 0


def test_predict_generates_72_features():
    """
    Verifica que 8 amostras × 9 features produzem
    exatamente 72 features.
    """

    model = MockML2Model()

    service = ML2Service(model)

    window = create_window()

    service.predict(window)

    received = model.received_features

    assert received.shape == (1, 72)


def test_predict_uses_correct_feature_order():
    """
    Verifica a ordem das features utilizada pelo ML2.

    A ordem deve ser:

    Irradiancia
    Temperatura
    Vout
    Iout
    Ipv
    Vpv
    Iload
    Pout
    Ibat
    """

    model = MockML2Model()

    service = ML2Service(model)

    window = create_window()

    service.predict(window)

    features = model.received_features

    assert features.shape == (1, 72)

    expected_first_sample = [
        800.0,
        25.0,
        14.5,
        8.0,
        7.0,
        30.0,
        5.0,
        100.0,
        3.0,
    ]

    assert features[0, :9].tolist() == expected_first_sample


def test_predict_preserves_temporal_order():
    """
    Verifica que as amostras são organizadas temporalmente
    como t1, t2, ..., t8.
    """

    model = MockML2Model()

    service = ML2Service(model)

    window = create_window()

    service.predict(window)

    features = model.received_features

    for index in range(8):
        start = index * 9
        end = start + 9

        expected = [
            800.0 + index,
            25.0 + index,
            14.5 + index,
            8.0 + index,
            7.0 + index,
            30.0 + index,
            5.0 + index,
            100.0 + index,
            3.0 + index,
        ]

        assert features[0, start:end].tolist() == expected


def test_predict_proba_returns_class_probabilities():
    """
    Verifica o retorno das probabilidades NORMAL e PSC.
    """

    model = MockML2Model(
        probability_normal=0.25,
        probability_psc=0.75,
    )

    service = ML2Service(model)

    window = create_window()

    probability_normal, probability_psc = service.predict_proba(window)

    assert probability_normal == 0.25
    assert probability_psc == 0.75


def test_predict_rejects_incomplete_window():
    """
    Verifica que uma janela com menos de 8 amostras
    é rejeitada.
    """

    model = MockML2Model()

    service = ML2Service(model)

    window = [create_sample(i) for i in range(7)]

    with pytest.raises(ValueError):
        service.predict(window)


def test_predict_rejects_window_with_more_than_8_samples():
    """
    Verifica que uma janela com mais de 8 amostras
    também é rejeitada.
    """

    model = MockML2Model()

    service = ML2Service(model)

    window = [create_sample(i) for i in range(9)]

    with pytest.raises(ValueError):
        service.predict(window)


def test_ml2_service_with_real_model_normal():
    """
    Teste de integração com o modelo ML2 real usando
    uma janela NORMAL validada anteriormente no Raspberry Pi.

    O modelo deve retornar:

        class = 0
        NORMAL ≈ 0.975
        PSC ≈ 0.025
    """

    model_path = Path("/home/ufuene/gdh-platform/models/ml2/RandomForest_ML2_cpu.pkl")

    if not model_path.exists():
        pytest.skip(f"Modelo ML2 real não encontrado: {model_path}")

    model = joblib.load(model_path)

    assert model.n_features_in_ == 72
    assert list(model.classes_) == [0, 1]
    assert model.n_estimators == 200
    assert model.max_depth == 20
    assert model.max_features == "sqrt"

    service = ML2Service(model)

    window = [
        create_sample(
            0,
        )
        for _ in range(8)
    ]

    diagnosis = service.predict(window)

    assert diagnosis in (0, 1)


def test_ml2_service_with_real_model_psc_json():
    """
    Teste de integração com uma janela PSC real extraída
    do conjunto de generalização utilizado para validação
    independente do modelo no Raspberry Pi.

    O arquivo JSON deve conter:

        label = 1
        features = 72 valores

    O teste reconstrói Samples a partir das 72 features para
    validar a cadeia:

        Samples
            ↓
        ML2Service
            ↓
        72 features
            ↓
        modelo real
    """

    model_path = Path("/home/ufuene/gdh-platform/models/ml2/RandomForest_ML2_cpu.pkl")

    window_path = Path("/home/ufuene/gdh-platform/ml2_psc_test_window.json")

    if not model_path.exists():
        pytest.skip(f"Modelo ML2 real não encontrado: {model_path}")

    if not window_path.exists():
        pytest.skip(f"Janela PSC de teste não encontrada: {window_path}")

    with window_path.open("r") as file:
        data = json.load(file)

    assert data["label"] == 1
    assert len(data["features"]) == 72

    features = data["features"]

    window = []

    for index in range(8):
        start = index * 9
        end = start + 9

        values = features[start:end]

        sample = Sample(
            timestamp=datetime.now(),
            irradiance=values[0],
            temperature=values[1],
            v_out=values[2],
            i_out=values[3],
            i_pv=values[4],
            v_pv=values[5],
            i_load=values[6],
            p_out=values[7],
            i_bat=values[8],
        )

        window.append(sample)

    assert len(window) == 8

    model = joblib.load(model_path)

    service = ML2Service(model)

    diagnosis = service.predict(window)

    probability_normal, probability_psc = service.predict_proba(window)

    assert diagnosis == 1

    assert isfinite(probability_normal)
    assert isfinite(probability_psc)

    assert isclose(
        probability_normal,
        0.0,
        abs_tol=1e-6,
    )

    assert isclose(
        probability_psc,
        1.0,
        abs_tol=1e-6,
    )
