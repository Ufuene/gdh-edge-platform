from math import isfinite
from pathlib import Path

import joblib

from app.inference.ml1_service import ML1Service


class MockML1Model:
    """
    Modelo simulado utilizado nos testes do ML1Service.
    """

    def __init__(self):
        self.received_features = None

    def predict(self, features):
        """
        Registra as entradas recebidas e retorna
        uma potência prevista simulada.
        """

        self.received_features = features

        return [120.0]


def test_ml1_service_creation():
    """
    Verifica a criação do ML1Service.
    """

    model = MockML1Model()

    service = ML1Service(model)

    assert service._model == model


def test_predict_returns_predicted_power():
    """
    Verifica se o ML1 retorna a potência de referência.
    """

    model = MockML1Model()

    service = ML1Service(model)

    predicted_power = service.predict(
        irradiance=850.0,
        temperature=30.0,
        v_out=14.6,
    )

    assert predicted_power == 120.0


def test_predict_uses_irradiance_temperature_and_vout():
    """
    Verifica se as três entradas do ML1 são encaminhadas
    corretamente ao modelo.
    """

    model = MockML1Model()

    service = ML1Service(model)

    service.predict(
        irradiance=900.0,
        temperature=32.0,
        v_out=14.7,
    )

    assert model.received_features == [[900.0, 32.0, 14.7]]


def test_predict_is_point_by_point():
    """
    Verifica que cada chamada ao ML1 corresponde a uma
    única amostra, sem construção de janela temporal.
    """

    model = MockML1Model()

    service = ML1Service(model)

    result_1 = service.predict(
        irradiance=800.0,
        temperature=25.0,
        v_out=14.5,
    )

    result_2 = service.predict(
        irradiance=900.0,
        temperature=30.0,
        v_out=14.7,
    )

    assert result_1 == 120.0
    assert result_2 == 120.0

    assert model.received_features == [[900.0, 30.0, 14.7]]


def test_ml1_service_with_real_model():
    """
    Teste de integração do ML1Service com o modelo ML1 real.

    O teste carrega o RandomForestRegressor treinado para
    execução em CPU no Raspberry Pi e verifica uma inferência
    real através do ML1Service.

    O modelo foi treinado com as características:

        [Irradiancia, Temperatura, Vout]

    e possui 100 estimadores.

    Este teste deve ser executado no Raspberry Pi onde o
    artefato real do modelo está disponível.
    """

    model_path = Path("/home/ufuene/gdh-platform/models/ml1/RandomForest_ML1_CPU.pkl")

    if not model_path.exists():
        import pytest

        pytest.skip(f"Modelo ML1 real não encontrado: {model_path}")

    model = joblib.load(model_path)

    assert model.n_features_in_ == 3
    assert model.n_estimators == 100

    service = ML1Service(model)

    predicted_power = service.predict(
        irradiance=62.8,
        temperature=12.6445,
        v_out=12.034455,
    )

    assert isinstance(predicted_power, float)
    assert isfinite(predicted_power)

    # Resultado de referência obtido anteriormente
    # diretamente no Raspberry Pi com o mesmo modelo.
    assert abs(predicted_power - 14.93917673) < 1e-6
