from datetime import datetime

from app.data_manager.sample import Sample
from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import WindowManager, WindowState
from app.inference.ml2_service import ML2Service


class MockML2Model:
    """
    Modelo ML2 simulado para testar a integração
    WindowManager → ML2Service.
    """

    def __init__(self, diagnosis=1):
        self.diagnosis = diagnosis
        self.received_features = None

    def predict(self, features):
        """
        Registra as features recebidas e retorna
        um diagnóstico simulado.
        """

        self.received_features = features

        return [self.diagnosis]


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


def test_window_manager_provides_complete_window_to_ml2():
    """
    Verifica a integração básica entre WindowManager e ML2Service.

    O WindowManager deve construir uma janela de exatamente
    8 amostras, que então é entregue ao ML2Service.
    """

    buffer = SampleBuffer()

    window_manager = WindowManager(buffer)

    model = MockML2Model(diagnosis=1)

    ml2_service = ML2Service(model)

    for index in range(12):
        buffer.add_sample(create_sample(index))

    window_manager.notify_deviation(7)

    window_manager.add_sample()

    assert window_manager._state == WindowState.ACTIVE_EVENT
    assert window_manager.has_complete_window()

    window = window_manager.get_current_window()

    assert window is not None
    assert len(window) == 8

    diagnosis = ml2_service.predict(window)

    assert diagnosis == 1

    assert model.received_features is not None
    assert model.received_features.shape == (1, 72)
