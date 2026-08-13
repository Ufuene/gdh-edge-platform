from datetime import datetime

from app.data_manager.sample import Sample
from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import WindowManager, WindowState
from app.inference.deviation_detector import DeviationDetector
from app.inference.inference_orchestrator import InferenceOrchestrator


class MockML1Service:
    """
    Serviço ML1 simulado utilizado nos testes do
    InferenceOrchestrator.
    """

    def __init__(self, predicted_power=120.0):
        self.predicted_power = predicted_power
        self.received_inputs = None

    def predict(
        self,
        irradiance: float,
        temperature: float,
        v_out: float,
    ) -> float:
        """
        Registra as entradas recebidas e retorna
        uma potência prevista simulada.
        """

        self.received_inputs = {
            "irradiance": irradiance,
            "temperature": temperature,
            "v_out": v_out,
        }

        return self.predicted_power


def create_sample(power: float) -> Sample:
    """
    Cria uma amostra de teste com a potência especificada.
    """

    return Sample(
        timestamp=datetime.now(),
        irradiance=850,
        temperature=30,
        v_pv=34.5,
        i_pv=7.8,
        v_out=14.6,
        i_out=8.2,
        i_bat=3.1,
        i_load=5.0,
        p_out=power,
    )


def create_orchestrator(predicted_power=120.0):
    """
    Cria um InferenceOrchestrator com todos os componentes
    necessários para os testes.
    """

    ml1_service = MockML1Service(
        predicted_power=predicted_power,
    )

    detector = DeviationDetector(
        power_threshold=5.0,
    )

    buffer = SampleBuffer()

    window_manager = WindowManager(buffer)

    orchestrator = InferenceOrchestrator(
        ml1_service=ml1_service,
        deviation_detector=detector,
        window_manager=window_manager,
    )

    return (
        orchestrator,
        ml1_service,
        detector,
        window_manager,
    )


def test_orchestrator_creation():
    """
    Verifica a criação correta do InferenceOrchestrator.
    """

    (
        orchestrator,
        ml1_service,
        detector,
        window_manager,
    ) = create_orchestrator()

    assert orchestrator._ml1_service == ml1_service
    assert orchestrator._deviation_detector == detector
    assert orchestrator._window_manager == window_manager


def test_process_sample_without_deviation():
    """
    Verifica que uma amostra sem desvio não ativa
    o WindowManager.
    """

    (
        orchestrator,
        ml1_service,
        _,
        window_manager,
    ) = create_orchestrator(
        predicted_power=118.0,
    )

    sample = create_sample(120.0)

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=0,
    )

    assert result is False

    assert window_manager._state == WindowState.IDLE

    assert window_manager._deviation_index is None

    assert ml1_service.received_inputs == {
        "irradiance": 850,
        "temperature": 30,
        "v_out": 14.6,
    }


def test_process_sample_with_deviation():
    """
    Verifica que um desvio detectado pelo DeviationDetector
    é encaminhado ao WindowManager com o índice correto.
    """

    (
        orchestrator,
        ml1_service,
        _,
        window_manager,
    ) = create_orchestrator(
        predicted_power=120.0,
    )

    sample = create_sample(100.0)

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=0,
    )

    assert result is True

    assert window_manager._state == WindowState.WAITING_WINDOW

    assert window_manager._deviation_index == 0

    assert ml1_service.received_inputs == {
        "irradiance": 850,
        "temperature": 30,
        "v_out": 14.6,
    }


def test_process_sample_when_error_equals_threshold():
    """
    Verifica que atingir exatamente o limiar não caracteriza
    um desvio.

    A condição utilizada pelo DeviationDetector é:

        error > power_threshold
    """

    (
        orchestrator,
        _,
        _,
        window_manager,
    ) = create_orchestrator(
        predicted_power=120.0,
    )

    sample = create_sample(115.0)

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=0,
    )

    assert result is False

    assert window_manager._state == WindowState.IDLE

    assert window_manager._deviation_index is None


def test_process_sample_passes_correct_sample_index():
    """
    Verifica que o índice fornecido ao orquestrador é encaminhado
    corretamente ao WindowManager quando ocorre um desvio.
    """

    (
        orchestrator,
        _,
        _,
        window_manager,
    ) = create_orchestrator(
        predicted_power=120.0,
    )

    sample = create_sample(100.0)

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=7,
    )

    assert result is True

    assert window_manager._state == WindowState.WAITING_WINDOW

    assert window_manager._deviation_index == 7


def test_process_sample_uses_ml1_output_as_predicted_power():
    """
    Verifica explicitamente que o valor produzido pelo ML1
    é utilizado como potência de referência pelo
    DeviationDetector.

    O ML1 retorna 120 W e a amostra apresenta 100 W.
    Portanto:

        |100 - 120| = 20 W

    Como o limiar é 5 W, deve existir desvio.
    """

    (
        orchestrator,
        ml1_service,
        _,
        window_manager,
    ) = create_orchestrator(
        predicted_power=120.0,
    )

    sample = create_sample(100.0)

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=3,
    )

    assert result is True

    assert ml1_service.received_inputs == {
        "irradiance": 850,
        "temperature": 30,
        "v_out": 14.6,
    }

    assert window_manager._state == WindowState.WAITING_WINDOW

    assert window_manager._deviation_index == 3


def test_process_sample_is_point_by_point():
    """
    Verifica que o ML1 é executado individualmente para
    cada amostra processada.

    O orquestrador não constrói janelas para o ML1.
    """

    (
        orchestrator,
        ml1_service,
        _,
        window_manager,
    ) = create_orchestrator(
        predicted_power=120.0,
    )

    sample_1 = create_sample(120.0)

    result_1 = orchestrator.process_sample(
        sample=sample_1,
        sample_index=0,
    )

    assert result_1 is False

    assert window_manager._state == WindowState.IDLE

    assert ml1_service.received_inputs == {
        "irradiance": 850,
        "temperature": 30,
        "v_out": 14.6,
    }

    sample_2 = create_sample(100.0)

    result_2 = orchestrator.process_sample(
        sample=sample_2,
        sample_index=1,
    )

    assert result_2 is True

    assert window_manager._state == WindowState.WAITING_WINDOW

    assert window_manager._deviation_index == 1

    assert ml1_service.received_inputs == {
        "irradiance": 850,
        "temperature": 30,
        "v_out": 14.6,
    }
