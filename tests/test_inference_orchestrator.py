from datetime import datetime

from app.data_manager.sample import Sample
from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import WindowManager, WindowState
from app.inference.deviation_detector import DeviationDetector
from app.inference.inference_orchestrator import InferenceOrchestrator
from app.inference.inference_result import InferenceResult


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


class MockML2Service:
    """
    Serviço ML2 simulado utilizado nos testes do
    InferenceOrchestrator.
    """

    def __init__(self, diagnosis=1):
        self.diagnosis = diagnosis
        self.received_window = None

    def predict(self, window):
        """
        Registra a janela recebida e retorna
        um diagnóstico simulado.
        """

        self.received_window = window

        return self.diagnosis


def create_sample(
    power: float,
    index: int = 0,
) -> Sample:
    """
    Cria uma amostra de teste.
    """

    return Sample(
        timestamp=datetime.now(),
        irradiance=850.0 + index,
        temperature=30.0 + index,
        v_pv=34.5 + index,
        i_pv=7.8 + index,
        v_out=14.6 + index,
        i_out=8.2 + index,
        i_bat=3.1 + index,
        i_load=5.0 + index,
        p_out=power,
    )


def create_orchestrator(
    predicted_power=120.0,
    diagnosis=1,
):
    """
    Cria um InferenceOrchestrator com todos os componentes
    necessários para os testes.
    """

    ml1_service = MockML1Service(
        predicted_power=predicted_power,
    )

    ml2_service = MockML2Service(
        diagnosis=diagnosis,
    )

    detector = DeviationDetector(
        power_threshold=5.0,
    )

    buffer = SampleBuffer()

    window_manager = WindowManager(buffer)

    orchestrator = InferenceOrchestrator(
        sample_buffer=buffer,
        ml1_service=ml1_service,
        ml2_service=ml2_service,
        deviation_detector=detector,
        window_manager=window_manager,
    )

    return (
        orchestrator,
        ml1_service,
        ml2_service,
        detector,
        window_manager,
        buffer,
    )


def test_orchestrator_creation():
    """
    Verifica a criação correta do InferenceOrchestrator.
    """

    (
        orchestrator,
        ml1_service,
        ml2_service,
        detector,
        window_manager,
        buffer,
    ) = create_orchestrator()

    assert orchestrator._sample_buffer == buffer
    assert orchestrator._ml1_service == ml1_service
    assert orchestrator._ml2_service == ml2_service
    assert orchestrator._deviation_detector == detector
    assert orchestrator._window_manager == window_manager


def test_process_sample_without_deviation():
    """
    Verifica que uma amostra sem desvio:

    - é adicionada ao SampleBuffer;
    - é processada pelo ML1;
    - não ativa o WindowManager;
    - não executa o ML2;
    - retorna um InferenceResult sem diagnóstico.
    """

    (
        orchestrator,
        ml1_service,
        ml2_service,
        _,
        window_manager,
        buffer,
    ) = create_orchestrator(
        predicted_power=118.0,
    )

    sample = create_sample(120.0)

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=0,
    )

    assert isinstance(result, InferenceResult)

    assert result.predicted_power == 118.0
    assert result.deviation == 2.0
    assert result.deviation_detected is False
    assert result.diagnosis is None

    assert buffer.size() == 1
    assert window_manager._state == WindowState.IDLE
    assert window_manager._deviation_index is None

    assert sample.predicted_power == 118.0
    assert sample.deviation == 2.0
    assert sample.deviation_detected is False
    assert sample.diagnosis is None

    assert ml1_service.received_inputs == {
        "irradiance": 850.0,
        "temperature": 30.0,
        "v_out": 14.6,
    }

    assert ml2_service.received_window is None


def test_process_sample_with_early_deviation_does_not_start_event():
    """
    Verifica que um desvio detectado antes da existência
    do contexto temporal mínimo não inicia um evento.

    Com previous_samples=3, o índice 0 não possui
    três amostras anteriores.

    O ML1 e o DeviationDetector continuam funcionando,
    mas o WindowManager permanece em IDLE.
    """

    (
        orchestrator,
        _,
        ml2_service,
        _,
        window_manager,
        buffer,
    ) = create_orchestrator(
        predicted_power=120.0,
    )

    sample = create_sample(100.0)

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=0,
    )

    assert isinstance(result, InferenceResult)

    assert result.predicted_power == 120.0
    assert result.deviation == 20.0
    assert result.deviation_detected is True
    assert result.diagnosis is None

    assert buffer.size() == 1

    assert window_manager._state == WindowState.IDLE
    assert window_manager._deviation_index is None

    assert ml2_service.received_window is None

    assert sample.predicted_power == 120.0
    assert sample.deviation == 20.0
    assert sample.deviation_detected is True
    assert sample.diagnosis is None


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
        ml2_service,
        _,
        window_manager,
        _,
    ) = create_orchestrator(
        predicted_power=120.0,
    )

    sample = create_sample(115.0)

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=0,
    )

    assert isinstance(result, InferenceResult)

    assert result.predicted_power == 120.0
    assert result.deviation == 5.0
    assert result.deviation_detected is False
    assert result.diagnosis is None

    assert window_manager._state == WindowState.IDLE
    assert window_manager._deviation_index is None
    assert ml2_service.received_window is None


def test_process_sample_passes_correct_sample_index():
    """
    Verifica que o índice fornecido ao orquestrador é encaminhado
    corretamente ao WindowManager quando ocorre um desvio.
    """

    (
        orchestrator,
        _,
        _,
        _,
        window_manager,
        _,
    ) = create_orchestrator(
        predicted_power=120.0,
    )

    sample = create_sample(100.0)

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=7,
    )

    assert isinstance(result, InferenceResult)
    assert result.deviation_detected is True

    assert window_manager._state == WindowState.WAITING_WINDOW
    assert window_manager._deviation_index == 7


def test_process_sample_uses_ml1_output_as_predicted_power():
    """
    Verifica explicitamente que o valor produzido pelo ML1
    é utilizado como potência de referência pelo
    DeviationDetector.
    """

    (
        orchestrator,
        ml1_service,
        _,
        _,
        window_manager,
        _,
    ) = create_orchestrator(
        predicted_power=120.0,
    )

    sample = create_sample(
        100.0,
        index=3,
    )

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=3,
    )

    assert isinstance(result, InferenceResult)

    assert result.predicted_power == 120.0
    assert result.deviation == 20.0
    assert result.deviation_detected is True
    assert result.diagnosis is None

    assert ml1_service.received_inputs == {
        "irradiance": 853.0,
        "temperature": 33.0,
        "v_out": 17.6,
    }

    assert window_manager._state == WindowState.WAITING_WINDOW
    assert window_manager._deviation_index == 3


def test_process_sample_is_point_by_point_for_ml1():
    """
    Verifica que o ML1 continua sendo executado
    individualmente para cada amostra.

    Também verifica que desvios detectados durante o
    warm-up não iniciam eventos temporais.
    """

    (
        orchestrator,
        ml1_service,
        ml2_service,
        _,
        window_manager,
        buffer,
    ) = create_orchestrator(
        predicted_power=120.0,
    )

    sample_1 = create_sample(
        120.0,
        index=0,
    )

    result_1 = orchestrator.process_sample(
        sample=sample_1,
        sample_index=0,
    )

    assert isinstance(result_1, InferenceResult)
    assert result_1.deviation_detected is False
    assert result_1.diagnosis is None

    assert buffer.size() == 1
    assert window_manager._state == WindowState.IDLE

    assert ml1_service.received_inputs == {
        "irradiance": 850.0,
        "temperature": 30.0,
        "v_out": 14.6,
    }

    assert ml2_service.received_window is None

    sample_2 = create_sample(
        100.0,
        index=1,
    )

    result_2 = orchestrator.process_sample(
        sample=sample_2,
        sample_index=1,
    )

    assert isinstance(result_2, InferenceResult)
    assert result_2.deviation_detected is True
    assert result_2.diagnosis is None

    assert buffer.size() == 2

    # Índice 1 ainda não possui três amostras anteriores.
    assert window_manager._state == WindowState.IDLE
    assert window_manager._deviation_index is None

    assert ml1_service.received_inputs == {
        "irradiance": 851.0,
        "temperature": 31.0,
        "v_out": 15.6,
    }

    assert ml2_service.received_window is None


def test_process_sample_builds_window_and_runs_ml2():
    """
    Verifica o fluxo completo:

        Sample
        ↓
        SampleBuffer
        ↓
        ML1
        ↓
        DeviationDetector
        ↓
        WindowManager
        ↓
        8 Samples
        ↓
        ML2
        ↓
        InferenceResult
    """

    (
        orchestrator,
        ml1_service,
        ml2_service,
        _,
        window_manager,
        buffer,
    ) = create_orchestrator(
        predicted_power=120.0,
        diagnosis=1,
    )

    results = []

    for index in range(8):
        power = 100.0 if index == 3 else 120.0

        sample = create_sample(
            power=power,
            index=index,
        )

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

        results.append(result)

    assert buffer.size() == 8

    assert window_manager._state == WindowState.ACTIVE_EVENT
    assert window_manager.has_complete_window()

    window = window_manager.get_current_window()

    assert window is not None
    assert len(window) == 8

    assert ml2_service.received_window == window

    # A amostra que disparou o evento está na quarta posição.
    assert window[3].p_out == 100.0

    for index, sample in enumerate(window):
        expected_power = 100.0 if index == 3 else 120.0
        assert sample.p_out == expected_power

    # O resultado da amostra que disparou o evento
    # ainda não possui diagnóstico.
    assert results[3].deviation_detected is True
    assert results[3].diagnosis is None

    # O diagnóstico torna-se disponível quando a janela
    # completa é construída na amostra 7.
    final_result = results[7]

    assert isinstance(final_result, InferenceResult)

    assert final_result.predicted_power == 120.0
    assert final_result.deviation == 0.0
    assert final_result.deviation_detected is False
    assert final_result.diagnosis == 1

    assert ml1_service.received_inputs == {
        "irradiance": 857.0,
        "temperature": 37.0,
        "v_out": 21.6,
    }


def test_process_sample_builds_normal_ml2_diagnosis():
    """
    Verifica que o diagnóstico NORMAL produzido pelo ML2
    é propagado através do InferenceResult.
    """

    (
        orchestrator,
        _,
        ml2_service,
        _,
        window_manager,
        _,
    ) = create_orchestrator(
        predicted_power=120.0,
        diagnosis=0,
    )

    results = []

    for index in range(8):
        power = 100.0 if index == 3 else 120.0

        sample = create_sample(
            power=power,
            index=index,
        )

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

        results.append(result)

    assert window_manager.has_complete_window()
    assert ml2_service.received_window is not None

    assert results[-1].diagnosis == 0
