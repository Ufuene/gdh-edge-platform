from datetime import datetime

from app.data_manager.sample import Sample
from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import WindowManager, WindowState
from app.inference.deviation_detector import DeviationDetector
from app.inference.inference_orchestrator import InferenceOrchestrator


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


def test_orchestrator_creation():
    """
    Verifica a criação correta do InferenceOrchestrator.
    """

    detector = DeviationDetector(
        power_threshold=5.0,
    )

    buffer = SampleBuffer()

    window_manager = WindowManager(buffer)

    orchestrator = InferenceOrchestrator(
        deviation_detector=detector,
        window_manager=window_manager,
    )

    assert orchestrator._deviation_detector == detector
    assert orchestrator._window_manager == window_manager


def test_process_sample_without_deviation():
    """
    Verifica que uma amostra sem desvio não ativa
    o WindowManager.
    """

    detector = DeviationDetector(
        power_threshold=5.0,
    )

    buffer = SampleBuffer()

    window_manager = WindowManager(buffer)

    orchestrator = InferenceOrchestrator(
        deviation_detector=detector,
        window_manager=window_manager,
    )

    sample = create_sample(120.0)

    buffer.add_sample(sample)

    result = orchestrator.process_sample(
        sample=sample,
        predicted_power=118.0,
        sample_index=0,
    )

    assert result is False

    assert window_manager._state == WindowState.IDLE

    assert window_manager._deviation_index is None


def test_process_sample_with_deviation():
    """
    Verifica que um desvio detectado é encaminhado
    ao WindowManager com o índice correto.
    """

    detector = DeviationDetector(
        power_threshold=5.0,
    )

    buffer = SampleBuffer()

    window_manager = WindowManager(buffer)

    orchestrator = InferenceOrchestrator(
        deviation_detector=detector,
        window_manager=window_manager,
    )

    sample = create_sample(100.0)

    buffer.add_sample(sample)

    result = orchestrator.process_sample(
        sample=sample,
        predicted_power=120.0,
        sample_index=0,
    )

    assert result is True

    assert window_manager._state == WindowState.WAITING_WINDOW

    assert window_manager._deviation_index == 0


def test_process_sample_when_error_equals_threshold():
    """
    Verifica que atingir exatamente o limiar não caracteriza
    um desvio.

    A condição utilizada pelo DeviationDetector é:

        error > power_threshold
    """

    detector = DeviationDetector(
        power_threshold=5.0,
    )

    buffer = SampleBuffer()

    window_manager = WindowManager(buffer)

    orchestrator = InferenceOrchestrator(
        deviation_detector=detector,
        window_manager=window_manager,
    )

    sample = create_sample(115.0)

    buffer.add_sample(sample)

    result = orchestrator.process_sample(
        sample=sample,
        predicted_power=120.0,
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

    detector = DeviationDetector(
        power_threshold=5.0,
    )

    buffer = SampleBuffer()

    window_manager = WindowManager(buffer)

    orchestrator = InferenceOrchestrator(
        deviation_detector=detector,
        window_manager=window_manager,
    )

    sample = create_sample(100.0)

    buffer.add_sample(sample)

    result = orchestrator.process_sample(
        sample=sample,
        predicted_power=120.0,
        sample_index=7,
    )

    assert result is True

    assert window_manager._state == WindowState.WAITING_WINDOW

    assert window_manager._deviation_index == 7
