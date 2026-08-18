from datetime import datetime

from app.data_manager.sample import Sample
from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import WindowManager
from app.inference.deviation_detector import DeviationDetector
from app.inference.inference_orchestrator import InferenceOrchestrator


class FakeML1Service:
    """
    ML1 determinístico para o teste.
    """

    def __init__(self, predicted_power=58.8):
        self.predicted_power = predicted_power

    def predict(
        self,
        irradiance,
        temperature,
        v_out,
    ):
        return self.predicted_power


class FakeML2Service:
    """
    ML2 determinístico para o teste.
    """

    def predict(self, window):
        assert len(window) == 8
        return 1


def create_sample(
    index: int,
    p_out: float = 58.8,
) -> Sample:
    """
    Cria uma amostra identificável pelo índice.
    """

    return Sample(
        timestamp=datetime.fromtimestamp(float(index)),
        irradiance=800.0,
        temperature=35.0,
        v_pv=30.0,
        i_pv=5.0,
        v_out=14.7,
        i_out=4.0,
        i_bat=2.0,
        i_load=2.0,
        p_out=p_out,
    )


def create_orchestrator(
    predicted_power=58.8,
):
    """
    Cria um InferenceOrchestrator mínimo para o teste.
    """

    sample_buffer = SampleBuffer()

    window_manager = WindowManager(
        sample_buffer,
        previous_samples=3,
        future_samples=4,
    )

    deviation_detector = DeviationDetector(
        power_threshold=5.0,
    )

    return InferenceOrchestrator(
        sample_buffer=sample_buffer,
        ml1_service=FakeML1Service(predicted_power=predicted_power),
        ml2_service=FakeML2Service(),
        deviation_detector=deviation_detector,
        window_manager=window_manager,
    )


def test_ml2_result_preserves_the_exact_window_samples():
    """
    Verifica que o InferenceResult preserva exatamente
    as oito amostras utilizadas pelo ML2.

    A janela do ML2 é criada pelo WindowManager a partir
    de um evento de desvio.

    Com:

        previous_samples = 3
        future_samples = 4

    e desvio no índice 3, a janela será:

        [0..7]
    """

    orchestrator = create_orchestrator()

    results = []

    for index in range(8):

        # ------------------------------------------------------
        # Desvio no índice 3.
        #
        # ML1 prevê 58.8.
        # A amostra no índice 3 possui 50.0.
        #
        # deviation = 8.8 > threshold 5.0
        # ------------------------------------------------------

        if index == 3:
            sample = create_sample(
                index=index,
                p_out=50.0,
            )
        else:
            sample = create_sample(
                index=index,
                p_out=58.8,
            )

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

        results.append(result)

    # ----------------------------------------------------------
    # A janela [0..7] deve ser construída na chegada
    # da amostra de índice 7.
    # ----------------------------------------------------------

    result = results[-1]

    assert result.ml2_executed is True
    assert result.diagnosis == 1

    assert result.window_available is True

    assert result.window_start == 0
    assert result.window_end == 7

    # ----------------------------------------------------------
    # A janela deve conter exatamente oito amostras.
    # ----------------------------------------------------------

    assert result.window_samples is not None
    assert len(result.window_samples) == 8

    # ----------------------------------------------------------
    # Verificar a sequência exata das amostras.
    # ----------------------------------------------------------

    timestamps = [sample.timestamp for sample in result.window_samples]

    expected_timestamps = [create_sample(index).timestamp for index in range(8)]

    assert timestamps == expected_timestamps


def test_no_ml2_execution_means_no_window_samples():
    """
    Verifica que, quando ML2 ainda não foi executado,
    window_samples permanece None.
    """

    orchestrator = create_orchestrator()

    sample = create_sample(0)

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=0,
    )

    assert result.ml2_executed is False
    assert result.diagnosis is None
    assert result.window_available is False

    assert result.window_samples is None
