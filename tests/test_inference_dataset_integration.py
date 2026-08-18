from datetime import datetime, timedelta

from app.data_manager.sample import Sample
from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import WindowManager
from app.datasets.dataset_manager import DatasetManager
from app.inference.deviation_detector import DeviationDetector
from app.inference.inference_orchestrator import InferenceOrchestrator


class FakeML1Service:
    """
    ML1 determinístico para o teste de integração.
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
    ML2 determinístico para o teste de integração.
    """

    def predict(self, window):
        assert len(window) == 8
        return 1


def create_sample(sequence: int) -> Sample:
    """
    Cria uma amostra saudável identificável pela sequência.
    """

    return Sample(
        timestamp=datetime(2026, 8, 17) + timedelta(seconds=sequence),
        irradiance=800.0,
        temperature=35.0,
        v_pv=30.0,
        i_pv=5.0,
        v_out=14.7,
        i_out=4.0,
        i_bat=2.0,
        i_load=2.0,
        p_out=58.8,
    )


def create_orchestrator():
    """
    Cria o InferenceOrchestrator mínimo utilizado
    pelo teste de integração.
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
        ml1_service=FakeML1Service(),
        ml2_service=FakeML2Service(),
        deviation_detector=deviation_detector,
        window_manager=window_manager,
    )


def test_healthy_inference_result_flows_into_dataset_manager():
    """
    Verifica o fluxo:

        InferenceOrchestrator
                ↓
        InferenceResult
                ↓
        DatasetManager

    Para oito amostras saudáveis:

        8 registros ML1
        +
        1 janela ML2 normal
    """

    orchestrator = create_orchestrator()

    dataset_manager = DatasetManager()

    results = []

    # ----------------------------------------------------------
    # Processar oito amostras saudáveis.
    # ----------------------------------------------------------

    for sequence in range(1, 9):

        sample = create_sample(sequence)

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=sequence - 1,
        )

        results.append(result)

        ml1_record, ml2_window = dataset_manager.process_sample(
            sample=sample,
            result=result,
        )

        # ------------------------------------------------------
        # Todas as amostras saudáveis devem entrar no ML1.
        # ------------------------------------------------------

        assert ml1_record is not None

        # ------------------------------------------------------
        # A janela normal somente deve surgir na oitava
        # amostra elegível.
        # ------------------------------------------------------

        if sequence < 8:
            assert ml2_window is None
        else:
            assert ml2_window is not None

    # ----------------------------------------------------------
    # Dataset ML1
    # ----------------------------------------------------------

    assert len(dataset_manager.get_ml1_records()) == 8

    # ----------------------------------------------------------
    # Dataset ML2
    # ----------------------------------------------------------

    assert len(dataset_manager.get_ml2_records()) == 1

    window = dataset_manager.get_ml2_records()[0]

    assert window.window_id == "normal-000001"

    assert window.window_start == 1

    assert window.window_end == 8

    assert window.label == 0

    assert len(window.samples) == 8

    # ----------------------------------------------------------
    # Verificar que as oito amostras preservadas pelo
    # DatasetManager são exatamente as oito amostras recebidas.
    # ----------------------------------------------------------

    actual_timestamps = [sample.timestamp for sample in window.samples]

    expected_timestamps = [
        create_sample(sequence).timestamp for sequence in range(1, 9)
    ]

    assert actual_timestamps == expected_timestamps

    # ----------------------------------------------------------
    # O resultado da inferência continua independente do
    # DatasetManager.
    # ----------------------------------------------------------

    assert results[-1].deviation_detected is False

    assert results[-1].ml2_executed is False
