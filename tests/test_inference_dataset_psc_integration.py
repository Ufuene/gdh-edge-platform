from datetime import datetime

from app.data_manager.sample import Sample
from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import WindowManager
from app.datasets.dataset_manager import DatasetManager
from app.inference.deviation_detector import DeviationDetector
from app.inference.inference_orchestrator import InferenceOrchestrator


class FakeML1Service:
    """
    ML1 determinístico para o teste de integração PSC.
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
    ML2 determinístico para o teste de integração PSC.

    O diagnóstico 1 representa PSC.
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


def create_orchestrator():
    """
    Cria o InferenceOrchestrator mínimo utilizado
    pelo teste.
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


def test_psc_inference_result_flows_into_dataset_manager():
    """
    Verifica o fluxo completo de uma janela PSC:

        InferenceOrchestrator
                ↓
        InferenceResult
                ↓
        DatasetManager
                ↓
        ML2DatasetWindow
                ↓
        label = 1

    O desvio ocorre no índice 3.

    Com:

        previous_samples = 3
        future_samples = 4

    a janela construída será:

        [0..7]
    """

    orchestrator = create_orchestrator()

    dataset_manager = DatasetManager()

    results = []

    # ----------------------------------------------------------
    # Processar oito amostras.
    #
    # O índice 3 apresenta desvio suficiente para iniciar
    # o evento PSC.
    # ----------------------------------------------------------

    for index in range(8):

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

        ml1_record, ml2_window = dataset_manager.process_sample(
            sample=sample,
            result=result,
        )

        # ------------------------------------------------------
        # Todas as amostras sem desvio são elegíveis para ML1.
        #
        # A amostra do índice 3 apresenta desvio e, portanto,
        # não é elegível.
        # ------------------------------------------------------

        if index == 3:
            assert ml1_record is None
        else:
            assert ml1_record is not None

        # ------------------------------------------------------
        # A janela PSC somente existe quando o contexto temporal
        # completo estiver disponível.
        # ------------------------------------------------------

        if index < 7:
            assert ml2_window is None
        else:
            assert ml2_window is not None

    # ==========================================================
    # RESULTADO FINAL DA INFERÊNCIA
    # ==========================================================

    result = results[-1]

    assert result.ml2_executed is True
    assert result.diagnosis == 1

    assert result.window_available is True

    assert result.window_start == 0
    assert result.window_end == 7

    # ==========================================================
    # A janela efetivamente utilizada pelo ML2 deve estar
    # preservada no InferenceResult.
    # ==========================================================

    assert result.window_samples is not None
    assert len(result.window_samples) == 8

    actual_timestamps = [
        sample.timestamp
        for sample in result.window_samples
    ]

    expected_timestamps = [
        create_sample(index).timestamp
        for index in range(8)
    ]

    assert actual_timestamps == expected_timestamps

    # ==========================================================
    # DATASET ML1
    # ==========================================================

    assert len(dataset_manager.get_ml1_records()) == 7

    # ==========================================================
    # DATASET ML2
    # ==========================================================

    assert len(dataset_manager.get_ml2_records()) == 1

    window = dataset_manager.get_ml2_records()[0]

    assert window.window_id == "psc-000001"

    assert window.window_start == 0
    assert window.window_end == 7

    assert window.label == 1

    assert len(window.samples) == 8

    # ==========================================================
    # A janela armazenada no dataset deve ser exatamente a
    # mesma janela preservada pelo InferenceResult.
    # ==========================================================

    dataset_timestamps = [
        sample.timestamp
        for sample in window.samples
    ]

    assert dataset_timestamps == expected_timestamps
