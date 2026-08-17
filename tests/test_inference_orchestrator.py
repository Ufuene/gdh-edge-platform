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


def test_event_lifecycle_with_hysteresis():
    """
    Verifica o ciclo integrado de um evento:

        desvio válido
            ↓
        WAITING_WINDOW
            ↓
        janela completa
            ↓
        ACTIVE_EVENT
            ↓
        perda do desvio
            ↓
        HYSTERESIS
            ↓
        novo desvio
            ↓
        ACTIVE_EVENT
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
        diagnosis=1,
    )

    results = []

    # ==========================================================
    # Amostras 0, 1 e 2:
    # ainda não existe contexto suficiente para iniciar evento.
    # ==========================================================

    for index in range(3):
        sample = create_sample(
            power=120.0,
            index=index,
        )

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

        results.append(result)

        assert window_manager._state == WindowState.IDLE

    # ==========================================================
    # Índice 3:
    # primeiro desvio válido.
    # ==========================================================

    sample = create_sample(
        power=100.0,
        index=3,
    )

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=3,
    )

    results.append(result)

    assert result.deviation_detected is True
    assert result.diagnosis is None

    assert window_manager._state == WindowState.WAITING_WINDOW
    assert window_manager._deviation_index == 3

    # ==========================================================
    # Índices 4, 5 e 6:
    # fornecem contexto futuro.
    # ==========================================================

    for index in range(4, 7):
        sample = create_sample(
            power=120.0,
            index=index,
        )

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

        results.append(result)

        assert window_manager._state == WindowState.WAITING_WINDOW

    # ==========================================================
    # Índice 7:
    # quarta amostra futura.
    #
    # A janela [0..7] fica completa.
    # ==========================================================

    sample = create_sample(
        power=120.0,
        index=7,
    )

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=7,
    )

    results.append(result)

    assert buffer.size() == 8

    assert window_manager.has_complete_window()
    assert window_manager._state == WindowState.ACTIVE_EVENT

    assert ml2_service.received_window is not None
    assert len(ml2_service.received_window) == 8

    assert result.diagnosis == 1

    # ==========================================================
    # A amostra 7 completa a primeira janela.
    #
    # Embora não exista mais desvio nesta amostra, ela pertence
    # ao processo de construção/classificação da janela.
    # Portanto, o evento permanece ACTIVE_EVENT.
    # ==========================================================

    assert result.deviation_detected is False

    assert window_manager._state == WindowState.ACTIVE_EVENT
    assert window_manager._hysteresis_counter == 0

    # ==========================================================
    # Índice 8:
    # primeira amostra posterior à janela completa sem desvio.
    #
    # Agora sim começa a histerese.
    # ==========================================================

    sample = create_sample(
        power=120.0,
        index=8,
    )

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=8,
    )

    assert result.deviation_detected is False

    assert window_manager._state == WindowState.HYSTERESIS
    assert window_manager._hysteresis_counter == 1

    # ==========================================================
    # Índice 9:
    # novo desvio durante a histerese.
    #
    # O evento deve ser reativado.
    # ==========================================================

    sample = create_sample(
        power=100.0,
        index=9,
    )

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=9,
    )

    assert result.deviation_detected is True

    assert window_manager._state == WindowState.ACTIVE_EVENT
    assert window_manager._hysteresis_counter == 0


def test_event_lifecycle_ends_after_hysteresis():
    """
    Verifica que um evento ativo termina após
    hysteresis_samples ciclos consecutivos sem desvio.
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
        diagnosis=1,
    )

    # ==========================================================
    # Criar uma janela válida com desvio no índice 3.
    # ==========================================================

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

    # ==========================================================
    # A janela deve estar completa e o evento ativo.
    # ==========================================================

    assert buffer.size() == 8
    assert window_manager.has_complete_window()
    assert window_manager._state == WindowState.ACTIVE_EVENT

    assert ml2_service.received_window is not None
    assert result.diagnosis == 1

    # ==========================================================
    # A primeira amostra posterior à janela sem desvio
    # inicia a histerese.
    # ==========================================================

    sample = create_sample(
        power=120.0,
        index=8,
    )

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=8,
    )

    assert result.deviation_detected is False

    assert window_manager._state == WindowState.HYSTERESIS
    assert window_manager._hysteresis_counter == 1

    # ==========================================================
    # Mais 6 amostras sem desvio.
    #
    # O contador deve chegar a 7, mas ainda não encerrar.
    # ==========================================================

    for index in range(9, 15):

        sample = create_sample(
            power=120.0,
            index=index,
        )

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

        assert result.deviation_detected is False

    assert window_manager._state == WindowState.HYSTERESIS
    assert window_manager._hysteresis_counter == 7

    # ==========================================================
    # O oitavo ciclo sem desvio encerra o evento.
    # ==========================================================

    sample = create_sample(
        power=120.0,
        index=15,
    )

    result = orchestrator.process_sample(
        sample=sample,
        sample_index=15,
    )

    assert result.deviation_detected is False

    assert window_manager._state == WindowState.IDLE
    assert window_manager._deviation_index is None
    assert window_manager._current_window is None
    assert window_manager._hysteresis_counter == 0


def test_orchestrator_does_not_reclassify_consumed_window():
    """
    Verifica que uma janela já consumida pelo pipeline
    não é enviada novamente ao ML2.

    Fluxo:

        primeira janela
            ↓
        ML2
            ↓
        consume
            ↓
        novas amostras insuficientes para nova janela
            ↓
        ML2 não deve ser chamado novamente
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
        diagnosis=1,
    )

    # ==========================================================
    # Construir a primeira janela [0..7]
    # ==========================================================

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

    # ==========================================================
    # A primeira janela deve ter sido classificada.
    # ==========================================================

    assert window_manager.has_complete_window()
    assert window_manager.is_current_window_consumed() is True

    first_window = window_manager.get_current_window()

    assert first_window is not None
    assert len(first_window) == 8

    assert ml2_service.received_window is first_window

    assert results[7].diagnosis == 1

    # ==========================================================
    # Adicionar algumas amostras posteriores, mas ainda sem
    # completar uma segunda janela.
    # ==========================================================

    for index in range(8, 12):

        sample = create_sample(
            power=120.0,
            index=index,
        )

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

        assert result.diagnosis is None

    # ==========================================================
    # A janela anterior continua armazenada, mas permanece
    # consumida.
    # ==========================================================

    assert window_manager.has_complete_window()
    assert window_manager.is_current_window_consumed() is True

    assert window_manager.get_current_window() is first_window

    assert buffer.size() == 12

    # O ML2 não deve ter recebido uma nova janela.
    assert ml2_service.received_window is first_window


def test_orchestrator_builds_next_non_overlapping_window_after_consumption():
    """
    Verifica a integração entre o InferenceOrchestrator e o
    mecanismo de construção de janelas não sobrepostas.

    Primeira janela:

        [0, 1, 2, 3, 4, 5, 6, 7]

    Segunda janela:

        [8, 9, 10, 11, 12, 13, 14, 15]
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
        diagnosis=1,
    )

    # ==========================================================
    # Primeira janela.
    #
    # Desvio no índice 3.
    # ==========================================================

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

    assert result.diagnosis == 1

    assert window_manager.has_complete_window()
    assert window_manager.is_current_window_consumed() is True

    assert window_manager.get_current_window_start_index() == 0
    assert window_manager.get_current_window_end_index() == 7
    assert window_manager.get_next_window_start_index() == 8

    first_window = window_manager.get_current_window()

    assert first_window is not None
    assert len(first_window) == 8

    # ==========================================================
    # Forçar explicitamente a construção da próxima janela.
    #
    # O WindowManager já possui a regra temporal:
    #
    # próxima janela = start 8
    # ==========================================================

    for index in range(8, 16):
        buffer.add_sample(
            create_sample(
                power=120.0,
                index=index,
            )
        )

    window_manager.build_next_window()

    # ==========================================================
    # A segunda janela deve ser não sobreposta.
    # ==========================================================

    assert window_manager.has_complete_window()

    second_window = window_manager.get_current_window()

    assert second_window is not None
    assert len(second_window) == 8

    assert window_manager.get_current_window_start_index() == 8
    assert window_manager.get_current_window_end_index() == 15

    assert second_window is not first_window

    for position, sample in enumerate(second_window):
        assert sample is buffer.get_sample(8 + position)

    # ==========================================================
    # A segunda janela ainda não foi consumida.
    # ==========================================================

    assert window_manager.is_current_window_consumed() is False

    # O ML2 ainda não recebeu a segunda janela, pois ela foi
    # construída diretamente pelo WindowManager.
    assert ml2_service.received_window is first_window


def test_orchestrator_classifies_second_non_overlapping_window():
    """
    Verifica o ciclo completo de duas janelas não sobrepostas
    através do pipeline.

    Janela 1:

        [0..7]

    Janela 2:

        [8..15]

    Cada janela deve ser classificada exatamente uma vez.
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
        diagnosis=1,
    )

    # ==========================================================
    # Primeira janela [0..7].
    #
    # Desvio no índice 3.
    # ==========================================================

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

    assert result.diagnosis == 1

    first_window = window_manager.get_current_window()

    assert first_window is not None
    assert len(first_window) == 8
    assert window_manager.is_current_window_consumed() is True

    assert ml2_service.received_window is first_window

    # ==========================================================
    # Encerrar explicitamente o ciclo da primeira janela.
    #
    # O WindowManager deve permitir a construção da próxima
    # janela somente depois do consumo.
    # ==========================================================

    for index in range(8, 16):

        buffer.add_sample(
            create_sample(
                power=120.0,
                index=index,
            )
        )

    window_manager.build_next_window()

    assert window_manager.has_complete_window()

    second_window = window_manager.get_current_window()

    assert second_window is not None
    assert len(second_window) == 8

    assert window_manager.get_current_window_start_index() == 8
    assert window_manager.get_current_window_end_index() == 15

    assert second_window is not first_window

    assert window_manager.is_current_window_consumed() is False

    # ==========================================================
    # O Orchestrator deve poder consumir/classificar a nova
    # janela disponível.
    #
    # Como a segunda janela já foi construída fora do fluxo
    # process_sample(), executamos explicitamente o mesmo
    # mecanismo de consumo utilizado pelo Orchestrator.
    # ==========================================================

    diagnosis = ml2_service.predict(second_window)

    assert diagnosis == 1

    window_manager.consume_current_window()

    assert window_manager.is_current_window_consumed() is True

    assert ml2_service.received_window is second_window

    # ==========================================================
    # As duas janelas são distintas e não sobrepostas.
    # ==========================================================

    assert first_window[0] is buffer.get_sample(0)
    assert first_window[-1] is buffer.get_sample(7)

    assert second_window[0] is buffer.get_sample(8)
    assert second_window[-1] is buffer.get_sample(15)


def test_process_sample_builds_next_non_overlapping_window():
    """
    Verifica a integração do InferenceOrchestrator com o mecanismo
    de consumo e construção de janelas não sobrepostas.

    Primeira janela:

        [0, 1, 2, 3, 4, 5, 6, 7]

    Segunda janela:

        [8, 9, 10, 11, 12, 13, 14, 15]

    O objetivo é garantir que:

    1. a primeira janela seja classificada pelo ML2;
    2. a primeira janela seja consumida;
    3. as oito amostras seguintes sejam processadas;
    4. uma segunda janela seja construída automaticamente;
    5. a segunda janela seja enviada novamente ao ML2;
    6. nenhuma amostra da primeira janela seja reutilizada.
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
        diagnosis=1,
    )

    results = []

    # ==========================================================
    # PRIMEIRA JANELA
    #
    # Desvio no índice 3:
    #
    # [0, 1, 2, 3, 4, 5, 6, 7]
    # ==========================================================

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

    # ==========================================================
    # A primeira janela deve ter sido classificada.
    # ==========================================================

    assert buffer.size() == 8

    assert window_manager.has_complete_window()

    assert window_manager.get_current_window_start_index() == 0

    assert window_manager.get_current_window_end_index() == 7

    assert ml2_service.received_window is not None
    assert len(ml2_service.received_window) == 8

    first_window = list(ml2_service.received_window)

    assert first_window[0] == buffer.get_sample(0)
    assert first_window[7] == buffer.get_sample(7)

    assert results[7].diagnosis == 1

    # ==========================================================
    # A primeira janela deve ter sido consumida.
    # ==========================================================

    assert window_manager.is_current_window_consumed() is True

    # ==========================================================
    # SEGUNDA JANELA
    #
    # Adicionar as oito amostras seguintes:
    #
    # [8, 9, 10, 11, 12, 13, 14, 15]
    # ==========================================================

    for index in range(8, 16):

        sample = create_sample(
            power=120.0,
            index=index,
        )

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

        results.append(result)

    # ==========================================================
    # O buffer deve conter as 16 amostras.
    # ==========================================================

    assert buffer.size() == 16

    # ==========================================================
    # A segunda janela deve ter sido construída.
    # ==========================================================

    assert window_manager.has_complete_window()

    assert window_manager.get_current_window_start_index() == 8

    assert window_manager.get_current_window_end_index() == 15

    second_window = window_manager.get_current_window()

    assert second_window is not None
    assert len(second_window) == 8

    # ==========================================================
    # A segunda janela não pode conter nenhuma amostra
    # da primeira janela.
    # ==========================================================

    for position, expected_index in enumerate(range(8, 16)):

        assert second_window[position] == buffer.get_sample(expected_index)

    # ==========================================================
    # A segunda janela também deve ter sido enviada ao ML2.
    # ==========================================================

    assert ml2_service.received_window is second_window

    # ==========================================================
    # O diagnóstico deve estar disponível na amostra 15,
    # que completa a segunda janela.
    # ==========================================================

    assert results[15].diagnosis == 1

    # ==========================================================
    # A segunda janela deve ter sido consumida.
    # ==========================================================

    assert window_manager.is_current_window_consumed() is True


def test_process_sample_does_not_build_next_window_early():
    """
    Verifica que o InferenceOrchestrator não constrói a próxima
    janela antes de existirem as oito novas amostras necessárias.

    Após a primeira janela [0..7] ser consumida, apenas sete
    amostras novas [8..14] não são suficientes.

    Portanto, a segunda janela [8..15] ainda não pode existir.
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
        diagnosis=1,
    )

    # ==========================================================
    # Primeira janela [0..7]
    # ==========================================================

    for index in range(8):

        power = 100.0 if index == 3 else 120.0

        sample = create_sample(
            power=power,
            index=index,
        )

        orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

    assert buffer.size() == 8

    assert window_manager.get_current_window_start_index() == 0

    assert window_manager.get_current_window_end_index() == 7

    assert ml2_service.received_window is not None

    first_window = ml2_service.received_window

    assert len(first_window) == 8

    assert window_manager.is_current_window_consumed() is True

    # ==========================================================
    # Adicionar somente sete novas amostras:
    #
    # [8, 9, 10, 11, 12, 13, 14]
    # ==========================================================

    for index in range(8, 15):

        sample = create_sample(
            power=120.0,
            index=index,
        )

        result = orchestrator.process_sample(
            sample=sample,
            sample_index=index,
        )

        # Nenhum novo diagnóstico deve surgir porque
        # a segunda janela ainda não está completa.
        assert result.diagnosis is None

    # ==========================================================
    # Ainda existem somente sete amostras para a segunda janela.
    # ==========================================================

    assert buffer.size() == 15

    assert window_manager.get_current_window_start_index() == 0

    assert window_manager.get_current_window_end_index() == 7

    # A janela armazenada continua sendo a primeira.
    assert window_manager.get_current_window() is first_window

    # O ML2 não deve ter recebido uma nova janela.
    assert ml2_service.received_window is first_window

    # A primeira janela continua consumida.
    assert window_manager.is_current_window_consumed() is True
