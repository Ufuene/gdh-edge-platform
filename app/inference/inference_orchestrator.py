"""
inference_orchestrator.py

Implementa a camada de orquestração do fluxo de inferência
do Gêmeo Digital Híbrido (GDH).

O InferenceOrchestrator coordena o fluxo entre:

    Sample
       |
       v
SampleBuffer
       |
       v
    ML1Service
       |
       v
DeviationDetector
       |
       v
WindowManager
       |
       v
    ML2Service
       |
       v
InferenceResult

Responsabilidades:

- receber uma amostra;
- encaminhar a amostra ao SampleBuffer;
- utilizar o ML1Service para gerar a potência de referência
  (Pref);
- utilizar o DeviationDetector para calcular o desvio;
- informar ao WindowManager quando um desvio for detectado;
- verificar a disponibilidade de uma janela completa;
- executar o ML2 quando uma janela estiver disponível;
- consumir explicitamente a janela após sua classificação;
- solicitar a construção da próxima janela não sobreposta
  quando houver amostras suficientes;
- informar ao WindowManager o estado do desvio para atualização
  da histerese;
- produzir um InferenceResult.

O InferenceOrchestrator NÃO implementa os modelos ML1 ou ML2.
O InferenceOrchestrator NÃO calcula diretamente as regras internas
dos modelos.
O InferenceOrchestrator NÃO constrói diretamente as janelas.
O InferenceOrchestrator NÃO controla a histerese.

Essas responsabilidades pertencem aos componentes especializados.
"""

from app.data_manager.sample import Sample
from app.data_manager.sample_buffer import SampleBuffer
from app.data_manager.window_manager import WindowManager
from app.inference.deviation_detector import DeviationDetector
from app.inference.inference_result import InferenceResult
from app.inference.ml1_service import ML1Service
from app.inference.ml2_service import ML2Service


class InferenceOrchestrator:
    """
    Coordena o fluxo completo de inferência do GDH.
    """

    def __init__(
        self,
        sample_buffer: SampleBuffer,
        ml1_service: ML1Service,
        ml2_service: ML2Service,
        deviation_detector: DeviationDetector,
        window_manager: WindowManager,
    ):
        """
        Inicializa o InferenceOrchestrator.
        """

        self._sample_buffer = sample_buffer
        self._ml1_service = ml1_service
        self._ml2_service = ml2_service
        self._deviation_detector = deviation_detector
        self._window_manager = window_manager

    def process_sample(
        self,
        sample: Sample,
        sample_index: int,
    ) -> InferenceResult:
        """
        Processa uma nova amostra através do fluxo completo
        de inferência.

        Fluxo:

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
            ML2
                ↓
            consumo da janela
                ↓
            próxima janela não sobreposta
                ↓
            atualização da histerese
                ↓
            InferenceResult
        """

        # ==========================================================
        # 1. Armazenar a nova amostra
        # ==========================================================

        self._sample_buffer.add_sample(sample)

        # ==========================================================
        # 2. ML1 — inferência ponto a ponto
        # ==========================================================

        predicted_power = self._ml1_service.predict(
            irradiance=sample.irradiance,
            temperature=sample.temperature,
            v_out=sample.v_out,
        )

        sample.predicted_power = predicted_power

        # ==========================================================
        # 3. Calcular desvio
        # ==========================================================

        deviation = self._deviation_detector.compute_error(
            measured_power=sample.p_out,
            predicted_power=predicted_power,
        )

        deviation_detected = self._deviation_detector.has_deviation(
            measured_power=sample.p_out,
            predicted_power=predicted_power,
        )

        sample.deviation = deviation
        sample.deviation_detected = deviation_detected

        # ==========================================================
        # 4. Informar desvio ao WindowManager
        # ==========================================================

        if deviation_detected:
            self._window_manager.notify_deviation(deviation_index=sample_index)

        # ==========================================================
        # 5. Verificar se já existia uma janela completa antes
        # desta amostra.
        #
        # Essa informação permite distinguir:
        #
        # - primeira construção de janela;
        # - construção posterior de janela.
        # ==========================================================

        window_was_complete = self._window_manager.has_complete_window()

        # ==========================================================
        # 6. Tentar construir a primeira janela do evento.
        # ==========================================================

        self._window_manager.add_sample()

        window_is_complete = self._window_manager.has_complete_window()

        window_became_complete = not window_was_complete and window_is_complete

        # ==========================================================
        # 7. Executar ML2 para uma janela nova ainda não consumida.
        #
        # Isso cobre a primeira janela do evento.
        # ==========================================================

        diagnosis = None

        if window_is_complete and self._window_manager.has_unconsumed_window():
            window = self._window_manager.get_current_window()

            diagnosis = self._ml2_service.predict(window)

            sample.diagnosis = diagnosis

            # ------------------------------------------------------
            # Consumir a janela após a classificação.
            # ------------------------------------------------------

            self._window_manager.consume_current_window()

        # ==========================================================
        # 8. Construir a próxima janela não sobreposta.
        #
        # Exemplo:
        #
        # primeira:
        #     [0..7]
        #
        # segunda:
        #     [8..15]
        #
        # terceira:
        #     [16..23]
        #
        # A operação é delegada integralmente ao WindowManager.
        # ==========================================================

        next_window_built = self._window_manager.build_next_window()

        # ==========================================================
        # 9. Se uma nova janela foi construída, executar ML2 e
        # consumi-la.
        # ==========================================================

        if next_window_built:

            window = self._window_manager.get_current_window()

            diagnosis = self._ml2_service.predict(window)

            sample.diagnosis = diagnosis

            self._window_manager.consume_current_window()

        # ==========================================================
        # 10. Atualizar a histerese.
        #
        # A primeira amostra que completa a primeira janela do
        # evento não inicia imediatamente a contagem da histerese.
        #
        # Porém, uma janela subsequente construída posteriormente
        # NÃO deve bloquear a atualização da histerese.
        #
        # Portanto, a condição correta é:
        #
        #     if not window_became_complete:
        #
        # e NÃO:
        #
        #     if not window_became_complete
        #        and not next_window_built:
        #
        # A segunda condição faria com que a construção de uma
        # janela subsequente impedisse a atualização da histerese.
        # ==========================================================

        if not window_became_complete:

            self._window_manager.update_hysteresis(
                deviation_detected=deviation_detected
            )

        # ==========================================================
        # 11. Produzir resultado agregado.
        # ==========================================================

        return InferenceResult(
            predicted_power=predicted_power,
            deviation=deviation,
            deviation_detected=deviation_detected,
            diagnosis=diagnosis,
        )
