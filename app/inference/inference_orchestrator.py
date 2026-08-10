"""
inference_orchestrator.py

Implementa a camada de orquestração do fluxo de inferência
do Gêmeo Digital Híbrido (GDH).

O InferenceOrchestrator coordena o fluxo entre:

    Sample
       |
       v
DeviationDetector
       |
       v
WindowManager

Responsabilidades:

- receber uma amostra;
- utilizar o DeviationDetector para verificar a existência
  de um desvio;
- informar ao WindowManager o índice da amostra quando
  um desvio for detectado.

O InferenceOrchestrator NÃO executa o ML1.
O InferenceOrchestrator NÃO executa o ML2.
O InferenceOrchestrator NÃO calcula diretamente o erro.
O InferenceOrchestrator NÃO constrói janelas.
O InferenceOrchestrator NÃO controla a histerese.

Essas responsabilidades pertencem aos componentes especializados.
"""

from app.data_manager.sample import Sample
from app.data_manager.window_manager import WindowManager
from app.inference.deviation_detector import DeviationDetector


class InferenceOrchestrator:
    """
    Coordena o fluxo entre a amostra, o detector de desvios
    e o WindowManager.
    """

    def __init__(
        self,
        deviation_detector: DeviationDetector,
        window_manager: WindowManager,
    ):
        """
        Inicializa o InferenceOrchestrator.

        Parameters
        ----------
        deviation_detector : DeviationDetector
            Detector responsável por determinar se existe
            desvio entre a potência medida e a potência prevista.

        window_manager : WindowManager
            Gerenciador responsável pelo tratamento temporal
            dos eventos de desvio.
        """

        self._deviation_detector = deviation_detector
        self._window_manager = window_manager

    def process_sample(
        self,
        sample: Sample,
        predicted_power: float,
        sample_index: int,
    ) -> bool:
        """
        Processa uma amostra utilizando a potência prevista.

        Nesta etapa, a potência prevista é fornecida diretamente
        ao orquestrador. Posteriormente, essa entrada será
        substituída pela saída do ML1Service.

        Parameters
        ----------
        sample : Sample
            Amostra adquirida do sistema fotovoltaico.

        predicted_power : float
            Potência prevista pelo ML1.

        sample_index : int
            Índice da amostra no SampleBuffer.

        Returns
        -------
        bool
            True se um desvio foi detectado.
            False caso contrário.
        """

        deviation_detected = self._deviation_detector.has_deviation(
            measured_power=sample.p_out,
            predicted_power=predicted_power,
        )

        if deviation_detected:
            self._window_manager.notify_deviation(deviation_index=sample_index)

        return deviation_detected
