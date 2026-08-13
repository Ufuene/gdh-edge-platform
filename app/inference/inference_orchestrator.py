"""
inference_orchestrator.py

Implementa a camada de orquestração do fluxo de inferência
do Gêmeo Digital Híbrido (GDH).

O InferenceOrchestrator coordena o fluxo entre:

    Sample
       |
       v
    ML1Service
       |
       v
    DeviationDetector
       |
       v
    WindowManager

Responsabilidades:

- receber uma amostra;
- utilizar o ML1Service para gerar a potência de referência
  (Pref);
- utilizar o DeviationDetector para verificar a existência
  de um desvio;
- informar ao WindowManager o índice da amostra quando
  um desvio for detectado.

O InferenceOrchestrator NÃO carrega o modelo ML1.
O InferenceOrchestrator NÃO executa diretamente o modelo.
O InferenceOrchestrator NÃO calcula diretamente o erro.
O InferenceOrchestrator NÃO constrói janelas.
O InferenceOrchestrator NÃO controla a histerese.

Essas responsabilidades pertencem aos componentes especializados.
"""

from app.data_manager.sample import Sample
from app.data_manager.window_manager import WindowManager
from app.inference.deviation_detector import DeviationDetector
from app.inference.ml1_service import ML1Service


class InferenceOrchestrator:
    """
    Coordena o fluxo entre a amostra, o ML1Service,
    o detector de desvios e o WindowManager.
    """

    def __init__(
        self,
        ml1_service: ML1Service,
        deviation_detector: DeviationDetector,
        window_manager: WindowManager,
    ):
        """
        Inicializa o InferenceOrchestrator.

        Parameters
        ----------
        ml1_service : ML1Service
            Serviço responsável por executar a inferência
            ponto a ponto do ML1.

        deviation_detector : DeviationDetector
            Detector responsável por determinar se existe
            desvio entre a potência medida e a potência prevista.

        window_manager : WindowManager
            Gerenciador responsável pelo tratamento temporal
            dos eventos de desvio.
        """

        self._ml1_service = ml1_service
        self._deviation_detector = deviation_detector
        self._window_manager = window_manager

    def process_sample(
        self,
        sample: Sample,
        sample_index: int,
    ) -> bool:
        """
        Processa uma amostra através do fluxo ML1 → desvio.

        O ML1 recebe as três características utilizadas durante
        o treinamento:

            [Irradiancia, Temperatura, Vout]

        e produz a potência de referência:

            Pref

        Essa potência é então comparada com a potência medida
        da amostra pelo DeviationDetector.

        Parameters
        ----------
        sample : Sample
            Amostra adquirida do sistema fotovoltaico.

        sample_index : int
            Índice da amostra no SampleBuffer.

        Returns
        -------
        bool
            True se um desvio foi detectado.
            False caso contrário.
        """

        predicted_power = self._ml1_service.predict(
            irradiance=sample.irradiance,
            temperature=sample.temperature,
            v_out=sample.v_out,
        )

        deviation_detected = self._deviation_detector.has_deviation(
            measured_power=sample.p_out,
            predicted_power=predicted_power,
        )

        if deviation_detected:
            self._window_manager.notify_deviation(deviation_index=sample_index)

        return deviation_detected
