"""
Inference Runtime

Responsabilidade:

    receber uma mensagem PVDataMessage
    proveniente da camada MQTT e encaminhar
    sua Sample para o InferenceOrchestrator.

Este componente NÃO implementa:

- MQTT;
- ML1;
- ML2;
- detecção de desvio;
- construção de janela;
- histerese.

Essas responsabilidades permanecem nos componentes
especializados do pipeline.
"""

from app.inference.inference_orchestrator import InferenceOrchestrator


class InferenceRuntime:
    """
    Ponte entre a camada de comunicação MQTT e o
    InferenceOrchestrator.
    """

    def __init__(self, orchestrator: InferenceOrchestrator):
        """
        Parameters
        ----------
        orchestrator:
            Instância do InferenceOrchestrator responsável
            pela execução do pipeline de inferência.
        """

        self._orchestrator = orchestrator

        # Índice interno do pipeline.
        #
        # Não corresponde diretamente ao seq do ESP32.
        self._sample_index = 0

    @property
    def sample_index(self) -> int:
        """
        Retorna o próximo índice interno do pipeline.
        """

        return self._sample_index

    def process_message(self, message):
        """
        Processa uma mensagem PVDataMessage.

        A Sample contida na mensagem é encaminhada
        ao InferenceOrchestrator.

        O índice interno é incrementado somente após
        o processamento da amostra.
        """

        sample = message.sample

        sample_index = self._sample_index

        result = self._orchestrator.process_sample(
            sample=sample,
            sample_index=sample_index,
        )

        self._sample_index += 1

        return result
