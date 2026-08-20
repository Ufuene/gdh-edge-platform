"""
aws_publisher.py

Publisher MQTT do GDH Edge para o AWS IoT Core.

Responsabilidades:

- publicar payloads JSON no AWS IoT Core;
- utilizar uma conexão AWSIoTClient já estabelecida;
- utilizar o tópico configurado para a plataforma.

Este módulo NÃO:

- estabelece conexão MQTT/TLS;
- configura certificados;
- constrói payloads;
- executa ML1;
- executa ML2;
- acessa o S3 diretamente.

Fluxo:

    GDH Edge
       |
       | AWSIoTClient
       v
    MQTT/TLS
       |
       v
    AWSPublisher
       |
       | publish()
       v
    AWS IoT Core
       |
       | regra existente: pv/data
       v
      S3
"""

import json

from app.cloud.aws_client import AWSIoTClient, AWS_IOT_TOPIC


class AWSPublisher:
    """
    Publicador MQTT do GDH Edge para o AWS IoT Core.
    """

    def __init__(
        self,
        client: AWSIoTClient,
        topic: str = AWS_IOT_TOPIC,
    ):
        """
        Inicializa o publisher.

        Parameters
        ----------
        client:
            Instância do AWSIoTClient responsável pela conexão
            MQTT/TLS.

        topic:
            Tópico MQTT utilizado para publicação.
        """

        self._client = client
        self._topic = topic

    # ========================================================
    # PUBLICAÇÃO
    # ========================================================

    def publish(
        self,
        payload: dict,
    ) -> bool:
        """
        Publica um payload JSON no AWS IoT Core.

        Parameters
        ----------
        payload:
            Dicionário contendo os dados a serem enviados.

        Returns
        -------
        bool
            True quando a publicação foi aceita pelo cliente MQTT.
        """

        if not self._client.connected:
            raise RuntimeError("AWS IoT Client não está conectado ao AWS IoT Core.")

        # ----------------------------------------------------
        # Serialização
        # ----------------------------------------------------

        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # ----------------------------------------------------
        # Publicação MQTT
        # ----------------------------------------------------

        info = self._client.client.publish(
            self._topic,
            payload_json,
            qos=0,
            retain=False,
        )

        # ----------------------------------------------------
        # Verificação do resultado MQTT
        # ----------------------------------------------------

        if info.rc == 0:
            print()
            print("=" * 60)
            print("AWS IOT CORE - PUBLICAÇÃO")
            print("=" * 60)
            print(f"Topic        : {self._topic}")
            print(f"Payload bytes: {len(payload_json.encode('utf-8'))}")
            print("Resultado    : SUCCESS")

            return True

        print()
        print("=" * 60)
        print("AWS IOT CORE - FALHA NA PUBLICAÇÃO")
        print("=" * 60)
        print(f"Topic        : {self._topic}")
        print(f"Resultado    : rc={info.rc}")

        return False
