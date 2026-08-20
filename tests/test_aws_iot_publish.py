import time

from app.cloud.aws_client import AWSIoTClient
from app.cloud.aws_publisher import AWSPublisher


def test_publish_test_payload_to_aws_iot_core():
    """
    Testa a publicação real de um payload mínimo no AWS IoT Core.

    Fluxo:

        Raspberry Pi
            |
            | MQTT/TLS
            v
        AWSIoTClient
            |
            v
        AWSPublisher
            |
            | pv/data
            v
        AWS IoT Core
            |
            | regra existente
            v
        S3

    Este teste NÃO utiliza ainda o payload definitivo do GDH.

    O objetivo é comprovar somente que o Raspberry Pi consegue
    publicar uma mensagem no tópico MQTT já utilizado pela
    infraestrutura AWS existente.
    """

    client = AWSIoTClient()

    publisher = AWSPublisher(client)

    print()
    print("=" * 60)
    print("GDH EDGE - AWS IOT CORE PUBLISH TEST")
    print("=" * 60)

    # ----------------------------------------------------------
    # Conectar
    # ----------------------------------------------------------

    client.connect()

    try:
        # ------------------------------------------------------
        # Payload mínimo de teste
        # ------------------------------------------------------

        payload = {
            "schema_version": "test-1.0",
            "source": "gdh-edge-rpi",
            "device_id": "GDH_EDGE_RASPBERRY_PI",
            "timestamp": int(time.time() * 1000),
            "sequence": 1,
            "test": True,
            "message": "GDH Edge AWS IoT Core publish test",
        }

        print()
        print("Payload:")
        print(payload)

        # ------------------------------------------------------
        # Publicar através do AWSPublisher
        # ------------------------------------------------------

        result = publisher.publish(payload)

        print()
        print(f"Resultado da publicação: {result}")

        assert result is True

        print()
        print("=" * 60)
        print("PUBLICAÇÃO AWS IOT CORE: SUCESSO")
        print("=" * 60)

    finally:
        # ------------------------------------------------------
        # Desconectar
        # ------------------------------------------------------

        client.disconnect()
