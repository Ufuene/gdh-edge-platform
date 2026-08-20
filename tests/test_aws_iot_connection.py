"""
test_aws_iot_connection.py

Teste manual de integração com o AWS IoT Core.

Este teste verifica somente:

    Raspberry Pi
        |
        | MQTT/TLS
        v
    AWS IoT Core

Não publica payloads e não envolve ML1, ML2 ou S3.
"""

import time

from app.cloud.aws_client import AWSIoTClient


def test_aws_iot_connection():
    """
    Verifica se o Raspberry Pi consegue estabelecer uma conexão
    MQTT/TLS real com o AWS IoT Core.
    """

    aws = AWSIoTClient()

    try:
        aws.connect()

        # ------------------------------------------------------
        # Aguarda o callback MQTT processar a conexão.
        # ------------------------------------------------------

        timeout = 10
        start = time.time()

        while not aws.connected:

            if time.time() - start > timeout:
                raise AssertionError("Timeout aguardando conexão com AWS IoT Core.")

            time.sleep(0.2)

        assert aws.connected is True

    finally:
        aws.disconnect()
