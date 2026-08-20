"""
aws_client.py

Cliente MQTT/TLS para comunicação do GDH Edge
com o AWS IoT Core.

Responsabilidades:

- estabelecer conexão MQTT segura com o AWS IoT Core;
- utilizar os certificados já provisionados no Raspberry Pi;
- manter a conexão disponível para o AWS Publisher.

Este módulo NÃO:

- constrói payloads;
- executa ML1;
- executa ML2;
- acessa o S3 diretamente;
- implementa regras de negócio.

Fluxo:

    GDH Edge
       |
       | MQTT/TLS
       v
    AWS IoT Core
       |
       | regra existente: pv/data
       v
      S3
"""

from pathlib import Path
import time

import paho.mqtt.client as mqtt

# ============================================================
# CONFIGURAÇÃO AWS IoT
# ============================================================

AWS_IOT_ENDPOINT = "a3abc97jt72yue-ats.iot.us-east-1.amazonaws.com"

AWS_IOT_PORT = 8883

AWS_IOT_TOPIC = "pv/data"

AWS_IOT_CLIENT_ID = "GDH_EDGE_RASPBERRY_PI"


# ============================================================
# CERTIFICADOS
# ============================================================

PROJECT_ROOT = Path("/home/ufuene/gdh-platform")

AWS_CERT_DIR = PROJECT_ROOT / "config/aws/iot"

AWS_ROOT_CA = AWS_CERT_DIR / "AmazonRootCA1.pem"

AWS_DEVICE_CERT = AWS_CERT_DIR / "device.crt"

AWS_PRIVATE_KEY = AWS_CERT_DIR / "private.key"


# ============================================================
# CLIENTE AWS
# ============================================================


class AWSIoTClient:
    """
    Cliente MQTT/TLS do GDH Edge para o AWS IoT Core.
    """

    def __init__(
        self,
        endpoint: str = AWS_IOT_ENDPOINT,
        port: int = AWS_IOT_PORT,
        client_id: str = AWS_IOT_CLIENT_ID,
    ):
        """
        Inicializa o cliente AWS IoT.

        Parameters
        ----------
        endpoint:
            Endpoint MQTT do AWS IoT Core.

        port:
            Porta MQTT/TLS.

        client_id:
            Identificador MQTT do Raspberry Pi.
        """

        self._endpoint = endpoint
        self._port = port
        self._client_id = client_id

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        self._connected = False

        self._configure_tls()

    # ========================================================
    # CONFIGURAÇÃO TLS
    # ========================================================

    def _configure_tls(self) -> None:
        """
        Configura autenticação TLS utilizando os certificados
        provisionados no Raspberry Pi.
        """

        # ----------------------------------------------------
        # Verificar certificados
        # ----------------------------------------------------

        required_files = (
            AWS_ROOT_CA,
            AWS_DEVICE_CERT,
            AWS_PRIVATE_KEY,
        )

        for path in required_files:

            if not path.exists():
                raise FileNotFoundError(f"Certificado AWS não encontrado: {path}")

        # ----------------------------------------------------
        # Configurar TLS
        # ----------------------------------------------------

        self._client.tls_set(
            ca_certs=str(AWS_ROOT_CA),
            certfile=str(AWS_DEVICE_CERT),
            keyfile=str(AWS_PRIVATE_KEY),
        )

    # ========================================================
    # CONEXÃO
    # ========================================================

    def connect(self) -> None:
        """
        Estabelece conexão segura com o AWS IoT Core.

        O método somente retorna quando o callback MQTT
        confirmar efetivamente a conexão.
        """

        print()
        print("=" * 60)
        print("GDH EDGE - AWS IoT CORE")
        print("=" * 60)

        print(f"Endpoint     : {self._endpoint}")
        print(f"Porta        : {self._port}")
        print(f"Client ID    : {self._client_id}")
        print(f"Topic        : {AWS_IOT_TOPIC}")

        print()
        print("Certificados:")
        print(f"CA           : {AWS_ROOT_CA}")
        print(f"Device cert  : {AWS_DEVICE_CERT}")
        print(f"Private key  : {AWS_PRIVATE_KEY}")

        print()
        print("Conectando ao AWS IoT Core...")

        self._client.connect(
            self._endpoint,
            self._port,
            keepalive=60,
        )

        # ----------------------------------------------------
        # O loop MQTT precisa ser executado para que os
        # callbacks e o estado da conexão sejam processados.
        # ----------------------------------------------------

        self._client.loop_start()

        # ----------------------------------------------------
        # A conexão MQTT é confirmada de forma assíncrona pelo
        # callback _on_connect().
        #
        # Aguarda até que a conexão seja efetivamente
        # confirmada antes de retornar.
        # ----------------------------------------------------

        timeout = 10.0
        start_time = time.monotonic()

        while not self._connected:

            if time.monotonic() - start_time >= timeout:

                self._client.loop_stop()

                raise ConnectionError("Timeout aguardando conexão com o AWS IoT Core.")

            time.sleep(0.05)

    # ========================================================
    # DESCONEXÃO
    # ========================================================

    def disconnect(self) -> None:
        """
        Encerra a conexão com o AWS IoT Core.
        """

        if self._client.is_connected():

            self._client.disconnect()

        self._client.loop_stop()

        self._connected = False

        print()
        print("AWS IoT Core desconectado.")

    # ========================================================
    # ESTADO
    # ========================================================

    @property
    def connected(self) -> bool:
        """
        Retorna True quando o cliente está conectado ao AWS IoT.
        """

        return self._connected

    # ========================================================
    # CALLBACK CONNECT
    # ========================================================

    def _on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties,
    ):
        """
        Callback executado após tentativa de conexão.
        """

        print()
        print("-" * 60)
        print("AWS IoT CORE - CONEXÃO")
        print("-" * 60)

        print(f"Endpoint     : {self._endpoint}")
        print(f"Porta        : {self._port}")
        print(f"Client ID    : {self._client_id}")
        print(f"Resultado    : {reason_code}")

        if reason_code == 0:

            self._connected = True

            print("Status       : CONECTADO")

        else:

            self._connected = False

            print("Status       : FALHA NA CONEXÃO")

    # ========================================================
    # CALLBACK DISCONNECT
    # ========================================================

    def _on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties,
    ):
        """
        Callback executado quando a conexão é encerrada.
        """

        self._connected = False

        print()
        print("AWS IoT Core desconectado.")
        print(f"Reason code  : {reason_code}")

    # ========================================================
    # ACESSO AO CLIENTE MQTT
    # ========================================================

    @property
    def client(self) -> mqtt.Client:
        """
        Retorna o cliente MQTT interno.

        O publisher utilizará este cliente para publicar
        os payloads no AWS IoT Core.
        """

        return self._client
