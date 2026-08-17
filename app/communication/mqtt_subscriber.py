"""
mqtt_subscriber.py

Subscriber MQTT do GDH Edge.

Responsabilidades:
    Mosquitto
        ↓
    receber payload MQTT
        ↓
    decodificar JSON
        ↓
    validar contrato ESP32 -> Raspberry
        ↓
    converter payload em Sample
        ↓
    entregar Sample ao handler da aplicação.

Este módulo NÃO implementa:
    - ML1
    - ML2
    - DeviationDetector
    - WindowManager
    - lógica de diagnóstico

O núcleo de inferência permanece desacoplado.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import paho.mqtt.client as mqtt

from app.data_manager.sample import Sample

# ============================================================
# CONFIGURAÇÃO MQTT
# ============================================================

MQTT_BROKER = "192.168.15.21"
MQTT_PORT = 1883
MQTT_TOPIC = "pv/data"
MQTT_CLIENT_ID = "GDH_EDGE_SUBSCRIBER"


# ============================================================
# EXCEÇÃO DE PAYLOAD
# ============================================================


class PayloadValidationError(ValueError):
    """
    Erro relacionado à estrutura ou aos dados do payload.
    """


# ============================================================
# REPRESENTAÇÃO DA MENSAGEM
# ============================================================


@dataclass
class PVDataMessage:
    """
    Representa uma mensagem válida recebida do ESP32.

    O Sample é utilizado pelo núcleo de inferência.

    device_id e sequence permanecem disponíveis como
    metadados da comunicação.
    """

    device_id: str
    timestamp: datetime
    sequence: int
    sample: Sample


# ============================================================
# SUBSCRIBER
# ============================================================


class MQTTSubscriber:
    """
    Subscriber MQTT do GDH Edge.

    Fluxo:

        MQTT
          ↓
        JSON
          ↓
        validação
          ↓
        PVDataMessage
          ↓
        handler
    """

    def __init__(
        self,
        broker: str = MQTT_BROKER,
        port: int = MQTT_PORT,
        topic: str = MQTT_TOPIC,
        client_id: str = MQTT_CLIENT_ID,
        message_handler: Optional[Callable[[PVDataMessage, int], None]] = None,
    ):
        self._broker = broker
        self._port = port
        self._topic = topic

        self._message_handler = message_handler

        # Índice utilizado pelo SampleBuffer / WindowManager.
        #
        # NÃO utilizar diretamente o "seq" do ESP32.
        self._sample_index = 0

        self._last_sequence: Optional[int] = None

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

    # ========================================================
    # CONEXÃO
    # ========================================================

    def connect(self) -> None:
        """
        Conecta ao broker MQTT.
        """

        print("=" * 60)
        print("GDH EDGE - MQTT SUBSCRIBER")
        print("=" * 60)

        print(f"Broker       : {self._broker}")
        print(f"Porta        : {self._port}")
        print(f"Topic        : {self._topic}")

        print()
        print("Conectando ao Mosquitto...")

        self._client.connect(
            self._broker,
            self._port,
            keepalive=60,
        )

    # ========================================================
    # LOOP
    # ========================================================

    def loop_forever(self) -> None:
        """
        Mantém o subscriber executando continuamente.
        """

        print()
        print("Subscriber iniciado.")
        print("Aguardando mensagens MQTT...")
        print()

        self._client.loop_forever()

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
        Callback executado após conexão com o broker.
        """

        print()
        print("=" * 60)
        print("MQTT CONECTADO")
        print("=" * 60)

        print(f"Broker       : {self._broker}")
        print(f"Porta        : {self._port}")
        print(f"Topic        : {self._topic}")
        print(f"Resultado    : {reason_code}")

        if reason_code == 0:

            result = client.subscribe(self._topic)

            print()
            print("Subscribe executado.")
            print(f"Topic        : {self._topic}")
            print(f"Resultado    : {result}")

        else:

            print()
            print("ERRO: conexão MQTT não foi aceita.")

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

        print()
        print("MQTT DESCONNECTED")
        print(f"Reason code  : {reason_code}")

    # ========================================================
    # CALLBACK MESSAGE
    # ========================================================

    def _on_message(
        self,
        client,
        userdata,
        message,
    ):
        """
        Processa uma mensagem MQTT recebida.
        """

        print()
        print("-" * 60)
        print("MENSAGEM MQTT RECEBIDA")
        print("-" * 60)

        print(f"Topic        : {message.topic}")
        print(f"Payload bytes: {len(message.payload)}")

        try:

            pv_message = self.parse_payload(message.payload)

        except PayloadValidationError as exc:

            print()
            print("PAYLOAD INVALIDO")
            print(f"Motivo       : {exc}")

            return

        except Exception as exc:

            print()
            print("ERRO inesperado ao processar payload")
            print(f"Erro         : {exc}")

            return

        # ----------------------------------------------------
        # Sequência
        # ----------------------------------------------------

        sequence = pv_message.sequence

        if self._last_sequence is not None:

            if sequence <= self._last_sequence:

                print()
                print("AVISO: sequência não crescente.")
                print(f"Anterior     : {self._last_sequence}")
                print(f"Atual        : {sequence}")

        self._last_sequence = sequence

        # ----------------------------------------------------
        # Informações recebidas
        # ----------------------------------------------------

        print()
        print("PAYLOAD VALIDADO")

        print(f"Device ID    : {pv_message.device_id}")

        print(f"Timestamp    : " f"{pv_message.timestamp.isoformat()}")

        print(f"Sequence     : {pv_message.sequence}")

        print()
        print("SAMPLE")

        print(f"Irradiance   : " f"{pv_message.sample.irradiance}")

        print(f"Temperature  : " f"{pv_message.sample.temperature}")

        print(f"Vpv          : " f"{pv_message.sample.v_pv}")

        print(f"Ipv          : " f"{pv_message.sample.i_pv}")

        print(f"Vout         : " f"{pv_message.sample.v_out}")

        print(f"Iout         : " f"{pv_message.sample.i_out}")

        print(f"Ibat         : " f"{pv_message.sample.i_bat}")

        print(f"Iload        : " f"{pv_message.sample.i_load}")

        print(f"Pout         : " f"{pv_message.sample.p_out}")

        # ----------------------------------------------------
        # Entregar à aplicação
        # ----------------------------------------------------

        if self._message_handler is not None:

            current_index = self._sample_index

            self._message_handler(
                pv_message,
                current_index,
            )

        # ----------------------------------------------------
        # Incrementar índice local
        # ----------------------------------------------------

        self._sample_index += 1

    # ========================================================
    # PARSER / VALIDADOR
    # ========================================================

    @staticmethod
    def parse_payload(
        payload: bytes,
    ) -> PVDataMessage:
        """
        Decodifica e valida o payload ESP32 -> Raspberry.

        Payload esperado:

        {
            "d_id": "PV01",
            "timestamp": 1723456789123,
            "seq": 1,
            "data": {
                "G": ...,
                "T": ...,
                "Vout": ...,
                "Iout": ...,
                "Ipv": ...,
                "Vpv": ...,
                "Iload": ...,
                "Pout": ...,
                "Ibat": ...
            },
            "quality": {
                "valid": true,
                "code": 0
            }
        }
        """

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        try:

            decoded = payload.decode("utf-8")

        except UnicodeDecodeError as exc:

            raise PayloadValidationError("payload não está em UTF-8") from exc

        try:

            data = json.loads(decoded)

        except json.JSONDecodeError as exc:

            raise PayloadValidationError("payload não contém JSON válido") from exc

        if not isinstance(data, dict):

            raise PayloadValidationError("payload JSON deve ser um objeto")

        # ----------------------------------------------------
        # Campos principais
        # ----------------------------------------------------

        required_top_level = (
            "d_id",
            "timestamp",
            "seq",
            "data",
            "quality",
        )

        for field in required_top_level:

            if field not in data:

                raise PayloadValidationError(f"campo obrigatório ausente: {field}")

        # ----------------------------------------------------
        # Device ID
        # ----------------------------------------------------

        device_id = data["d_id"]

        if not isinstance(device_id, str):
            raise PayloadValidationError("d_id deve ser string")

        if not device_id:
            raise PayloadValidationError("d_id não pode estar vazio")

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_raw = data["timestamp"]

        if not isinstance(
            timestamp_raw,
            (int, float),
        ):
            raise PayloadValidationError("timestamp deve ser numérico")

        if not math.isfinite(float(timestamp_raw)):
            raise PayloadValidationError("timestamp inválido")

        timestamp_seconds = float(timestamp_raw) / 1000.0

        try:

            timestamp = datetime.fromtimestamp(
                timestamp_seconds,
                tz=timezone.utc,
            )

        except (OverflowError, OSError, ValueError) as exc:

            raise PayloadValidationError("timestamp fora do intervalo válido") from exc

        # ----------------------------------------------------
        # Sequence
        # ----------------------------------------------------

        sequence = data["seq"]

        if not isinstance(sequence, int):

            raise PayloadValidationError("seq deve ser inteiro")

        if sequence < 0:

            raise PayloadValidationError("seq não pode ser negativo")

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        measurements = data["data"]

        if not isinstance(measurements, dict):

            raise PayloadValidationError("data deve ser objeto")

        required_measurements = (
            "G",
            "T",
            "Vout",
            "Iout",
            "Ipv",
            "Vpv",
            "Iload",
            "Pout",
            "Ibat",
        )

        for field in required_measurements:

            if field not in measurements:

                raise PayloadValidationError(f"medição ausente: {field}")

        # ----------------------------------------------------
        # QUALITY
        # ----------------------------------------------------

        quality = data["quality"]

        if not isinstance(quality, dict):

            raise PayloadValidationError("quality deve ser objeto")

        if "valid" not in quality:

            raise PayloadValidationError("quality.valid ausente")

        if "code" not in quality:

            raise PayloadValidationError("quality.code ausente")

        if quality["valid"] is not True:

            raise PayloadValidationError("payload marcado como inválido")

        if quality["code"] != 0:

            raise PayloadValidationError(f"quality.code inválido: {quality['code']}")

        # ----------------------------------------------------
        # Converter medições
        # ----------------------------------------------------

        values = {}

        for field in required_measurements:

            value = measurements[field]

            if not isinstance(
                value,
                (int, float),
            ):

                raise PayloadValidationError(f"{field} deve ser numérico")

            value = float(value)

            if not math.isfinite(value):

                raise PayloadValidationError(f"{field} possui valor não finito")

            values[field] = value

        # ----------------------------------------------------
        # Criar Sample
        # ----------------------------------------------------

        sample = Sample(
            timestamp=timestamp,
            irradiance=values["G"],
            temperature=values["T"],
            v_pv=values["Vpv"],
            i_pv=values["Ipv"],
            v_out=values["Vout"],
            i_out=values["Iout"],
            i_bat=values["Ibat"],
            i_load=values["Iload"],
            p_out=values["Pout"],
        )

        return PVDataMessage(
            device_id=device_id,
            timestamp=timestamp,
            sequence=sequence,
            sample=sample,
        )


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================


def main():
    """
    Executa o subscriber em modo de teste.

    Nesta etapa ainda não ligamos o subscriber ao
    InferenceOrchestrator.

    O objetivo é validar:

        ESP32
          ↓
        Mosquitto
          ↓
        Subscriber
          ↓
        Parser
          ↓
        Sample
    """

    subscriber = MQTTSubscriber()

    subscriber.connect()
    subscriber.loop_forever()


if __name__ == "__main__":
    main()
