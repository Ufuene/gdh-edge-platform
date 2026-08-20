import json
from datetime import datetime
import time

from app.cloud.aws_client import AWSIoTClient
from app.cloud.aws_publisher import AWSPublisher
from app.cloud.gdh_cloud_payload import build_gdh_cloud_payload
from app.data_manager.sample import Sample
from app.datasets.ml1_dataset import ML1DatasetRecord
from app.datasets.ml2_dataset import ML2DatasetWindow
from app.inference.inference_result import InferenceResult


def create_sample() -> Sample:
    """
    Cria uma amostra representativa da telemetria do GDH.
    """

    return Sample(
        timestamp=datetime.now(),
        irradiance=1040.23,
        temperature=47.63,
        v_pv=24.4825,
        i_pv=4.5538,
        v_out=12.0281,
        i_out=9.6643,
        i_bat=-4.8276,
        i_load=14.4917,
        p_out=116.2428,
    )


def test_publish_gdh_payload_to_aws_iot_core():
    """
    Publica um payload GDH completo no AWS IoT Core.

    Fluxo validado:

        Sample
            +
        InferenceResult
            +
        ML1DatasetRecord
            +
        ML2DatasetWindow
            |
            v
        GDHCloudPayload
            |
            v
        AWSPublisher
            |
            | MQTT/TLS
            v
        AWS IoT Core
            |
            | regra existente: pv/data
            v
        S3

    Este teste representa a primeira integração real entre
    os resultados produzidos pelo GDH Edge e a infraestrutura
    AWS já existente.
    """

    sample = create_sample()

    # ==========================================================
    # RESULTADO DA INFERÊNCIA
    # ==========================================================

    result = InferenceResult(
        predicted_power=117.10,
        deviation=0.8572,
        deviation_detected=True,
        diagnosis=1,
        ml2_executed=True,
        window_available=True,
        window_start=0,
        window_end=7,
    )

    # ==========================================================
    # DATASET ML1
    # ==========================================================

    ml1_record = ML1DatasetRecord.from_sample(sample)

    # ==========================================================
    # DATASET ML2
    # ==========================================================

    ml2_window = ML2DatasetWindow.from_window(
        window_id="psc-000001",
        window_start=0,
        window_end=7,
        samples=[sample],
        label=1,
    )

    # ==========================================================
    # CONSTRUIR PAYLOAD GDH
    # ==========================================================

    payload = build_gdh_cloud_payload(
        sample=sample,
        result=result,
        ml1_record=ml1_record,
        ml2_window=ml2_window,
        device_id="GDH_EDGE_RASPBERRY_PI",
        sequence=int(time.time()),
    )

    # ==========================================================
    # VALIDAR SERIALIZAÇÃO JSON
    # ==========================================================

    payload_json = json.dumps(payload)

    assert isinstance(payload_json, str)

    # ==========================================================
    # MOSTRAR PAYLOAD
    # ==========================================================

    print()
    print("=" * 60)
    print("GDH EDGE - PUBLICAÇÃO DO PAYLOAD GDH")
    print("=" * 60)

    print()
    print("Payload GDH:")

    print(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
    )

    # ==========================================================
    # CLIENTE AWS
    # ==========================================================

    client = AWSIoTClient()

    publisher = AWSPublisher(client)

    # ==========================================================
    # CONEXÃO
    # ==========================================================

    client.connect()

    try:
        # ======================================================
        # PUBLICAÇÃO
        # ======================================================

        result_publish = publisher.publish(payload)

        print()
        print("=" * 60)
        print("RESULTADO DA PUBLICAÇÃO")
        print("=" * 60)

        print(f"Publicado no AWS IoT Core: {result_publish}")

        assert result_publish is True

        print()
        print("=" * 60)
        print("GDH PAYLOAD → AWS IOT CORE: SUCESSO")
        print("=" * 60)

    finally:
        client.disconnect()
