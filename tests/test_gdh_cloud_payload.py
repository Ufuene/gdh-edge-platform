from datetime import datetime

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
        timestamp=datetime(2026, 8, 20, 12, 0, 0),
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


def test_gdh_cloud_payload_preserves_telemetry_and_inference():
    """
    Verifica o contrato mínimo do payload GDH destinado à nuvem.

    O teste valida:

    - ordem das grandezas de telemetria;
    - Pout real;
    - potência prevista pelo ML1;
    - desvio;
    - diagnóstico ML2;
    - registro ML1;
    - janela ML2.
    """

    sample = create_sample()

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

    ml1_record = ML1DatasetRecord.from_sample(sample)

    ml2_window = ML2DatasetWindow.from_window(
        window_id="psc-000001",
        window_start=0,
        window_end=7,
        samples=[sample],
        label=1,
    )

    payload = build_gdh_cloud_payload(
        sample=sample,
        result=result,
        ml1_record=ml1_record,
        ml2_window=ml2_window,
        device_id="GDH_EDGE_RASPBERRY_PI",
        sequence=1,
    )

    # ==========================================================
    # METADATA
    # ==========================================================

    assert payload["schema_version"] == "gdh-edge-1.0"

    assert payload["metadata"]["device_id"] == "GDH_EDGE_RASPBERRY_PI"

    assert payload["metadata"]["sequence"] == 1

    # ==========================================================
    # TELEMETRIA
    # ==========================================================

    telemetry = payload["telemetry"]

    assert list(telemetry.keys()) == [
        "Irradiancia",
        "Temperatura",
        "Vout",
        "Iout",
        "Ipv",
        "Vpv",
        "Iload",
        "Pout",
        "Ibat",
    ]

    assert telemetry["Irradiancia"] == 1040.23
    assert telemetry["Temperatura"] == 47.63
    assert telemetry["Vout"] == 12.0281
    assert telemetry["Iout"] == 9.6643
    assert telemetry["Ipv"] == 4.5538
    assert telemetry["Vpv"] == 24.4825
    assert telemetry["Iload"] == 14.4917
    assert telemetry["Pout"] == 116.2428
    assert telemetry["Ibat"] == -4.8276

    # ==========================================================
    # ML1
    # ==========================================================

    assert payload["inference"]["ML1"]["Pref"] == 117.10

    # ==========================================================
    # DESVIO
    # ==========================================================

    assert payload["inference"]["deviation"]["value"] == 0.8572

    assert payload["inference"]["deviation"]["detected"] is True

    # ==========================================================
    # ML2
    # ==========================================================

    assert payload["inference"]["ML2"]["executed"] is True

    assert payload["inference"]["ML2"]["diagnosis"] == 1

    # ==========================================================
    # JANELA
    # ==========================================================

    assert payload["inference"]["window"]["available"] is True

    assert payload["inference"]["window"]["start"] == 0

    assert payload["inference"]["window"]["end"] == 7

    # ==========================================================
    # DATASET ML1
    # ==========================================================

    assert payload["datasets"]["ML1"]["eligible"] is True

    assert payload["datasets"]["ML1"]["record"] is not None

    assert payload["datasets"]["ML1"]["record"]["irradiance"] == 1040.23

    assert payload["datasets"]["ML1"]["record"]["temperature"] == 47.63

    assert payload["datasets"]["ML1"]["record"]["v_out"] == 12.0281

    assert payload["datasets"]["ML1"]["record"]["p_out"] == 116.2428

    # ==========================================================
    # DATASET ML2
    # ==========================================================

    assert payload["datasets"]["ML2"]["available"] is True

    assert payload["datasets"]["ML2"]["window"] is not None

    assert payload["datasets"]["ML2"]["window"]["window_id"] == "psc-000001"

    assert payload["datasets"]["ML2"]["window"]["window_start"] == 0

    assert payload["datasets"]["ML2"]["window"]["window_end"] == 7

    assert payload["datasets"]["ML2"]["window"]["label"] == 1
