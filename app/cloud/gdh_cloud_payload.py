"""
gdh_cloud_payload.py

Constrói o payload mínimo do GDH Edge destinado ao AWS IoT Core.

Este módulo NÃO:

- executa ML1;
- executa ML2;
- detecta desvios;
- constrói janelas;
- cria datasets;
- publica na AWS.

Ele somente transforma os objetos já produzidos pela arquitetura
em um dicionário serializável para posterior publicação MQTT.

Fluxo:

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
    dict
       |
       v
    JSON
"""

from typing import Optional

from app.data_manager.sample import Sample
from app.datasets.ml1_dataset import ML1DatasetRecord
from app.datasets.ml2_dataset import ML2DatasetWindow
from app.inference.inference_result import InferenceResult

# ============================================================
# TELEMETRIA
# ============================================================


def sample_to_telemetry(sample: Sample) -> dict:
    """
    Converte uma Sample para o bloco de telemetria.

    A ordem das grandezas é deliberadamente preservada de acordo
    com o dataset e o payload utilizados na arquitetura:

        Irradiancia
        Temperatura
        Vout
        Iout
        Ipv
        Vpv
        Iload
        Pout
        Ibat
    """

    return {
        "Irradiancia": sample.irradiance,
        "Temperatura": sample.temperature,
        "Vout": sample.v_out,
        "Iout": sample.i_out,
        "Ipv": sample.i_pv,
        "Vpv": sample.v_pv,
        "Iload": sample.i_load,
        "Pout": sample.p_out,
        "Ibat": sample.i_bat,
    }


# ============================================================
# DATASET ML1
# ============================================================


def ml1_record_to_dict(
    record: ML1DatasetRecord,
) -> dict:
    """
    Serializa um registro do dataset ML1.
    """

    return record.to_dict()


# ============================================================
# DATASET ML2
# ============================================================


def ml2_window_to_dict(
    window: ML2DatasetWindow,
) -> dict:
    """
    Serializa uma janela do dataset ML2.

    A serialização utiliza o próprio objeto ML2DatasetWindow,
    evitando duplicação da lógica existente.
    """

    return window.to_dict()


# ============================================================
# PAYLOAD GDH
# ============================================================


def build_gdh_cloud_payload(
    sample: Sample,
    result: InferenceResult,
    ml1_record: Optional[ML1DatasetRecord] = None,
    ml2_window: Optional[ML2DatasetWindow] = None,
    device_id: str = "GDH_EDGE_RASPBERRY_PI",
    sequence: int = 0,
) -> dict:
    """
    Constrói o payload mínimo do GDH Edge para a nuvem.

    Parameters
    ----------
    sample:
        Amostra de telemetria processada pelo pipeline.

    result:
        Resultado produzido pelo InferenceOrchestrator.

    ml1_record:
        Registro ML1 produzido pelo DatasetManager, quando elegível.

    ml2_window:
        Janela ML2 produzida pelo DatasetManager, quando disponível.

    device_id:
        Identificador do sistema Edge.

    sequence:
        Número sequencial da mensagem.

    Returns
    -------
    dict
        Payload serializável em JSON.
    """

    return {
        "schema_version": "gdh-edge-1.0",
        "metadata": {
            "device_id": device_id,
            "timestamp": sample.timestamp.isoformat(),
            "sequence": sequence,
        },
        "telemetry": sample_to_telemetry(sample),
        "inference": {
            "ML1": {
                "Pref": result.predicted_power,
            },
            "deviation": {
                "value": result.deviation,
                "detected": result.deviation_detected,
            },
            "ML2": {
                "executed": result.ml2_executed,
                "diagnosis": result.diagnosis,
            },
            "window": {
                "available": result.window_available,
                "start": result.window_start,
                "end": result.window_end,
            },
        },
        "datasets": {
            "ML1": {
                "eligible": ml1_record is not None,
                "record": (
                    ml1_record_to_dict(ml1_record) if ml1_record is not None else None
                ),
            },
            "ML2": {
                "available": ml2_window is not None,
                "window": (
                    ml2_window_to_dict(ml2_window) if ml2_window is not None else None
                ),
            },
        },
    }
