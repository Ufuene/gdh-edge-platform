from datetime import datetime

from app.data_manager.sample import Sample


def test_sample_creation():

    sample = Sample(
        timestamp=datetime.now(),
        irradiance=850,
        temperature=30,
        v_pv=34.5,
        i_pv=7.8,
        v_out=14.6,
        i_out=8.2,
        i_bat=3.1,
        i_load=5.0,
        p_out=120.0,
    )

    assert sample.irradiance == 850
    assert sample.temperature == 30
    assert sample.v_pv == 34.5
    assert sample.i_pv == 7.8
    assert sample.v_out == 14.6
    assert sample.i_out == 8.2
    assert sample.i_bat == 3.1
    assert sample.i_load == 5.0
    assert sample.p_out == 120.0

    assert sample.predicted_power is None
    assert sample.deviation is None
    assert sample.deviation_detected is False
    assert sample.diagnosis is None

    print("✓ Sample criado com sucesso!")
