from datetime import datetime

from app.data_manager.sample import Sample
from app.data_manager.sample_buffer import SampleBuffer


def test_add_sample():

    buffer = SampleBuffer()

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

    buffer.add_sample(sample)

    assert not buffer.is_empty()
    assert buffer.size() == 1


def test_empty_buffer():

    buffer = SampleBuffer()

    assert buffer.is_empty()
    assert buffer.size() == 0
