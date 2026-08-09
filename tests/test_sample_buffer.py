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


def test_get_latest():

    buffer = SampleBuffer()

    sample1 = Sample(
        timestamp=datetime.now(),
        irradiance=800,
        temperature=25,
        v_pv=32.0,
        i_pv=7.0,
        v_out=14.5,
        i_out=8.0,
        i_bat=2.5,
        i_load=5.0,
        p_out=110.0,
    )

    sample2 = Sample(
        timestamp=datetime.now(),
        irradiance=900,
        temperature=30,
        v_pv=35.0,
        i_pv=8.0,
        v_out=14.7,
        i_out=8.3,
        i_bat=3.0,
        i_load=5.3,
        p_out=125.0,
    )

    buffer.add_sample(sample1)
    buffer.add_sample(sample2)

    latest = buffer.get_latest()

    assert latest == sample2


def test_get_last_samples():

    buffer = SampleBuffer()

    samples = []

    for i in range(5):

        sample = Sample(
            timestamp=datetime.now(),
            irradiance=800 + i,
            temperature=25,
            v_pv=30,
            i_pv=7,
            v_out=14.5,
            i_out=8,
            i_bat=3,
            i_load=5,
            p_out=100 + i,
        )

        samples.append(sample)
        buffer.add_sample(sample)

    last_samples = buffer.get_last_samples(3)

    assert len(last_samples) == 3

    assert last_samples[0] == samples[2]
    assert last_samples[1] == samples[3]
    assert last_samples[2] == samples[4]


def test_has_minimum_samples():

    buffer = SampleBuffer()

    for i in range(5):

        sample = Sample(
            timestamp=datetime.now(),
            irradiance=800 + i,
            temperature=25,
            v_pv=30,
            i_pv=7,
            v_out=14.5,
            i_out=8,
            i_bat=3,
            i_load=5,
            p_out=100 + i,
        )

        buffer.add_sample(sample)

    assert buffer.has_minimum_samples(3)
    assert buffer.has_minimum_samples(5)
    assert not buffer.has_minimum_samples(6)


def test_get_sample():

    buffer = SampleBuffer()

    samples = []

    for i in range(5):

        sample = Sample(
            timestamp=datetime.now(),
            irradiance=800 + i,
            temperature=25,
            v_pv=30,
            i_pv=7,
            v_out=14.5,
            i_out=8,
            i_bat=3,
            i_load=5,
            p_out=100 + i,
        )

        samples.append(sample)
        buffer.add_sample(sample)

    assert buffer.get_sample(0) == samples[0]
    assert buffer.get_sample(2) == samples[2]
    assert buffer.get_sample(-1) == samples[4]


def test_get_samples_range():

    buffer = SampleBuffer()

    samples = []

    for i in range(5):

        sample = Sample(
            timestamp=datetime.now(),
            irradiance=800 + i,
            temperature=25,
            v_pv=30,
            i_pv=7,
            v_out=14.5,
            i_out=8,
            i_bat=3,
            i_load=5,
            p_out=100 + i,
        )

        samples.append(sample)
        buffer.add_sample(sample)

    subset = buffer.get_samples_range(1, 4)

    assert len(subset) == 3
    assert subset[0] == samples[1]
    assert subset[1] == samples[2]
    assert subset[2] == samples[3]


def test_get_all():

    buffer = SampleBuffer()

    for i in range(5):

        sample = Sample(
            timestamp=datetime.now(),
            irradiance=800 + i,
            temperature=25,
            v_pv=30,
            i_pv=7,
            v_out=14.5,
            i_out=8,
            i_bat=3,
            i_load=5,
            p_out=100 + i,
        )

        buffer.add_sample(sample)

    all_samples = buffer.get_all()

    assert len(all_samples) == 5


def test_clear():

    buffer = SampleBuffer()

    for i in range(5):

        sample = Sample(
            timestamp=datetime.now(),
            irradiance=800 + i,
            temperature=25,
            v_pv=30,
            i_pv=7,
            v_out=14.5,
            i_out=8,
            i_bat=3,
            i_load=5,
            p_out=100 + i,
        )

        buffer.add_sample(sample)

    buffer.clear()

    assert buffer.is_empty()
    assert buffer.size() == 0
