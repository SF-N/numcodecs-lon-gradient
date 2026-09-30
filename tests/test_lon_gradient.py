import numcodecs
import numcodecs.registry
import numpy as np
import pytest

INNER = dict(id="eb_quantize", eb="$eb_abs", codec=dict(id="zlib", level=1))


def test_from_config():
    codec = numcodecs.registry.get_codec(
        dict(id="lon_gradient", eb=1e-6, codec=INNER, spacing=0.25, stencil=5)
    )
    assert codec.__class__.__name__ == "LongitudeGradientCodec"
    assert codec.__class__.__module__ == "numcodecs_lon_gradient"
    config = codec.get_config()
    assert (
        config["codec"] == INNER
        and config["stencil"] == 5
        and config["spacing"] == 0.25
    )
    assert numcodecs.registry.get_codec(config).get_config() == config


def test_invalid():
    from numcodecs_lon_gradient import LongitudeGradientCodec

    with pytest.raises(ValueError):
        LongitudeGradientCodec(eb=0.0, codec=INNER)
    with pytest.raises(ValueError):
        LongitudeGradientCodec(eb=1.0, codec=INNER, stencil=0)
    with pytest.raises(ValueError):
        LongitudeGradientCodec(eb=1.0, codec=INNER, shrink=1.0)
    with pytest.raises(TypeError):
        LongitudeGradientCodec(eb=1.0, codec=INNER).encode(np.arange(20))
    with pytest.raises(ValueError):
        LongitudeGradientCodec(eb=1.0, codec=INNER, stencil=5).encode(np.zeros((4, 25)))
    with pytest.raises(ValueError):
        LongitudeGradientCodec(eb=1.0, codec=INNER).encode(np.array([[1.0, np.nan]]))


def test_map():
    from numcodecs_lon_gradient import LongitudeGradientCodec

    codec = LongitudeGradientCodec(eb=1e-5, codec=INNER, spacing=0.25, stencil=5)
    mapped = codec.map(lambda c: c)
    config = mapped.get_config()
    # the marker is restored in the (now fully explicit) inner configuration
    assert config["codec"]["id"] == "eb_quantize" and config["codec"]["eb"] == "$eb_abs"
    assert config["codec"]["codec"] == INNER["codec"]
    data = periodic_field((40, 120))
    np.testing.assert_array_equal(
        np.asarray(mapped.decode(mapped.encode(data))),
        np.asarray(codec.decode(codec.encode(data))),
    )


def derivative(x, k, spacing):
    return (np.roll(x, -k, axis=-1) - np.roll(x, k, axis=-1)) / (2 * k * spacing)


def check_roundtrip(
    data: np.ndarray, eb: float, spacing: float, stencil: int, **kwargs
):
    codec = numcodecs.registry.get_codec(
        dict(
            id="lon_gradient",
            eb=eb,
            codec=INNER,
            spacing=spacing,
            stencil=stencil,
            **kwargs,
        )
    )

    encoded = codec.encode(data)
    decoded = np.asarray(codec.decode(encoded))

    assert decoded.dtype == data.dtype
    assert decoded.shape == data.shape

    error = np.abs(
        derivative(decoded, stencil, spacing) - derivative(data, stencil, spacing)
    )
    assert np.all(error <= eb)

    out = np.empty_like(data)
    codec.decode(encoded, out=out)
    np.testing.assert_array_equal(out, decoded)

    return len(encoded)


def periodic_field(shape, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    lon = np.linspace(0, 2 * np.pi, shape[-1], endpoint=False)
    lat = np.linspace(-1, 1, shape[-2])
    field = np.exp(-(lat[:, None] ** 2)) * (
        1.0 + 0.5 * np.sin(3 * lon) + 0.2 * np.cos(7 * lon)
    )
    if len(shape) == 3:
        field = np.stack([field * (1 + 0.1 * t) for t in range(shape[0])])
    return 1e-2 * field + noise * rng.normal(size=shape)


def test_roundtrip():
    data = periodic_field((2, 40, 120), noise=1e-5)
    for eb in (1e-4, 1e-5, 1e-6):
        check_roundtrip(data, eb, 0.25, 5)
    check_roundtrip(data, 1e-5, 0.25, 5, closure_flips=False)
    check_roundtrip(data, 1e-5, 1.0, 1)
    check_roundtrip(data, 1e-5, 0.5, 3)
    check_roundtrip(data[0], 1e-5, 0.25, 5)
    check_roundtrip(data[0].astype(np.float32), 1e-4, 0.25, 5)
    check_roundtrip(data[0, 0], 1e-5, 0.25, 5)
    check_roundtrip(np.zeros((3, 20)), 1e-5, 0.25, 5)


def test_wider_step_than_pointwise():
    # bounding the derivative allows a 2x wider quantisation step than
    # bounding the values pointwise with eb * 2k * spacing / 2
    data = periodic_field((60, 240), noise=1e-5)
    eb, spacing, k = 1e-5, 0.25, 5
    size = check_roundtrip(data, eb, spacing, k)
    pointwise = numcodecs.registry.get_codec(
        dict(
            id="eb_quantize",
            eb=eb * 2 * k * spacing / 2,
            codec=dict(id="zlib", level=1),
        )
    )
    assert size < len(pointwise.encode(data))
