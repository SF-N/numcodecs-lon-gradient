[![image](https://img.shields.io/github/actions/workflow/status/SF-N/numcodecs-lon-gradient/ci.yml?branch=main)](https://github.com/SF-N/numcodecs-lon-gradient/actions/workflows/ci.yml?query=branch%3Amain)
[![image](https://img.shields.io/pypi/v/numcodecs-lon-gradient.svg)](https://pypi.python.org/pypi/numcodecs-lon-gradient)
[![image](https://img.shields.io/pypi/l/numcodecs-lon-gradient.svg)](https://github.com/SF-N/numcodecs-lon-gradient/blob/main/LICENSE)
[![image](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FSF-N%2Fnumcodecs-lon-gradient%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://pypi.python.org/pypi/numcodecs-lon-gradient)
[![image](https://readthedocs.org/projects/numcodecs-lon-gradient/badge/?version=latest)](https://numcodecs-lon-gradient.readthedocs.io/en/latest/?badge=latest)

# numcodecs-lon-gradient

`LongitudeGradientCodec` for the [`numcodecs`] buffer compression API.

The `LongitudeGradientCodec` is a meta-codec that bounds the absolute error of the centred finite-difference derivative along a periodic axis (the last axis, e.g. longitude), `(x[i+k] - x[i-k]) / (2 k spacing)`, instead of the values themselves. Only the stride-`2k` differences `x[i+k] - x[i-k]` are constrained, so they are compressed with an inner absolute-error-bounded codec (with a bound of `eb * 2k * spacing`, twice the pointwise bound that would otherwise be needed) and the values are reconstructed by integrating the differences along each residue class of the periodic axis. The integration constants are free with respect to the requirement and are stored coarsely (class means); the periodic closure is enforced by a linear ramp on decoding and closure-aware rounding in the encoder, which verifies the derivative bound before finishing.

```python
from numcodecs_lon_gradient import LongitudeGradientCodec

# derivative along longitude (0.25 deg spacing, stencil x[i+5] - x[i-5]) within 1e-6 per degree
codec = LongitudeGradientCodec(
    eb=1e-6,
    spacing=0.25,
    stencil=5,
    codec=dict(
        id="eb_quantize", eb="$eb_abs", codec=dict(id="context_mixing.residuals")
    ),
)
```

The inner `codec` configuration contains a marker (`eb_abs_marker`, default `"$eb_abs"`) that is replaced by the translated absolute error bound, like in [`numcodecs-pw-ratio`](https://numcodecs-pw-ratio.readthedocs.io). NaN and infinite values are not supported.

[`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).


## Funding

The `numcodecs-lon-gradient` package has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
