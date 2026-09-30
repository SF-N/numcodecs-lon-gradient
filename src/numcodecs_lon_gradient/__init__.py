"""
[`LongitudeGradientCodec`][numcodecs_lon_gradient.LongitudeGradientCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["LongitudeGradientCodec"]

import copy
import math
import zlib
from collections.abc import Callable
from functools import reduce
from io import BytesIO

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
from numba import njit
from numcodecs.abc import Codec
from numcodecs_combinators.abc import CodecCombinatorMixin
from typing_extensions import Buffer  # MSPV 3.12


def _as_slices(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) == 0:
        return (1, 1, 1)
    if len(shape) == 1:
        return (1, 1, shape[0])
    return (reduce(lambda a, b: a * b, shape[:-2], 1), shape[-2], shape[-1])


def _replace_marker(config: object, marker: object, value: object) -> object:
    """Replace values equal to `marker` in a (nested) codec config by `value`."""
    if isinstance(config, dict):
        return {
            k: (
                value
                if (type(v) is type(marker) and v == marker)
                else _replace_marker(v, marker, value)
            )
            for k, v in config.items()
        }
    if isinstance(config, (list, tuple)):
        return type(config)(_replace_marker(v, marker, value) for v in config)
    return config


def _differences(x: np.ndarray, k: int) -> np.ndarray:
    """Periodic stride-2k differences D[i] = x[i+k] - x[i-k] along the last axis."""
    return np.roll(x, -k, axis=-1) - np.roll(x, k, axis=-1)


def _reconstruct(
    D: np.ndarray, anchors: np.ndarray, k: int, anchor_step: float
) -> np.ndarray:
    """Integrate the differences along each residue class (mod 2k) of the
    periodic axis; the per-class closure defect is spread as a linear ramp
    and the class means are set from the anchors."""
    T, Y, X = D.shape
    n = 2 * k
    K = X // n
    Droll = np.roll(D, -k, axis=2).reshape(T, Y, K, n)
    S = Droll.sum(axis=2, keepdims=True)
    Deff = Droll - S / K
    z = np.zeros((T, Y, K, n), np.float64)
    z[:, :, 1:, :] = np.cumsum(Deff[:, :, :-1, :], axis=2)
    z -= z.mean(axis=2, keepdims=True)
    z += (anchors.astype(np.float64) * anchor_step)[:, :, None, :]
    return z.reshape(T, Y, X)


@njit(cache=True)
def _optimise_classes(e_r, q_r, w, T, Y, K, n, max_flips):
    """For every residue class pick the number m of rounding flips (of the m
    largest same-sign quantisation errors) that minimises the final maximum
    error after the closure ramp, max_k |e'_k - mean(e')|. In place."""
    vals = np.empty(K, np.float64)
    idxs = np.empty(K, np.int64)
    flipped = np.zeros(K, np.bool_)
    for t in range(T):
        for y in range(Y):
            for c in range(n):
                S = 0.0
                for k in range(K):
                    S += e_r[t, y, k, c]
                sgn = 1.0 if S > 0 else -1.0
                cnt = 0
                for k in range(K):
                    ek = e_r[t, y, k, c]
                    if (ek > 0 and sgn > 0) or (ek < 0 and sgn < 0):
                        vals[cnt] = -abs(ek)
                        idxs[cnt] = k
                        cnt += 1
                order = np.argsort(vals[:cnt])
                best_m = 0
                mean = S / K
                best_err = 0.0
                for k in range(K):
                    v = abs(e_r[t, y, k, c] - mean)
                    if v > best_err:
                        best_err = v
                mmax = min(cnt, max_flips)
                for k in range(K):
                    flipped[k] = False
                for m in range(1, mmax + 1):
                    flipped[idxs[order[m - 1]]] = True
                    Sp = S - m * w * sgn
                    mean = Sp / K
                    err = 0.0
                    for k in range(K):
                        ek = e_r[t, y, k, c]
                        if flipped[k]:
                            ek -= sgn * w
                        v = abs(ek - mean)
                        if v > err:
                            err = v
                    if err < best_err:
                        best_err = err
                        best_m = m
                for m in range(best_m):
                    k = idxs[order[m]]
                    e_r[t, y, k, c] -= sgn * w
                    q_r[t, y, k, c] -= sgn


def _closure_aware_round(D: np.ndarray, w: float, k: int) -> np.ndarray:
    """Round D onto the grid min(D) + w*q, choosing the rounding direction of
    near-tie values per residue class so that the closure ramp stays small."""
    T, Y, X = D.shape
    n = 2 * k
    K = X // n
    ymin = float(D.min())
    q = np.rint((D - ymin) / w)
    e = (ymin + w * q) - D
    e_r = np.ascontiguousarray(np.roll(e, -k, axis=2).reshape(T, Y, K, n))
    q_r = np.ascontiguousarray(np.roll(q, -k, axis=2).reshape(T, Y, K, n))
    _optimise_classes(e_r, q_r, w, T, Y, K, n, 48)
    return ymin + w * np.roll(q_r.reshape(T, Y, X), k, axis=2)


class LongitudeGradientCodec(Codec, CodecCombinatorMixin):
    r"""
    Meta-codec that bounds the absolute error of the centred finite-difference
    derivative along a periodic axis (e.g. longitude), instead of the values
    themselves.

    For data `x[..., i]` with grid spacing `spacing` and stencil half-width
    `k`, the derivative

    $$ \frac{dx}{d\lambda}[i] = \frac{x[i+k] - x[i-k]}{2 k \cdot \text{spacing}} $$

    (indices periodic) is reconstructed with an absolute error of at most `eb`.
    Only the stride-`2k` differences `D[i] = x[i+k] - x[i-k]` are constrained,
    so they are compressed with the inner `codec` with an absolute error bound
    of `eb * 2k * spacing` (translated via the `eb_abs_marker`, like
    [`numcodecs-pw-ratio`](https://numcodecs-pw-ratio.readthedocs.io)), which
    is twice the pointwise bound that would otherwise be needed. The values
    are reconstructed by integrating the differences along each residue class
    (mod `2k`) of the periodic axis. The integration constants are free with
    respect to the requirement and are stored coarsely (the class means, with
    step `anchor_step`) so that the reconstruction stays close to the data.

    The periodic closure (the differences of a class must sum to zero) is
    enforced by spreading the closure defect as a linear ramp on decoding; the
    encoder rounds near-tie values per class so that the defect stays small
    (`closure_flips`), shrinks the quantisation step starting from
    `2 * (1 - shrink)` times the bound, and verifies the derivative bound
    before finishing.

    The last axis is the periodic axis and its length must be a multiple of
    `2k`. NaN and infinite values are not supported.

    Parameters
    ----------
    eb : float
        The positive absolute error bound on the derivative.
    codec : dict
        The configuration of the codec that encodes the differences with an
        absolute error bound; it must contain the `eb_abs_marker`, which is
        replaced by the translated absolute error bound.
    eb_abs_marker : str, optional
        The marker for the absolute error bound in `codec`.
    spacing : float, optional
        The grid spacing along the periodic axis (in the derivative's units).
    stencil : int, optional
        The half-width `k` of the difference stencil.
    anchor_step : None | float, optional
        The quantisation step of the class means, or [`None`][None] for
        `0.8 * eb * 2k * spacing`.
    shrink : float, optional
        The initial relative shrinking of the quantisation step (in [0, 1)).
    closure_flips : bool, optional
        Whether to use closure-aware rounding in the encoder.
    """

    __slots__: tuple[str, ...] = (
        "_eb",
        "_codec",
        "_eb_abs_marker",
        "_spacing",
        "_stencil",
        "_anchor_step",
        "_shrink",
        "_closure_flips",
    )
    _eb: float
    _codec: dict
    _eb_abs_marker: str
    _spacing: float
    _stencil: int
    _anchor_step: None | float
    _shrink: float
    _closure_flips: bool

    codec_id: str = "lon_gradient"  # type: ignore

    def __init__(
        self,
        *,
        eb: float,
        codec: dict,
        eb_abs_marker: str = "$eb_abs",
        spacing: float = 1.0,
        stencil: int = 1,
        anchor_step: None | float = None,
        shrink: float = 0.04,
        closure_flips: bool = True,
    ) -> None:
        if not (math.isfinite(eb) and eb > 0):
            raise ValueError("eb must be finite and positive")
        if not (math.isfinite(spacing) and spacing > 0):
            raise ValueError("spacing must be finite and positive")
        if int(stencil) < 1:
            raise ValueError("stencil must be positive")
        if anchor_step is not None and not (
            math.isfinite(anchor_step) and anchor_step > 0
        ):
            raise ValueError("anchor_step must be finite and positive")
        if not (0 <= shrink < 1):
            raise ValueError("shrink must be in [0, 1)")
        if not isinstance(codec, dict):
            raise TypeError(
                "codec must be a configuration dict containing the eb_abs_marker"
            )

        self._eb = float(eb)
        self._codec = copy.deepcopy(codec)
        self._eb_abs_marker = eb_abs_marker
        self._spacing = float(spacing)
        self._stencil = int(stencil)
        self._anchor_step = None if anchor_step is None else float(anchor_step)
        self._shrink = float(shrink)
        self._closure_flips = bool(closure_flips)

    @property
    def _eb_diff(self) -> float:
        return self._eb * 2 * self._stencil * self._spacing

    def _inner(self, eb_abs: float) -> Codec:
        config = _replace_marker(self._codec, self._eb_abs_marker, float(eb_abs))
        assert isinstance(config, dict)
        return numcodecs.registry.get_codec(config)

    def encode(self, buf: Buffer) -> bytes:
        """
        Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Floating-point data to be encoded. May be any object supporting
            the new-style buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded data as a bytestring.
        """

        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape

        if not np.issubdtype(dtype, np.floating):
            raise TypeError("can only encode floating point values")
        if not np.all(np.isfinite(a)):
            raise ValueError("cannot encode non-finite values, mask them first")

        k = self._stencil
        n = 2 * k
        T, Y, X = _as_slices(shape)
        if X % n != 0:
            raise ValueError(f"the periodic axis length {X} must be a multiple of {n}")
        K = X // n
        x3 = np.ascontiguousarray(a.astype(np.float64).reshape(T, Y, X))

        eb_diff = self._eb_diff
        anchor_step = (
            self._anchor_step if self._anchor_step is not None else 0.8 * eb_diff
        )
        D = _differences(x3, k)

        # class means as integer anchors, delta coded along the classes
        means = x3.reshape(T, Y, K, n).mean(axis=2)
        anchors = np.rint(means / anchor_step).astype(np.int64)
        deltas = np.diff(anchors, axis=2, prepend=0)
        deltas[:, :, 0] = np.diff(anchors[:, :, 0], axis=1, prepend=0)
        anchor_bytes = zlib.compress(deltas.astype("<i8").tobytes(), 9)

        shrink = self._shrink
        while True:
            step = 2.0 * eb_diff * (1.0 - shrink)
            inner = self._inner(step / 2.0)
            D_in = _closure_aware_round(D, step, k) if self._closure_flips else D
            encoded = numcodecs.compat.ensure_ndarray(inner.encode(D_in))
            D_dec = np.asarray(
                numcodecs.compat.ensure_ndarray(inner.decode(encoded)), dtype=np.float64
            ).reshape(T, Y, X)
            x_dec = (
                _reconstruct(D_dec, anchors, k, anchor_step)
                .astype(dtype)
                .astype(np.float64)
            )
            error = np.abs(_differences(x_dec, k) - D)
            if error.size == 0 or error.max() <= eb_diff:
                break
            shrink += 0.02
            if shrink >= 0.5:
                raise ValueError(
                    f"cannot satisfy the error bound {self._eb} with dtype {dtype}"
                )

        # message: dtype shape anchor-step shrink anchors-len anchors
        #          encoded-dtype encoded-shape [padding] encoded
        message: list[bytes | bytearray] = []

        message.append(leb128.u.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(shape)))
        for s in shape:
            message.append(leb128.u.encode(s))

        message.append(np.array([anchor_step, shrink], dtype="<f8").tobytes())

        message.append(leb128.u.encode(len(anchor_bytes)))
        message.append(anchor_bytes)

        message.append(leb128.u.encode(len(encoded.dtype.str)))
        message.append(encoded.dtype.str.encode("ascii"))

        message.append(leb128.u.encode(encoded.ndim))
        for s in encoded.shape:
            message.append(leb128.u.encode(s))

        # insert padding to align with encoded itemsize
        message.append(
            b"\0"
            * (
                encoded.dtype.itemsize
                - (sum(len(m) for m in message) % encoded.itemsize)
            )
        )

        # ensure that the encoded values are encoded in little endian binary
        message.append(encoded.astype(encoded.dtype.newbyteorder("<")).tobytes())

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol.
        """

        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        dtype = np.dtype(b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii"))
        shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )

        anchor_step, shrink = np.frombuffer(b_io.read(16), dtype="<f8", count=2)

        anchor_bytes = b_io.read(leb128.u.decode_reader(b_io)[0])

        encoded_dtype = np.dtype(
            b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
        )
        encoded_shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        encoded_size = reduce(lambda a, b: a * b, encoded_shape, 1)

        # remove padding to align with encoded itemsize
        b_io.read(encoded_dtype.itemsize - (b_io.tell() % encoded_dtype.itemsize))

        encoded = (
            np.frombuffer(
                b_io.read(encoded_size * encoded_dtype.itemsize),
                dtype=encoded_dtype.newbyteorder("<"),
                count=encoded_size,
            )
            .astype(encoded_dtype)
            .reshape(encoded_shape)
        )

        k = self._stencil
        n = 2 * k
        T, Y, X = _as_slices(shape)

        deltas = (
            np.frombuffer(zlib.decompress(anchor_bytes), dtype="<i8")
            .astype(np.int64)
            .reshape(T, Y, n)
        )
        anchors: np.ndarray = deltas.copy()
        anchors[:, :, 0] = np.cumsum(deltas[:, :, 0], axis=1)
        anchors = np.cumsum(anchors, axis=2)

        step = 2.0 * self._eb_diff * (1.0 - float(shrink))
        D_dec = np.asarray(
            numcodecs.compat.ensure_ndarray(self._inner(step / 2.0).decode(encoded)),
            dtype=np.float64,
        ).reshape(T, Y, X)

        decoded = (
            _reconstruct(D_dec, anchors, k, float(anchor_step))
            .reshape(shape)
            .astype(dtype)
        )

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of this meta-codec.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this codec from the returned config.

        Returns
        -------
        config : dict
            Configuration of this meta-codec.
        """

        return dict(
            id=type(self).codec_id,
            eb=self._eb,
            codec=copy.deepcopy(self._codec),
            eb_abs_marker=self._eb_abs_marker,
            spacing=self._spacing,
            stencil=self._stencil,
            anchor_step=self._anchor_step,
            shrink=self._shrink,
            closure_flips=self._closure_flips,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(eb={self._eb!r}, codec={self._codec!r}, eb_abs_marker={self._eb_abs_marker!r}, spacing={self._spacing!r}, stencil={self._stencil!r}, anchor_step={self._anchor_step!r}, shrink={self._shrink!r}, closure_flips={self._closure_flips!r})"

    def map(self, mapper: Callable[[Codec], Codec]) -> "LongitudeGradientCodec":
        """
        Apply the `mapper` to the inner codec (instantiated with a placeholder
        error bound of 1, then mapped, and converted back to a configuration
        in which the bound is replaced by the `eb_abs_marker` again).

        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that is applied to the inner codec.

        Returns
        -------
        mapped : LongitudeGradientCodec
            The mapped meta-codec.
        """

        placeholder = 0x7EB_A85_5EED * 1e-300  # unlikely to occur in a config
        mapped = mapper(self._inner(placeholder)).get_config()
        config = _replace_marker(mapped, placeholder, self._eb_abs_marker)
        assert isinstance(config, dict)
        return LongitudeGradientCodec(
            eb=self._eb,
            codec=config,
            eb_abs_marker=self._eb_abs_marker,
            spacing=self._spacing,
            stencil=self._stencil,
            anchor_step=self._anchor_step,
            shrink=self._shrink,
            closure_flips=self._closure_flips,
        )


numcodecs.registry.register_codec(LongitudeGradientCodec)
