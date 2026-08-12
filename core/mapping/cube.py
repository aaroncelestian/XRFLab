"""
Decode Oxford INCA SmartMap ListData hyperspectral cubes.

Format (reverse-engineered from XGT/INCA .ipj samples):
- Header includes version float 1.9 and map width/height
- Body is a sequence of gzip members interleaved with channel descriptors
- Small (~34 byte) descriptors name the next channel after a gzip bitmap
- Large gaps hold uncompressed "type 3/15" channel bitmaps for high-count
  channels; a trailing incomplete descriptor's payload is the next gzip
- Each channel image is bit-packed into uint32 words (1/2/4/8 bits per pixel)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import zlib
except ImportError:  # pragma: no cover
    zlib = None


@dataclass
class SpectrumCube:
    """Hyperspectral cube: channels × height × width (uint16 counts)."""

    data: np.ndarray  # (n_channels, height, width)
    ev_per_channel: float = 10.0  # matches INCA/XGT 4096-bin spectra
    energy_offset_ev: float = 0.0

    def __post_init__(self):
        self.data = np.asarray(self.data)
        if self.data.ndim != 3:
            raise ValueError("cube data must be (n_channels, height, width)")

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def height(self) -> int:
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        return int(self.data.shape[2])

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    def energy_axis_kev(self) -> np.ndarray:
        ch = np.arange(self.n_channels, dtype=np.float64)
        return (self.energy_offset_ev + ch * self.ev_per_channel) / 1000.0

    def spectrum_at(self, x: int, y: int) -> np.ndarray:
        """Counts spectrum at pixel (x=col, y=row)."""
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        return self.data[:, y, x].astype(np.float64)

    def sum_spectrum(self) -> np.ndarray:
        return self.data.sum(axis=(1, 2)).astype(np.float64)

    def roi_map(self, ch0: int, ch1: int) -> np.ndarray:
        """Sum channels [ch0, ch1] inclusive → 2D map."""
        ch0 = max(0, int(ch0))
        ch1 = min(self.n_channels - 1, int(ch1))
        if ch1 < ch0:
            ch0, ch1 = ch1, ch0
        return self.data[ch0 : ch1 + 1].sum(axis=0).astype(np.float64)

    def roi_map_energy(self, e0_kev: float, e1_kev: float) -> np.ndarray:
        """Sum channels covering [e0, e1] keV."""
        axis = self.energy_axis_kev()
        mask = (axis >= e0_kev) & (axis <= e1_kev)
        if not np.any(mask):
            ch = int(np.argmin(np.abs(axis - 0.5 * (e0_kev + e1_kev))))
            return self.data[ch].astype(np.float64)
        return self.data[mask].sum(axis=0).astype(np.float64)

    def mean_spectrum_line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        n_points: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample spectra along a line; return (distances, mean spectrum)."""
        length = float(np.hypot(x1 - x0, y1 - y0))
        if n_points is None:
            n_points = max(2, int(np.ceil(length)) + 1)
        xs = np.linspace(x0, x1, n_points)
        ys = np.linspace(y0, y1, n_points)
        acc = np.zeros(self.n_channels, dtype=np.float64)
        for x, y in zip(xs, ys):
            acc += self.spectrum_at(int(round(x)), int(round(y)))
        acc /= max(n_points, 1)
        dist = np.linspace(0.0, length, n_points)
        return dist, acc


def decode_listdata(
    raw: bytes,
    *,
    n_channels: int = 4096,
    expected_sum: Optional[np.ndarray] = None,
) -> SpectrumCube:
    """
    Decode a SmartMap ListData stream into a SpectrumCube.

    Args:
        raw: Full ListData stream bytes
        n_channels: Channel axis length (4096 for these XGT files)
        expected_sum: Optional totals for validating type-3 blocks
            (n_channels,) totals for validating type-3 blocks
            (same length as cube axis; 2× length is auto-rebinned)
    """
    if zlib is None:
        raise ImportError("zlib is required to decode SmartMap ListData")
    if len(raw) < 64:
        raise ValueError("ListData too small")

    width, height = struct.unpack_from("<II", raw, 16)
    if width == 0 or height == 0 or width > 8192 or height > 8192:
        raise ValueError(f"invalid ListData dims {width}x{height}")

    npix = width * height
    expected_sum = _normalize_expected_sum(expected_sum, n_channels)

    cube = np.zeros((n_channels, npix), dtype=np.uint16)
    start = raw.find(b"\x1f\x8b\x08")
    if start < 0:
        raise ValueError("No gzip members found in ListData")

    pos = start
    ch = 0
    pending_ch: Optional[int] = None

    while pos < len(raw) - 2 and raw[pos] == 0x1F:
        deco = zlib.decompressobj(31)  # gzip
        try:
            chunk = deco.decompress(raw[pos:]) + deco.flush()
        except zlib.error as exc:
            raise ValueError(f"gzip decompress failed at {pos}: {exc}") from exc
        used = len(raw[pos:]) - len(deco.unused_data)
        next_pos = pos + used
        nxt = raw.find(b"\x1f\x8b\x08", next_pos)
        gap = raw[next_pos:nxt] if nxt >= 0 else raw[next_pos:]

        img = _decode_gzip_channel(chunk, npix)
        if img is not None:
            if pending_ch is not None and 0 <= pending_ch < n_channels:
                cube[pending_ch] = img
                pending_ch = None
            elif 0 <= ch < n_channels:
                cube[ch] = img

        if len(gap) > 100 and expected_sum is not None:
            filled, pending_ch = _parse_type3_block(
                gap, height, width, expected_sum, npix
            )
            for c, plane in filled.items():
                if 0 <= c < n_channels:
                    cube[c] = plane
            if pending_ch is None:
                # Next gzip uses small-descriptor channel index if present
                if len(gap) >= 26:
                    # Prefer max filled + 1 when dense block consumed fully
                    ch = max(filled) + 1 if filled else ch + 1
                else:
                    ch = ch + 1
            # else: next gzip supplies pending_ch; leave ch unused
        elif len(gap) >= 26:
            ch = struct.unpack_from("<H", gap, 24)[0]
        else:
            ch += 1

        if nxt < 0:
            break
        pos = nxt

    data = cube.reshape(n_channels, height, width)
    return SpectrumCube(data=data, ev_per_channel=10.0)


def _normalize_expected_sum(
    expected_sum: Optional[np.ndarray], n_channels: int
) -> Optional[np.ndarray]:
    if expected_sum is None:
        return None
    expected_sum = np.asarray(expected_sum, dtype=np.float64)
    if expected_sum.shape[0] == n_channels:
        return expected_sum
    if expected_sum.shape[0] == n_channels * 2:
        return expected_sum.reshape(n_channels, 2).sum(axis=1)
    return None


def _decode_gzip_channel(chunk: bytes, npix: int) -> Optional[np.ndarray]:
    if len(chunk) < 8:
        return None
    nbytes = struct.unpack_from("<I", chunk, 0)[0]
    payload = chunk[4 : 4 + nbytes]
    return _decode_packed(payload, npix)


def _decode_packed(payload: bytes, npix: int) -> Optional[np.ndarray]:
    """Bit-packed pixels in little-endian uint32 words (1/2/4/8 bpp)."""
    if not payload:
        return np.zeros(npix, dtype=np.uint16)
    bits = None
    for cand in (1, 2, 4, 8):
        ppw = 32 // cand
        need = ((npix + ppw - 1) // ppw) * 4
        if need <= len(payload) <= need + 16:
            bits = cand
            break
    if bits is None:
        return None
    ppw = 32 // bits
    nwords = (npix + ppw - 1) // ppw
    need = nwords * 4
    if len(payload) < need:
        return None
    words = np.frombuffer(payload[:need], dtype="<u4")
    img = np.zeros(nwords * ppw, dtype=np.uint32)
    mask = (1 << bits) - 1
    for j in range(ppw):
        img[j::ppw] = (words >> (j * bits)) & mask
    return img[:npix].astype(np.uint16)


def _u16_header(buf: bytes, off: int, n: int = 20) -> Optional[np.ndarray]:
    need = n * 2
    if off + 2 > len(buf):
        return None
    chunk = buf[off : off + need]
    if len(chunk) < need:
        chunk = chunk + bytes(need - len(chunk))
    return np.frombuffer(chunk, dtype="<u2")


def _is_valid_record(
    u16: np.ndarray,
    height: int,
    width: int,
    expected_sum: np.ndarray,
) -> bool:
    if u16 is None or len(u16) < 13:
        return False
    count = int(u16[5])
    hh = int(u16[7])
    ww = int(u16[9])
    ch = int(u16[12])
    return (
        hh == height
        and ww == width
        and 0 <= ch < len(expected_sum)
        and count == int(expected_sum[ch])
    )


def _parse_type3_block(
    buf: bytes,
    height: int,
    width: int,
    expected_sum: np.ndarray,
    npix: int,
) -> Tuple[Dict[int, np.ndarray], Optional[int]]:
    """
    Parse uncompressed type-3/15 records from an inter-gzip gap.

    Returns (channel→image, pending_channel_or_None).
    pending_channel means the trailing record has no inline payload and the
    next gzip member is that channel's bitmap.
    """
    out: Dict[int, np.ndarray] = {}
    pending: Optional[int] = None
    records: List[Tuple[int, int, int]] = []  # (offset, channel, count)

    off = 0
    n = len(buf)
    while off + 26 <= n:
        if buf[off : off + 2] != b"\x03\x00":
            off += 1
            continue
        u16 = _u16_header(buf, off)
        if not _is_valid_record(u16, height, width, expected_sum):
            off += 1
            continue
        records.append((off, int(u16[12]), int(u16[5])))
        off += 2

    for i, (roff, ch, count) in enumerate(records):
        end = records[i + 1][0] if i + 1 < len(records) else n
        rem = end - roff
        decoded = None
        for hdr_size in (26, 28, 30, 32, 34):
            if roff + hdr_size >= end:
                continue
            payload = buf[roff + hdr_size : end]
            img = _decode_packed(payload, npix)
            if img is not None and int(img.sum()) == count:
                decoded = img
                break
        if decoded is not None:
            out[ch] = decoded
        elif rem <= 40 and i == len(records) - 1:
            # Trailing descriptor only — payload is the following gzip
            pending = ch
    return out, pending
