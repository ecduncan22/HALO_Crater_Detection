"""Tile grid along one axis, and which part of each tile is kept in the output.

Tiles overlap when stride < tile_size. Every output pixel is taken from exactly one
tile: the overlap between neighbouring tiles is split down the middle ("center"
merge), so predictions near tile edges, where the network has less context, are
discarded wherever a neighbouring tile covers them. With stride == tile_size this
reduces to plain non-overlapping tiling (the original notebook's behaviour).
"""
from __future__ import annotations


def tile_offsets(length: int, tile: int, stride: int, edge_mode: str = "shift") -> list[int]:
    """Start offsets of tiles covering [0, length).

    edge_mode="shift": the last tile is moved back so it ends exactly at `length`
        (no padding, unless the image is smaller than one tile).
    edge_mode="pad":   offsets are 0, stride, 2*stride, ... < length; tiles running past
        the edge are zero-padded (original notebook behaviour).
    """
    if length <= 0:
        return []
    if edge_mode == "pad":
        return list(range(0, length, stride))
    if length <= tile:
        return [0]
    offs = list(range(0, length - tile + 1, stride))
    if offs[-1] + tile < length:
        offs.append(length - tile)
    return offs


def keep_ranges(offsets: list[int], length: int, tile: int) -> list[tuple[int, int]]:
    """For each tile, the [start, end) range (image coordinates) that it contributes.

    Ranges are contiguous, non-overlapping and together cover [0, length).
    """
    n = len(offsets)
    bounds = [0]
    for i in range(1, n):
        prev_end = min(offsets[i - 1] + tile, length)
        cur_start = offsets[i]
        if cur_start >= prev_end:  # no overlap
            bounds.append(cur_start)
        else:
            bounds.append((cur_start + prev_end) // 2)
    bounds.append(length)
    return [(bounds[i], bounds[i + 1]) for i in range(n)]
