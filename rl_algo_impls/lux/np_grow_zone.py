from typing import NamedTuple

import numpy as np


class GrowZoneCarry(NamedTuple):
    own_zone: np.ndarray  # bool[W, H]
    growing_zone: np.ndarray  # bool[W, H]
    growable_zone: np.ndarray  # bool[W, H]


def has_growing_zones(carry: GrowZoneCarry) -> np.ndarray:  # bool[1]
    return carry.growing_zone.any()


def grow_own_zone(carry: GrowZoneCarry) -> GrowZoneCarry:
    own_zone, growing_zone, growable_zone = carry
    # bool[1+W+1, 1+H+1]
    padded_own_zone = np.pad(own_zone, ((1, 1), (1, 1)))
    # bool[W, H]
    spread_zone = (
        np.roll(padded_own_zone, -1, axis=-2)
        | np.roll(padded_own_zone, 1, axis=-2)
        | np.roll(padded_own_zone, -1, axis=-1)
        | np.roll(padded_own_zone, 1, axis=-1)
    )[1:-1, 1:-1]
    growing_zone = spread_zone & ~own_zone & growable_zone
    own_zone = own_zone | growing_zone
    return GrowZoneCarry(own_zone, growing_zone, growable_zone)


def fill_valid_regions(valid_masks: np.ndarray) -> np.ndarray:
    # int[1, H]
    x_left = np.where(
        valid_masks.any(-2, keepdims=True),
        np.argmax(valid_masks, axis=-2, keepdims=True),
        valid_masks.shape[-1],
    )
    # int[W, 1]
    y_up = np.where(
        valid_masks.any(-1, keepdims=True),
        np.argmax(valid_masks, axis=-1, keepdims=True),
        valid_masks.shape[-2],
    )
    x_right = np.where(
        valid_masks.any(-2, keepdims=True),
        valid_masks.shape[-1] - np.argmax(valid_masks[::-1, :], axis=-2, keepdims=True),
        0,
    )
    y_down = np.where(
        valid_masks.any(-1, keepdims=True),
        valid_masks.shape[-2] - np.argmax(valid_masks[:, ::-1], axis=-1, keepdims=True),
        0,
    )
    # bool[W, 1]
    x_fill = (
        np.arange(valid_masks.shape[-2])[:, None] >= x_left.min(-1, keepdims=True)
    ) & (np.arange(valid_masks.shape[-2])[:, None] < x_right.max(-1, keepdims=True))
    # bool[1, H]
    y_fill = (
        np.arange(valid_masks.shape[-1])[None, :] >= y_up.min(-2, keepdims=True)
    ) & (np.arange(valid_masks.shape[-1])[None, :] < y_down.max(-2, keepdims=True))
    # bool[W, H]
    filled_valid_masks = x_fill & y_fill
    return filled_valid_masks
