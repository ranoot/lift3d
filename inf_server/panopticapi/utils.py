"""Vendored subset of panopticapi.utils (cocodataset/panopticapi).

Only ``rgb2id`` / ``id2rgb`` / ``IdGenerator`` are reproduced -- the symbols
DVIS_Plus imports at module load. These are faithful to upstream (the COCO
panoptic color<->id encoding). The inf_server inference path does not call them;
they exist so the DVIS dataset/eval modules import cleanly.
"""

import numpy as np


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


class IdGenerator:
    """Assigns unique colors/ids per segment, matching upstream behaviour."""

    def __init__(self, categories):
        self.taken_colors = {(0, 0, 0)}
        self.categories = categories
        for category in self.categories.values():
            if category["isthing"] == 0:
                self.taken_colors.add(tuple(category["color"]))

    def get_color(self, cat_id):
        def random_color(base, max_dist=30):
            new_color = base + np.random.randint(
                low=-max_dist, high=max_dist + 1, size=3
            )
            return tuple(np.maximum(0, np.minimum(255, new_color)))

        category = self.categories[cat_id]
        if category["isthing"] == 0:
            return category["color"]
        base_color_array = category["color"]
        base_color = tuple(base_color_array)
        if base_color not in self.taken_colors:
            self.taken_colors.add(base_color)
            return base_color
        while True:
            color = random_color(base_color_array)
            if color not in self.taken_colors:
                self.taken_colors.add(color)
                return color

    def get_id(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color)

    def get_id_and_color(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color), color
