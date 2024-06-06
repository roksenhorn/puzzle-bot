import os
import json

from common.config import *


def tighten_or_relax(solution_path, output_path):
    for file in os.listdir(solution_path):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(solution_path, file), "r") as f:
            data = json.load(f)

        x, y = data["solution_x"], data["solution_y"]
        dest_x, dest_y = data["dest_photo_space_incenter"]

        cumulative_padding_x = x * TIGHTEN_RELAX_PX_W
        cumulative_padding_y = y * TIGHTEN_RELAX_PX_H
        dest_x += cumulative_padding_x
        dest_y += cumulative_padding_y
        data["dest_photo_space_incenter"] = [dest_x, dest_y]

        with open(os.path.join(output_path, file), "w") as f:
            json.dump(data, f)
        print(f"Relaxed {file} by ({cumulative_padding_x}, {cumulative_padding_y})")
