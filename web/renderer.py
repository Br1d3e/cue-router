from datetime import datetime
from pathlib import Path

import pooltool as pt
from pooltool.ani.animate import FrameStepper
from pooltool.ani.globals import Global
from pooltool.ani.image.interface import get_graphics_texture, image_array_from_texture
from pooltool.ani.hud import HUDElement, hud
from panda3d.core import OrthographicLens
from PIL import Image

OUTPUT_BASE_DIR = Path(__file__).parent / "frames"

def render_frames(shot: pt.System, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_BASE_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stepper = FrameStepper()
    try:
        frames, frame_count = stepper.iterator(shot, size=(1280, 720), fps=30)
        texture = get_graphics_texture()
        hud.init()
        hud.elements[HUDElement.help_text].help_hint.hide()
        hud.update_cue(shot.cue, shot.balls[shot.cue.cue_ball_id])

        camera = Global.base.camera

        w, l = shot.table.w, shot.table.l

        camera.setPos(Global.render, w / 2, l / 2, 3.0)
        camera.setHpr(Global.render, 90, -90, 0)

        lens = OrthographicLens()

        film_width = l * 1.05
        film_height = film_width * 720 / 1280

        lens.setFilmSize(film_width, film_height)
        lens.setNearFar(0.1, 20)

        Global.base.cam.node().setLens(lens)

        for i in frames:
            frame = image_array_from_texture(texture)
            filename = output_dir / f"frame_{i:05d}.png"
            Image.fromarray(frame).save(filename)

        return output_dir
    finally:
        stepper.destroy()
