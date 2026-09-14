# DuckVLM integration notes

Raw RGB uses the existing `duck_play.perception.headcam.render_headcam_rgb` / LocalSim head-camera
path and passed the render-differencing test.

The existing `duck_play.perception.camera.DuckHeadCam.project()` uses a camera right vector that is
horizontally opposite to MuJoCo's rendered image. This affects target UV used by affordance overlay,
not the raw RGB frame. The minimal patch is in `duckvlm_headcam_alignment.patch`; apply it in the main
project and rerun `duck_vlm/tests` plus the three scene validators.
