import os
import torch
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

# Create the video generation pipeline.
pipe = Video2WorldPipeline.from_config(
    config=get_cosmos_predict2_video2world_pipeline(model_size="2B"),
    dit_path=get_cosmos_predict2_video2world_checkpoint(model_size="2B"),
)

# Specify the input image path and text prompt.
image_path = "assets/video2world/example_input.jpg"
prompt = "A high-definition video captures the precision of robotic welding in an industrial setting. The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation."

# Run the video generation pipeline.
video = pipe(input_path=image_path, prompt=prompt)

# Save the resulting output video.
os.makedirs("output", exist_ok=True)
save_image_or_video(video, "output/test.mp4", fps=16)