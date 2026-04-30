from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import time_mirror
from moviepy.editor import concatenate_videoclips
from pathlib import Path

input_path = r"C:\Users\vamsi\VamsiResearchProject\antmaze_videos\antmaze_episode_02.mp4"

start = 0
end = 2

output_dir = Path("Ant Skills")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"clip_{start}_to_{end}_loop.mp4"

clip = VideoFileClip(input_path)

subclip = clip.subclip(start, end)
final = concatenate_videoclips([subclip, subclip])

final.write_videofile(str(output_path), codec="libx264")

clip.close()