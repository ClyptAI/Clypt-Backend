from pipeline.phase_2_embed import get_native_multimodal_embedding

vec = get_native_multimodal_embedding(
    text="test transcript",
    video_uri="gs://clypt-test-bucket/phase_1a/video.mp4",
    start_sec=0,
    end_sec=10,
)
print(f"Dim: {len(vec)}, first 5: {vec[:5]}")
