from moviepy import editor as movp

class VideoAudioHandler:
            
    def extract_audio_from_video(video_file_path: str, output_location: str):
        video_file = movp.VideoFileClip(video_file_path)
        
        if video_file is not None:
            video_file.audio.write_audiofile(output_location)
    