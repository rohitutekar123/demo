# # # # from fastapi import FastAPI, File, UploadFile, HTTPException
# # # # import shutil
# # # # import os
# # # # from moviepy.editor import VideoFileClip, concatenate_videoclips
# # # # import whisper
# # # # from transformers import MarianMTModel, MarianTokenizer
# # # # from scenedetect import VideoManager, SceneManager
# # # # from scenedetect.detectors import ContentDetector
# # # # from gtts import gTTS
# # # # from pydub import AudioSegment
# # # # import subprocess
# # # # import logging

# # # # app = FastAPI()

# # # # # Setup logging for debugging
# # # # logging.basicConfig(level=logging.INFO)

# # # # # Allowed video extensions
# # # # ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov"]

# # # # # Folder paths
# # # # UPLOAD_FOLDER = "uploads"
# # # # OUTPUT_FOLDER = "outputs"
# # # # TRANSCRIPTION_FOLDER = "transcriptions"
# # # # SUMMARY_FOLDER = "summaries"
# # # # AUDIO_FOLDER = "audios"

# # # # # Ensure directories exist
# # # # for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TRANSCRIPTION_FOLDER, SUMMARY_FOLDER, AUDIO_FOLDER]:
# # # #     os.makedirs(folder, exist_ok=True)

# # # # # Function to check if file is a valid video
# # # # def is_video_file(filename: str):
# # # #     ext = os.path.splitext(filename)[1].lower()
# # # #     return ext in ALLOWED_VIDEO_EXTENSIONS

# # # # # Step 1: Improved Scene Detection with dynamic thresholding
# # # # def detect_scenes(video_path):
# # # #     video_manager = VideoManager([video_path])
# # # #     scene_manager = SceneManager()
# # # #     scene_manager.add_detector(ContentDetector(threshold=20.0))  # Lower threshold for finer detection

# # # #     video_manager.set_downscale_factor()
# # # #     video_manager.start()
# # # #     scene_manager.detect_scenes(frame_source=video_manager)
# # # #     scene_list = scene_manager.get_scene_list()
# # # #     video_manager.release()

# # # #     scenes = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
    
# # # #     if not scenes:
# # # #         raise ValueError("No scenes detected, adjust detection settings.")
    
# # # #     logging.info(f"Detected {len(scenes)} scenes.")
# # # #     return scenes

# # # # # Step 2: Create a summary video by selecting important scenes
# # # # def create_summary(video_path, scene_list):
# # # #     video = VideoFileClip(video_path)

# # # #     # Select key scenes based on length and importance
# # # #     key_clips = [video.subclip(start, end) for start, end in scene_list]
# # # #     key_clips = sorted(key_clips, key=lambda clip: clip.duration, reverse=True)

# # # #     # Pick the top 10 scenes based on duration or all if less
# # # #     summary_clips = key_clips[:10] if len(key_clips) > 10 else key_clips
# # # #     summary_clip = concatenate_videoclips(summary_clips, method="compose")

# # # #     summarized_video_path = os.path.join(OUTPUT_FOLDER, "summarized_video.mp4")
# # # #     summary_clip.write_videofile(summarized_video_path, codec="libx264")

# # # #     video.close()  # Close the video to release resources
# # # #     logging.info(f"Summarized video created at {summarized_video_path}")

# # # #     return summarized_video_path

# # # # # Step 3: Use Whisper large model for better transcription accuracy
# # # # def transcribe_audio(video_path):
# # # #     audio_path = os.path.join(AUDIO_FOLDER, "extracted_audio.mp3")
# # # #     video = VideoFileClip(video_path)

# # # #     if video.audio is not None:
# # # #         video.audio.write_audiofile(audio_path)
# # # #     else:
# # # #         raise ValueError("The video does not contain any audio.")

# # # #     video.close()

# # # #     model = whisper.load_model("large")  # Switch to a larger model for better transcription
# # # #     result = model.transcribe(audio_path)
# # # #     transcription = result["text"]
    
# # # #     transcription_file = os.path.join(TRANSCRIPTION_FOLDER, "transcription.txt")
# # # #     with open(transcription_file, "w") as f:
# # # #         f.write(transcription)

# # # #     logging.info(f"Transcription completed and saved to {transcription_file}")
# # # #     return transcription, result['segments']

# # # # # Step 4: Translate text to Hindi in batches for better efficiency
# # # # def translate_text_to_hindi(transcription_segments):
# # # #     model_name = 'Helsinki-NLP/opus-mt-en-hi'
# # # #     tokenizer = MarianTokenizer.from_pretrained(model_name)
# # # #     model = MarianMTModel.from_pretrained(model_name)

# # # #     translated_segments = []
# # # #     for segment in transcription_segments:
# # # #         text = segment['text']
# # # #         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)  # Enable truncation for long texts
# # # #         translated = model.generate(**inputs)
# # # #         hindi_text = tokenizer.decode(translated[0], skip_special_tokens=True)
# # # #         translated_segments.append((hindi_text, segment['start'], segment['end']))

# # # #     logging.info(f"Translation to Hindi completed for {len(translated_segments)} segments.")
# # # #     return translated_segments

# # # # # Step 5: Create synchronized Hindi audio using gTTS with proper timing
# # # # def create_hindi_audio(translated_segments):
# # # #     hindi_audio_path = os.path.join(AUDIO_FOLDER, "hindi_audio.mp3")
# # # #     combined_audio = AudioSegment.silent(duration=0)  # Start with an empty audio segment

# # # #     for hindi_text, start, end in translated_segments:
# # # #         tts = gTTS(hindi_text, lang='hi')
# # # #         segment_audio_path = os.path.join(AUDIO_FOLDER, f"hindi_segment_{start}_{end}.mp3")
# # # #         tts.save(segment_audio_path)

# # # #         # Load the audio segment
# # # #         segment_audio = AudioSegment.from_file(segment_audio_path)

# # # #         # Calculate the desired duration in milliseconds
# # # #         desired_duration = (end - start) * 1000  # Convert seconds to milliseconds

# # # #         # Adjust the audio segment to fit the desired duration
# # # #         if len(segment_audio) > desired_duration:
# # # #             segment_audio = segment_audio[:desired_duration]  # Trim if too long
# # # #         else:
# # # #             segment_audio = segment_audio + AudioSegment.silent(duration=desired_duration - len(segment_audio))  # Pad if too short

# # # #         combined_audio += segment_audio

# # # #     # Export the final combined audio
# # # #     combined_audio.export(hindi_audio_path, format="mp3")

# # # #     logging.info(f"Hindi audio created and saved at {hindi_audio_path}")
# # # #     return hindi_audio_path

# # # # # Step 6: Merge summarized video and Hindi audio using subprocess and ffmpeg
# # # # def merge_audio_video_ffmpeg(video_path, audio_path):
# # # #     output_video_path = os.path.join(OUTPUT_FOLDER, "final_output_video.mp4")
    
# # # #     video_path = os.path.abspath(video_path)
# # # #     audio_path = os.path.abspath(audio_path)

# # # #     video_duration = VideoFileClip(video_path).duration
# # # #     audio_duration = AudioSegment.from_file(audio_path).duration_seconds

# # # #     # Align durations
# # # #     if audio_duration < video_duration:
# # # #         silence_duration = (video_duration - audio_duration) * 1000  # in milliseconds
# # # #         silence_segment = AudioSegment.silent(duration=silence_duration)
# # # #         combined_audio = AudioSegment.from_file(audio_path) + silence_segment
# # # #     else:
# # # #         combined_audio = AudioSegment.from_file(audio_path)[:int(video_duration * 1000)]  # trim to video duration

# # # #     # Increase volume by 6dB if needed
# # # #     combined_audio = combined_audio + 6  # Boost volume by 6dB
# # # #     adjusted_audio_path = os.path.join(AUDIO_FOLDER, "adjusted_hindi_audio.mp3")
# # # #     combined_audio.export(adjusted_audio_path, format="mp3")

# # # #     try:
# # # #         logging.info(f"Merging video: {video_path} with audio: {adjusted_audio_path}")
        
# # # #         command = [
# # # #             'ffmpeg', '-loglevel', 'verbose', '-i', video_path, '-i', adjusted_audio_path,
# # # #             '-map', '0:v:0', '-map', '1:a:0',
# # # #             '-c:v', 'libx264', '-c:a', 'aac', '-b:a', '192k',
# # # #             '-shortest', output_video_path
# # # #         ]

# # # #         subprocess.run(command, check=True)
        
# # # #         logging.info(f"Output video saved at: {output_video_path}")
# # # #         return output_video_path

# # # #     except subprocess.CalledProcessError as e:
# # # #         logging.error(f"An error occurred during the merging process: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail=f"An error occurred during the merging process: {str(e)}")

# # # # # Endpoint to upload video and process it
# # # # @app.post("/upload-video/")
# # # # async def upload_video(file: UploadFile = File(...)):
# # # #     if not is_video_file(file.filename):
# # # #         raise HTTPException(status_code=400, detail="Invalid file format. Please upload a valid video file.")
    
# # # #     video_path = os.path.join(UPLOAD_FOLDER, file.filename)
# # # #     with open(video_path, "wb") as buffer:
# # # #         shutil.copyfileobj(file.file, buffer)

# # # #     try:
# # # #         scene_list = detect_scenes(video_path)
# # # #         summarized_video_path = create_summary(video_path, scene_list)
# # # #         transcription, transcription_segments = transcribe_audio(summarized_video_path)
    
# # # #         # Translate the transcription segments to Hindi
# # # #         translated_segments = translate_text_to_hindi(transcription_segments)
    
# # # #         # Create Hindi audio from the translated segments
# # # #         hindi_audio_path = create_hindi_audio(translated_segments)

# # # #         # Merge the summarized video with the Hindi audio
# # # #         final_video_path = merge_audio_video_ffmpeg(summarized_video_path, hindi_audio_path)

# # # #         return {"message": "Video processed successfully", "final_video_path": final_video_path}

# # # #     except Exception as e:
# # # #         logging.error(f"An error occurred: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")



# # # #   ---------------------------------------------------------------- new code 1 here  to handle   ----------------------------------------------------------------

# # # from fastapi import FastAPI, File, UploadFile, HTTPException
# # # import shutil
# # # import os
# # # from moviepy.editor import VideoFileClip, concatenate_videoclips
# # # import whisper
# # # from transformers import MarianMTModel, MarianTokenizer
# # # from scenedetect import VideoManager, SceneManager
# # # from scenedetect.detectors import ContentDetector
# # # from gtts import gTTS
# # # from pydub import AudioSegment
# # # import subprocess
# # # import logging

# # # app = FastAPI()

# # # # Setup logging for debugging
# # # logging.basicConfig(level=logging.INFO)

# # # # Allowed video extensions
# # # ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov"]

# # # # Folder paths
# # # UPLOAD_FOLDER = "uploads"
# # # OUTPUT_FOLDER = "outputs"
# # # TRANSCRIPTION_FOLDER = "transcriptions"
# # # SUMMARY_FOLDER = "summaries"
# # # AUDIO_FOLDER = "audios"

# # # # Ensure directories exist
# # # for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TRANSCRIPTION_FOLDER, SUMMARY_FOLDER, AUDIO_FOLDER]:
# # #     os.makedirs(folder, exist_ok=True)

# # # # Function to check if file is a valid video
# # # def is_video_file(filename: str):
# # #     ext = os.path.splitext(filename)[1].lower()
# # #     return ext in ALLOWED_VIDEO_EXTENSIONS

# # # # Step 1: Improved Scene Detection with dynamic thresholding
# # # def detect_scenes(video_path):
# # #     video_manager = VideoManager([video_path])
# # #     scene_manager = SceneManager()
# # #     scene_manager.add_detector(ContentDetector(threshold=20.0))  # Lower threshold for finer detection

# # #     video_manager.set_downscale_factor()
# # #     video_manager.start()
# # #     scene_manager.detect_scenes(frame_source=video_manager)
# # #     scene_list = scene_manager.get_scene_list()
# # #     video_manager.release()

# # #     scenes = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
    
# # #     if not scenes:
# # #         raise ValueError("No scenes detected, adjust detection settings.")
    
# # #     logging.info(f"Detected {len(scenes)} scenes.")
# # #     return scenes

# # # # Step 2: Create a summary video by selecting important scenes
# # # def create_summary(video_path, scene_list):
# # #     video = VideoFileClip(video_path)

# # #     # Select key scenes based on length and importance
# # #     key_clips = [video.subclip(start, end) for start, end in scene_list]
# # #     key_clips = sorted(key_clips, key=lambda clip: clip.duration, reverse=True)

# # #     # Pick the top 10 scenes based on duration or all if less
# # #     summary_clips = key_clips[:10] if len(key_clips) > 10 else key_clips
# # #     summary_clip = concatenate_videoclips(summary_clips, method="compose")

# # #     summarized_video_path = os.path.join(OUTPUT_FOLDER, "summarized_video.mp4")
# # #     summary_clip.write_videofile(summarized_video_path, codec="libx264")

# # #     video.close()  # Close the video to release resources
# # #     logging.info(f"Summarized video created at {summarized_video_path}")

# # #     return summarized_video_path

# # # # Step 3: Use Whisper large model for better transcription accuracy
# # # def transcribe_audio(video_path):
# # #     audio_path = os.path.join(AUDIO_FOLDER, "extracted_audio.mp3")
# # #     video = VideoFileClip(video_path)

# # #     if video.audio is not None:
# # #         video.audio.write_audiofile(audio_path)
# # #     else:
# # #         raise ValueError("The video does not contain any audio.")

# # #     video.close()

# # #     model = whisper.load_model("large")  # Switch to a larger model for better transcription
# # #     result = model.transcribe(audio_path)
# # #     transcription = result["text"]
    
# # #     transcription_file = os.path.join(TRANSCRIPTION_FOLDER, "transcription.txt")
# # #     with open(transcription_file, "w") as f:
# # #         f.write(transcription)

# # #     logging.info(f"Transcription completed and saved to {transcription_file}")
# # #     return transcription, result['segments']

# # # # Step 4: Translate text to Hindi in batches for better efficiency
# # # def translate_text_to_hindi(transcription_segments):
# # #     model_name = 'Helsinki-NLP/opus-mt-en-hi'
# # #     tokenizer = MarianTokenizer.from_pretrained(model_name)
# # #     model = MarianMTModel.from_pretrained(model_name)

# # #     translated_segments = []
# # #     for segment in transcription_segments:
# # #         text = segment['text']
# # #         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)  # Enable truncation for long texts
# # #         translated = model.generate(**inputs)
# # #         hindi_text = tokenizer.decode(translated[0], skip_special_tokens=True)
# # #         translated_segments.append((hindi_text, segment['start'], segment['end']))

# # #     logging.info(f"Translation to Hindi completed for {len(translated_segments)} segments.")
# # #     return translated_segments

# # # # Step 5: Create synchronized Hindi audio using gTTS with proper timing
# # # def create_hindi_audio(translated_segments, delay_duration=0.5):  # Delay duration in seconds
# # #     hindi_audio_path = os.path.join(AUDIO_FOLDER, "hindi_audio.mp3")
# # #     combined_audio = AudioSegment.silent(duration=0)  # Start with an empty audio segment

# # #     for hindi_text, start, end in translated_segments:
# # #         tts = gTTS(hindi_text, lang='hi')
# # #         segment_audio_path = os.path.join(AUDIO_FOLDER, f"hindi_segment_{start}_{end}.mp3")
# # #         tts.save(segment_audio_path)

# # #         # Load the audio segment
# # #         segment_audio = AudioSegment.from_file(segment_audio_path)

# # #         # Calculate the desired duration in milliseconds
# # #         desired_duration = (end - start) * 1000  # Convert seconds to milliseconds

# # #         # Adjust the audio segment to fit the desired duration
# # #         if len(segment_audio) > desired_duration:
# # #             segment_audio = segment_audio[:desired_duration]  # Trim if too long
# # #         else:
# # #             segment_audio = segment_audio + AudioSegment.silent(duration=desired_duration - len(segment_audio))  # Pad if too short

# # #         # Add delay after each segment
# # #         delay_segment = AudioSegment.silent(duration=int(delay_duration * 1000))  # Convert seconds to milliseconds
# # #         combined_audio += segment_audio + delay_segment  # Add segment and delay

# # #     # Export the final combined audio
# # #     combined_audio.export(hindi_audio_path, format="mp3")

# # #     logging.info(f"Hindi audio created and saved at {hindi_audio_path}")
# # #     return hindi_audio_path

# # # # Step 6: Merge summarized video and Hindi audio using subprocess and ffmpeg
# # # def merge_audio_video_ffmpeg(video_path, audio_path):
# # #     output_video_path = os.path.join(OUTPUT_FOLDER, "final_output_video.mp4")
    
# # #     video_path = os.path.abspath(video_path)
# # #     audio_path = os.path.abspath(audio_path)

# # #     video_duration = VideoFileClip(video_path).duration
# # #     audio_duration = AudioSegment.from_file(audio_path).duration_seconds

# # #     # Align durations
# # #     if audio_duration < video_duration:
# # #         silence_duration = (video_duration - audio_duration) * 1000  # in milliseconds
# # #         silence_segment = AudioSegment.silent(duration=silence_duration)
# # #         combined_audio = AudioSegment.from_file(audio_path) + silence_segment
# # #     else:
# # #         combined_audio = AudioSegment.from_file(audio_path)[:int(video_duration * 1000)]  # trim to video duration

# # #     # Increase volume by 6dB if needed
# # #     combined_audio = combined_audio + 6  # Boost volume by 6dB
# # #     adjusted_audio_path = os.path.join(AUDIO_FOLDER, "adjusted_hindi_audio.mp3")
# # #     combined_audio.export(adjusted_audio_path, format="mp3")

# # #     try:
# # #         logging.info(f"Merging video: {video_path} with audio: {adjusted_audio_path}")
        
# # #         command = [
# # #             'ffmpeg', '-loglevel', 'verbose', '-i', video_path, '-i', adjusted_audio_path,
# # #             '-map', '0:v:0', '-map', '1:a:0',
# # #             '-c:v', 'libx264', '-c:a', 'aac', '-b:a', '192k',
# # #             '-shortest', output_video_path
# # #         ]

# # #         subprocess.run(command, check=True)
        
# # #         logging.info(f"Output video saved at: {output_video_path}")
# # #         return output_video_path

# # #     except subprocess.CalledProcessError as e:
# # #         logging.error(f"An error occurred during the merging process: {str(e)}")
# # #         raise HTTPException(status_code=500, detail=f"An error occurred during the merging process: {str(e)}")

# # # # Endpoint to upload video and process it
# # # @app.post("/upload-video/")
# # # async def upload_video(file: UploadFile = File(...)):
# # #     if not is_video_file(file.filename):
# # #         raise HTTPException(status_code=400, detail="Invalid file format. Please upload a valid video file.")
    
# # #     video_path = os.path.join(UPLOAD_FOLDER, file.filename)
# # #     with open(video_path, "wb") as buffer:
# # #         shutil.copyfileobj(file.file, buffer)

# # #     try:
# # #         scene_list = detect_scenes(video_path)
# # #         summarized_video_path = create_summary(video_path, scene_list)
# # #         transcription, transcription_segments = transcribe_audio(summarized_video_path)
    
# # #         # Translate the transcription segments to Hindi
# # #         translated_segments = translate_text_to_hindi(transcription_segments)
    
# # #         # Create Hindi audio from the translated segments with a delay of 0.5 seconds
# # #         hindi_audio_path = create_hindi_audio(translated_segments, delay_duration=0.5)

# # #         # Merge the summarized video with the Hindi audio
# # #         final_video_path = merge_audio_video_ffmpeg(summarized_video_path, hindi_audio_path)

# # #         return {"final_video_path": final_video_path}

# # #     except Exception as e:
# # #         logging.error(f"An error occurred during processing: {str(e)}")
# # #         raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")


# # # # https://chatgpt.com/share/6728d160-6104-8008-bf90-261ee8ae19b2


# # # ---------------------------------------------------------------- new code2 here-the fine tuning part --------------------------------

# # from fastapi import FastAPI, File, UploadFile, HTTPException
# # import shutil
# # import os
# # from moviepy.editor import VideoFileClip, concatenate_videoclips
# # import whisper
# # from transformers import MarianMTModel, MarianTokenizer
# # from scenedetect import VideoManager, SceneManager
# # from scenedetect.detectors import ContentDetector
# # from gtts import gTTS
# # from pydub import AudioSegment
# # import subprocess
# # import logging

# # app = FastAPI()

# # # Setup logging for debugging
# # logging.basicConfig(level=logging.INFO)

# # # Allowed video extensions
# # ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov"]

# # # Folder paths
# # UPLOAD_FOLDER = "uploads"
# # OUTPUT_FOLDER = "outputs"
# # TRANSCRIPTION_FOLDER = "transcriptions"
# # SUMMARY_FOLDER = "summaries"
# # AUDIO_FOLDER = "audios"

# # # Ensure directories exist
# # for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TRANSCRIPTION_FOLDER, SUMMARY_FOLDER, AUDIO_FOLDER]:
# #     os.makedirs(folder, exist_ok=True)

# # # Function to check if file is a valid video
# # def is_video_file(filename: str):
# #     ext = os.path.splitext(filename)[1].lower()
# #     return ext in ALLOWED_VIDEO_EXTENSIONS

# # # Step 1: Improved Scene Detection with dynamic thresholding
# # def detect_scenes(video_path):
# #     video_manager = VideoManager([video_path])
# #     scene_manager = SceneManager()
# #     scene_manager.add_detector(ContentDetector(threshold=20.0))  # Lower threshold for finer detection

# #     video_manager.set_downscale_factor()
# #     video_manager.start()
# #     scene_manager.detect_scenes(frame_source=video_manager)
# #     scene_list = scene_manager.get_scene_list()
# #     video_manager.release()

# #     scenes = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
    
# #     if not scenes:
# #         raise ValueError("No scenes detected, adjust detection settings.")
    
# #     logging.info(f"Detected {len(scenes)} scenes.")
# #     return scenes

# # # Step 2: Create a summary video by selecting important scenes
# # def create_summary(video_path, scene_list):
# #     video = VideoFileClip(video_path)

# #     # Select key scenes based on length and importance
# #     key_clips = [video.subclip(start, end) for start, end in scene_list]
# #     key_clips = sorted(key_clips, key=lambda clip: clip.duration, reverse=True)

# #     # Pick the top 10 scenes based on duration or all if less
# #     summary_clips = key_clips[:10] if len(key_clips) > 10 else key_clips
# #     summary_clip = concatenate_videoclips(summary_clips, method="compose")

# #     summarized_video_path = os.path.join(OUTPUT_FOLDER, "summarized_video.mp4")
# #     summary_clip.write_videofile(summarized_video_path, codec="libx264")

# #     video.close()  # Close the video to release resources
# #     logging.info(f"Summarized video created at {summarized_video_path}")

# #     return summarized_video_path

# # # Step 3: Use Whisper large model for better transcription accuracy
# # def transcribe_audio(video_path):
# #     audio_path = os.path.join(AUDIO_FOLDER, "extracted_audio.mp3")
# #     video = VideoFileClip(video_path)

# #     if video.audio is not None:
# #         video.audio.write_audiofile(audio_path)
# #     else:
# #         raise ValueError("The video does not contain any audio.")

# #     video.close()

# #     model = whisper.load_model("large")  # Switch to a larger model for better transcription
# #     result = model.transcribe(audio_path)
# #     transcription = result["text"]
    
# #     transcription_file = os.path.join(TRANSCRIPTION_FOLDER, "transcription.txt")
# #     with open(transcription_file, "w") as f:
# #         f.write(transcription)

# #     logging.info(f"Transcription completed and saved to {transcription_file}")
# #     return transcription, result['segments']

# # # Step 4: Translate text to Hindi in batches for better efficiency
# # def translate_text_to_hindi(transcription_segments):
# #     model_name = 'Helsinki-NLP/opus-mt-en-hi'
# #     tokenizer = MarianTokenizer.from_pretrained(model_name)
# #     model = MarianMTModel.from_pretrained(model_name)

# #     translated_segments = []
# #     for segment in transcription_segments:
# #         text = segment['text']
# #         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)  # Enable truncation for long texts
# #         translated = model.generate(**inputs)
# #         hindi_text = tokenizer.decode(translated[0], skip_special_tokens=True)
# #         translated_segments.append((hindi_text, segment['start'], segment['end']))

# #     logging.info(f"Translation to Hindi completed for {len(translated_segments)} segments.")
# #     return translated_segments

# # # Step 5: Create synchronized Hindi audio using gTTS with proper timing
# # def create_hindi_audio(translated_segments, delay=500):
# #     hindi_audio_path = os.path.join(AUDIO_FOLDER, "hindi_audio.mp3")
# #     combined_audio = AudioSegment.silent(duration=0)  # Start with an empty audio segment

# #     for hindi_text, start, end in translated_segments:
# #         tts = gTTS(hindi_text, lang='hi')
# #         segment_audio_path = os.path.join(AUDIO_FOLDER, f"hindi_segment_{start}_{end}.mp3")
# #         tts.save(segment_audio_path)

# #         # Load the audio segment
# #         segment_audio = AudioSegment.from_file(segment_audio_path)

# #         # Calculate the desired duration in milliseconds
# #         desired_duration = (end - start) * 1000  # Convert seconds to milliseconds

# #         # Adjust the audio segment to fit the desired duration
# #         if len(segment_audio) > desired_duration:
# #             segment_audio = segment_audio[:desired_duration]  # Trim if too long
# #         else:
# #             segment_audio = segment_audio + AudioSegment.silent(duration=desired_duration - len(segment_audio))  # Pad if too short

# #         # Add a delay between segments
# #         segment_audio += AudioSegment.silent(duration=delay)

# #         combined_audio += segment_audio

# #     # Export the final combined audio
# #     combined_audio.export(hindi_audio_path, format="mp3")

# #     logging.info(f"Hindi audio created and saved at {hindi_audio_path}")
# #     return hindi_audio_path

# # # Step 6: Merge summarized video and Hindi audio using subprocess and ffmpeg
# # def merge_audio_video_ffmpeg(video_path, audio_path):
# #     output_video_path = os.path.join(OUTPUT_FOLDER, "final_output_video.mp4")
    
# #     video_path = os.path.abspath(video_path)
# #     audio_path = os.path.abspath(audio_path)

# #     video_duration = VideoFileClip(video_path).duration
# #     audio_duration = AudioSegment.from_file(audio_path).duration_seconds

# #     # Align durations
# #     if audio_duration < video_duration:
# #         silence_duration = (video_duration - audio_duration) * 1000  # in milliseconds
# #         silence_segment = AudioSegment.silent(duration=silence_duration)
# #         combined_audio = AudioSegment.from_file(audio_path) + silence_segment
# #     else:
# #         combined_audio = AudioSegment.from_file(audio_path)[:int(video_duration * 1000)]  # trim to video duration

# #     # Save the final audio
# #     final_audio_path = os.path.join(AUDIO_FOLDER, "final_hindi_audio.mp3")
# #     combined_audio.export(final_audio_path, format="mp3")

# #     # Use ffmpeg to combine audio and video
# #     command = f'ffmpeg -i "{video_path}" -i "{final_audio_path}" -c:v copy -c:a aac -strict experimental "{output_video_path}"'
# #     subprocess.run(command, shell=True, check=True)

# #     logging.info(f"Final video with Hindi audio created at {output_video_path}")
# #     return output_video_path

# # # FastAPI endpoint to process the video file
# # @app.post("/process-video/")
# # async def process_video(file: UploadFile = File(...)):
# #     if not is_video_file(file.filename):
# #         raise HTTPException(status_code=400, detail="Invalid video file format")

# #     video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
# #     with open(video_path, "wb") as buffer:
# #         shutil.copyfileobj(file.file, buffer)

# #     try:
# #         scene_list = detect_scenes(video_path)
# #         summarized_video_path = create_summary(video_path, scene_list)
# #         transcription, segments = transcribe_audio(summarized_video_path)
# #         translated_segments = translate_text_to_hindi(segments)
# #         hindi_audio_path = create_hindi_audio(translated_segments)
# #         final_output_video_path = merge_audio_video_ffmpeg(summarized_video_path, hindi_audio_path)

# #         return {"status": "success", "output_video": final_output_video_path}
# #     except Exception as e:
# #         logging.error(f"Error processing video: {e}")
# #         raise HTTPException(status_code=500, detail=f"An error occurred while processing the video: {str(e)}")



# # ---------------------------------------------------------------- new code 3 here-the finle management function ----------------------------------------------------------------

# from fastapi import FastAPI, File, UploadFile, HTTPException
# import shutil
# import os
# import uuid  # For generating unique identifiers
# from datetime import datetime  # For timestamp
# from moviepy.editor import VideoFileClip, concatenate_videoclips
# import whisper
# from transformers import MarianMTModel, MarianTokenizer
# from scenedetect import VideoManager, SceneManager
# from scenedetect.detectors import ContentDetector
# from gtts import gTTS
# from pydub import AudioSegment
# import subprocess
# import logging

# app = FastAPI()

# # Setup logging for debugging
# logging.basicConfig(level=logging.INFO)

# # Allowed video extensions
# ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov"]

# # Ensure a base directory exists for uploads and outputs
# BASE_UPLOAD_FOLDER = "uploads"
# BASE_OUTPUT_FOLDER = "outputs"
# os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)

# # Function to check if the file is a valid video
# def is_video_file(filename: str):
#     ext = os.path.splitext(filename)[1].lower()
#     return ext in ALLOWED_VIDEO_EXTENSIONS

# # Function to create a unique folder for each video upload using UUID or timestamp
# def create_unique_output_folder():
#     # Use a combination of timestamp and UUID for unique folder names
#     unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
#     unique_folder = os.path.join(BASE_OUTPUT_FOLDER, unique_id)
#     os.makedirs(unique_folder, exist_ok=True)
#     return unique_folder

# # Step 1: Scene Detection (Same as before)
# def detect_scenes(video_path):
#     video_manager = VideoManager([video_path])
#     scene_manager = SceneManager()
#     scene_manager.add_detector(ContentDetector(threshold=20.0))

#     video_manager.set_downscale_factor()
#     video_manager.start()
#     scene_manager.detect_scenes(frame_source=video_manager)
#     scene_list = scene_manager.get_scene_list()
#     video_manager.release()

#     scenes = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
    
#     if not scenes:
#         raise ValueError("No scenes detected, adjust detection settings.")
    
#     logging.info(f"Detected {len(scenes)} scenes.")
#     return scenes

# # Step 2: Create a summary video by selecting important scenes (Same as before)
# def create_summary(video_path, scene_list, output_folder):
#     video = VideoFileClip(video_path)
#     key_clips = [video.subclip(start, end) for start, end in scene_list]
#     key_clips = sorted(key_clips, key=lambda clip: clip.duration, reverse=True)
#     summary_clips = key_clips[:10] if len(key_clips) > 10 else key_clips
#     summary_clip = concatenate_videoclips(summary_clips, method="compose")

#     summarized_video_path = os.path.join(output_folder, "summarized_video.mp4")
#     summary_clip.write_videofile(summarized_video_path, codec="libx264")

#     video.close()
#     logging.info(f"Summarized video created at {summarized_video_path}")

#     return summarized_video_path

# # Step 3: Transcription (Same as before)
# def transcribe_audio(video_path, output_folder):
#     audio_path = os.path.join(output_folder, "extracted_audio.mp3")
#     video = VideoFileClip(video_path)

#     if video.audio is not None:
#         video.audio.write_audiofile(audio_path)
#     else:
#         raise ValueError("The video does not contain any audio.")

#     video.close()

#     model = whisper.load_model("large")
#     result = model.transcribe(audio_path)
#     transcription = result["text"]

#     transcription_file = os.path.join(output_folder, "transcription.txt")
#     with open(transcription_file, "w") as f:
#         f.write(transcription)

#     logging.info(f"Transcription completed and saved to {transcription_file}")
#     return transcription, result['segments']

# # Step 4: Translate transcription segments to Hindi (Same as before)
# def translate_text_to_hindi(transcription_segments, output_folder):
#     model_name = 'Helsinki-NLP/opus-mt-en-hi'
#     tokenizer = MarianTokenizer.from_pretrained(model_name)
#     model = MarianMTModel.from_pretrained(model_name)

#     translated_segments = []
#     for segment in transcription_segments:
#         text = segment['text']
#         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         translated = model.generate(**inputs)
#         hindi_text = tokenizer.decode(translated[0], skip_special_tokens=True)
#         translated_segments.append((hindi_text, segment['start'], segment['end']))

#     logging.info(f"Translation to Hindi completed for {len(translated_segments)} segments.")
#     return translated_segments

# # Step 5: Create Hindi audio using gTTS and synchronize with timestamps (Same as before)
# def create_hindi_audio(translated_segments, output_folder, delay=500):
#     hindi_audio_path = os.path.join(output_folder, "hindi_audio.mp3")
#     combined_audio = AudioSegment.silent(duration=0)

#     for hindi_text, start, end in translated_segments:
#         tts = gTTS(hindi_text, lang='hi')
#         segment_audio_path = os.path.join(output_folder, f"hindi_segment_{start}_{end}.mp3")
#         tts.save(segment_audio_path)

#         segment_audio = AudioSegment.from_file(segment_audio_path)
#         desired_duration = (end - start) * 1000
#         if len(segment_audio) > desired_duration:
#             segment_audio = segment_audio[:desired_duration]
#         else:
#             segment_audio = segment_audio + AudioSegment.silent(duration=desired_duration - len(segment_audio))
#         segment_audio += AudioSegment.silent(duration=delay)

#         combined_audio += segment_audio

#     combined_audio.export(hindi_audio_path, format="mp3")

#     logging.info(f"Hindi audio created and saved at {hindi_audio_path}")
#     return hindi_audio_path

# # Step 6: Merge summarized video and Hindi audio using ffmpeg (Same as before)
# def merge_audio_video_ffmpeg(video_path, audio_path, output_folder):
#     output_video_path = os.path.join(output_folder, "final_output_video.mp4")
#     command = f'ffmpeg -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac -strict experimental "{output_video_path}"'
#     subprocess.run(command, shell=True, check=True)

#     logging.info(f"Final video with Hindi audio created at {output_video_path}")
#     return output_video_path

# # FastAPI endpoint to process video upload and generate unique outputs
# @app.post("/process-video/")
# async def process_video(file: UploadFile = File(...)):
#     if not is_video_file(file.filename):
#         raise HTTPException(status_code=400, detail="Invalid video file format")

#     # Create a unique output folder for this upload session
#     unique_output_folder = create_unique_output_folder()

#     # Save the uploaded video in a unique folder
#     video_path = os.path.join(BASE_UPLOAD_FOLDER, file.filename)
#     with open(video_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     try:
#         # Run the entire video processing pipeline
#         scene_list = detect_scenes(video_path)
#         summarized_video_path = create_summary(video_path, scene_list, unique_output_folder)
#         transcription, segments = transcribe_audio(summarized_video_path, unique_output_folder)
#         translated_segments = translate_text_to_hindi(segments, unique_output_folder)
#         hindi_audio_path = create_hindi_audio(translated_segments, unique_output_folder)
#         final_output_video_path = merge_audio_video_ffmpeg(summarized_video_path, hindi_audio_path, unique_output_folder)

#         return {
#             "status": "success",
#             "unique_output_folder": unique_output_folder,
#             "output_video": final_output_video_path
#         }
#     except Exception as e:
#         logging.error(f"Error processing video: {e}")
#         raise HTTPException(status_code=500, detail=f"An error occurred while processing the video: {str(e)}")


# ---------------------------------------------------------------- final processing with hindi video processing ------------------------
from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
import uuid  # For generating unique identifiers
from datetime import datetime  # For timestamp
from moviepy.editor import VideoFileClip, concatenate_videoclips
import whisper
from transformers import MarianMTModel, MarianTokenizer
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from gtts import gTTS
from pydub import AudioSegment
import subprocess
import logging

app = FastAPI()

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)

# Allowed video extensions
ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov"]

# Ensure a base directory exists for uploads and outputs
BASE_UPLOAD_FOLDER = "uploads"
BASE_OUTPUT_FOLDER = "outputs"
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)

# Function to check if the file is a valid video
def is_video_file(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_VIDEO_EXTENSIONS

# Function to create a unique folder for each video upload using UUID or timestamp
def create_unique_output_folder():
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    unique_folder = os.path.join(BASE_OUTPUT_FOLDER, unique_id)
    os.makedirs(unique_folder, exist_ok=True)
    return unique_folder

# Step 1: Scene Detection (Same as before)
def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=20.0))

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()

    scenes = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
    
    if not scenes:
        raise ValueError("No scenes detected, adjust detection settings.")
    
    logging.info(f"Detected {len(scenes)} scenes.")
    return scenes

# Step 2: Create a summary video by selecting important scenes (Same as before)
def create_summary(video_path, scene_list, output_folder):
    video = VideoFileClip(video_path)
    key_clips = [video.subclip(start, end) for start, end in scene_list]
    key_clips = sorted(key_clips, key=lambda clip: clip.duration, reverse=True)
    summary_clips = key_clips[:10] if len(key_clips) > 10 else key_clips
    summary_clip = concatenate_videoclips(summary_clips, method="compose")

    summarized_video_path = os.path.join(output_folder, "summarized_video.mp4")
    summary_clip.write_videofile(summarized_video_path, codec="libx264")

    video.close()
    logging.info(f"Summarized video created at {summarized_video_path}")

    return summarized_video_path

# Step 3: Transcription (Same as before)
def transcribe_audio(video_path, output_folder):
    audio_path = os.path.join(output_folder, "extracted_audio.mp3")
    video = VideoFileClip(video_path)

    if video.audio is not None:
        video.audio.write_audiofile(audio_path)
    else:
        raise ValueError("The video does not contain any audio.")

    video.close()

    model = whisper.load_model("large")
    result = model.transcribe(audio_path)
    transcription = result["text"]

    transcription_file = os.path.join(output_folder, "transcription.txt")
    with open(transcription_file, "w") as f:
        f.write(transcription)

    logging.info(f"Transcription completed and saved to {transcription_file}")
    return transcription, result['segments']

# Step 4: Translate transcription segments to Hindi (Same as before)
def translate_text_to_hindi(transcription_segments, output_folder):
    model_name = 'Helsinki-NLP/opus-mt-en-hi'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translated_segments = []
    for segment in transcription_segments:
        text = segment['text']
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        hindi_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_segments.append((hindi_text, segment['start'], segment['end']))

    logging.info(f"Translation to Hindi completed for {len(translated_segments)} segments.")
    return translated_segments

# Step 5: Create Hindi audio using gTTS and synchronize with timestamps (Same as before)
def create_hindi_audio(translated_segments, output_folder, delay=500):
    hindi_audio_path = os.path.join(output_folder, "hindi_audio.mp3")
    combined_audio = AudioSegment.silent(duration=0)

    for hindi_text, start, end in translated_segments:
        tts = gTTS(hindi_text, lang='hi')
        segment_audio_path = os.path.join(output_folder, f"hindi_segment_{start}_{end}.mp3")
        tts.save(segment_audio_path)

        segment_audio = AudioSegment.from_file(segment_audio_path)
        desired_duration = (end - start) * 1000
        if len(segment_audio) > desired_duration:
            segment_audio = segment_audio[:desired_duration]
        else:
            segment_audio = segment_audio + AudioSegment.silent(duration=desired_duration - len(segment_audio))
        segment_audio += AudioSegment.silent(duration=delay)

        combined_audio += segment_audio

    combined_audio.export(hindi_audio_path, format="mp3")

    logging.info(f"Hindi audio created and saved at {hindi_audio_path}")
    return hindi_audio_path

# Step 6: Merge summarized video and Hindi audio using ffmpeg (Fix to include Hindi audio and exclude original audio)
def merge_audio_video_ffmpeg(video_path, hindi_audio_path, output_folder):
    output_video_path = os.path.join(output_folder, "final_output_with_hindi_audio.mp4")

    # FFmpeg command to map the video and Hindi audio and disable the original audio
    command = (
        f'ffmpeg -i "{video_path}" -i "{hindi_audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac '
        f'-shortest -y "{output_video_path}"'
    )

    # Log the command being executed for troubleshooting
    logging.info(f"Running FFmpeg command: {command}")

    try:
        # Capture FFmpeg logs and errors
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during FFmpeg execution: {e.stderr.decode('utf-8')}")
        raise Exception(f"FFmpeg merge failed: {e.stderr.decode('utf-8')}")

    logging.info(f"Final video with Hindi audio created at {output_video_path}")
    return output_video_path

# FastAPI endpoint
@app.post("/uploadvideo/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Ensure the uploaded file is a video
        if not is_video_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a valid video file.")

        # Create a unique output folder for this upload
        output_folder = create_unique_output_folder()

        # Save the uploaded video
        video_path = os.path.join(output_folder, file.filename)
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Detect scenes in the video
        scenes = detect_scenes(video_path)

        # Create summarized video
        summarized_video_path = create_summary(video_path, scenes, output_folder)

        # Transcribe the audio
        transcription, transcription_segments = transcribe_audio(summarized_video_path, output_folder)

        # Translate to Hindi
        translated_segments = translate_text_to_hindi(transcription_segments, output_folder)

        # Create Hindi audio
        hindi_audio_path = create_hindi_audio(translated_segments, output_folder)

        # Merge video and Hindi audio
        final_video_path = merge_audio_video_ffmpeg(summarized_video_path, hindi_audio_path, output_folder)

        return {"message": "Video processed successfully", "final_video": final_video_path}

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the video: {str(e)}")
