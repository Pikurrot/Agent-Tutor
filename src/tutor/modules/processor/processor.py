from __future__ import annotations
import pickle
import json
import cv2
import concurrent.futures
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.signal import find_peaks
from faster_whisper import WhisperModel
from pdf2image import convert_from_path
from docling.document_converter import DocumentConverter

from tutor.utils.paths import MODELS_CACHE_DIR
from tutor.utils.file_handler import save_json


class Processor:
    def __init__(self, config: dict):
        self.config = config
        self.my_config = config["processor_config"]
        self.doc_converter = DocumentConverter()
        self.transcription_model = WhisperModel(self.my_config["transcription_model"], download_root=MODELS_CACHE_DIR)

    def process_pdf(
        self,
        file_path: str,
        output_path: str
    ):
        doc = self.doc_converter.convert(file_path)
        pass

    def _compute_pixel_diff(
        self,
        video_path: str,
        save_path: Path
    ) -> tuple[np.ndarray, int, float]:
        # check if already computed
        if save_path.exists():
            print("Loading pixel diff from ", save_path)
            with open(save_path, "rb") as f:
                data = pickle.load(f)
                diff_scores, frames_to_skip, fps = data["diff_scores"], data["frames_to_skip"], data["fps"]
                return np.array(diff_scores), frames_to_skip, fps

        resize_width = self.my_config["pixel_diff_resize_width"]
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        diff_scores = []
        
        ret, prev_frame = cap.read()
        if not ret:
            return None, None, None
        
        aspect_ratio = prev_frame.shape[0] / prev_frame.shape[1]
        dim = (resize_width, int(resize_width * aspect_ratio))
        prev_gray = cv2.cvtColor(cv2.resize(prev_frame, dim), cv2.COLOR_BGR2GRAY)
        
        frames_to_skip = int(fps)
        current_frame_idx = frames_to_skip
        
        with tqdm(total=total_frames, desc="Frames processed", unit="frame") as pbar:
            pbar.update(frames_to_skip) 
            
            while current_frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                gray = cv2.cvtColor(cv2.resize(frame, dim), cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, prev_gray)
                score = np.mean(diff)
                
                diff_scores.append(score)
                prev_gray = gray
                
                current_frame_idx += frames_to_skip
                pbar.update(frames_to_skip)

        cap.release()
        
        if self.my_config["pixel_diff_save"]:
            print("Saving pixel diff to ", save_path)
            with open(save_path, "wb") as f:
                pickle.dump({"diff_scores": diff_scores, "frames_to_skip": frames_to_skip, "fps": fps}, f)
        
        return np.array(diff_scores), frames_to_skip, fps

    def _find_slide_transitions(
        self,
        diff_scores: np.ndarray,
        frames_to_skip: int,
        fps: float,
        save_path: Path
    ) -> list[int]:
        # check if already computed
        if save_path.exists():
            print("Loading slide transitions from ", save_path)
            with open(save_path, "rb") as f:
                slide_transition_frames = pickle.load(f)["slide_transition_frames"]
                return slide_transition_frames

        threshold_multiplier = self.my_config["slide_transition_threshold_multiplier"]
        custom_threshold = self.my_config["slide_transition_custom_threshold"]
        distance_sec = self.my_config["slide_transition_distance_sec"]
        
        if diff_scores is None or len(diff_scores) == 0:
            print("Error: Invalid difference scores provided.")
            return []

        # Calculate the threshold
        if custom_threshold is not None:
            threshold = custom_threshold
        else:
            no_outliers_diff_scores = diff_scores[diff_scores < np.percentile(diff_scores, 95)]
            mean_score = np.mean(no_outliers_diff_scores)
            std_score = np.std(no_outliers_diff_scores)
            threshold = mean_score + (threshold_multiplier * std_score)
        
        # Calculate distance in array indices (since 1 index != 1 frame)
        index_rate_per_sec = fps / frames_to_skip
        distance_in_indices = max(1, int(distance_sec * index_rate_per_sec))
        
        # Find the peaks
        peaks, _ = find_peaks(
            diff_scores, 
            height=threshold, 
            distance=distance_in_indices
        )
        
        # Map back to exact video frames
        slide_transition_frames = (peaks + 1) * frames_to_skip
        slide_transition_frames = slide_transition_frames.tolist()
        
        if self.my_config["slide_transition_save"]:
            print("Saving slide transitions to ", save_path)
            with open(save_path, "wb") as f:
                pickle.dump({"slide_transition_frames": slide_transition_frames}, f)
        
        return slide_transition_frames

    def _extract_frames_from_video(
        self,
        video_path: str,
        frame_indices: list[int]
    ) -> list[np.ndarray]:
        convert_to_rgb = self.my_config["extract_frames_from_video_convert_to_rgb"]
        cap = cv2.VideoCapture(video_path)
        extracted_images = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                if convert_to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                extracted_images.append(frame)
                
        cap.release()
        
        return extracted_images

    def _process_single_image_for_matching(
        self,
        img_a: np.ndarray,
        sift: cv2.SIFT,
        flann: cv2.FlannBasedMatcher,
        list_b_features: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> tuple[int, int]:
        color_tolerance = self.my_config["matching_color_tolerance"]
        # Add these to your config, or rely on these defaults
        rel_threshold = self.my_config.get("matching_rel_threshold", 0.6) 
        gap_tolerance = self.my_config.get("matching_gap_tolerance", 2)
        
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if len(img_a.shape) == 3 else img_a
        kp_a, des_a = sift.detectAndCompute(gray_a, None)
        
        if des_a is None or len(kp_a) < 4:
            return None, None

        # Extract coordinates once
        pts_a_all = np.array([kp.pt for kp in kp_a], dtype=np.float32)

        # Store scores for ALL video frames instead of just tracking the single best
        num_b = len(list_b_features)
        scores = np.zeros(num_b, dtype=np.int32)

        for j, (kp_b, des_b, img_b, pts_b_all) in enumerate(list_b_features):
            if des_b is None or len(kp_b) < 4:
                continue
                
            # 1. FLANN Matching (Much faster than Brute Force)
            matches = flann.knnMatch(des_a, des_b, k=2)
            
            # 2. Lowe's ratio test
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            if len(good_matches) < 4:
                continue
                
            # Get matching coordinates
            query_idx = [m.queryIdx for m in good_matches]
            train_idx = [m.trainIdx for m in good_matches]
            
            src_pts = pts_a_all[query_idx].reshape(-1, 1, 2)
            dst_pts = pts_b_all[train_idx].reshape(-1, 1, 2)
            
            # 3. RANSAC
            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is None:
                continue
                
            # 4. VECTORIZED Color Validation (Huge speedup)
            mask_bool = mask.ravel().astype(bool)
            
            # Filter points that survived RANSAC
            valid_src = src_pts[mask_bool].reshape(-1, 2).astype(int)
            valid_dst = dst_pts[mask_bool].reshape(-1, 2).astype(int)
            
            # Check boundaries cleanly
            bounds_a = (valid_src[:, 0] >= 0) & (valid_src[:, 0] < img_a.shape[1]) & \
                    (valid_src[:, 1] >= 0) & (valid_src[:, 1] < img_a.shape[0])
            bounds_b = (valid_dst[:, 0] >= 0) & (valid_dst[:, 0] < img_b.shape[1]) & \
                    (valid_dst[:, 1] >= 0) & (valid_dst[:, 1] < img_b.shape[0])
                    
            valid_bounds = bounds_a & bounds_b
            
            final_src = valid_src[valid_bounds]
            final_dst = valid_dst[valid_bounds]
            
            if len(final_src) == 0:
                continue
                
            # Extract colors and compute distance across all points at once
            colors_a = img_a[final_src[:, 1], final_src[:, 0]].astype(np.int32)
            colors_b = img_b[final_dst[:, 1], final_dst[:, 0]].astype(np.int32)
            
            color_dists = np.linalg.norm(colors_a - colors_b, axis=1)
            
            # Save the score to the array
            scores[j] = np.sum(color_dists <= color_tolerance)

        # Sequence detecting logic
        max_score = np.max(scores)
        
        # If no valid matches were found at all
        if max_score == 0:
            return None, None
            
        max_idx = int(np.argmax(scores))
        
        # Any frame scoring at least this percentage of the max score belongs to the slide
        threshold = max_score * rel_threshold
        
        start_idx = max_idx
        current_gap = 0
        
        # Walk backward from the peak to find the chronological beginning of the slide sequence
        for i in range(max_idx - 1, -1, -1):
            if scores[i] >= threshold:
                start_idx = i
                current_gap = 0 # Reset gap tolerance because we found a valid frame
            else:
                current_gap += 1
                # If we hit too many bad frames in a row, we assume the slide group has ended
                if current_gap > gap_tolerance:
                    break
                    
        return start_idx, max_score


    def _find_most_similar_images(
        self,
        list_a: list[np.ndarray],
        list_b: list[np.ndarray],
        save_path: Path
    ) -> tuple[list[int], list[int]]:
        # check if already computed
        if save_path.exists():
            print("Loading most similar images from ", save_path)
            with open(save_path, "rb") as f:
                data = pickle.load(f)
                best_matches_indices = data["best_matches_indices"]
                scores = data["scores"]
                return best_matches_indices, scores

        sift = cv2.SIFT_create()
        
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50) 
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        list_b_features = []
        for img_b in list_b:
            gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if len(img_b.shape) == 3 else img_b
            kp_b, des_b = sift.detectAndCompute(gray_b, None)
            # Pre-extract coordinates to save time in the loop
            pts_b_all = np.array([kp.pt for kp in kp_b], dtype=np.float32) if kp_b else np.array([])
            list_b_features.append((kp_b, des_b, img_b, pts_b_all))
            
        best_matches_indices = []
        scores = []
        
        # Parallel processing using ThreadPoolExecutor (or ProcessPoolExecutor for heavy CPU loads)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks
            futures = [executor.submit(self._process_single_image_for_matching, img_a, sift, flann, list_b_features) 
                    for img_a in list_a]
            
            # Retrieve results in order
            for future in futures:
                best_b_idx, best_score = future.result()
                
                # Handle cases where no SIFT features were found safely
                if best_b_idx is None:
                    best_matches_indices.append(None)
                    scores.append(0)
                else:
                    best_matches_indices.append(int(best_b_idx))
                    scores.append(int(best_score))

        if self.my_config["matching_save"]:
            print("Saving most similar images to ", save_path)
            with open(save_path, "wb") as f:
                pickle.dump({"best_matches_indices": best_matches_indices, "scores": scores}, f)

        return best_matches_indices, scores

    def _generate_timelines(
        self,
        video_path: str,
        slide_transition_frames: list[int],
        best_matches_indices: list[int],
        save_path: Path
    ) -> tuple[list[dict], dict[int, list]]:
        # check if already computed
        if save_path.exists():
            print("Loading video timeline from ", save_path)
            with open(save_path, "r") as f:
                data = json.load(f)
                video_timeline, slide_timestamps = data["video_timeline"], data["slide_timestamps"]
                return video_timeline, {int(k): v for k, v in slide_timestamps.items()}

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        valid_matches = [idx for idx in best_matches_indices if idx is not None]
        unique_best_matches_indices = list(set(valid_matches))
        
        pdf_to_segment = {int(pdf_idx): int(vid_idx) for pdf_idx, vid_idx in enumerate(best_matches_indices) if vid_idx is not None}
        
        segment_to_pdf_raw = {int(vid_idx): [] for vid_idx in unique_best_matches_indices}
        for pdf_idx, vid_idx in pdf_to_segment.items():
            segment_to_pdf_raw[vid_idx].append(pdf_idx)
            
        segment_to_pdf = {i: v for i, v in enumerate(sorted(segment_to_pdf_raw.values()))}

        if len(unique_best_matches_indices) > 0:
            transition_frames_sel = sorted(np.array(slide_transition_frames)[unique_best_matches_indices].tolist())
        else:
            transition_frames_sel = []
            
        segment_boundaries = [0] + transition_frames_sel[1:] + [total_frames]
        n_segments = min(len(transition_frames_sel), len(segment_boundaries) - 1)
        
        video_timeline = []     
        slide_timestamps = {}   

        for i in range(n_segments):
            start_frame = segment_boundaries[i]
            end_frame = segment_boundaries[i+1]
            
            start_time = start_frame / fps
            end_time = end_frame / fps
            
            assigned_pdfs = segment_to_pdf.get(i)
            
            video_timeline.append({
                'segment_index': i,
                'start_time_sec': round(start_time, 2),
                'end_time_sec': round(end_time, 2),
                'pdf_slide_index': assigned_pdfs
            })

            if assigned_pdfs is not None:
                for assigned_pdf in assigned_pdfs:
                    if assigned_pdf not in slide_timestamps:
                        slide_timestamps[assigned_pdf] = []
                    slide_timestamps[assigned_pdf].append((round(start_time, 2), round(end_time, 2)))

        if self.my_config["timelines_save"]:
            print("Saving video timeline to ", save_path)
            save_json(save_path, {"video_timeline": video_timeline, "slide_timestamps": slide_timestamps})

        return video_timeline, slide_timestamps

    def _transcribe_video(
        self,
        video_path: str,
        save_path: Path
    ) -> list[tuple[float, float, str]]:
        # check if already computed
        if save_path.exists():
            print("Loading transcription from ", save_path)
            with open(save_path, "r") as f:
                data = json.load(f)
                return data["segments"]

        segments = []
        
        transcription_config = self.my_config["transcription_config"]

        segments_generator, info = self.transcription_model.transcribe(
            video_path,
            language=transcription_config["language"],
            
            # VAD
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=transcription_config["min_silence_duration_ms"],
                speech_pad_ms=transcription_config["speech_pad_ms"],
            ),
            
            # Anti-Hallucination
            condition_on_previous_text=transcription_config["condition_on_previous_text"],
            no_speech_threshold=transcription_config["no_speech_threshold"],
            log_prob_threshold=transcription_config["log_prob_threshold"],
        )
        
        for segment in tqdm(segments_generator, desc="Transcribing video"):
            segments.append([segment.start, segment.end, segment.text])
        
        if self.my_config["transcription_save"]:
            print("Saving transcription to ", save_path)
            save_json(save_path, {"segments": segments})

        return segments

    def _get_segments_for_slide(
        self,
        slide_time_ranges: list[tuple[float, float]],
        transcription_segments: list[tuple[float, float, str]]
    ) -> list[tuple[float, float, str]]:
        matched_segments = []

        # If the slide never appeared (None or empty list), return nothing
        if not slide_time_ranges:
            return matched_segments

        for seg in transcription_segments:
            seg_start, seg_end, _ = seg[0], seg[1], seg[2]
            
            # Check if this audio segment overlaps with ANY of the times this slide was on screen
            for slide_start, slide_end in slide_time_ranges:
                # The Overlap Condition
                if seg_start < slide_end and seg_end > slide_start:
                    matched_segments.append(seg)
                    break  # We found a match, move on to the next audio segment

        return matched_segments

    def _get_segments_for_all_slides(
        self,
        slide_timestamps: dict[int, list[tuple[float, float]]],
        segments: list[tuple[float, float, str]]
    ) -> dict[int, list[tuple[float, float, str]]]:
        segments_for_slides = {}
        for slide_idx, slide_time_ranges in slide_timestamps.items():
            segments_for_slides[slide_idx] = self._get_segments_for_slide(slide_time_ranges, segments)
        return segments_for_slides

    def merge_video_matches(
        self,
        all_matches_indices: list[list[int]],
        all_scores: list[list[float]],
        video_names: list[str],
        save_path: Path
    ) -> tuple[list[list[tuple[str, int]]], list[list[float]]]:
        # check if already computed
        if save_path.exists():
            print("Loading merged matches from", save_path)
            with open(save_path, "r") as f:
                merged_data = json.load(f)
                merged_indices = merged_data["merged_indices"]
                merged_scores = merged_data["merged_scores"]
                return merged_indices, merged_scores
            
        abs_thresh = self.my_config["merge_asb_thresh"]
        rel_thresh = self.my_config["merge_rel_thresh"]
        merged_indices = []
        merged_scores = []
        
        num_slides = len(all_scores[0])
        num_videos = len(all_scores)
        
        for i in range(num_slides):
            valid_appearances = []
            valid_scores = []
            
            slide_scores = [all_scores[v][i] for v in range(num_videos)]
            max_score = max(slide_scores)
            
            if max_score < abs_thresh:
                merged_indices.append([])
                merged_scores.append([])
                continue
                
            for v in range(num_videos):
                s = slide_scores[v]
                idx = all_matches_indices[v][i]
                
                if s >= abs_thresh and s >= (rel_thresh * max_score):
                    valid_appearances.append((video_names[v], int(idx)))
                    valid_scores.append(float(s))
                    
            merged_indices.append(valid_appearances)
            merged_scores.append(valid_scores)
            
        if self.my_config["merge_save"]:
            print("Saving merged matches to ", save_path)
            save_json(save_path, {"merged_indices": merged_indices, "merged_scores": merged_scores})
            
        return merged_indices, merged_scores

    def process_video(
        self,
        video_path: str,
        pdf_path: str
    ) -> dict[int, list[tuple[float, float, str]]]:
        video_dir = Path(video_path).parent
        video_name = Path(video_path).stem
        subject_dir = video_dir.parent.parent
        intermediate_dir = subject_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_dir = subject_dir / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_name}_slides_transcriptions.json"

        # check if already computed
        if output_path.exists():
            print("Loading segments for slides from ", output_path)
            with open(output_path, "r") as f:
                return json.load(f)

        # Compute pixel diff between each frame and previous frame
        pixel_diff_save_path = intermediate_dir / f"{video_name}_pixel_diff.pkl"
        pixel_diff, frames_to_skip, fps = self._compute_pixel_diff(video_path, pixel_diff_save_path)
        # Find slide transitions when pixel diff is above threshold
        slide_transition_save_path = intermediate_dir / f"{video_name}_slide_transition.pkl"
        slide_transition_frames = self._find_slide_transitions(pixel_diff, frames_to_skip, fps, slide_transition_save_path)
        # Extract frames where slide transitions happen
        video_images = self._extract_frames_from_video(video_path, slide_transition_frames)
        # Extract images from corresponding PDF slides
        pdf_images = convert_from_path(pdf_path)
        pdf_images = [np.array(img) for img in pdf_images]
        # Find the most similar video frame for each pdf slide
        matching_save_path = intermediate_dir / f"{video_name}_matching.pkl"
        best_matches_indices, scores = self._find_most_similar_images(pdf_images, video_images, matching_save_path)
        # Separate video into segments (video_timeline)
        # and map each segment to the corresponding pdf slide (slide_timestamps)
        video_timeline_save_path = intermediate_dir / f"{video_name}_video_timeline.json"
        video_timeline, slide_timestamps = self._generate_timelines(video_path, slide_transition_frames, best_matches_indices, video_timeline_save_path)
        # Transcribe video into text segments
        transcription_save_path = intermediate_dir / f"{video_name}_transcription.json"
        segments = self._transcribe_video(video_path, transcription_save_path)
        # Get text segments for each slide
        segments_for_slides = self._get_segments_for_all_slides(slide_timestamps, segments)
        save_json(output_path, segments_for_slides)
        return segments_for_slides

    def process_videos(
        self,
        video_paths: list[str],
        pdf_path: str
    ) -> dict[int, dict[str, list[tuple[float, float, str]]]]:
        
        pdf_name = Path(pdf_path).stem
        # Assuming all videos belong to the same subject directory structure
        subject_dir = Path(video_paths[0]).parent.parent.parent
        intermediate_dir = subject_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_dir = subject_dir / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{pdf_name}_all_slides_transcriptions.json"

        if output_path.exists():
            print("Loading aggregated segments for slides from ", output_path)
            with open(output_path, "r") as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}

        print("Extracting PDF images...")
        pdf_images = convert_from_path(pdf_path)
        pdf_images = [np.array(img) for img in pdf_images]
        
        all_matches_indices = [] # says for each pdf slide, which video frame it is most similar to
        all_scores = [] # says how similar the video frame is to the pdf slide
        video_names = []
        video_data = {}
        
        # Process visual features for all videos
        for video_path in video_paths:
            video_name = Path(video_path).stem
            video_names.append(video_name)
            print(f"\n--- Processing Video: {video_name} ---")
            
            pixel_diff_save_path = intermediate_dir / f"{video_name}_pixel_diff.pkl"
            pixel_diff, frames_to_skip, fps = self._compute_pixel_diff(video_path, pixel_diff_save_path)
            
            slide_transition_save_path = intermediate_dir / f"{video_name}_slide_transition.pkl"
            slide_transition_frames = self._find_slide_transitions(pixel_diff, frames_to_skip, fps, slide_transition_save_path)
            
            video_images = self._extract_frames_from_video(video_path, slide_transition_frames)
            
            matching_save_path = intermediate_dir / f"{video_name}_matching.pkl"
            best_matches_indices, scores = self._find_most_similar_images(pdf_images, video_images, matching_save_path)
            
            all_matches_indices.append(best_matches_indices)
            all_scores.append(scores)
            video_data[video_name] = {"slide_transition_frames": slide_transition_frames}
            
        # Merge the results
        print("\n--- Merging Matches Across Videos ---")
        merged_save_path = intermediate_dir / f"{pdf_name}_merged_matches.json"
        merged_indices, merged_scores = self.merge_video_matches(all_matches_indices, all_scores, video_names, merged_save_path)
        
        # Generate Timelines and Audio Segments
        final_segments_for_slides = {i: {} for i in range(len(pdf_images))}
        
        for v, video_path in enumerate(video_paths):
            video_name = video_names[v]
            print(f"\n--- Generating Timeline & Audio for: {video_name} ---")
            
            # Create a filtered list of match indices JUST for this video based on the merge
            filtered_matches = []
            for i in range(len(pdf_images)):
                assigned_segment = None
                for vid_name, seg_idx in merged_indices[i]:
                    if vid_name == video_name:
                        assigned_segment = seg_idx
                        break
                filtered_matches.append(assigned_segment)
                
            video_timeline_save_path = intermediate_dir / f"{video_name}_video_timeline.json"
            video_timeline, slide_timestamps = self._generate_timelines(
                video_path, 
                video_data[video_name]["slide_transition_frames"], 
                filtered_matches, 
                video_timeline_save_path
            )
            
            transcription_save_path = intermediate_dir / f"{video_name}_transcription.json"
            segments = self._transcribe_video(video_path, transcription_save_path)
            
            segments_for_slides = self._get_segments_for_all_slides(slide_timestamps, segments)
            
            # Append this video's segments to the master slide dictionary
            for slide_idx, slide_segments in segments_for_slides.items():
                if slide_segments: 
                    final_segments_for_slides[slide_idx][video_name] = slide_segments

        # Save the final aggregated result
        save_json(output_path, final_segments_for_slides)
        print("\nProcessing Complete!")
        
        return final_segments_for_slides
