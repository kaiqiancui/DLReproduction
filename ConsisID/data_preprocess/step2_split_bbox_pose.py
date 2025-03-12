import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import math
from moviepy import VideoFileClip
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Video Processing Parameters")
    parser.add_argument('--input_video_folder', type=str, default='step0/videos', help='Directory containing input videos (default: input_videos)')
    parser.add_argument('--input_json_folder', type=str, default='step0/json', help='Directory containing JSON files for bbox (default: step0/bbox)')
    parser.add_argument('--output_video_folder', type=str, default='step1/videos', help='Directory to store output videos (default: step1/videos)')
    parser.add_argument('--output_json_folder', type=str, default='step1/json', help='Directory to store output JSON files (default: step1/bbox)')
    parser.add_argument('--num_processes', type=int, default=1, help="Max number of parallel workers")
    args = parser.parse_args()
    return args


def is_face_large_enough_v2(face_boxes, threshold=0):
    for box in face_boxes:
        width = box['box']['x2'] - box['box']['x1']
        height = box['box']['y2'] - box['box']['y1']
        if width > threshold and height > threshold:
            return True
    return False


def extract_useful_frames(json_file, video_file_path, min_valid_frames=49, tolerance=5):
    with open(json_file, 'r') as f:
        data = json.load(f)

    useful_frames = []
    current_segment = []
    non_face_count = 0

    cap = cv2.VideoCapture(video_file_path)
    cap.release()

    for frame_num in range(len(data)):
        if str(frame_num) in data and data[str(frame_num)]['face']:
            face_boxes = data[str(frame_num)]['face']
            if is_face_large_enough_v2(face_boxes):
                current_segment.append(frame_num)
                non_face_count = 0
            else:
                if current_segment:
                    if non_face_count < tolerance:
                        current_segment.append(frame_num)
                        non_face_count += 1
                    else:
                        while non_face_count > 0:
                            if not is_face_large_enough_v2(data[str(current_segment[-1])]['face']):
                                current_segment.pop()
                                non_face_count -= 1
                            else:
                                break
                        if len(current_segment) >= min_valid_frames:
                            useful_frames.append(current_segment)
                        current_segment = []
                        non_face_count = 0
        else:
            if current_segment:
                if non_face_count < tolerance:
                    current_segment.append(frame_num)
                    non_face_count += 1
                else:
                    while non_face_count > 0:
                        if not is_face_large_enough_v2(data[str(current_segment[-1])]['face']):
                            current_segment.pop()
                            non_face_count -= 1
                        else:
                            break
                    if len(current_segment) >= min_valid_frames:
                        useful_frames.append(current_segment)
                    current_segment = []
                    non_face_count = 0

    if current_segment and len(current_segment) >= min_valid_frames:
        while non_face_count > 0:
            if not is_face_large_enough_v2(data[str(current_segment[-1])]['face']):
                current_segment.pop()
                non_face_count -= 1
            else:
                break
        if len(current_segment) >= min_valid_frames:
            useful_frames.append(current_segment)

    return useful_frames


def is_valid_frame(frame_data):
    for person in frame_data:
        visible = person['keypoints']['visible']
        # is_full_face = all(visible[i] >= 0.6 for i in range(3))  # 0: Nose, 1: Left Eye, 2: Right Eye
        is_half_face = (
            visible[0] >= 0.6 and 
            (visible[1] >= 0.6 or visible[2] >= 0.6)  # Nose and at least one eye visible
        )
        if is_half_face:
            return True
    return False


def merge_segments(useful_frames_bbox, segments_pose):
    merged_segments = []
    for bbox_segment in useful_frames_bbox:
        for pose_segment in segments_pose:
            # Find overlap between bbox and pose segments
            overlap = set(bbox_segment) & set(pose_segment)
            if overlap:
                # Merge the segment if there is overlap
                merged_segment = sorted(overlap)
                merged_segments.append(merged_segment)
    return merged_segments


def process_and_save_video(input_video_path, merged_segments, input_json_data, output_video_folder, output_json_folder):
    video_name = os.path.basename(input_video_path).replace('.mp4', '')
    video = VideoFileClip(input_video_path)

    segments_to_process = []
    for segment in merged_segments:
        start_frame = segment[0]
        end_frame = segment[-1]
        output_video_file = os.path.join(output_video_folder, f"{video_name}_{start_frame}_{end_frame}.mp4")
        output_bbox_file = os.path.join(output_json_folder, f"{video_name}_{start_frame}_{end_frame}.json")

        if not (os.path.exists(output_video_file) and os.path.exists(output_bbox_file)):
            segments_to_process.append(segment)

    if not segments_to_process:
        print("All segments already processed. Skipping video processing.")
        return

    for idx, segment in enumerate(segments_to_process):
        start_frame = segment[0]
        end_frame = segment[-1]
        start_time = start_frame / video.fps
        end_time = (end_frame + 1) / video.fps

        output_video_file = os.path.join(output_video_folder, f"{video_name}_{start_frame}_{end_frame}.mp4")
        output_bbox_file = os.path.join(output_json_folder, f"{video_name}_{start_frame}_{end_frame}.json")

        if not os.path.exists(output_video_file):
            video.subclipped(start_time, end_time).write_videofile(output_video_file, codec="libx264")

        if not os.path.exists(output_bbox_file):
            segment_json = {str(new_idx): input_json_data[str(original_idx)] for new_idx, original_idx in enumerate(segment)}
            with open(output_bbox_file, 'w') as f:
                json.dump(segment_json, f)

    print("Processing completed for the necessary segments.")


def extract_valid_segments_from_filtered_data(filtered_pose_json_data, min_valid_frames=49, tolerance=5):
    valid_segments = []
    current_segment = []
    consecutive_invalid_count = 0
    started_segment = False

    frame_keys = sorted(filtered_pose_json_data.keys(), key=lambda x: int(x))
    
    i = 0
    while i < len(frame_keys):
        frame_idx = frame_keys[i]
        frame_data = filtered_pose_json_data[frame_idx]

        if is_valid_frame(frame_data):
            if not started_segment:
                started_segment = True
            current_segment.append(int(frame_idx))
            consecutive_invalid_count = 0
        else:
            if not started_segment:
                i += 1
                continue

            consecutive_invalid_count += 1
            if consecutive_invalid_count <= tolerance:
                current_segment.append(int(frame_idx))
            else:
                trim_idx = len(current_segment) - 1
                while trim_idx >= 0:
                    tail_key = str(current_segment[trim_idx])
                    if is_valid_frame(filtered_pose_json_data[tail_key]):
                        break
                    trim_idx -= 1

                final_segment = current_segment[:trim_idx+1]
                if len(final_segment) >= min_valid_frames:
                    valid_segments.append(final_segment)

                # Restart segment and rollback to recheck frames
                rollback_start = max(0, i - tolerance)
                i = rollback_start - 1  # `-1` because `i` will be incremented in the next iteration

                current_segment = []
                consecutive_invalid_count = 0
                started_segment = False

        i += 1

    if len(current_segment) >= min_valid_frames:
        trim_idx = len(current_segment) - 1
        while trim_idx >= 0:
            tail_key = str(current_segment[trim_idx])
            if is_valid_frame(filtered_pose_json_data[tail_key]):
                break
            trim_idx -= 1

        final_segment = current_segment[:trim_idx+1]
        if len(final_segment) >= min_valid_frames:
            valid_segments.append(final_segment)

    return valid_segments


def process_video(input_video_path, input_json_path, output_video_folder, output_json_folder):
    video_name = os.path.basename(input_video_path).replace('.mp4', '')
    bbox_json_file = os.path.join(input_json_path, f"{video_name}.json")

    with open(bbox_json_file, 'r') as f:
        json_data = json.load(f)

    # Step 1: Extract useful frames from bbox data
    useful_frames_bbox = extract_useful_frames(bbox_json_file, input_video_path, tolerance=math.ceil(0.05*len(json_data)))

    # Step 2: Filter pose data based on the useful frames from bbox
    filtered_pose_json_data = {str(idx): json_data[str(idx)]['pose'] for segment in useful_frames_bbox for idx in segment}

    # Step 3: Extract valid segments from filtered pose data
    segments_pose = extract_valid_segments_from_filtered_data(filtered_pose_json_data, tolerance=math.ceil(0.025*len(json_data)))

    # Step 4: Merge bbox and pose segments
    merged_segments = merge_segments(useful_frames_bbox, segments_pose)

    # Step 5: Process and save merged segments
    process_and_save_video(input_video_path, merged_segments, json_data, output_video_folder, output_json_folder)


def main():
    args = parse_args()

    os.makedirs(args.output_video_folder, exist_ok=True)
    os.makedirs(args.output_json_folder, exist_ok=True)

    video_files = [f for f in os.listdir(args.input_video_folder) if f.endswith(".mp4")]

    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        futures = [
            executor.submit(process_video, os.path.join(args.input_video_folder, video_file), args.input_json_folder, args.output_video_folder, args.output_json_folder)
            for video_file in video_files
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()


if __name__ == "__main__":
    main()
