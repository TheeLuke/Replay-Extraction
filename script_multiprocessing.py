import cv2
import cv2.version
import numpy as np
import pytesseract
import json
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import os
import time
import multiprocessing # Added import
import traceback # Keep for potential errors within worker logic

##### Configuration
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#### Parameters
PARAMS = {
    'ALPHA': 3, 'P_BINARIZE': 2.5, 'L_AVG': 5, 'T1_OCR_CONF': 0.75,
    'T2_OCR_CHARS': 3, 'N_GT_MIN_LEN': 7, 'N_RL_MIN_DUR': 235,
    'N_RU_MAX_DUR': 800, 'T_L_HIST_DIFF': 0.025, 'T_U_HIST_ACCUM': 3.5,
    'HIST_COMP_METHOD': cv2.HISTCMP_CHISQR, 'SC_ROI_Y_START': 0,
    'SC_ROI_Y_END': 1080, 'SC_ROI_X_START': 0, 'SC_ROI_X_END': 1920,
}

# --- Functions (extract_frames, preprocess_frame_for_sc, detect_gradual_transitions,
# --- identify_candidate_replay_segments, detect_score_caption_absence,
# --- evaluate_replay_detection, export_replays_to_video) remain the same
# --- as in the previous version (where MemoryError occurred). Ensure you have those definitions.

#### 1. Frame Extraction (Keep as is)
def extract_frames(video_path):
    """Extracts frames from a video file."""
    print(f"Extracting frames from: {video_path}")
    # Check if file exists before trying to open
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return [], 0.0
    if not os.path.isfile(video_path):
        print(f"Error: Provided path is not a file: {video_path}")
        return [], 0.0

    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get total frames for progress
    print(f"Video Info: FPS={fps}, Total Frames={total_frames_video}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # Break if read failed or end of video
            break
        frames.append(frame)
        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"  Extracted {frame_count}/{total_frames_video} frames...")

    cap.release()
    # Verify frame count
    if frame_count != len(frames):
        print(f"Warning: Number of frames read ({len(frames)}) differs from internal count ({frame_count}). Using {len(frames)}.")
    if total_frames_video > 0 and len(frames) != total_frames_video:
         print(f"Warning: Number of extracted frames ({len(frames)}) differs from video metadata ({total_frames_video}).")

    print(f"Finished extraction. Total frames extracted: {len(frames)}, FPS: {fps}")
    return frames, fps

#### 2. SC Preprocessing (Keep as is)
def preprocess_frame_for_sc(frame, alpha_param):
    """Performs preprocessing steps on a single frame for SC detection."""
    if frame is None: return None # Handle None input
    try:
        # Grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Downsample by factor of 2
        height, width = gray.shape
        new_height = (height // 2) * 2
        new_width = (width // 2) * 2
        gray_resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        downsampled = cv2.resize(gray_resized, (new_width // 2, new_height // 2), interpolation=cv2.INTER_LINEAR)

        # Illumination adjustment using top-hat filtering
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (alpha_param, alpha_param))
        morph_open = cv2.morphologyEx(downsampled, cv2.MORPH_OPEN, structuring_element)
        illum_adjusted = cv2.subtract(downsampled, morph_open) # I_adj = I - (I open SE)
        return illum_adjusted
    except cv2.error as e:
         # print(f"  Warning: OpenCV error during preprocessing: {e}.") # Reduce noise
         return None # Return None on error
    except Exception as e:
        # print(f"  Warning: General error during preprocessing: {e}.") # Reduce noise
        return None # Return None on error


#### GT Detection
def detect_gradual_transitions(frames, PARAMS):
    print("Detecting Gradual Transitions (GTs)...")
    print(f"  Using Hist Comparison: {PARAMS['HIST_COMP_METHOD']}, T_L: {PARAMS['T_L_HIST_DIFF']}, T_U: {PARAMS['T_U_HIST_ACCUM']}")
    gt_segments = []
    if not frames:
        return gt_segments

    hist_size = [256] # Luminance histogram
    hist_range = [0, 256]
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_hist = cv2.calcHist([prev_gray], [0], None, hist_size, hist_range)
    cv2.normalize(prev_hist, prev_hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1) # Normalize

    potential_gt_start = -1
    accum_hist_diff = 0

    for i in range(1, len(frames)):
        current_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        current_hist = cv2.calcHist([current_gray], [0], None, hist_size, hist_range)
        cv2.normalize(current_hist, current_hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

        successive_diff = cv2.compareHist(prev_hist, current_hist, PARAMS['HIST_COMP_METHOD'])

        is_potential_start = False
        if PARAMS['HIST_COMP_METHOD'] == cv2.HISTCMP_CORREL:
            is_potential_start = successive_diff < PARAMS['T_L_HIST_DIFF'] # Correlation: Lower means different
        else:
             is_potential_start = successive_diff > PARAMS['T_L_HIST_DIFF'] # High difference suggests start

        if is_potential_start: # Potential start of GT
            if potential_gt_start == -1:
                potential_gt_start = i - 1
                print(f"  Potential GT start at frame {potential_gt_start} (Diff: {successive_diff:.4f})")
            # Accumulate difference during potential GT
            accum_hist_diff += successive_diff
        else:
             # If we were in a potential GT and the difference drops below threshold again
            if potential_gt_start != -1:
                print(f"  Potential GT end near frame {i-1} (Diff: {successive_diff:.4f}, Accum Diff: {accum_hist_diff:.4f})")
                # Check if accumulated difference exceeds TU
                if accum_hist_diff > PARAMS['T_U_HIST_ACCUM']:
                    gt_end = i - 1
                    # Check if GT duration is sufficient
                    if (gt_end - potential_gt_start + 1) >= PARAMS['N_GT_MIN_LEN']:
                        gt_segments.append((potential_gt_start, gt_end))
                        print(f"    --> Detected GT: Frames {potential_gt_start} to {gt_end} (Duration: {gt_end - potential_gt_start + 1}, AccumDiff: {accum_hist_diff:.4f})")
                    else:
                         print(f"    --> Discarded GT: Too short (Duration {gt_end - potential_gt_start + 1} < {PARAMS['N_GT_MIN_LEN']})")
                else:
                     print(f"    --> Discarded GT: AccumDiff {accum_hist_diff:.4f} <= {PARAMS['T_U_HIST_ACCUM']}")

                # Reset regardless of whether it met TU threshold
                potential_gt_start = -1
                accum_hist_diff = 0

        prev_hist = current_hist
        if i % 1000 == 0:
             print(f"  Processed {i} frames for GT detection...")
        
    return gt_segments

#### Identify Candidate Replay Segments (Keep as is)
def identify_candidate_replay_segments(gt_segments, params):
    print("Identifying Candidate Replay Segments (RSs)...")
    candidate_rs = []
    if len(gt_segments) < 2:
        print("  Not enough GTs found (< 2) to identify replay segments.")
        return candidate_rs

    gt_segments.sort(key=lambda x: x[0])

    for i in range(len(gt_segments) - 1):
        gt1_start, gt1_end = gt_segments[i]
        gt2_start, gt2_end = gt_segments[i+1]

        actual_replay_content_start = gt1_end + 1
        actual_replay_content_end = gt2_start - 1

        if actual_replay_content_start <= actual_replay_content_end:
             actual_replay_duration = actual_replay_content_end - actual_replay_content_start + 1
             if params['N_RL_MIN_DUR'] <= actual_replay_duration <= params['N_RU_MAX_DUR']:
                  candidate_rs.append((actual_replay_content_start, actual_replay_content_end))

    print(f"Finished candidate RS identification. Found {len(candidate_rs)} candidates.")
    return candidate_rs

#### SC Detection (Keep as is)
def detect_score_caption_absence(frames_segment, params, tesseract_cmd):
    """ Detects the absence of a Score Caption (SC) within a given segment of frames. """
    try:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    except Exception as e:
        print(f"Warning: Could not set tesseract command path in worker: {e}")

    if not frames_segment: return True

    if not isinstance(frames_segment[0], np.ndarray):
         print("Warning: detect_score_caption_absence received invalid frames_segment[0].")
         return False

    processed_frames = []
    alpha_p = params.get('ALPHA', 3)
    for f in frames_segment:
         processed = preprocess_frame_for_sc(f, alpha_p)
         processed_frames.append(processed)

    avg_images = []
    num_frames = len(processed_frames)
    l_avg = params.get('L_AVG', 5)
    for i in range(num_frames):
        if processed_frames[i] is None: avg_images.append(None); continue
        start = max(0, i - l_avg // 2)
        end = min(num_frames, i + l_avg // 2 + 1)
        window = processed_frames[start:end]
        valid_window = [img for img in window if img is not None and img.size > 0]
        if valid_window:
            try: # Add try block for shape check
                h, w = valid_window[0].shape
                consistent_window = [img for img in valid_window if img.shape == (h, w)]
                if consistent_window:
                     avg_img = np.mean(np.array(consistent_window, dtype=np.float32), axis=0).astype(np.uint8)
                     avg_images.append(avg_img)
                else: avg_images.append(processed_frames[i])
            except Exception as shape_e: # Catch potential errors if valid_window[0] is bad
                # print(f"Warning: Error accessing shape in averaging window {i}: {shape_e}") # Reduce noise
                avg_images.append(processed_frames[i]) # Fallback
        else: avg_images.append(processed_frames[i])

    sc_detected_in_segment = False
    t1_conf_scaled = params.get('T1_OCR_CONF', 0.75) * 100
    t2_chars = params.get('T2_OCR_CHARS', 3)
    roi_y_start = params.get('SC_ROI_Y_START', 0)
    roi_y_end = params.get('SC_ROI_Y_END', 1080)
    roi_x_start = params.get('SC_ROI_X_START', 0)
    roi_x_end = params.get('SC_ROI_X_END', 1920)
    p_bin = params.get('P_BINARIZE', 2.5)

    for i, avg_img in enumerate(avg_images):
        if avg_img is None: continue
        try:
            mean, std_dev = cv2.meanStdDev(avg_img)
            mean = mean[0][0]
            std_dev = std_dev[0][0] if std_dev[0][0] > 1e-6 else 1.0
            lower_bound = mean - p_bin * std_dev
            upper_bound = mean + p_bin * std_dev
            _, binary_img_inv = cv2.threshold(avg_img, upper_bound, 255, cv2.THRESH_BINARY_INV)
            _, binary_img_lower = cv2.threshold(avg_img, lower_bound, 255, cv2.THRESH_BINARY)
            binary_img = cv2.bitwise_and(binary_img_inv, binary_img_lower)
        except cv2.error as e:
            # print(f"Warning: Binarization error frame {i}: {e}") # Reduce noise
            continue

        roi_h, roi_w = binary_img.shape
        y_start_ds = max(0, roi_y_start // 2)
        y_end_ds = min(roi_h, roi_y_end // 2)
        x_start_ds = max(0, roi_x_start // 2)
        x_end_ds = min(roi_w, roi_x_end // 2)
        if y_start_ds >= y_end_ds or x_start_ds >= x_end_ds: continue
        roi_img = binary_img[y_start_ds:y_end_ds, x_start_ds:x_end_ds]
        if roi_img.size == 0: continue

        try:
            ocr_data = pytesseract.image_to_data(roi_img, lang='eng', config='--psm 7', output_type=pytesseract.Output.DICT)
            num_confident_chars = 0
            for j, conf_str in enumerate(ocr_data['conf']):
                try: conf = int(float(conf_str))
                except ValueError: conf = -1
                if conf >= t1_conf_scaled:
                    text = ocr_data['text'][j].strip()
                    if text: num_confident_chars += len(text)
            if num_confident_chars >= t2_chars:
                 sc_detected_in_segment = True; break
        except pytesseract.TesseractNotFoundError:
             print(f"FATAL ERROR (worker {os.getpid()}): Tesseract not found: {tesseract_cmd}")
             return False # Assume SC present
        except Exception as e:
            # print(f"Warning: OCR error frame {i} worker {os.getpid()}: {e}") # Reduce noise
            sc_detected_in_segment = True; break
    return not sc_detected_in_segment

#### Results Evaluation (Keep as is)
def evaluate_replay_detection(detected_replays, ground_truth_json_path, total_frames):
    """Evaluates the performance of the replay detection against ground truth."""
    print(f"\nEvaluating performance against: {ground_truth_json_path}")
    if not os.path.exists(ground_truth_json_path): print(f"Error: Ground truth file not found."); return None
    try:
        with open(ground_truth_json_path, 'r') as f: ground_truth = json.load(f)
        gt_replays = ground_truth.get("replays", [])
    except Exception as e: print(f"Error reading ground truth file: {e}"); return None

    if total_frames <= 0: print("Error: total_frames must be positive for evaluation."); return None

    y_true = np.zeros(total_frames, dtype=int)
    print("Processing Ground Truth Segments:")
    for start, end in gt_replays:
        start, end = int(start), int(end)
        if 0 <= start < total_frames and 0 <= end < total_frames and start <= end:
            y_true[start : end + 1] = 1
        else: print(f"  Warning: GT segment [{start}, {end}] invalid.")

    y_pred = np.zeros(total_frames, dtype=int)
    print("Processing Detected Segments:")
    for start, end in detected_replays:
         start, end = int(start), int(end)
         if 0 <= start < total_frames and 0 <= end < total_frames and start <= end:
             y_pred[start : end + 1] = 1
         else: print(f"  Warning: Detected segment [{start}, {end}] invalid.")

    labels = np.unique(np.concatenate((y_true, y_pred)))
    if len(labels) == 0: labels=[0, 1] # Handle case of no frames/no predictions
    elif len(labels) == 1: labels = sorted(list(labels) + [1 - labels[0]]) # Ensure two labels for CM

    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if cm.size == 4: tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1 and labels == [0, 1]: # Only TN or TP possible if only one class predicted/true
            tn = cm[0,0] if 0 in np.unique(y_true) else 0
            tp = cm[0,0] if 1 in np.unique(y_true) else 0
            fp, fn = 0, 0
        else: tn, fp, fn, tp = 0, 0, 0, 0 # Fallback
    except ValueError as e: print(f"Warning: Confusion matrix error: {e}. Setting counts to 0."); tn, fp, fn, tp = 0, 0, 0, 0

    precision = precision_score(y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    error_rate = (fp + fn) / total_frames if total_frames > 0 else 0

    results = {
        "true_positives": int(tp), "true_negatives": int(tn),
        "false_positives": int(fp), "false_negatives": int(fn),
        "precision": precision, "recall": recall, "accuracy": accuracy,
        "error_rate": error_rate, "total_frames": total_frames,
        "gt_replay_frames": int(np.sum(y_true)), "detected_replay_frames": int(np.sum(y_pred))
    }
    print("\n--- Evaluation Results ---")
    print(f"  Total Frames: {results['total_frames']}")
    print(f"  Ground Truth Replay Frames: {results['gt_replay_frames']}")
    print(f"  Detected Replay Frames: {results['detected_replay_frames']}")
    print(f"  TP: {results['true_positives']}, TN: {results['true_negatives']}, FP: {results['false_positives']}, FN: {results['false_negatives']}")
    print(f"  Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, Accuracy: {results['accuracy']:.4f}, Error Rate:{results['error_rate']:.4f}")
    print("-------------------------\n")
    return results

#### Export Video (Keep as is)
def export_replays_to_video(replay_segments, all_video_frames, fps, output_path):
    """ Combines frames from specified replay segments into a single video file. """
    print(f"\nExporting {len(replay_segments)} replay segments to: {output_path}")
    if not replay_segments: print("  No replay segments to export."); return
    # Check if all_video_frames is a non-empty list/iterable before proceeding
    if not all_video_frames or not hasattr(all_video_frames, '__getitem__'):
        print("  Error: Cannot export replays, 'all_video_frames' is invalid or empty.")
        return

    try:
        # Find first non-None frame to get dimensions
        first_valid_frame = next((f for f in all_video_frames if f is not None and isinstance(f, np.ndarray)), None)
        if first_valid_frame is None: print("  Error: No valid frames found to determine video size."); return
        height, width, layers = first_valid_frame.shape
        frame_size = (width, height)
    except Exception as e: print(f"  Error getting frame dimensions: {e}"); return

    # Ensure FPS is valid before creating VideoWriter
    if fps is None or fps <= 0:
        print(f"  Error: Invalid FPS ({fps}) for video export.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not out.isOpened(): print(f"  Error opening VideoWriter for path: {output_path}"); return

    print(f"  VideoWriter initialized (Codec: mp4v, FPS: {fps}, Size: {frame_size})")
    total_frames_written = 0
    for i, (start, end) in enumerate(replay_segments):
        # Add more robust index checking
        if not (isinstance(start, int) and isinstance(end, int) and
                0 <= start < len(all_video_frames) and 0 <= end < len(all_video_frames) and start <= end):
            print(f"    Warning: Skipping invalid segment indices type/range [{start}, {end}].")
            continue

        for frame_index in range(start, end + 1):
            try:
                frame = all_video_frames[frame_index]
                if frame is not None and frame.shape[0] == height and frame.shape[1] == width:
                     out.write(frame); total_frames_written += 1
            except IndexError: print(f"    Warning: Frame index {frame_index} out of bounds."); break
            except Exception as write_e: print(f"    Warning: Error writing frame {frame_index}: {write_e}") # Catch other write errors

    out.release()
    print(f"\nFinished exporting. Total frames written: {total_frames_written}")
    print(f"Replay video saved to: {output_path}")


#######################################
#### Multiprocessing Worker Function ####
#######################################

def process_segment_worker(args):
    """
    Worker function reads its own frames based on indices.
    Receives (video_path, params, tesseract_cmd, segment_indices).
    """
    # Unpack arguments
    video_path, params, tesseract_cmd, segment_indices = args
    start, end = segment_indices

    segment_frames = []
    cap = None # Initialize cap to None
    try:
        # print(f"Worker {os.getpid()} opening video: {video_path} for segment {start}-{end}") # Debug
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error in worker {os.getpid()}: Could not open video file {video_path}")
            return False # Indicate SC detected on error

        # Validate frame indices against video properties if possible
        # total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # if start >= total_vid_frames or end >= total_vid_frames:
        #      print(f"Warning in worker {os.getpid()}: Segment indices {start}-{end} out of bounds for video length {total_vid_frames}.")
             # Decide whether to return error or process up to end of video
             # return False # Safer to return error

        # Set the starting frame position
        # Using set might be inaccurate for some formats/codecs; reading sequentially might be more reliable but slower.
        # Let's try set first.
        set_success = cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        if not set_success:
            # print(f"Warning in worker {os.getpid()}: Failed to set video position to frame {start}. Reading sequentially.") # Reduce noise
            # If set fails, might need to read from beginning (very slow) or handle error
            # For now, let's try reading from current pos, hoping it's close enough or exact read handles it
            pass

        # Read frames for the segment
        current_frame_pos = start # Assume set worked or start from 0 if it failed badly
        frames_read_count = 0
        while current_frame_pos <= end:
             # Optimization: Check current position before reading if possible
             # actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # Reading pos can be slow
             # if actual_pos > end: break # Stop if we've somehow gone past the end

             ret, frame = cap.read()
             # Check if frame read was successful AND if we are within the desired segment range
             # (cap.read() increments position, so check pos *after* read or compare count)
             if not ret:
                 # print(f"Warning in worker {os.getpid()}: Failed to read frame at approx position {current_frame_pos} (target end: {end}).") # Reduce noise
                 break # End of video or read error

             # Append frame if we are within the target range (start <= current_frame_pos <= end)
             # This check handles cases where set(CAP_PROP_POS_FRAMES) wasn't exact
             if frames_read_count < (end - start + 1): # Read exactly the number of frames needed
                  segment_frames.append(frame)
                  frames_read_count += 1
             else:
                  break # Stop after reading the required number of frames

             current_frame_pos += 1 # Increment our logical position tracker

        # print(f"Worker {os.getpid()} read {len(segment_frames)} frames for segment {start}-{end}") # Debug

    except Exception as e:
        print(f"\n--- Exception during frame reading in Worker {os.getpid()} for segment {start}-{end} ---")
        traceback.print_exc()
        print(f"--- End Exception Traceback ---")
        return False # Indicate SC detected on error
    finally:
        if cap is not None and cap.isOpened():
            cap.release() # Ensure video capture is released

    # Check if we actually got frames
    if not segment_frames:
        print(f"Warning in worker {os.getpid()}: No frames read for segment {start}-{end}. Assuming SC present.")
        return False # Assume SC present if no frames could be read

    # Now, process the locally read frames
    try:
        is_sc_absent = detect_score_caption_absence(segment_frames, params, tesseract_cmd)
        return is_sc_absent
    except Exception as e:
        print(f"\n--- Exception during SC Detection in Worker {os.getpid()} for segment {start}-{end} ---")
        traceback.print_exc()
        print(f"--- End Exception Traceback ---")
        return False # Indicate SC detected on error


#################################
#### Replay Extraction Model ####
#################################

if __name__ == "__main__":
    multiprocessing.freeze_support()

    video_file = "test2.mp4" # INPUT: Video file path
    ground_truth_file = "ground_truth_annotations.json" # INPUT: Annotations path
    output_replay_video = "test1_replays_mp_v3.mp4" # INPUT: Output video path

    start_time = time.time()

    if not os.path.exists(video_file): print(f"FATAL: Video not found: {video_file}"); exit()
    perform_evaluation = os.path.exists(ground_truth_file)
    if not perform_evaluation: print(f"WARNING: GT file not found: {ground_truth_file}. Skipping evaluation.")

    # 1. Extract Frames (ONLY needed for non-parallel parts and final export)
    # We extract all frames for GT detection and export, but workers read their own.
    # Consider optimizing this later if memory is still an issue.
    all_frames_for_export, video_fps = extract_frames(video_file) # Renamed to clarify purpose
    if not all_frames_for_export: exit()
    total_video_frames = len(all_frames_for_export) # Use length of extracted frames as reliable count
    extract_end_time = time.time()
    print(f"Frame Extraction (Main) Time: {extract_end_time - start_time:.2f} seconds")

    # 2. Detect Gradual Transitions (Uses extracted frames)
    gt_segments = detect_gradual_transitions(all_frames_for_export, PARAMS)
    gt_detect_end_time = time.time()
    print(f"GT Detection Time: {gt_detect_end_time - extract_end_time:.2f} seconds")

    # 3. Identify Candidate Replay Segments
    candidate_segments = identify_candidate_replay_segments(gt_segments, PARAMS)
    candidate_id_end_time = time.time()
    print(f"Candidate Identification Time: {candidate_id_end_time - gt_detect_end_time:.2f} seconds")

    # 4. Detect Score Caption Absence (Parallelized - Workers Read Frames)
    final_replay_segments = []
    print("\nDetecting Score Caption (SC) Absence (using multiprocessing v3 - workers read frames)...")
    sc_detection_start_time = time.time()

    if not candidate_segments:
        print("  No candidate segments found. Skipping SC detection.")
    else:
        # Filter candidate_segments first (using total_video_frames from main extraction)
        valid_candidate_segments = [(s, e) for s, e in candidate_segments
                                    if (0 <= s < total_video_frames and 0 <= e < total_video_frames and s <= e)]
        if len(valid_candidate_segments) != len(candidate_segments):
            print(f"  Warning: Filtered out {len(candidate_segments) - len(valid_candidate_segments)} invalid candidate segments.")

        if not valid_candidate_segments:
             print("  No valid candidate segments to process.")
        else:
            # Prepare task arguments: video path, params, cmd, indices tuple
            task_args = [(video_file, PARAMS, TESSERACT_CMD, indices) for indices in valid_candidate_segments]

            num_workers = multiprocessing.cpu_count()
            print(f"  Initializing Pool with {num_workers} workers...")

            try:
                with multiprocessing.Pool(processes=num_workers) as pool:
                    print(f"  Mapping {len(task_args)} tasks to worker pool...")
                    results = pool.map(process_segment_worker, task_args)

                print("  Worker pool finished processing.")

                if len(results) != len(valid_candidate_segments):
                     print("Error: Mismatch between results and processed segments!")
                else:
                    for i, is_sc_absent in enumerate(results):
                        start, end = valid_candidate_segments[i]
                        if is_sc_absent:
                            final_replay_segments.append((start, end))

            except Exception as e:
                print(f"\nError during multiprocessing SC detection (v3): {e}")
                traceback.print_exc()
                print("Multiprocessing failed.")

    sc_detection_end_time = time.time()
    print(f"\nFinished SC detection. Identified {len(final_replay_segments)} final replay segments.")
    print(f"SC Detection Time (Parallel v3): {sc_detection_end_time - sc_detection_start_time:.2f} seconds")
    if final_replay_segments: final_replay_segments.sort(key=lambda x: x[0])
    else: print("  No final replay segments identified.")

    # 5. Evaluation (Uses total_video_frames)
    eval_start_time = time.time()
    if perform_evaluation:
        evaluation_metrics = evaluate_replay_detection(final_replay_segments, ground_truth_file, total_video_frames)
        if evaluation_metrics: print("\nEvaluation completed successfully.")
    else: print(f"\nSkipping evaluation step.")
    eval_end_time = time.time()
    if perform_evaluation: print(f"Evaluation Time: {eval_end_time - eval_start_time:.2f} seconds")

    # 6. Export Video (Uses all_frames_for_export extracted initially)
    export_start_time = time.time()
    if final_replay_segments and all_frames_for_export and video_fps is not None and video_fps > 0:
        export_replays_to_video(
            replay_segments=final_replay_segments, all_video_frames=all_frames_for_export,
            fps=video_fps, output_path=output_replay_video
        )
    else:
        reason = "No replays detected" if not final_replay_segments else "Frame data missing" if not all_frames_for_export else f"Invalid FPS ({video_fps})"
        print(f"\nSkipping replay export: {reason}.")
    export_end_time = time.time()
    if final_replay_segments: print(f"Export Time: {export_end_time - export_start_time:.2f} seconds")

    total_end_time = time.time()
    print(f"\nProcessing finished. Total execution time: {total_end_time - start_time:.2f} seconds")