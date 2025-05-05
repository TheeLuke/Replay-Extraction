import cv2
import cv2.version
import numpy as np
import pytesseract
import json
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import os
import time
import multiprocessing # For parallel processing
import traceback # For detailed error logging

#################################
#### Configuration & Globals ####
#################################

# --- User Configuration ---
# !! IMPORTANT !! Update this path to your Tesseract installation
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Define Input/Output Files (Update these paths as needed)
VIDEO_FILE_IN = "test2.mp4"
GROUND_TRUTH_FILE_IN = "ground_truth_annotations.json" # Optional, for evaluation
OUTPUT_REPLAY_VIDEO_OUT = "test1_replays_streamed.mp4"
# -------------------------

# --- Parameters ---
# Group parameters into a dictionary for easier passing
PARAMS = {
    # SC Preprocessing & Detection
    'ALPHA': 3,         # Structuring element size for top-hat filter
    'P_BINARIZE': 2.5,  # Constant for binarization threshold
    'L_AVG': 5,         # Window length for temporal running average (SC detection)
    'T1_OCR_CONF': 0.75, # Confidence threshold (0-1.0) for OCR character recognition
    'T2_OCR_CHARS': 3,  # Minimum number of confident characters for SC presence
    'SC_ROI_Y_START': 0,    # ROI Y start (original video coordinates)
    'SC_ROI_Y_END': 1080,   # ROI Y end
    'SC_ROI_X_START': 0,    # ROI X start
    'SC_ROI_X_END': 1920,   # ROI X end

    # GT Detection
    'N_GT_MIN_LEN': 7,  # Minimum length (frames) for a gradual transition
    'T_L_HIST_DIFF': 0.025, # Lower threshold for successive histogram difference
    'T_U_HIST_ACCUM': 3.5,  # Upper threshold for accumulative histogram difference
    'HIST_COMP_METHOD': cv2.HISTCMP_CHISQR, # Options: HISTCMP_CORREL, HISTCMP_BHATTACHARYYA, HISTCMP_CHISQR

    # Replay Segment Identification
    'N_RL_MIN_DUR': 235,# Minimum duration (frames) for a replay segment
    'N_RU_MAX_DUR': 800,# Maximum duration (frames) for a replay segment
}
# ------------------


#################################
##### Core Functions        #####
#################################

def get_video_metadata(video_path):
    """Gets metadata (fps, width, height, frame count) from a video file."""
    print(f"Getting metadata from: {video_path}")
    if not os.path.exists(video_path) or not os.path.isfile(video_path):
        print(f"Error: Video file not found or is not a file: {video_path}")
        return None, 0, 0, 0 # Return None for fps to indicate error

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None, 0, 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Basic validation
        if fps <= 0 or width <= 0 or height <= 0:
             print(f"Warning: Invalid metadata read from video (fps={fps}, w={width}, h={height}). Check video file.")
             fps = None if fps <=0 else fps # Treat invalid FPS as error indicator

        print(f"Video Metadata: FPS={fps}, Size=({width}x{height}), Total Frames={frame_count}")
        return fps, width, height, frame_count

    except Exception as e:
        print(f"Error getting video metadata: {e}")
        traceback.print_exc()
        return None, 0, 0, 0
    finally:
        if cap is not None and cap.isOpened():
            cap.release()

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

def detect_gradual_transitions_streaming(video_path, params, total_frames):
    """Detects Gradual Transitions by reading frames directly from the video."""
    print("Detecting Gradual Transitions (GTs) - Streaming Mode...")
    print(f"  Using Hist Comparison: {params['HIST_COMP_METHOD']}, T_L: {params['T_L_HIST_DIFF']}, T_U: {params['T_U_HIST_ACCUM']}")
    gt_segments = []

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path} for GT detection.")
            return gt_segments

        hist_size = [256] # Luminance histogram
        hist_range = [0, 256]

        # Read the first frame
        ret, prev_frame = cap.read()
        if not ret or prev_frame is None:
            print("Error: Could not read the first frame for GT detection.")
            return gt_segments

        # Initialize with the first frame's histogram
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_hist = cv2.calcHist([prev_gray], [0], None, hist_size, hist_range)
        cv2.normalize(prev_hist, prev_hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

        potential_gt_start = -1
        accum_hist_diff = 0
        processed_count = 1 # Frame count starts at 1 (we read frame 0)

        # Loop from the second frame onwards
        while True:
            ret, current_frame = cap.read()
            if not ret or current_frame is None:
                # End of video or read error
                # Check if a potential GT was ending at the last valid frame (processed_count - 1)
                if potential_gt_start != -1:
                     if accum_hist_diff > params['T_U_HIST_ACCUM']:
                         gt_end = processed_count - 1 # Ends at the last successfully read frame index
                         if (gt_end - potential_gt_start + 1) >= params['N_GT_MIN_LEN']:
                             gt_segments.append((potential_gt_start, gt_end))
                             print(f"    --> Detected GT at video end: Frames {potential_gt_start} to {gt_end}")
                break # Exit loop

            frame_index = processed_count # Current frame index (0-based)

            try:
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                current_hist = cv2.calcHist([current_gray], [0], None, hist_size, hist_range)
                cv2.normalize(current_hist, current_hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

                successive_diff = cv2.compareHist(prev_hist, current_hist, params['HIST_COMP_METHOD'])

                is_potential_start = False
                if params['HIST_COMP_METHOD'] == cv2.HISTCMP_CORREL:
                    is_potential_start = successive_diff < params['T_L_HIST_DIFF']
                else:
                     is_potential_start = successive_diff > params['T_L_HIST_DIFF']

                if is_potential_start:
                    if potential_gt_start == -1:
                        potential_gt_start = frame_index - 1 # GT starts at the previous frame index
                    accum_hist_diff += successive_diff
                else:
                    if potential_gt_start != -1:
                        # Potential GT ends here (frame_index - 1 was the last frame index in it)
                        if accum_hist_diff > params['T_U_HIST_ACCUM']:
                            gt_end = frame_index - 1
                            if (gt_end - potential_gt_start + 1) >= params['N_GT_MIN_LEN']:
                                gt_segments.append((potential_gt_start, gt_end))
                                print(f"    --> Detected GT: Frames {potential_gt_start} to {gt_end} (Duration: {gt_end - potential_gt_start + 1}, AccumDiff: {accum_hist_diff:.4f})")
                        # Reset potential GT state
                        potential_gt_start = -1
                        accum_hist_diff = 0

                # Current histogram becomes the previous one for the next iteration
                prev_hist = current_hist

            except cv2.error as e:
                # print(f"  Warning: OpenCV error processing frame {frame_index} for GT: {e}. Skipping comparison.") # Reduce noise
                if potential_gt_start != -1: potential_gt_start = -1; accum_hist_diff = 0
                # Don't update prev_hist if current frame failed, use last known good one
                pass

            processed_count += 1
            if processed_count % 1000 == 0:
                 progress = f"({processed_count}/{total_frames})" if total_frames > 0 else ""
                 print(f"  Processed {processed_count} frames for GT detection {progress}...")

    except Exception as e:
        print(f"An error occurred during streaming GT detection: {e}")
        traceback.print_exc()
    finally:
        if cap is not None and cap.isOpened():
            cap.release()

    print(f"Finished streaming GT detection. Found {len(gt_segments)} segments.")
    return gt_segments

def identify_candidate_replay_segments(gt_segments, params):
    """Identifies candidate replay segments based on detected GTs."""
    print("Identifying Candidate Replay Segments (RSs)...")
    candidate_rs = []
    if len(gt_segments) < 2:
        print("  Not enough GTs found (< 2) to identify replay segments.")
        return candidate_rs

    # Sort GTs just in case they are out of order (streaming might produce them slightly out of order if errors occur)
    gt_segments.sort(key=lambda x: x[0])

    for i in range(len(gt_segments) - 1):
        gt1_start, gt1_end = gt_segments[i]
        gt2_start, gt2_end = gt_segments[i+1]

        # Define the segment *between* the end of GT1 and start of GT2
        actual_replay_content_start = gt1_end + 1
        actual_replay_content_end = gt2_start - 1

        # Check if segment exists (gt2 must start after gt1 ends)
        if actual_replay_content_start <= actual_replay_content_end:
             actual_replay_duration = actual_replay_content_end - actual_replay_content_start + 1
             # Check if the duration of the content *between* GTs fits replay limits
             if params['N_RL_MIN_DUR'] <= actual_replay_duration <= params['N_RU_MAX_DUR']:
                  candidate_rs.append((actual_replay_content_start, actual_replay_content_end))
                  # print(f"    -> Added Candidate RS: Frames {actual_replay_content_start}-{actual_replay_content_end}, Duration: {actual_replay_duration}") # Reduce noise
             # else: print(f"    -> Discarded (Duration {actual_replay_duration} outside limits)") # Reduce noise
        # else: print(f"  Skipping adjacent/overlapping GTs {i} & {i+1}") # Reduce noise

    print(f"Finished candidate RS identification. Found {len(candidate_rs)} candidates.")
    return candidate_rs

def detect_score_caption_absence(frames_segment, params, tesseract_cmd):
    """ Detects the absence of a Score Caption (SC) within a given segment of frames. """
    try:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    except Exception as e:
        print(f"Warning: Could not set tesseract command path in worker: {e}")

    if not frames_segment: return True # No frames means no SC

    # Basic check on the input structure
    if not isinstance(frames_segment, list) or not frames_segment:
         print("Warning: detect_score_caption_absence received invalid frames_segment.")
         return False # Assume SC present

    if not isinstance(frames_segment[0], np.ndarray):
         # Check if it's a list of Nones perhaps?
         if all(f is None for f in frames_segment): return True # Treat segment of Nones as SC absent
         else:
            print("Warning: detect_score_caption_absence received invalid frames_segment[0] type.")
            return False # Assume SC present

    # --- Preprocessing ---
    processed_frames = []
    alpha_p = params.get('ALPHA', 3)
    for f in frames_segment:
         processed = preprocess_frame_for_sc(f, alpha_p) # Handles None frames internally
         processed_frames.append(processed)

    # --- Temporal Running Averaging ---
    avg_images = []
    num_frames = len(processed_frames)
    l_avg = params.get('L_AVG', 5)
    for i in range(num_frames):
        if processed_frames[i] is None: # If center frame is invalid, result is invalid
            avg_images.append(None)
            continue
        # Define window around i
        start = max(0, i - l_avg // 2)
        end = min(num_frames, i + l_avg // 2 + 1)
        window = processed_frames[start:end]
        # Filter only valid frames within the window
        valid_window = [img for img in window if img is not None and isinstance(img, np.ndarray) and img.size > 0]

        if valid_window:
            try: # Ensure shape compatibility before averaging
                h, w = valid_window[0].shape
                consistent_window = [img for img in valid_window if img.shape == (h, w)]
                if consistent_window: # Average if we have valid, consistent frames
                     avg_img = np.mean(np.array(consistent_window, dtype=np.float32), axis=0).astype(np.uint8)
                     avg_images.append(avg_img)
                else: # Fallback if shapes mismatched (should be rare with video)
                    avg_images.append(processed_frames[i]) # Use center frame if window inconsistent
            except Exception as shape_e:
                # print(f"Warning: Error during averaging window {i}: {shape_e}") # Reduce noise
                avg_images.append(processed_frames[i]) # Fallback to center frame on error
        else: # Fallback if no valid frames in window
             avg_images.append(processed_frames[i]) # Use center frame if window empty/invalid

    # --- OCR ---
    sc_detected_in_segment = False
    # Get parameters with defaults
    t1_conf_scaled = params.get('T1_OCR_CONF', 0.75) * 100
    t2_chars = params.get('T2_OCR_CHARS', 3)
    roi_y_start = params.get('SC_ROI_Y_START', 0)
    roi_y_end = params.get('SC_ROI_Y_END', 1080)
    roi_x_start = params.get('SC_ROI_X_START', 0)
    roi_x_end = params.get('SC_ROI_X_END', 1920)
    p_bin = params.get('P_BINARIZE', 2.5)

    for i, avg_img in enumerate(avg_images):
        if avg_img is None: continue # Skip frames that failed averaging

        # Binarization
        try:
            mean, std_dev = cv2.meanStdDev(avg_img)
            mean_val = mean[0][0] # Extract scalar value
            std_dev_val = std_dev[0][0] if std_dev[0][0] > 1e-6 else 1.0 # Avoid zero std dev
            lower_bound = mean_val - p_bin * std_dev_val
            upper_bound = mean_val + p_bin * std_dev_val
            # Perform thresholding (ensure avg_img is uint8)
            avg_img_u8 = avg_img.astype(np.uint8)
            _, binary_img_inv = cv2.threshold(avg_img_u8, upper_bound, 255, cv2.THRESH_BINARY_INV)
            _, binary_img_lower = cv2.threshold(avg_img_u8, lower_bound, 255, cv2.THRESH_BINARY)
            binary_img = cv2.bitwise_and(binary_img_inv, binary_img_lower)
        except cv2.error as e:
            # print(f"Warning: Binarization error frame {i}: {e}") # Reduce noise
            continue # Skip frame on error
        except Exception as e_bin: # Catch other potential errors (e.g., type issues)
            # print(f"Warning: Non-OpenCV Binarization error frame {i}: {e_bin}") # Reduce noise
            continue

        # ROI Extraction (from binarized, downsampled image)
        roi_h, roi_w = binary_img.shape
        # Adjust ROI based on downsampling (factor of 2)
        y_start_ds = max(0, roi_y_start // 2)
        y_end_ds = min(roi_h, roi_y_end // 2)
        x_start_ds = max(0, roi_x_start // 2)
        x_end_ds = min(roi_w, roi_x_end // 2)

        if y_start_ds >= y_end_ds or x_start_ds >= x_end_ds: continue # Skip if ROI is invalid
        roi_img = binary_img[y_start_ds:y_end_ds, x_start_ds:x_end_ds]
        if roi_img.size == 0: continue # Skip if ROI is empty

        # Perform OCR
        try:
            ocr_data = pytesseract.image_to_data(roi_img, lang='eng', config='--psm 7', output_type=pytesseract.Output.DICT)
            num_confident_chars = 0
            for j, conf_str in enumerate(ocr_data['conf']):
                try: conf = int(float(conf_str))
                except ValueError: conf = -1 # Handle non-numeric confidence
                if conf >= t1_conf_scaled: # Check confidence threshold
                    text = ocr_data['text'][j].strip()
                    if text: # Count non-empty text
                        num_confident_chars += len(text)

            # Check if enough confident characters were found
            if num_confident_chars >= t2_chars:
                 sc_detected_in_segment = True
                 break # SC found, exit loop for this segment

        except pytesseract.TesseractNotFoundError:
             # This is critical, print prominently
             print(f"FATAL ERROR (worker {os.getpid()}): Tesseract executable not found or not in PATH. Check TESSERACT_CMD: {tesseract_cmd}")
             # Should ideally stop everything or signal main process, but difficult from worker.
             # Returning False assumes SC might be present.
             return False
        except Exception as e:
            # Log other OCR errors but treat as potential SC presence to be safe
            # print(f"Warning: OCR error frame {i} worker {os.getpid()}: {e}") # Reduce noise
            # traceback.print_exc() # Uncomment for more detail if needed
            sc_detected_in_segment = True
            break # Stop processing this segment on unexpected OCR error

    # Return True if SC Absent (loop completed without detection), False if SC Present (loop broken)
    return not sc_detected_in_segment

def evaluate_replay_detection(detected_replays, ground_truth_json_path, total_frames):
    """Evaluates the performance of the replay detection against ground truth."""
    print(f"\nEvaluating performance against: {ground_truth_json_path}")
    if not os.path.exists(ground_truth_json_path):
        print(f"Error: Ground truth file not found at {ground_truth_json_path}")
        return None
    if total_frames <= 0:
        print(f"Error: Cannot evaluate with invalid total_frames: {total_frames}")
        return None

    try:
        with open(ground_truth_json_path, 'r') as f:
            ground_truth = json.load(f)
        # Ensure 'replays' key exists and is a list
        gt_replays = ground_truth.get("replays", [])
        if not isinstance(gt_replays, list):
             print("Error: Ground truth 'replays' field is not a list.")
             return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {ground_truth_json_path}")
        return None
    except Exception as e:
        print(f"Error reading ground truth file: {e}")
        return None

    # Create frame-level labels
    y_true = np.zeros(total_frames, dtype=int) # 0 = Live, 1 = Replay
    print("Processing Ground Truth Segments:")
    for segment in gt_replays:
        try:
            # Handle potential variations in GT format (list or tuple)
            start, end = int(segment[0]), int(segment[1])
            # Validate indices strictly
            if 0 <= start < total_frames and 0 <= end < total_frames and start <= end:
                y_true[start : end + 1] = 1
            else:
                 print(f"  Warning: Ground truth segment [{start}, {end}] out of bounds or invalid (Total frames: {total_frames}). Skipping.")
        except (ValueError, TypeError, IndexError) as e:
             print(f"  Warning: Skipping invalid ground truth segment format '{segment}': {e}")


    y_pred = np.zeros(total_frames, dtype=int)
    print("Processing Detected Segments:")
    for segment in detected_replays:
         try:
             start, end = int(segment[0]), int(segment[1])
             # Validate indices strictly
             if 0 <= start < total_frames and 0 <= end < total_frames and start <= end:
                 y_pred[start : end + 1] = 1
             else:
                 print(f"  Warning: Detected segment [{start}, {end}] out of bounds or invalid (Total frames: {total_frames}). Skipping.")
         except (ValueError, TypeError, IndexError) as e:
              print(f"  Warning: Skipping invalid detected segment format '{segment}': {e}")


    # Calculate confusion matrix and metrics
    try:
        # Ensure labels cover all possible values (0 and 1) for consistency
        labels_present = np.unique(np.concatenate((y_true, y_pred)))
        cm_labels = sorted(list(set(labels_present) | {0, 1})) # Ensure 0 and 1 are included

        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

        if len(cm_labels) == 2: # Standard case [0, 1]
             tn, fp, fn, tp = cm.ravel()
        elif len(cm_labels) == 1: # Only one class present in both true/pred
             if cm_labels[0] == 0: tn, fp, fn, tp = cm[0,0], 0, 0, 0 # All TN
             else: tn, fp, fn, tp = 0, 0, 0, cm[0,0] # All TP
        else: # Should not happen if cm_labels includes 0 and 1
            tn, fp, fn, tp = 0, 0, 0, 0

        precision = precision_score(y_true, y_pred, labels=cm_labels, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=cm_labels, pos_label=1, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        # Ensure total_frames is non-zero before division
        error_rate = (fp + fn) / total_frames if total_frames > 0 else 0

        results = {
            "true_positives": int(tp), "true_negatives": int(tn),
            "false_positives": int(fp), "false_negatives": int(fn),
            "precision": precision, "recall": recall, "accuracy": accuracy,
            "error_rate": error_rate, "total_frames": total_frames,
            "gt_replay_frames": int(np.sum(y_true)), "detected_replay_frames": int(np.sum(y_pred))
        }
    except Exception as e:
        print(f"Error during metrics calculation: {e}")
        traceback.print_exc()
        return None

    # Print results
    print("\n--- Evaluation Results ---")
    print(f"  Total Frames: {results['total_frames']}")
    print(f"  Ground Truth Replay Frames: {results['gt_replay_frames']}")
    print(f"  Detected Replay Frames: {results['detected_replay_frames']}")
    print(f"  TP: {results['true_positives']}, TN: {results['true_negatives']}, FP: {results['false_positives']}, FN: {results['false_negatives']}")
    print(f"  Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, Accuracy: {results['accuracy']:.4f}, Error Rate:{results['error_rate']:.4f}")
    print("-------------------------\n")

    return results

def export_replays_to_video_streaming(replay_segments, video_path, fps, width, height, output_path):
    """
    Combines frames from specified replay segments into a single video file
    by reading frames directly from the video file.
    """
    print(f"\nExporting {len(replay_segments)} replay segments to: {output_path} (Streaming Read)")
    if not replay_segments: print("  No replay segments to export."); return
    if not os.path.exists(video_path): print(f"  Error: Video file not found: {video_path}"); return
    if not (fps is not None and fps > 0 and width is not None and width > 0 and height is not None and height > 0):
        print(f"  Error: Invalid metadata for export (fps={fps}, w={width}, h={height})"); return

    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not out.isOpened():
        print(f"  Error: Could not open VideoWriter for path: {output_path}")
        print("  Check if the directory exists and you have permissions.")
        return

    cap = None
    total_frames_written = 0
    print(f"  VideoWriter initialized (Codec: mp4v, FPS: {fps}, Size: {frame_size})")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Error opening video file {video_path} for export."); return

        # Sort segments to potentially optimize seeking (though effectiveness varies)
        replay_segments.sort(key=lambda x: x[0])

        for i, segment in enumerate(replay_segments):
             try:
                start, end = int(segment[0]), int(segment[1])
                if not (0 <= start <= end): # Basic validation
                     print(f"    Warning: Skipping invalid segment indices [{start}, {end}] in segment {i+1}.")
                     continue
             except (ValueError, TypeError, IndexError) as e:
                  print(f"    Warning: Skipping invalid segment format '{segment}' in segment {i+1}: {e}")
                  continue

             print(f"  Writing segment {i+1}/{len(replay_segments)}: Frames {start} to {end}")

             # Attempt to seek to the start frame
             set_success = cap.set(cv2.CAP_PROP_POS_FRAMES, start)
             if not set_success:
                 print(f"    Warning: Failed to seek accurately to frame {start} for segment {i+1}. Read may be inaccurate or slow.")
                 # If seek fails, subsequent reads might start from wrong place or read sequentially from beginning.

             # Read and write frames for this specific segment
             frames_written_this_segment = 0
             # Loop from start index up to and including end index
             for frame_index in range(start, end + 1):
                 # Check current position before reading? Can be slow, might not be reliable.
                 # actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                 # if actual_pos != frame_index: print(f"Pos mismatch: expected {frame_index}, got {actual_pos}")

                 ret, frame = cap.read()
                 if not ret or frame is None:
                     print(f"    Warning: Failed to read frame {frame_index} for segment {i+1}. Stopping segment export.")
                     break # Stop writing this segment if a frame fails

                 # Ensure frame dimensions match expected output size
                 if frame.shape[0] == height and frame.shape[1] == width:
                     out.write(frame)
                     total_frames_written += 1
                     frames_written_this_segment += 1
                 else:
                     print(f"    Warning: Skipping frame {frame_index} due to mismatched dimensions (read {frame.shape[1]}x{frame.shape[0]}, expected {width}x{height})")

                 # Add a small sleep if I/O seems overwhelmed? Usually not necessary.
                 # time.sleep(0.001)

             print(f"    -> Wrote {frames_written_this_segment} frames for segment {i+1}.")

    except Exception as e:
        print(f"An error occurred during streaming export: {e}")
        traceback.print_exc()
    finally:
        # Ensure resources are released
        if cap is not None and cap.isOpened():
            cap.release()
            # print("  Video capture released.") # Debug
        if out is not None and out.isOpened():
            out.release()
            # print("  Video writer released.") # Debug

    print(f"\nFinished streaming export. Total frames written: {total_frames_written}")
    if total_frames_written > 0:
        print(f"Replay video saved to: {output_path}")
    else:
        # If no frames written, delete the empty file?
        if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
            try: os.remove(output_path); print(f"Removed empty output file: {output_path}")
            except OSError as e_rem: print(f"Warning: Could not remove empty output file: {e_rem}")
        else:
            print(f"No frames were written to {output_path}. Output file may not exist or is not empty.")


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
            print(f"Error in worker {os.getpid()}: Could not open video file {video_path} for segment {start}-{end}")
            return False # Indicate SC detected on error

        # Attempt to set the starting frame position
        # Note: Seeking accuracy depends heavily on video format and OpenCV backend.
        set_success = cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        if not set_success:
            # If seek fails, reading sequentially from start might still work but can be slow if previous reads left off far away.
            # Reading frame-by-frame using cap.read() generally moves to the next frame correctly.
            # print(f"Warning in worker {os.getpid()}: Failed to set video position to frame {start} for {start}-{end}. Reading sequentially.") # Reduce noise
            pass # Continue reading, hoping cap.read() handles it

        # Read frames for the segment
        frames_to_read = (end - start + 1)
        frames_read_count = 0
        current_frame_pos_estimate = start # Track where we think we are

        while frames_read_count < frames_to_read:
            ret, frame = cap.read()
            if not ret:
                # print(f"Warning in worker {os.getpid()}: Failed to read frame at approx position {current_frame_pos_estimate} (target end: {end}) during segment {start}-{end}. Read {frames_read_count}/{frames_to_read} frames.") # Reduce noise
                break # End of video or read error before segment finished

            # We assume the frame read corresponds to the desired sequence after the seek.
            # More robust checks could involve reading frame number if available/reliable, but adds overhead.
            segment_frames.append(frame)
            frames_read_count += 1
            current_frame_pos_estimate += 1

        # Optional: Check if expected number of frames were read
        if frames_read_count != frames_to_read:
             print(f"Warning in worker {os.getpid()}: Read {frames_read_count} frames for segment {start}-{end}, expected {frames_to_read}.")

        # print(f"Worker {os.getpid()} read {len(segment_frames)} frames for segment {start}-{end}") # Debug

    except Exception as e:
        print(f"\n--- Exception during frame reading in Worker {os.getpid()} for segment {start}-{end} ---")
        traceback.print_exc()
        print(f"--- End Exception Traceback ---")
        return False # Indicate SC detected on error
    finally:
        if cap is not None and cap.isOpened():
            cap.release() # Ensure video capture is released

    # Check if we actually got any frames
    if not segment_frames:
        # print(f"Warning in worker {os.getpid()}: No frames read for segment {start}-{end}. Assuming SC present.") # Reduce noise
        return False # Assume SC present if no frames could be read

    # Now, process the locally read frames using the SC detection logic
    try:
        # print(f"Worker {os.getpid()} starting SC detection for segment {start}-{end}") # Debug
        is_sc_absent = detect_score_caption_absence(segment_frames, params, tesseract_cmd)
        # print(f"Worker {os.getpid()} finished SC detection for segment {start}-{end}, result={is_sc_absent}") # Debug
        return is_sc_absent
    except Exception as e:
        print(f"\n--- Exception during SC Detection in Worker {os.getpid()} for segment {start}-{end} ---")
        traceback.print_exc()
        print(f"--- End Exception Traceback ---")
        return False # Indicate SC detected on processing error


#################################
#### Main Execution Block    ####
#################################

if __name__ == "__main__":
    # --- Important for multiprocessing, especially on Windows ---
    multiprocessing.freeze_support()

    # Use global constants for file paths defined at the top
    video_file = VIDEO_FILE_IN
    ground_truth_file = GROUND_TRUTH_FILE_IN
    output_replay_video = OUTPUT_REPLAY_VIDEO_OUT

    # --- Record start time ---
    main_start_time = time.time()

    # --- Basic File Checks ---
    if not os.path.exists(video_file):
        print(f"FATAL: Input video file not found at {video_file}")
        exit()
    perform_evaluation = os.path.exists(ground_truth_file)
    if not perform_evaluation:
        print(f"WARNING: Ground truth file not found at {ground_truth_file}. Evaluation will be skipped.")
    # -------------------------

    # 1. Get Video Metadata (No frame loading here)
    video_fps, video_width, video_height, total_video_frames = get_video_metadata(video_file)
    if video_fps is None or total_video_frames <= 0:
         print("FATAL: Failed to retrieve valid video metadata. Exiting.")
         exit()
    metadata_end_time = time.time()
    print(f"Metadata Retrieval Time: {metadata_end_time - main_start_time:.2f} seconds")

    # 2. Detect Gradual Transitions (Streaming)
    gt_segments = detect_gradual_transitions_streaming(video_file, PARAMS, total_video_frames)
    gt_detect_end_time = time.time()
    print(f"GT Detection Time (Streaming): {gt_detect_end_time - metadata_end_time:.2f} seconds")

    # 3. Identify Candidate Replay Segments (Uses GT results)
    candidate_segments = identify_candidate_replay_segments(gt_segments, PARAMS)
    candidate_id_end_time = time.time()
    print(f"Candidate Identification Time: {candidate_id_end_time - gt_detect_end_time:.2f} seconds")

    # 4. Detect Score Caption Absence (Parallelized - Workers Read Frames)
    final_replay_segments = []
    print("\nDetecting Score Caption (SC) Absence (using multiprocessing - workers read frames)...")
    sc_detection_start_time = time.time()
    if not candidate_segments:
        print("  No candidate segments found. Skipping SC detection.")
    else:
        # Filter candidate_segments for validity *before* creating tasks
        valid_candidate_segments = []
        for seg in candidate_segments:
             try:
                 s, e = int(seg[0]), int(seg[1])
                 if 0 <= s < total_video_frames and 0 <= e < total_video_frames and s <= e:
                     valid_candidate_segments.append((s, e))
                 else:
                      print(f"  Warning: Filtering out invalid candidate segment indices [{s}, {e}] (total frames: {total_video_frames}).")
             except (ValueError, TypeError, IndexError):
                  print(f"  Warning: Filtering out invalid candidate segment format: {seg}")

        if not valid_candidate_segments:
            print("  No valid candidate segments remain after filtering.")
        else:
            # Prepare task arguments: (video_path, params_dict, tesseract_cmd_path, indices_tuple)
            task_args = [(video_file, PARAMS, TESSERACT_CMD, indices) for indices in valid_candidate_segments]

            num_workers = multiprocessing.cpu_count() # Use all available cores
            print(f"  Initializing Pool with {num_workers} workers...")

            results = [] # Initialize results list
            try:
                # Use context manager for the pool
                with multiprocessing.Pool(processes=num_workers) as pool:
                    print(f"  Mapping {len(task_args)} SC detection tasks to worker pool...")
                    # `map` blocks until all results are ready
                    results = pool.map(process_segment_worker, task_args)

                print("  Worker pool finished SC processing.")

            except Exception as e:
                print(f"\n--- Error during multiprocessing SC detection phase ---")
                traceback.print_exc()
                print("--- Multiprocessing failed. Results may be incomplete. ---")

            # Process results (even if pool failed, process what we got)
            if len(results) != len(valid_candidate_segments):
                 print(f"Error/Warning: Number of results ({len(results)}) does not match number of processed segments ({len(valid_candidate_segments)})!")
                 # Attempt to process based on results count, but matching might be wrong
                 # For safety, maybe clear final_replay_segments if counts mismatch?
                 # final_replay_segments = [] # Uncomment to discard partial results on error
            else:
                print("  Processing results from workers...")
                for i, is_sc_absent in enumerate(results):
                    if is_sc_absent: # is_sc_absent is True if SC was absent (i.e., it's a replay)
                        final_replay_segments.append(valid_candidate_segments[i]) # Add the (start, end) tuple

    # --- SC Detection Timing ---
    sc_detection_end_time = time.time()
    print(f"\nFinished SC detection phase. Identified {len(final_replay_segments)} final replay segments.")
    print(f"SC Detection Time (Parallel): {sc_detection_end_time - sc_detection_start_time:.2f} seconds")
    if final_replay_segments:
        final_replay_segments.sort(key=lambda x: x[0]) # Sort results by start frame
        # print("  Final Replay Segments:") # Optional: Print segments
        # for s, e in final_replay_segments: print(f"    - Frames {s} to {e}")
    else:
        print("  No final replay segments identified.")
    # ---------------------------


    # 5. Evaluation (Uses total_video_frames from metadata)
    eval_start_time = time.time()
    if perform_evaluation:
        print("\nStarting Evaluation...")
        evaluation_metrics = evaluate_replay_detection(final_replay_segments, ground_truth_file, total_video_frames)
        if evaluation_metrics: print("\nEvaluation completed successfully.")
        else: print("\nEvaluation failed or produced no metrics.")
    else:
        print(f"\nSkipping evaluation step (Ground truth file not found or specified).")
    eval_end_time = time.time()
    if perform_evaluation: print(f"Evaluation Time: {eval_end_time - eval_start_time:.2f} seconds")
    # ---------------------------

    # 6. Export Video (Streaming Read - uses metadata and video path)
    export_start_time = time.time()
    if final_replay_segments and video_fps is not None and video_fps > 0:
        print("\nStarting Replay Export...")
        export_replays_to_video_streaming(
            replay_segments=final_replay_segments,
            video_path=video_file,
            fps=video_fps,
            width=video_width,
            height=video_height,
            output_path=output_replay_video
        )
    else:
        reason = "No replays detected" if not final_replay_segments else f"Invalid metadata for export (fps={video_fps})"
        print(f"\nSkipping replay export: {reason}.")
    export_end_time = time.time()
    # Only print export time if export was attempted
    if final_replay_segments and video_fps is not None and video_fps > 0:
        print(f"Export Time (Streaming): {export_end_time - export_start_time:.2f} seconds")
    # ---------------------------

    # --- Total Execution Time ---
    main_end_time = time.time()
    print(f"\nProcessing finished. Total execution time: {main_end_time - main_start_time:.2f} seconds")
    # ----------------------------