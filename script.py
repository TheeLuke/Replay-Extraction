import cv2
import numpy as np
import pytesseract
import json
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import os

##### Configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#### Parameters
ALPHA = 3         # Structuring element size for top-hat filter
P_BINARIZE = 2.5  # Constant for binarization threshold
L_AVG = 5         # Window length for temporal running average
T1_OCR_CONF = 0.65 # Confidence threshold (0-1.0) for OCR character recognition
T2_OCR_CHARS = 15  # Minimum number of recognized characters for SC presence
N_GT_MIN_LEN = 7 # Minimum length (frames) for a gradual transition
N_RL_MIN_DUR = 200 # Minimum duration (frames) for a replay segment 
N_RU_MAX_DUR = 800 # Maximum duration (frames) for a replay segment 

# Thresholds for GT detection
T_L_HIST_DIFF = 0.025 # Lower threshold for successive histogram difference
T_U_HIST_ACCUM = 5 # Upper threshold for accumulative histogram difference

# Histogram Comparison Method
HIST_COMP_METHOD = cv2.HISTCMP_CHISQR # Options: HISTCMP_CORREL, HISTCMP_BHATTACHARYYA, HISTCMP_CHISQR

# Score Caption (SC) Region of Interest (ROI)
SC_ROI_Y_START = 0 
SC_ROI_Y_END = 1080 
SC_ROI_X_START = 0 
SC_ROI_X_END = 1920 


#### 1. Frame Extraction
def extract_frames(video_path):
    """Extracts frames from a video file."""
    print(f"Extracting frames from: {video_path}")
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"  Extracted {frame_count} frames...")

    cap.release()
    print(f"Finished extraction. Total frames: {len(frames)}, FPS: {fps}")
    return frames, fps

#### 2. SC Preprocessing
def preprocess_frame_for_sc(frame):
    """Performs preprocessing steps on a single frame for SC detection."""
    # Grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Downsample by factor of 2
    height, width = gray.shape
    # Ensure dimensions are even before dividing
    new_height = (height // 2) * 2
    new_width = (width // 2) * 2
    gray_resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    downsampled = cv2.resize(gray_resized, (new_width // 2, new_height // 2), interpolation=cv2.INTER_LINEAR)


    # Illumination adjustment using top-hat filtering
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ALPHA, ALPHA))
    try:
        morph_open = cv2.morphologyEx(downsampled, cv2.MORPH_OPEN, structuring_element)
        illum_adjusted = cv2.subtract(downsampled, morph_open) # I_adj = I - (I open SE)
    except cv2.error as e:
         print(f"  Warning: OpenCV error during morphology: {e}. Skipping illumination adjustment for this frame.")
         illum_adjusted = downsampled # Fallback to downsampled image

    return illum_adjusted

#### GT Detection
def detect_gradual_transitions(frames):
    print("Detecting Gradual Transitions (GTs)...")
    print(f"  Using Hist Comparison: {HIST_COMP_METHOD}, T_L: {T_L_HIST_DIFF}, T_U: {T_U_HIST_ACCUM}")
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

        successive_diff = cv2.compareHist(prev_hist, current_hist, HIST_COMP_METHOD)

        is_potential_start = False
        if HIST_COMP_METHOD == cv2.HISTCMP_CORREL:
            is_potential_start = successive_diff < T_L_HIST_DIFF # Correlation: Lower means different
        else:
             is_potential_start = successive_diff > T_L_HIST_DIFF # High difference suggests start

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
                if accum_hist_diff > T_U_HIST_ACCUM:
                    gt_end = i - 1
                    # Check if GT duration is sufficient
                    if (gt_end - potential_gt_start + 1) >= N_GT_MIN_LEN:
                        gt_segments.append((potential_gt_start, gt_end))
                        print(f"    --> Detected GT: Frames {potential_gt_start} to {gt_end} (Duration: {gt_end - potential_gt_start + 1}, AccumDiff: {accum_hist_diff:.4f})")
                    else:
                         print(f"    --> Discarded GT: Too short (Duration {gt_end - potential_gt_start + 1} < {N_GT_MIN_LEN})")
                else:
                     print(f"    --> Discarded GT: AccumDiff {accum_hist_diff:.4f} <= {T_U_HIST_ACCUM}")

                # Reset regardless of whether it met TU threshold
                potential_gt_start = -1
                accum_hist_diff = 0

        prev_hist = current_hist
        if i % 1000 == 0:
             print(f"  Processed {i} frames for GT detection...")


def identify_candidate_replay_segments(gt_segments, total_frames):
    print("Identifying Candidate Replay Segments (RSs)...")
    candidate_rs = []
    if len(gt_segments) < 2:
        print("  Not enough GTs found (< 2) to identify replay segments.")
        return candidate_rs

    # Sort GTs just in case they are out of order
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
             print(f"  Considering segment between GTs: Frames {actual_replay_content_start}-{actual_replay_content_end}, Duration: {actual_replay_duration}")

             # Check if the duration of the content *between* GTs fits replay limits
             if N_RL_MIN_DUR <= actual_replay_duration <= N_RU_MAX_DUR:
                  candidate_rs.append((actual_replay_content_start, actual_replay_content_end))
                  print(f"    -> Added as Candidate RS.")
             else:
                  print(f"    -> Discarded (Duration {actual_replay_duration} outside [{N_RL_MIN_DUR}, {N_RU_MAX_DUR}])")
        else:
             print(f"  Skipping segment between GTs {i} ({gt1_end}) and {i+1} ({gt2_start}): End is not after Start.")


    print(f"Finished candidate RS identification. Found {len(candidate_rs)} candidates.")
    return candidate_rs

#### SC Detection
def detect_score_caption_absence(frames_segment):
    if not frames_segment:
        print("    Segment has no frames, assuming SC absent.")
        return True # No frames means no SC

    processed_frames = [preprocess_frame_for_sc(f) for f in frames_segment]

    # Temporal Running Averaging
    avg_images = []
    num_frames = len(processed_frames)
    for i in range(num_frames):
        start = max(0, i - L_AVG // 2)
        end = min(num_frames, i + L_AVG // 2 + 1)
        window = processed_frames[start:end]
        # Filter out None values if preprocessing failed for some frames
        valid_window = [img for img in window if img is not None and img.size > 0]

        if valid_window:
            # Ensure all frames in window have same dimensions before averaging
            h, w = valid_window[0].shape
            consistent_window = [img for img in valid_window if img.shape == (h, w)]
            if consistent_window:
                 avg_img = np.mean(np.array(consistent_window, dtype=np.float32), axis=0).astype(np.uint8)
                 avg_images.append(avg_img)
            elif processed_frames[i] is not None: # Fallback if window is invalid
                 avg_images.append(processed_frames[i])
            else:
                 avg_images.append(None) # Add placeholder if center frame is invalid
        else:
             avg_images.append(processed_frames[i]) # Fallback if window is empty

    sc_detected_in_segment = False
    print(f"    Applying OCR to ROI [{SC_ROI_Y_START}:{SC_ROI_Y_END}, {SC_ROI_X_START}:{SC_ROI_X_END}] with T1={T1_OCR_CONF}, T2={T2_OCR_CHARS}")
    for i, avg_img in enumerate(avg_images):
        if avg_img is None:
            print(f"    Skipping OCR for frame {i}: Invalid average image.")
            continue # Skip if averaging failed

        # Image Binarization using mean and std dev
        try:
            mean, std_dev = cv2.meanStdDev(avg_img)
            mean = mean[0][0]
            std_dev = std_dev[0][0] if std_dev[0][0] > 1e-6 else 1.0 # Avoid division by zero or near-zero std dev
            lower_bound = mean - P_BINARIZE * std_dev
            upper_bound = mean + P_BINARIZE * std_dev
            # Pixels within [mean-p*std, mean+p*std] are 0 (background), others are 1 (potential text) -> map to 255
            _, binary_img_inv = cv2.threshold(avg_img, upper_bound, 255, cv2.THRESH_BINARY_INV)
            _, binary_img_lower = cv2.threshold(avg_img, lower_bound, 255, cv2.THRESH_BINARY)
            binary_img = cv2.bitwise_and(binary_img_inv, binary_img_lower) # Combine thresholds
        except cv2.error as e:
            print(f"    Warning: OpenCV error during binarization for frame {i}: {e}. Skipping frame.")
            continue


        #### OCR wtih ROI
        roi_h, roi_w = binary_img.shape
        y_start = max(0, SC_ROI_Y_START)
        y_end = min(roi_h, SC_ROI_Y_END)
        x_start = max(0, SC_ROI_X_START)
        x_end = min(roi_w, SC_ROI_X_END)

        if y_start >= y_end or x_start >= x_end:
             print(f"    Warning: Invalid ROI dimensions for frame {i} after clamping. Skipping OCR.")
             continue

        roi_img = binary_img[y_start:y_end, x_start:x_end]



        if roi_img.size == 0:
             print(f"    Warning: ROI image is empty for frame {i}. Skipping OCR.")
             continue

        try:
            # Use psm 6 assuming a single uniform block of text
            # want to test with psm 7 though
            ocr_data = pytesseract.image_to_data(roi_img, lang='eng', config='--psm 6', output_type=pytesseract.Output.DICT)
            num_chars_recognized = 0
            sum_confidence = 0
            num_confident_chars = 0
            recognized_texts = []


            for j, conf_str in enumerate(ocr_data['conf']):
                try:
                    conf = int(float(conf_str)) # Tesseract confidence is 0-100
                except ValueError:
                    conf = -1 # Handle non-integer confidence values if any

                if conf > 0: # Consider only boxes with confidence > 0
                    text = ocr_data['text'][j].strip()
                    if text: # Check if recognized text is not just whitespace
                        recognized_texts.append(text)
                        text_len = len(text)
                        num_chars_recognized += text_len
                        if conf >= (T1_OCR_CONF * 100):
                             sum_confidence += conf * text_len # Weight confidence sum by char length
                             num_confident_chars += text_len

            # Calculate average confidence only for characters meeting the threshold T1
            avg_confidence_norm = (sum_confidence / num_confident_chars) / 100.0 if num_confident_chars > 0 else 0.0

            # Debug print OCR results for this frame
            if num_chars_recognized > 0:
                 print(f"      Frame {i}: OCR found: '{' '.join(recognized_texts)}' (Chars: {num_chars_recognized}, Confident Chars: {num_confident_chars}, AvgConf: {avg_confidence_norm:.2f})")

            # Decision: SC present if enough chars recognized meeting confidence threshold
            # We check if the count of *confident* characters meets T2
            if num_confident_chars >= T2_OCR_CHARS:
                 print(f"    --> Frame {i}: SC DETECTED (Confident Chars {num_confident_chars} >= {T2_OCR_CHARS})")
                 sc_detected_in_segment = True
                 break # If SC found in any frame, the segment is not a replay

        except pytesseract.TesseractNotFoundError:
             print("    FATAL ERROR: Tesseract executable not found or not in PATH. Please install Tesseract and/or set pytesseract.pytesseract.tesseract_cmd.")
             # Decide how to handle: raise error, return default? Let's stop here.
             raise
        except Exception as e:
            print(f"    Error during OCR on ROI frame {i}: {e}")
            # Decide how to handle OCR errors - assume no SC? Or skip frame? Let's assume no SC detected on error.
            pass # Continue processing other frames

    # Final decision for the segment
    if sc_detected_in_segment:
         print("    Segment Result: SC DETECTED in at least one frame.")
         return False # SC Present
    else:
         print("    Segment Result: SC Absent in all processed frames.")
         return True # SC Absent

#### Results Evaluation
def evaluate_replay_detection(detected_replays, ground_truth_json_path, total_frames):
    """Evaluates the performance of the replay detection against ground truth."""
    print(f"\nEvaluating performance against: {ground_truth_json_path}")
    if not os.path.exists(ground_truth_json_path):
         print(f"Error: Ground truth file not found at {ground_truth_json_path}")
         return None

    try:
        with open(ground_truth_json_path, 'r') as f:
            ground_truth = json.load(f)
        gt_replays = ground_truth.get("replays", [])
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {ground_truth_json_path}")
        return None
    except Exception as e:
        print(f"Error reading ground truth file: {e}")
        return None

    # Create frame-level labels
    y_true = np.zeros(total_frames, dtype=int) # 0 = Live, 1 = Replay
    print("Processing Ground Truth Segments:")
    for start, end in gt_replays:
        start, end = int(start), int(end)
        print(f"  GT Segment: {start} - {end}")
        if 0 <= start < total_frames and 0 <= end < total_frames and start <= end:
            y_true[start : end + 1] = 1
        else:
             print(f"  Warning: Ground truth segment [{start}, {end}] out of bounds or invalid (Total frames: {total_frames}). Skipping.")

    y_pred = np.zeros(total_frames, dtype=int)
    print("Processing Detected Segments:")
    for start, end in detected_replays:
         start, end = int(start), int(end)
         print(f"  Detected Segment: {start} - {end}")
         if 0 <= start < total_frames and 0 <= end < total_frames and start <= end:
             y_pred[start : end + 1] = 1
         else:
             print(f"  Warning: Detected segment [{start}, {end}] out of bounds or invalid (Total frames: {total_frames}). Skipping.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    error_rate = (fp + fn) / total_frames if total_frames > 0 else 0

    results = {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "total_frames": total_frames,
        "gt_replay_frames": int(np.sum(y_true)),
        "detected_replay_frames": int(np.sum(y_pred))
    }

    print("\n--- Evaluation Results ---")
    print(f"  Total Frames: {results['total_frames']}")
    print(f"  Ground Truth Replay Frames: {results['gt_replay_frames']}")
    print(f"  Detected Replay Frames: {results['detected_replay_frames']}")
    print(f"  TP (Replay as Replay): {results['true_positives']}")
    print(f"  TN (Live as Live):     {results['true_negatives']}")
    print(f"  FP (Live as Replay):  {results['false_positives']}")
    print(f"  FN (Replay as Live):  {results['false_negatives']}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Error Rate:{results['error_rate']:.4f}")
    print("-------------------------\n")

    return results

#################################
#### Replay Extraction Model ####
#################################

if __name__ == "__main__":
    video_file = "test1.mp4"
    ground_truth_file = "ground_truth_annotations.json"

    if not os.path.exists(video_file):
         print(f"FATAL ERROR: Video file not found at {video_file}")
         exit()
    if not os.path.exists(ground_truth_file):
        print(f"WARNING: Ground truth file not found at {ground_truth_file}. Evaluation will be skipped.")
        perform_evaluation = False
    else:
        perform_evaluation = True

    # 1. Extract Frames
    all_frames, video_fps = extract_frames(video_file)
    if not all_frames:
        exit()
    total_video_frames = len(all_frames)

    # 2. Detect Gradual Transitions
    gt_segments = detect_gradual_transitions(all_frames)

    # 3. Identify Candidate Replay Segments
    candidate_segments = identify_candidate_replay_segments(gt_segments, total_video_frames)

    # 4. Detect Score Caption Absence in Candidates
    final_replay_segments = []
    print("\nDetecting Score Caption (SC) Absence in candidates...")
    if not candidate_segments:
        print("  No candidate segments found. Skipping SC detection.")
    else:
        for i, (start, end) in enumerate(candidate_segments):
            print(f"\n--- Processing Candidate RS {i+1}/{len(candidate_segments)}: Frames {start} to {end} ---")
            # Ensure indices are within bounds
            if start < 0 or end >= total_video_frames or start > end:
                 print(f"  Skipping invalid candidate segment indices: {start}-{end}")
                 continue

            segment_frames = all_frames[start : end + 1]
            if not segment_frames:
                 print("  Skipping candidate segment: No frames extracted for these indices.")
                 continue

            is_sc_absent = detect_score_caption_absence(segment_frames, start)

            if is_sc_absent:
                print(f"  --> RESULT: SC Absent. Adding segment [{start}, {end}] as Final Replay.")
                final_replay_segments.append((start, end))
            else:
                print(f"  --> RESULT: SC Present. Discarding segment [{start}, {end}].")

    print(f"\nFinished SC detection. Identified {len(final_replay_segments)} final replay segments:")
    if final_replay_segments:
        for start, end in final_replay_segments:
             print(f"  - Replay: Frames {start} to {end}")
    else:
        print("  No final replay segments identified.")


    if perform_evaluation:
        evaluation_metrics = evaluate_replay_detection(final_replay_segments, ground_truth_file, total_video_frames)
        if evaluation_metrics:
             print("\nEvaluation completed successfully.")
    else:
        print(f"\nSkipping evaluation step.")

    print("\nProcessing finished.")