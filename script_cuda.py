import cv2
# Check if CUDA is available before importing cv2.cuda
cuda_available = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
if cuda_available:
    print(f"CUDA is available. OpenCV version: {cv2.__version__}")
    try:
        print(f"CUDA Device Count: {cv2.cuda.getCudaEnabledDeviceCount()}")
        # Optional: Print device info
        # for i in range(cv2.cuda.getCudaEnabledDeviceCount()):
        #     cv2.cuda.printCudaDeviceInfo(i)
    except Exception as e:
        print(f"Error accessing CUDA device info: {e}")
        cuda_available = False # Treat as unavailable if info check fails
else:
    print(f"CUDA is not available or no CUDA-enabled devices found. Running on CPU. OpenCV version: {cv2.__version__}")

import numpy as np
import pytesseract
import json
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import os
import time

##### Configuration
# IMPORTANT: Update this path to your Tesseract installation
# Make sure this path is correct for your system
try:
    # Example for Windows, adjust if needed for Linux/macOS
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Verify Tesseract can be found
    tesseract_version = pytesseract.get_tesseract_version()
    print(f"Tesseract version {tesseract_version} found at: {pytesseract.pytesseract.tesseract_cmd}")
except pytesseract.TesseractNotFoundError:
    print(f"FATAL ERROR: Tesseract executable not found or incorrect path: {pytesseract.pytesseract.tesseract_cmd}")
    print("Please install Tesseract OCR and update the `pytesseract.pytesseract.tesseract_cmd` variable.")
    exit(1)
except Exception as e:
    print(f"Warning: Could not verify Tesseract installation: {e}")
    # Depending on the error, you might still want to exit if Tesseract is crucial
    # exit(1)

#### Parameters (Keep original parameters)
ALPHA = 3         # Structuring element size for top-hat filter
P_BINARIZE = 2.5  # Constant for binarization threshold
L_AVG = 5         # Window length for temporal running average
T1_OCR_CONF = 0.75 # Confidence threshold (0-1.0) for OCR character recognition
T2_OCR_CHARS = 3 # Minimum number of recognized characters for SC presence
N_GT_MIN_LEN = 7 # Minimum length (frames) for a gradual transition
N_RL_MIN_DUR = 235 # Minimum duration (frames) for a replay segment
N_RU_MAX_DUR = 800 # Maximum duration (frames) for a replay segment

# Thresholds for GT detection
T_L_HIST_DIFF = 0.025 # Lower threshold for successive histogram difference
T_U_HIST_ACCUM = 3.5 # Upper threshold for accumulative histogram difference

# Histogram Comparison Method
HIST_COMP_METHOD = cv2.HISTCMP_CHISQR # Options: HISTCMP_CORREL, HISTCMP_BHATTACHARYYA, HISTCMP_CHISQR

# Score Caption (SC) Region of Interest (ROI) - Not used in the provided SC detection logic
# SC_ROI_Y_START = 0
# SC_ROI_Y_END = 1080
# SC_ROI_X_START = 0
# SC_ROI_X_END = 1920

#### 1. Frame Extraction (Remains on CPU, typically I/O bound)
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
        if frame is None: # Add check for None frames
             print(f"Warning: Read invalid frame at index {frame_count}. Skipping.")
             continue
        frames.append(frame) # Keep frames as NumPy arrays on CPU initially
        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"  Extracted {frame_count} frames...")

    cap.release()
    print(f"Finished extraction. Total frames: {len(frames)}, FPS: {fps}")
    return frames, fps

#### 2. SC Preprocessing (CUDA VERSION)
def preprocess_frame_for_sc_cuda(frame_cpu):
    """Performs preprocessing steps on a single frame for SC detection using CUDA."""
    if not cuda_available:
        raise RuntimeError("CUDA is not available, cannot run CUDA preprocessing.")
    if frame_cpu is None:
        print("  Warning: Received None frame for preprocessing. Returning None.")
        return None
    try:
        # Upload frame to GPU
        frame_gpu = cv2.cuda_GpuMat()
        frame_gpu.upload(frame_cpu)

        # Grayscale conversion
        gray_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY)

        # Downsample by factor of 2 (ensure dimensions are even for potential INTER_AREA)
        height, width = gray_gpu.size()[::-1] # GpuMat size is (width, height)
        new_height = height // 2
        new_width = width // 2

        # Use INTER_LINEAR or INTER_AREA for downsampling. INTER_AREA might be better for quality.
        downsampled_gpu = cv2.cuda.resize(gray_gpu, (new_width, new_height), interpolation=cv2.INTER_LINEAR) # Or cv2.INTER_AREA

        # Illumination adjustment using top-hat filtering
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ALPHA, ALPHA))
        # Create CUDA morphology filter for MORPH_OPEN (equivalent to Erode then Dilate)
        # Note: Top-hat is usually I - open(I).
        morph_filter_open = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, downsampled_gpu.type(), structuring_element)
        morph_open_gpu = morph_filter_open.apply(downsampled_gpu)

        # I_adj = I - (I open SE)
        illum_adjusted_gpu = cv2.cuda.subtract(downsampled_gpu, morph_open_gpu)

        # Clean up intermediate GPU mats explicitly (optional but good practice)
        del frame_gpu, gray_gpu, morph_filter_open, morph_open_gpu, downsampled_gpu

        return illum_adjusted_gpu # Return GpuMat

    except cv2.error as e:
         print(f"  Warning: OpenCV CUDA error during preprocessing: {e}. Returning None.")
         # Consider releasing potentially created GpuMats here if an error occurred mid-function
         return None
    except Exception as e:
        print(f"  Warning: Generic error during preprocessing: {e}. Returning None.")
        return None


#### GT Detection (CUDA Conversion/Hist Calc, CPU Norm/Compare) - KEEPS CPU NORMALIZATION FIX
def detect_gradual_transitions_cuda(frames):
    print("Detecting Gradual Transitions (GTs) using CUDA...")
    if not cuda_available:
        raise RuntimeError("CUDA is not available, cannot run CUDA GT detection.")

    print(f"  Using Hist Comparison: {HIST_COMP_METHOD}, T_L: {T_L_HIST_DIFF}, T_U: {T_U_HIST_ACCUM}")
    gt_segments = []
    if not frames:
        print("  No frames provided for GT detection.")
        return gt_segments

    hist_size = [256] # Luminance histogram
    hist_range = [0, 256]
    prev_hist_cpu_norm = None # Store the CPU-normalized previous histogram

    # --- Process first valid frame ---
    first_frame_index = -1
    frame0_gpu = None # Initialize to allow cleanup in case of error
    prev_gray_gpu = None
    prev_hist_gpu = None
    for idx, first_frame in enumerate(frames):
         if first_frame is not None:
              try:
                  frame0_gpu = cv2.cuda_GpuMat()
                  frame0_gpu.upload(first_frame)
                  prev_gray_gpu = cv2.cuda.cvtColor(frame0_gpu, cv2.COLOR_BGR2GRAY)
                  # Calculate histogram on GPU
                  prev_hist_gpu = cv2.cuda.calcHist(prev_gray_gpu, None)
                  # Download raw histogram
                  prev_hist_cpu_raw = prev_hist_gpu.download()

                  # --- FIX: Normalize and type cast on CPU ---
                  if prev_hist_cpu_raw is None:
                       raise ValueError("Downloaded initial raw histogram is None.")

                  # Convert to float32 first
                  prev_hist_cpu_float = prev_hist_cpu_raw.astype(np.float32)
                  # Normalize in-place on CPU
                  cv2.normalize(prev_hist_cpu_float, prev_hist_cpu_float, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
                  prev_hist_cpu_norm = prev_hist_cpu_float # Store the normalized version
                  # ------------------------------------------

                  del prev_hist_cpu_raw # Cleanup raw CPU hist
                  first_frame_index = idx
                  print(f"  Processed first valid frame at index {first_frame_index} for GT baseline.")
                  break # Exit loop once first frame is processed
              except cv2.error as e:
                  print(f"  Error processing first valid frame (index {idx}) for GT with CUDA: {e}. Trying next frame.")
                  prev_hist_cpu_norm = None # Reset baseline
              except Exception as e:
                  print(f"  Generic error processing first valid frame (index {idx}) for GT: {e}. Trying next frame.")
                  prev_hist_cpu_norm = None # Reset baseline
              finally: # Ensure cleanup even if errors occur
                  if frame0_gpu is not None: del frame0_gpu; frame0_gpu = None
                  if prev_gray_gpu is not None: del prev_gray_gpu; prev_gray_gpu = None
                  if prev_hist_gpu is not None: del prev_hist_gpu; prev_hist_gpu = None
         else:
              print(f"  Skipping None frame at index {idx} while searching for first valid frame.")

    if prev_hist_cpu_norm is None or first_frame_index == -1:
         print("  Error: Could not process any frame successfully to establish GT baseline. Aborting GT detection.")
         return gt_segments

    potential_gt_start = -1
    accum_hist_diff = 0

    # --- Process remaining frames ---
    current_frame_gpu = cv2.cuda_GpuMat()
    current_gray_gpu = cv2.cuda_GpuMat()
    current_hist_gpu = None # Initialize for cleanup

    for i in range(first_frame_index + 1, len(frames)):
        current_frame_cpu = frames[i]
        if current_frame_cpu is None:
             print(f"  Warning: Skipping None frame at index {i} during GT detection loop.")
             continue

        current_hist_cpu_norm = None # Ensure defined for update logic
        try:
            # Upload, Convert, Calculate Hist
            current_frame_gpu.upload(current_frame_cpu)
            current_gray_gpu = cv2.cuda.cvtColor(current_frame_gpu, cv2.COLOR_BGR2GRAY, dst=current_gray_gpu) # Use dst for potential reuse
            current_hist_gpu = cv2.cuda.calcHist(current_gray_gpu, None)
            # Download raw histogram
            current_hist_cpu_raw = current_hist_gpu.download()


            # --- FIX: Normalize and type cast on CPU ---
            if current_hist_cpu_raw is None:
                 print(f"  Warning: Downloaded raw histogram is None for frame {i}. Skipping comparison.")
                 continue
            # Convert to float32 first
            current_hist_cpu_float = current_hist_cpu_raw.astype(np.float32)
            # Normalize in-place on CPU
            cv2.normalize(current_hist_cpu_float, current_hist_cpu_float, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
            current_hist_cpu_norm = current_hist_cpu_float # Use this for comparison
            # ------------------------------------------
            del current_hist_cpu_raw # Delete raw hist


            # Compare histograms - NOW using CPU-normalized float32 arrays
            # Ensure previous histogram is valid before comparing
            if prev_hist_cpu_norm is None:
                 print(f"  Warning: Previous histogram is None at frame {i}, cannot compare. Updating prev hist.")
                 prev_hist_cpu_norm = current_hist_cpu_norm # Set current as previous for next step
                 continue # Skip comparison logic for this iteration

            successive_diff = cv2.compareHist(prev_hist_cpu_norm, current_hist_cpu_norm, HIST_COMP_METHOD)

            # --- GT Logic ---
            is_different = False
            # Handle potential NaN or Inf in diff if normalization/comparison had issues
            if not np.isfinite(successive_diff):
                 print(f"  Warning: Non-finite histogram difference ({successive_diff}) at frame {i}. Treating as 'not different'.")
                 is_different = False
            elif HIST_COMP_METHOD == cv2.HISTCMP_CORREL or HIST_COMP_METHOD == cv2.HISTCMP_INTERSECT:
                 # Lower correlation/intersection means more different
                 is_different = successive_diff < (1.0 - T_L_HIST_DIFF)
            else: # CHISQR, BHATTACHARYYA
                 # Higher distance means more different
                 is_different = successive_diff > T_L_HIST_DIFF

            if is_different:
                if potential_gt_start == -1:
                    potential_gt_start = i - 1 # GT starts at the *previous* frame
                # Use a consistent difference measure (distance from 'identical')
                if HIST_COMP_METHOD == cv2.HISTCMP_CORREL or HIST_COMP_METHOD == cv2.HISTCMP_INTERSECT:
                     diff_measure = 1.0 - successive_diff
                else:
                     diff_measure = successive_diff
                accum_hist_diff += diff_measure # Accumulate the difference measure
            else: # Frames are considered similar
                if potential_gt_start != -1: # Was in a potential GT
                    # Check if the accumulated difference meets the upper threshold
                    if accum_hist_diff > T_U_HIST_ACCUM:
                        gt_end = i - 1 # GT ended at the previous frame
                        # Check if the GT duration meets the minimum length
                        if (gt_end - potential_gt_start + 1) >= N_GT_MIN_LEN:
                            gt_segments.append((potential_gt_start, gt_end))
                            print(f"    --> Detected GT: Frames {potential_gt_start} to {gt_end} (Duration: {gt_end - potential_gt_start + 1}, AccumDiff: {accum_hist_diff:.4f})")
                    # Reset potential GT regardless of whether it met thresholds
                    potential_gt_start = -1
                    accum_hist_diff = 0

            # Update previous histogram FOR THE NEXT loop iteration
            # Important: Update *after* using it for comparison and GT logic
            prev_hist_cpu_norm = current_hist_cpu_norm
            # current_hist_cpu_norm is now prev_hist_cpu_norm, don't delete yet

        except cv2.error as e:
            print(f"  Warning: CUDA/OpenCV error processing frame {i} for GT: {e}. Resetting GT state.")
            potential_gt_start = -1 # Reset state on error
            accum_hist_diff = 0
            # Try to update prev_hist if possible, otherwise it remains from last good frame
            if current_hist_cpu_norm is not None:
                prev_hist_cpu_norm = current_hist_cpu_norm
            else:
                 print("    Cannot update previous histogram due to error.")
            # Continue to next frame
        except Exception as e:
             print(f"  Warning: Generic error processing frame {i} for GT: {e}. Resetting GT state.")
             potential_gt_start = -1
             accum_hist_diff = 0
             if current_hist_cpu_norm is not None:
                 prev_hist_cpu_norm = current_hist_cpu_norm
             else:
                 print("    Cannot update previous histogram due to error.")
             # Continue to next frame
        finally:
            # Clean up GPU histogram from this iteration
            if current_hist_gpu is not None:
                 del current_hist_gpu; current_hist_gpu = None

        if i % 1000 == 0:
             print(f"  Processed {i} frames for GT detection...")

    # --- Check for lingering potential GT at the end ---
    if potential_gt_start != -1:
        if accum_hist_diff > T_U_HIST_ACCUM:
            gt_end = len(frames) - 1 # GT extends to the last frame
            if (gt_end - potential_gt_start + 1) >= N_GT_MIN_LEN:
                gt_segments.append((potential_gt_start, gt_end))
                print(f"    --> Detected GT ending at last frame: Frames {potential_gt_start} to {gt_end} (Duration: {gt_end - potential_gt_start + 1}, AccumDiff: {accum_hist_diff:.4f})")

    # Clean up reusable GpuMats
    del current_frame_gpu, current_gray_gpu
    print(f"Finished GT detection. Found {len(gt_segments)} GTs.")
    return gt_segments

# Identifying candidate segments remains the same (logic based on GT indices)
def identify_candidate_replay_segments(gt_segments):
    print("Identifying Candidate Replay Segments (RSs)...")
    candidate_rs = []
    if len(gt_segments) < 2:
        print("  Not enough GTs found (< 2) to identify replay segments.")
        return candidate_rs

    # Sort GTs by start frame just in case
    gt_segments.sort(key=lambda x: x[0])

    for i in range(len(gt_segments) - 1):
        try:
            gt1_start, gt1_end = map(int, gt_segments[i])
            gt2_start, gt2_end = map(int, gt_segments[i+1])
        except (ValueError, TypeError):
             print(f"  Warning: Invalid GT segment format encountered near index {i}: {gt_segments[i]}. Skipping.")
             continue

        # Define the segment *between* the end of GT1 and start of GT2
        actual_replay_content_start = gt1_end + 1
        actual_replay_content_end = gt2_start - 1

        # Check if segment exists (gt2 must start after gt1 ends)
        if actual_replay_content_start <= actual_replay_content_end:
             actual_replay_duration = actual_replay_content_end - actual_replay_content_start + 1
             # print(f"  Considering segment between GTs: Frames {actual_replay_content_start}-{actual_replay_content_end}, Duration: {actual_replay_duration}") # Debug

             # Check if the duration of the content *between* GTs fits replay limits
             if N_RL_MIN_DUR <= actual_replay_duration <= N_RU_MAX_DUR:
                  candidate_rs.append((actual_replay_content_start, actual_replay_content_end))
                  print(f"    -> Added as Candidate RS: Frames {actual_replay_content_start}-{actual_replay_content_end}")
             # else: # Debug
             #     print(f"    -> Discarded (Duration {actual_replay_duration} outside [{N_RL_MIN_DUR}, {N_RU_MAX_DUR}])")
        # else: # Debug
             # print(f"  Skipping segment between GTs {i} ({gt1_end}) and {i+1} ({gt2_start}): End is not after Start or adjacent.")


    print(f"Finished candidate RS identification. Found {len(candidate_rs)} candidates.")
    return candidate_rs


#### SC Detection (CUDA Version) - PURE GPU AVERAGING (Corrected) ####
def detect_score_caption_absence_cuda(frames_segment_cpu):
    if not cuda_available:
        raise RuntimeError("CUDA is not available, cannot run CUDA SC detection.")

    if not frames_segment_cpu:
        print("    Segment has no frames, assuming SC absent.")
        return True # No frames means no SC

    num_frames = len(frames_segment_cpu)
    print(f"    Processing segment of {num_frames} frames for SC absence using CUDA (No ROI)...")

    # Preprocess all frames
    processed_frames_gpu = []
    original_dtypes = {} # Store original dtype for conversion later
    valid_indices = []
    for k, f_cpu in enumerate(frames_segment_cpu):
         if f_cpu is None:
             processed_frames_gpu.append(None)
             continue
         processed_gpu = preprocess_frame_for_sc_cuda(f_cpu)
         processed_frames_gpu.append(processed_gpu)
         if processed_gpu is not None:
             valid_indices.append(k)
             # Store the original dtype (assuming preprocess_frame_for_sc_cuda returns consistent type)
             # Use first valid frame's type as reference later
             if not original_dtypes: # Store only the first valid one encountered
                 original_dtypes['ref'] = processed_gpu.type() # Store type like cv2.CV_8U

    if not valid_indices:
        print("      Warning: Preprocessing failed for all frames. Assuming SC absent.")
        del processed_frames_gpu # Clean up list
        return True

    # Get a reference dtype (e.g., from the first valid processed frame)
    # Default to CV_8U if somehow no valid frames were processed but valid_indices isn't empty
    reference_dtype = original_dtypes.get('ref', cv2.CV_8U)
    print(f"      Reference data type for averaging output: {reference_dtype}")

    # --- Temporal Running Averaging ---
    avg_images_gpu = [None] * num_frames
    stream = cv2.cuda.Stream_Null() # Use default stream

    print(f"    Applying temporal averaging (L_AVG={L_AVG}) on GPU...")
    for i in range(num_frames):
        if processed_frames_gpu[i] is None:
            avg_images_gpu[i] = None # Ensure avg_images also gets None
            continue

        # Determine window boundaries
        start = max(0, i - L_AVG // 2)
        end = min(num_frames, i + L_AVG // 2 + 1)

        # Collect valid GpuMats within the window
        window_gpu_mats = [processed_frames_gpu[idx] for idx in range(start, end) if processed_frames_gpu[idx] is not None]

        # If window is empty or only contains the current frame, use the current frame directly
        if not window_gpu_mats or (len(window_gpu_mats) == 1 and window_gpu_mats[0] is processed_frames_gpu[i]):
            avg_images_gpu[i] = processed_frames_gpu[i] # Use the preprocessed frame as is
            continue

        # Filter for consistent shape and type within the window (using first mat as reference)
        ref_mat = window_gpu_mats[0]
        h, w = ref_mat.size()[::-1]
        dtype = ref_mat.type() # Type of preprocessed frames
        consistent_window_gpu = [img for img in window_gpu_mats if img.size()[::-1] == (h, w) and img.type() == dtype]

        if not consistent_window_gpu:
            print(f"      Warning: No consistent frames found in window for index {i}. Using center frame.")
            avg_images_gpu[i] = processed_frames_gpu[i] # Fallback to center frame
            continue

        count = float(len(consistent_window_gpu))
        if count <= 0: # Should not happen if consistent_window_gpu is not empty, but check anyway
             print(f"      Warning: Zero frames in consistent window for index {i}. Skipping averaging.")
             avg_images_gpu[i] = processed_frames_gpu[i] # Fallback
             continue

        # Initialize sum accumulator (must be float for accumulation)
        # Use the first consistent frame, converted to float
        temp_sum_gpu = consistent_window_gpu[0].convertTo(cv2.CV_32F, 1.0, 0.0, stream)

        # Accumulate sum (convert each subsequent frame to float before adding)
        for j in range(1, len(consistent_window_gpu)):
            # Create a temporary GpuMat for the float version to avoid modifying original list
            temp_float_gpu = consistent_window_gpu[j].convertTo(cv2.CV_32F, 1.0, 0.0, stream)
            temp_sum_gpu = cv2.cuda.add(temp_sum_gpu, temp_float_gpu, stream=stream)
            del temp_float_gpu # Clean up intermediate float conversion

        # --- PURE GPU AVERAGING AND TYPE CONVERSION using convertTo ---
        # 1. Calculate the scale factor (alpha) for averaging
        scale_factor = 1.0 / count

        # 2. Use convertTo to scale (alpha) and convert type (rtype) in one step
        #    The source is temp_sum_gpu (CV_32F)
        #    The destination type is reference_dtype (e.g., CV_8U)
        #    The scale factor alpha is 1.0 / count
        avg_gpu_result = temp_sum_gpu.convertTo(reference_dtype, alpha=scale_factor, beta=0.0, stream=stream)
        # -------------------------------------------------------------

        avg_images_gpu[i] = avg_gpu_result # Assign the final GpuMat

        # Clean up intermediate GPU sum mat for this iteration
        del temp_sum_gpu
        # avg_gpu_result is stored in avg_images_gpu, don't delete yet

    # Clean up stream object reference if needed (often managed contextually)
    # del stream

    # Clean up original preprocessed frames as they are now averaged or copied
    for mat in processed_frames_gpu:
        if mat is not None: del mat
    del processed_frames_gpu


    # --- Process averaged frames for OCR ---
    sc_detected_in_segment = False
    print(f"    Applying OCR to full binarized frames with T1={T1_OCR_CONF}, T2={T2_OCR_CHARS}")

    # Reusable GpuMats for thresholding
    binary_gpu_inv = cv2.cuda.GpuMat(); binary_gpu_lower = cv2.cuda.GpuMat(); binary_gpu = cv2.cuda.GpuMat()

    for i in range(num_frames):
        avg_img_gpu = avg_images_gpu[i] # This should be the final GpuMat from the averaging step
        if avg_img_gpu is None:
            # print(f"    Skipping OCR for frame {i}: Averaged image is None.") # Optional debug
            continue

        mean_stddev_gpu = None # For cleanup

        try: # Main processing try block for binarization and OCR
            # --- Binarization (GPU) ---
            # Calculate mean and std dev on GPU
            mean_stddev_gpu = cv2.cuda.meanStdDev(avg_img_gpu)
            # Download the result (small array)
            mean_stddev_cpu = mean_stddev_gpu.download()
            del mean_stddev_gpu; mean_stddev_gpu = None # Cleanup downloaded GpuMat

            # Check downloaded result and extract values safely
            if not isinstance(mean_stddev_cpu, np.ndarray) or mean_stddev_cpu.shape[1] < 2:
                 print(f"    Warning: Failed download or unexpected shape for mean/stddev frame {i}. Shape: {getattr(mean_stddev_cpu, 'shape', 'N/A')}. Skipping frame.")
                 continue

            mean_val = mean_stddev_cpu[0, 0]; std_dev_val = mean_stddev_cpu[0, 1]

            # Avoid division by zero or near-zero std dev
            std_dev_val = std_dev_val if std_dev_val > 1e-6 else 1.0

            # Calculate thresholds
            lower_bound = mean_val - P_BINARIZE * std_dev_val
            upper_bound = mean_val + P_BINARIZE * std_dev_val

            # Apply adaptive thresholding using calculated bounds on GPU
            # Thresholding: Keep pixels between lower_bound and upper_bound
            _, binary_gpu_inv = cv2.cuda.threshold(avg_img_gpu, upper_bound, 255, cv2.THRESH_BINARY_INV, dst=binary_gpu_inv)
            _, binary_gpu_lower = cv2.cuda.threshold(avg_img_gpu, lower_bound, 255, cv2.THRESH_BINARY, dst=binary_gpu_lower)
            # Combine the two thresholds: pixels must be < upper_bound AND > lower_bound
            binary_gpu = cv2.cuda.bitwise_and(binary_gpu_inv, binary_gpu_lower, dst=binary_gpu)

            if binary_gpu.empty(): # Check if result is valid
                 print(f"    Warning: Binarized image empty after bitwise_and frame {i}. Skipping.")
                 continue

            # --- OCR (CPU) ---
            # Download the final binarized image for Pytesseract
            full_binary_img_cpu = binary_gpu.download()
            if full_binary_img_cpu is None or full_binary_img_cpu.size == 0:
                 print(f"    Warning: Downloaded binary_img_cpu invalid/empty frame {i}. Skipping.")
                 continue

            try: # OCR try block
                # Use --psm 7: Treat the image as a single text line. Adjust if needed.
                # Tesseract works best on white background, black text. Our binary_gpu is likely this.
                ocr_data = pytesseract.image_to_data(full_binary_img_cpu, lang='eng', config='--psm 7', output_type=pytesseract.Output.DICT)

                num_confident_chars = 0
                # Iterate through detected words/characters
                for j, conf_str in enumerate(ocr_data['conf']):
                    try:
                        conf = int(float(conf_str)) # Confidence is usually string, convert safely
                    except ValueError:
                        conf = -1 # Treat non-numeric confidence as invalid

                    # Check confidence level (-1 indicates non-text block in tesseract data)
                    if conf >= 0: # Tesseract's confidence values are 0-100
                        text = ocr_data['text'][j].strip()
                        # Check if text is non-empty and confidence meets threshold
                        if text and conf >= (T1_OCR_CONF * 100):
                             num_confident_chars += len(text) # Count characters in confident text

                # Check if enough confident characters were found
                if num_confident_chars >= T2_OCR_CHARS:
                     #print(f"    --> Frame {i} (Avg'd): SC DETECTED (Chars {num_confident_chars} >= {T2_OCR_CHARS})")
                     sc_detected_in_segment = True
                     del full_binary_img_cpu # Cleanup CPU image early
                     break # Exit inner loop (frame processing) as SC is found

            except pytesseract.TesseractNotFoundError:
                print("    FATAL ERROR: Tesseract not found during OCR call. Ensure it's installed and path is correct.")
                raise # Re-raise the error to stop execution
            except Exception as e:
                print(f"    Error during OCR on binarized frame {i}: {e}")
                # Decide whether to continue or break; continuing might miss SC
                pass # Continue processing other frames in the segment
            finally:
                 # Ensure cleanup if OCR loop didn't break or if full_binary_img_cpu exists
                 if 'full_binary_img_cpu' in locals() and full_binary_img_cpu is not None:
                     # Check if it's a numpy array before deleting (in case it was assigned None)
                     if isinstance(full_binary_img_cpu, np.ndarray):
                         del full_binary_img_cpu


        except cv2.error as e: # Catch specific OpenCV errors during binarization
            print(f"    Warning: CUDA/OpenCV error processing frame {i} for binarization: {e}. Skipping.")
            continue # Skip to next frame
        except Exception as e: # Catch other errors during binarization
            print(f"    Warning: Generic error processing avg frame {i} before OCR: {e}. Skipping frame.")
            continue # Skip to next frame
        # No finally needed here for GPU mats as they are reused

    # Final cleanup of reusable and averaged mats
    del binary_gpu_inv, binary_gpu_lower, binary_gpu
    for gpu_mat in avg_images_gpu:
        if gpu_mat is not None: del gpu_mat
    del avg_images_gpu

    if sc_detected_in_segment:
        print("    Segment Result: SC DETECTED in at least one frame.")
        return False # SC is present (absence check fails)
    else:
        print("    Segment Result: SC Absent in all processed frames.")
        return True # SC is absent (absence check passes)

#### Results Evaluation (Remains on CPU)
def evaluate_replay_detection(detected_replays, ground_truth_json_path, total_frames):
    """Evaluates the performance of the replay detection against ground truth."""
    print(f"\nEvaluating performance against: {ground_truth_json_path}")
    if not os.path.exists(ground_truth_json_path):
         print(f"Error: Ground truth file not found at {ground_truth_json_path}")
         return None

    try:
        with open(ground_truth_json_path, 'r') as f:
            ground_truth = json.load(f)
        # Expecting format like {"replays": [[start1, end1], [start2, end2], ...]}
        gt_replays = ground_truth.get("replays", [])
        if not isinstance(gt_replays, list) or not all(isinstance(seg, list) and len(seg) == 2 for seg in gt_replays):
             print("Error: Ground truth 'replays' field is not a list of [start, end] pairs.")
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
             start, end = int(segment[0]), int(segment[1])
             # print(f"  GT Segment: {start} - {end}") # Debug
             # Clamp indices to valid range [0, total_frames - 1]
             start = max(0, start)
             end = min(total_frames - 1, end)
             if start <= end: # Ensure start is not after end after clamping
                 y_true[start : end + 1] = 1
             else:
                  print(f"  Warning: Ground truth segment [{segment[0]}, {segment[1]}] resulted in invalid range [{start}, {end}] after clamping to {total_frames} frames. Skipping.")
        except (ValueError, TypeError):
             print(f"  Warning: Invalid ground truth segment format: {segment}. Skipping.")


    y_pred = np.zeros(total_frames, dtype=int)
    print("Processing Detected Segments:")
    for segment in detected_replays:
        try:
             start, end = int(segment[0]), int(segment[1])
             # print(f"  Detected Segment: {start} - {end}") # Debug
             # Clamp indices to valid range [0, total_frames - 1]
             start = max(0, start)
             end = min(total_frames - 1, end)
             if start <= end:
                 y_pred[start : end + 1] = 1
             else:
                 print(f"  Warning: Detected segment [{segment[0]}, {segment[1]}] resulted in invalid range [{start}, {end}] after clamping to {total_frames} frames. Skipping.")
        except (ValueError, TypeError):
            print(f"  Warning: Invalid detected segment format: {segment}. Skipping.")


    # Ensure y_true and y_pred have the same length (should be total_frames if clamping worked)
    if len(y_true) != total_frames or len(y_pred) != total_frames:
        print(f"Warning: Label array lengths ({len(y_true)}, {len(y_pred)}) do not match total frames ({total_frames}). This indicates a potential issue.")
        # Attempt to proceed with the minimum length, but results might be skewed
        min_len = min(len(y_true), len(y_pred), total_frames)
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        eff_total_frames = min_len
    else:
        eff_total_frames = total_frames

    if eff_total_frames == 0:
        print("Error: Cannot evaluate with zero effective frames.")
        return None

    # Calculate metrics using sklearn functions (handle zero division)
    # Use labels=[0, 1] to ensure the confusion matrix has the correct dimensions even if one class isn't present.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Extract TP, TN, FP, FN safely
    if cm.size == 4:
         tn, fp, fn, tp = cm.ravel()
    else: # Handle unexpected matrix size (e.g., if only one label was ever present)
         print(f"Warning: Confusion matrix has unexpected shape: {cm.shape}. Estimating counts.")
         tp = np.sum((y_pred == 1) & (y_true == 1))
         tn = np.sum((y_pred == 0) & (y_true == 0))
         fp = np.sum((y_pred == 1) & (y_true == 0))
         fn = np.sum((y_pred == 0) & (y_true == 1))

    # Calculate metrics, ensuring zero_division parameter is set
    precision = precision_score(y_true, y_pred, labels=[1], pos_label=1, zero_division=0) # Precision for replay class (1)
    recall = recall_score(y_true, y_pred, labels=[1], pos_label=1, zero_division=0)       # Recall for replay class (1)
    accuracy = accuracy_score(y_true, y_pred)
    # F1 Score can be useful too
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    error_rate = 1.0 - accuracy


    results = {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "total_frames": total_frames, # Report original total
        "effective_eval_frames": eff_total_frames, # Report effective total used
        "gt_replay_frames": int(np.sum(y_true)),
        "detected_replay_frames": int(np.sum(y_pred))
    }

    print("\n--- Evaluation Results ---")
    print(f"  Total Frames: {results['total_frames']}")
    print(f"  Effective Frames Evaluated: {results['effective_eval_frames']}")
    print(f"  Ground Truth Replay Frames: {results['gt_replay_frames']}")
    print(f"  Detected Replay Frames: {results['detected_replay_frames']}")
    print(f"  Confusion Matrix (TN, FP, FN, TP): ({results['true_negatives']}, {results['false_positives']}, {results['false_negatives']}, {results['true_positives']})")
    print(f"  Accuracy (Overall):  {results['accuracy']:.4f}")
    print(f"  Precision (Replay):  {results['precision']:.4f}")
    print(f"  Recall (Replay):     {results['recall']:.4f}")
    print(f"  F1-Score (Replay):   {results['f1_score']:.4f}")
    print(f"  Error Rate (Overall):{results['error_rate']:.4f}")
    print("-------------------------\n")

    return results

#### Export Replays (Remains on CPU, writes NumPy frames)
def export_replays_to_video(replay_segments, all_video_frames, fps, output_path):
    """
    Combines frames from specified replay segments into a single video file.
    Expects all_video_frames to be a list of CPU NumPy arrays.
    """
    print(f"\nExporting {len(replay_segments)} replay segments to: {output_path}")
    if not replay_segments:
        print("  No replay segments to export.")
        return

    if not all_video_frames:
        print("  Error: Cannot export replays, the list of all video frames is empty.")
        return

    try:
        # Get frame size from the first valid frame more robustly
        first_valid_frame = None
        for f in all_video_frames:
             if f is not None and hasattr(f, 'shape') and len(f.shape) == 3:
                  first_valid_frame = f
                  break

        if first_valid_frame is None:
            print("  Error: Cannot determine frame size, no valid BGR frames found in 'all_video_frames'.")
            return
        height, width, layers = first_valid_frame.shape
        if layers != 3:
             print(f"  Error: Expected 3-channel BGR frames, but found {layers} channels.")
             return
        frame_size = (width, height)
    except Exception as e:
         print(f"  Error: Could not get frame dimensions from first valid frame: {e}")
         return

    # Ensure FPS is valid
    if not isinstance(fps, (int, float)) or fps <= 0:
         print(f"  Error: Invalid FPS value: {fps}. Cannot create video.")
         return

    # Choose codec - 'mp4v' is good for .mp4, 'XVID' for .avi might be more compatible sometimes
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID'
    # Ensure output path ends with .mp4 if using mp4v
    if not output_path.lower().endswith(".mp4") and fourcc == cv2.VideoWriter_fourcc(*'mp4v'):
        print(f"Warning: Output path '{output_path}' doesn't end with .mp4, but using 'mp4v' codec. Renaming to .mp4")
        output_path += ".mp4"


    out = cv2.VideoWriter(output_path, fourcc, float(fps), frame_size)

    if not out.isOpened():
         print(f"  Error: Could not open VideoWriter for path: {output_path}")
         print(f"  Check directory permissions, path validity, codec support (try 'XVID' if 'mp4v' fails), frame size: {frame_size}, FPS: {fps}")
         return

    print(f"  VideoWriter initialized (Codec: {'mp4v' if fourcc == cv2.VideoWriter_fourcc(*'mp4v') else 'XVID'}, FPS: {fps}, Size: {frame_size})")

    total_frames_written = 0
    for i, segment in enumerate(replay_segments):
        try:
             start, end = int(segment[0]), int(segment[1])
             print(f"  Writing segment {i+1}/{len(replay_segments)}: Frames {start} to {end}")

             # Validate segment indices against available frames
             if not (0 <= start < len(all_video_frames) and 0 <= end < len(all_video_frames) and start <= end):
                 print(f"    Warning: Skipping invalid segment indices [{start}, {end}] relative to {len(all_video_frames)} available frames.")
                 continue

             for frame_index in range(start, end + 1):
                 try:
                     frame = all_video_frames[frame_index]
                     # Ensure frame is valid and matches dimensions before writing
                     if frame is not None and hasattr(frame, 'shape') and frame.shape == (height, width, layers):
                          out.write(frame)
                          total_frames_written += 1
                     elif frame is None:
                          print(f"    Warning: Skipping None frame at index {frame_index}.")
                     else:
                          print(f"    Warning: Skipping frame at index {frame_index} due to unexpected shape {getattr(frame, 'shape', 'N/A')} (expected {(height, width, layers)}).")
                 except IndexError:
                      print(f"    Error: Frame index {frame_index} out of bounds while writing segment. Stopping segment.")
                      break # Stop processing this segment if an index is bad during iteration

        except (ValueError, TypeError):
             print(f"  Warning: Invalid segment format: {segment}. Skipping segment.")
             continue

    out.release() # Crucial to finalize the video file
    print(f"\nFinished exporting.")
    if total_frames_written > 0:
         print(f"Replay video with {total_frames_written} frames saved to: {output_path}")
    else:
         print(f"No frames were written to: {output_path}")


#################################
#### Replay Extraction Model ####
#################################

if __name__ == "__main__":
    # --- Configuration ---
    video_file = "test1.mp4" # INPUT: Replace with your video file path
    ground_truth_file = "ground_truth_annotations.json" # INPUT: Path to annotations file (optional)
    output_replay_video = "test1_replays_cuda_pure_gpu_output.mp4" # INPUT: Desired output video file path

    # --- Start Timer ---
    start_time = time.time()

    # --- Pre-checks ---
    if not cuda_available:
        print("FATAL ERROR: CUDA is not available or not detected by OpenCV. This script requires CUDA.")
        exit(1)
    if not os.path.exists(video_file):
         print(f"FATAL ERROR: Video file not found at {video_file}")
         exit(1)

    perform_evaluation = os.path.exists(ground_truth_file)
    if not perform_evaluation:
        print(f"INFO: Ground truth file not found at {ground_truth_file}. Evaluation will be skipped.")

    # --- Processing Pipeline ---
    all_frames = [] # Define scope outside try block
    video_fps = 0.0
    try:
        # 1. Extract Frames (CPU)
        all_frames, video_fps = extract_frames(video_file)
        if not all_frames or video_fps <= 0:
            print("FATAL ERROR: Frame extraction failed or yielded invalid FPS/no frames. Exiting.")
            exit(1)
        total_video_frames = len(all_frames)
        extract_end_time = time.time()
        print(f"Frame Extraction Time: {extract_end_time - start_time:.2f} seconds")

        # 2. Detect Gradual Transitions (CUDA)
        print("\n--- Starting GT Detection (CUDA) ---")
        gt_segments = detect_gradual_transitions_cuda(all_frames)
        gt_detect_end_time = time.time()
        print(f"GT Detection Time: {gt_detect_end_time - extract_end_time:.2f} seconds")

        # 3. Identify Candidate Replay Segments (CPU logic)
        print("\n--- Starting Candidate Identification ---")
        candidate_segments = identify_candidate_replay_segments(gt_segments)
        candidate_id_end_time = time.time()
        print(f"Candidate Identification Time: {candidate_id_end_time - gt_detect_end_time:.2f} seconds")

        # 4. Detect Score Caption Absence in Candidates (CUDA)
        final_replay_segments = []
        print("\n--- Starting SC Absence Detection (CUDA - Pure GPU Average) ---")
        sc_detection_start_time = time.time()
        if not candidate_segments:
            print("  No candidate segments found. Skipping SC detection.")
        else:
            for i, (start, end) in enumerate(candidate_segments):
                print(f"\n--- Processing Candidate RS {i+1}/{len(candidate_segments)}: Frames {start} to {end} ---")
                # Validate segment indices relative to extracted frames
                if not (0 <= start < total_video_frames and 0 <= end < total_video_frames and start <= end):
                     print(f"  Skipping invalid candidate segment indices: {start}-{end} (out of {total_video_frames} frames)")
                     continue

                # Get the segment frames (NumPy arrays)
                # Ensure slicing does not create an empty list if start==end+1
                segment_frames_cpu = all_frames[start : end + 1] if start <= end else []
                if not segment_frames_cpu or all(f is None for f in segment_frames_cpu): # Check if list is empty or all frames are None
                     print("  Skipping candidate segment: Frame slice is empty or contains only invalid frames.")
                     continue

                try:
                    # Call the CUDA SC detection function (now with pure GPU averaging)
                    is_sc_absent = detect_score_caption_absence_cuda(segment_frames_cpu)
                except pytesseract.TesseractNotFoundError:
                     # Should have been caught earlier, but is fatal here too.
                     print("FATAL ERROR: Tesseract not found during SC detection. Exiting.")
                     exit(1) # Stop script
                except RuntimeError as e: # Catch CUDA not available or other runtime errors
                     print(f"FATAL ERROR during SC Detection: {e}. Exiting.")
                     exit(1) # Stop script
                except Exception as e:
                     print(f"Error during SC detection for segment {start}-{end}: {e}. Assuming SC present (discarding).")
                     is_sc_absent = False # Treat errors as SC present to be safe

                # Process result
                if is_sc_absent:
                    print(f"  --> RESULT: SC Absent. Adding segment [{start}, {end}] as Final Replay.")
                    final_replay_segments.append((start, end))
                else:
                    print(f"  --> RESULT: SC Present (or error occurred). Discarding segment [{start}, {end}].")

                # Optional: Explicit garbage collection attempt if memory is tight
                # import gc
                # gc.collect()
                # cv2.cuda.current_context().collectGarbage() # Release unused GPU memory

        sc_detection_end_time = time.time()
        print(f"\nFinished SC detection. Identified {len(final_replay_segments)} final replay segments.")
        print(f"SC Detection Time: {sc_detection_end_time - sc_detection_start_time:.2f} seconds")
        if final_replay_segments:
            print("Final Replay Segments (Frame Indices):")
            for fs_start, fs_end in final_replay_segments:
                 print(f"  - {fs_start} to {fs_end}")
        else:
            print("  No final replay segments identified.")


        # 5. Evaluation (CPU)
        eval_start_time = time.time()
        if perform_evaluation:
            print("\n--- Starting Evaluation ---")
            evaluation_metrics = evaluate_replay_detection(final_replay_segments, ground_truth_file, total_video_frames)
            if evaluation_metrics:
                 print("\nEvaluation completed successfully.")
            else:
                 print("\nEvaluation failed or produced no results.")
        else:
            print(f"\nSkipping evaluation step (Ground truth file not found or specified).")
        eval_end_time = time.time()
        if perform_evaluation: print(f"Evaluation Time: {eval_end_time - eval_start_time:.2f} seconds")

        # 6. Export Video (CPU)
        export_start_time = time.time()
        if final_replay_segments and all_frames and video_fps > 0:
            print("\n--- Starting Replay Video Export ---")
            export_replays_to_video(
                replay_segments=final_replay_segments,
                all_video_frames=all_frames, # Pass the original NumPy frames
                fps=video_fps,
                output_path=output_replay_video
            )
        else:
            if not final_replay_segments:
                 print("\nSkipping replay export: No final replay segments identified.")
            else:
                 print("\nSkipping replay export: Frame data or FPS missing/invalid.")
        export_end_time = time.time()
        if final_replay_segments: print(f"Export Time: {export_end_time - export_start_time:.2f} seconds")


    except Exception as main_e:
         print(f"\n--- A critical error occurred during processing ---")
         print(f"Error Type: {type(main_e).__name__}")
         print(f"Error Details: {main_e}")
         # Optionally print traceback
         # import traceback
         # traceback.print_exc()
         print("Attempting cleanup...")

    finally:
        # --- Cleanup ---
        # Explicitly delete large frame list to free memory
        if 'all_frames' in locals():
            del all_frames
            # gc.collect() # Optional extra garbage collection

        # Release CUDA resources (Important especially if run in loops or long sessions)
        if cuda_available:
            try:
                # If you created specific CUDA objects like streams, filters outside functions, delete them here.
                # For GpuMats created inside functions, Python's garbage collector should handle them
                # if references are properly managed (e.g., using 'del').
                # Calling resetDevice() is a more forceful cleanup but might affect other CUDA contexts.
                # cv2.cuda.resetDevice()
                print("CUDA context cleanup (if needed) would happen here or automatically.")
            except Exception as cleanup_e:
                print(f"Warning: Error during CUDA cleanup: {cleanup_e}")

        # --- Record total end time ---
        total_end_time = time.time()
        print(f"\nProcessing finished. Total execution time: {total_end_time - start_time:.2f} seconds")