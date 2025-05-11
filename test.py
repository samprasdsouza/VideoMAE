import cv2
import os

def video_to_frames(video_path, output_folder, frame_interval=1):
    """
    Convert video to frames and check their sizes
    
    Args:
        video_path (str): Path to input video file
        output_folder (str): Directory to save frames
        frame_interval (int): Save every nth frame (1=every frame)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Info: {width}x{height} at {fps:.2f} fps, {total_frames} total frames")
    print(f"Saving every {frame_interval} frame(s) to '{output_folder}'")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video
        
        # Only process frames at the specified interval
        if frame_count % frame_interval == 0:
            # Save frame as image
            frame_file = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_file, frame)
            
            # Verify the saved image
            saved_frame = cv2.imread(frame_file)
            if saved_frame is not None:
                h, w = saved_frame.shape[:2]
                print(f"Saved frame {saved_count} ({w}x{h}) as {frame_file}")
            else:
                print(f"Warning: Failed to verify frame {saved_count}")
            
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Finished! Saved {saved_count} frames out of {total_frames} total frames")

# Example usage
if __name__ == "__main__":
    video_path = '/Users/apple/personal/CMU-research/VideoMAE/TODO-2/TODO-2.mp4'  # Remove quotes if path is pasted
    output_folder = "video_frames"
    
    # Convert every frame (change 1 to higher number to skip frames)
    video_to_frames(video_path, output_folder, frame_interval=1)