import cv2
import numpy as np
import os

def smooth(trajectory, window=30):
    smoothed_trajectory = np.copy(trajectory)

    # For the x, y, and angle trajectories:
    for i in range(3):
        # Define the filter
        window_size = 2 * window + 1
        f = np.ones(window_size)/window_size

        # Add padding to the boundaries
        trajectory_pad = np.lib.pad(trajectory[:,i], (window, window), 'edge')

        # Apply 1d convolution in time
        trajectory_smoothed = np.convolve(trajectory_pad, f, mode='same')
        smoothed_trajectory[:,i] = trajectory_smoothed[window:-window]

    return smoothed_trajectory

def zoom_in(frame, zoom=1.2):
    """ Zoom in while keeping the image centered """
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, zoom)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='*')
    parser.add_argument('--smoothing-window', type=int,   default=30)
    parser.add_argument('--max-corners',      type=int,   default=200)
    parser.add_argument('--quality-level',    type=float, default=0.01)
    parser.add_argument('--zoom',             type=float, default=1.2)
    parser.add_argument('--min-distance',     type=int,   default=30)
    parser.add_argument('--block-size',       type=int,   default=3)
    parser.add_argument('--show-frames',      action='store_true')
    args = parser.parse_args()

    if len(args.source) == 0:
        sources = ["./ThanksgivingOnTheL_Train.mp4"]
    else:
        sources = args.source

    for source in sources:
        if not os.path.isfile(source):
            print(f"{source} is not a valid file.")
            continue

        video = cv2.VideoCapture(source)

        # Get video dimensions and total number of frames.
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))

        # Use mp4v as the output video codec.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outfile = f'{os.path.splitext(source)[0]}_stabilized.mp4'
        out = cv2.VideoWriter(outfile, fourcc, fps, (w*2, h))

        transforms = []
        last_frame_bw = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)
        plot_pts=list()
        i=0
        while True:
          # Detect feature points in previous frame
          features = cv2.goodFeaturesToTrack(
                  last_frame_bw,
                  maxCorners=args.max_corners,
                  qualityLevel=args.quality_level,
                  minDistance=args.min_distance,
                  blockSize=args.block_size)

          success, frame = video.read()
          if not success:
            break

          # Convert to grayscale
          frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

          # Track features identified in the last frame.
          new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                  last_frame_bw,
                  frame_bw,
                  features,
                  None)

          # Using only feature points that exist in both the current
          # and previous frame, find transformation matrix between
          # feature positions in previous and current frames.
          valid, _ = np.where(status == 1)
          m, _ = cv2.estimateAffine2D(features[valid], new_features[valid])
          plot_pts.append(new_features[valid])
          # Translation and angle of rotation
          dx = m[0,2]
          dy = m[1,2]
          da = np.arctan2(m[1,0], m[0,0])

          transforms += [np.array([dx,dy,da])]

          # Move to next frame
          last_frame_bw = frame_bw

        # Integrate frame-to-frame transforms
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = smooth(trajectory)

        transforms_smooth = transforms + smoothed_trajectory - trajectory

        # Reset stream to first frame
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Write n_frames-1 transformed frames
        for dx, dy, da in transforms_smooth:
          # Read next frame
          success, frame = video.read()
          if not success:
            break

          # Make a transformation matrix out of smooth transform
          m = np.array([
              [np.cos(da), -np.sin(da), dx],
              [np.sin(da),  np.cos(da), dy],
          ])

          # Apply affine wrapping to the given frame
          frame_stabilized = cv2.warpAffine(frame, m, (w,h))

          # Fix border artifacts
          frame_stabilized = zoom_in(frame_stabilized, zoom=args.zoom)
          for pt in plot_pts[i]:
            cv2.circle(frame, tuple(pt[0]),3,(0,255,0),2)
          i+=1
          # Write the frame to the file
          frame_out = cv2.hconcat([frame, frame_stabilized])

          # If the image is too big, resize it.
          if(frame_out.shape[1] > 1920):
            frame_out = cv2.resize(
                    frame_out,
                    (frame_out.shape[1]/2, frame_out.shape[0]/2)
            )

          if args.show_frames:
              cv2.imshow("Before | After", frame_out)
              cv2.waitKey(30)

          out.write(frame_out)
        cv2.destroyAllWindows()
        out.release()
