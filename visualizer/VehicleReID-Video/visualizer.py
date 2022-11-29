import os
import cv2
import argparse
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(
            prog = 'Demo',
            description = 'Demo multi-camera tracking',
            epilog = 'Text at the bottom of help')

    parser.add_argument('-d', '--data', default='./AIC22_Track1_MTMC_Tracking/test/S06')              # path to data directory
    parser.add_argument('-p', '--pred', default='./tests/sample-test.txt')         # path to prediction file
    parser.add_argument('-o', '--out', default='./pred_visual_results')             # path to output folder

    parser.add_argument('-w', '--write', action='store_true')    # on/off flag
    
    return parser.parse_args


def get_captures(data_path, vids):
    """
        Return a list of camera captures based on video names
        Argument:
            data_path (str) : path to data directory
            vid (List[str]) : list of video strings
    """
    caps = []
    for vid in vids:
        vid_path = os.path.join(data_path, vid)
        cap = cv2.VideoCapture(vid_path)
        caps.append(cap)
    return caps

def read_tracklets(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()[:-1]
    
    tracklets = {}
    for line in lines:
        cam, tid, frame, xmin, ymin, width, height, _, _ = line[:-1].split(' ')
        
        if cam not in tracklets:
            tracklets[cam] = {}
        if frame not in  tracklets[cam]:
            tracklets[cam][frame] = []
        tracklets[cam][frame].append((int(tid), int(xmin), int(ymin), int(width), int(height)))

    return tracklets

def random_color():
    return list(np.random.random(size=3) * 256)

def draw_vehicle_bbox(frame, tlwh, color=(0,0,0), text_thickness=4):
    """ Draw a bounding box tagged with label on an input image.
    Args: 
        frame : (ndarray)
            Image to be drawn boxes on.
        tlwh : (ndarray)
            Bounding box coordinates of format `(min x, min y, width, height)`.
        color : (tuple)
            Bounding box color.
        text_thickness : (int)
            Bounding box & letter thickness.
    """

    x, y, w, h = tlwh
    pt1 = int(x), int(y)
    pt2 = int(x + w), int(y + h)
    cv2.rectangle(frame, pt1, pt2, color, text_thickness)
    
    return


def draw_vehicle_label(frame, tlwh, label, color=(127,127,127), text_thickness=6):
    """ Draw a bounding box tagged with label on an input image.
    Args: 
        frame : (ndarray)
            Image to be drawn boxes on.
        tlwh : (ndarray)
            Bounding box coordinates of format `(top-left x, top-left y, width, height)`.
        label : (str)
            The label to be annotated.
        color : (tuple)
            Bounding box color.
        text_thickness : (int)
            Bounding box & letter thickness.
    """

    x, y, w, h = tlwh

    def get_text_position(text, font_style, font_size, font_thickness, text_margin=4):
        # get boundary of this text
        textsize = cv2.getTextSize(text, font_style, font_size, font_thickness)[0]

        label_pt1 = int(x), int(y-textsize[1]-2*text_margin)
        label_pt2 = int(x+textsize[0]+2*text_margin), int(y)
        text_pt = int(x+text_margin), int(y-0.5*textsize[1]-0.5*text_margin)

        return label_pt1, label_pt2, text_pt

    if label is not None:
        vehicle_text = str(label)
        font_style = cv2.FONT_HERSHEY_DUPLEX
        font_size = 1

        label_pt1, label_pt2, text_pt = get_text_position(
            text=vehicle_text, font_style=font_style, 
            font_size=font_size, font_thickness=text_thickness)

        # cv2.rectangle(frame, label_pt1, label_pt2, color, -1)
        cv2.putText(frame, str(label), text_pt, font_style, font_size, color, text_thickness)

    return


def post_process(tracklets, cameras):
    track_cam, track_coloring = {}, {}
    for cam, frame_tracks in tracklets.items():
        if cam not in cameras:
            continue
        for frame, tracks in frame_tracks.items():
            for track in tracks:
                tid = track[0]
                if tid not in track_cam:
                    track_cam[tid] = [cam]
                    track_coloring[tid] = random_color()

                if cam != track_cam[tid][-1]:
                    track_cam[tid].append(cam)

    return track_cam, track_coloring


if __name__ == '__main__':
    args = get_arguments()

    cameras = ['41', '42', '43', '44']# , 'c045', 'c046']
    captures = get_captures(args.data, 
                        ['c041/vdo.avi', 'c042/vdo.avi', 'c043/vdo.avi',
                         'c044/vdo.avi', 'c045/vdo.avi', 'c046/vdo.avi'])
    
    if args.write:
        writer = cv2.VideoWriter(os.path.join(args.out, 'result.avi'), 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, (1320, 940))

    # tracked_boxes = ['']
    tracklets = read_tracklets(args.pred) # camera : frame : {(id, bounding box)}
    track_cam, track_coloring = post_process(tracklets, cameras)

    # Video streaming
    frame_id = 1
    while True:
        frames = []; stream = False
        for i in range(len(cameras)):
            cam = cameras[i]
            cap = captures[i]

            ret, frame = cap.read()

            try:
                tracklets_camframe = tracklets[cam][str(frame_id)]
                for tracklet in tracklets_camframe:
                    tid, xmin, ymin, width, height = tracklet
                    color = track_coloring[tid]
                    if len(track_cam[tid]) == 1:
                        continue

                    draw_vehicle_bbox(frame, (xmin, ymin, width, height), color)
                    draw_vehicle_label(frame, (xmin, ymin, width, height), str(tid), color, text_thickness=2)
            except:
                pass

            if not ret:
                frame = np.zeros(((450, 640, 3)), dtype=np.uint8)
            else:
                stream = True
                frame = cv2.resize(frame, (640, 450))

            frame = cv2.copyMakeBorder(frame,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
            frames.append(frame)
        
        frame_id += 1    
        
        # Concatenation of frames
        upper = cv2.hconcat([frames[1], frames[0]])
        lower = cv2.hconcat([frames[2], frames[3]])
        final_frame = cv2.vconcat([upper, lower])
        if args.write:
            writer.write(final_frame)
        if not stream:
            break
        
        cv2.imshow('videos', final_frame)
        __key__ = cv2.waitKey(1)
        if __key__ == ord('q'):
            break

    cv2.destroyAllWindows()
    if args.write:
        writer.release()

