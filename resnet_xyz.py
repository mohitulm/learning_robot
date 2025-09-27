#!/usr/bin/env python3
"""
RGBD object-detect → 3D coordinates publisher.

- Subscribes: RGB (compressed), Depth (16UC1 mm or 32FC1 m), CameraInfo
- Runs Faster R-CNN ResNet50-FPN (COCO) on RGB frames
- For each detection (score >= threshold):
    • Pick bbox center (optionally top-K detections)
    • Sample median depth in a small patch around center
    • Back-project to X,Y,Z (meters) using CameraInfo
    • Publish:
        - geometry_msgs/PointStamped on --xyz_topic (default: xyz_in)
        - std_msgs/String (JSON) on --label_topic (default: detected_object_json)
- Shows an annotated window (can be disabled with --no_viz)
- Keys: [q]=quit, [s]=save frames

Notes:
- If RGB and depth resolutions differ, (u,v) are scaled into depth space.
- Depth sampling uses a robust median over a (2r+1)x(2r+1) patch; set --r to tune.
"""
import os, time, argparse, json
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

# Torch / torchvision (same model family as your resnet.py)
import torch
import torchvision.transforms as T
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def _safe_class_name(i: int) -> str:
    if 0 <= i < len(COCO_INSTANCE_CATEGORY_NAMES):
        return COCO_INSTANCE_CATEGORY_NAMES[i]
    return f"class_{i}"

class RGBDDetectToXYZ(Node):
    def __init__(self, args):
        super().__init__('rgbd_detect_to_xyz')
        if not _HAS_TORCH:
            raise RuntimeError("torch/torchvision not available. Install torchvision with detection models.")

        self.bridge = CvBridge()
        self.rgb_topic   = args.rgb
        self.depth_topic = args.depth
        self.info_topic  = args.info
        self.xyz_topic   = args.xyz_topic
        self.label_topic = args.label_topic
        self.det_score   = float(args.score)
        self.topk        = int(args.topk)
        self.det_period  = 1.0 / float(args.det_fps)
        self.patch_r     = int(args.r)
        self.visualize   = not args.no_viz
        self.outdir      = args.out
        os.makedirs(self.outdir, exist_ok=True)

        # Latest frames & intrinsics
        self.rgb_bgr = None
        self.depth_m = None
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_frame_id = None

        # Publishers
        self.xyz_pub   = self.create_publisher(PointStamped, self.xyz_topic, 10)
        self.json_pub  = self.create_publisher(String, self.label_topic, 10)

        # Subs
        if self.rgb_topic:
            self.create_subscription(CompressedImage, self.rgb_topic, self.cb_rgb, 1)
        self.create_subscription(Image, self.depth_topic, self.cb_depth, 1)
        self.create_subscription(CameraInfo, self.info_topic, self.cb_info, 1)

        # Visualization windows
        if self.visualize:
            self.win_rgb   = "Detections (publishing XYZ automatically)"
            self.win_depth = "Depth"
            cv2.namedWindow(self.win_rgb,   cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.win_depth, cv2.WINDOW_NORMAL)

        # Torch model (match your resnet.py model family)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using torch device: {self.device}")
        try:
            # Newer torchvision
            self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        except Exception:
            # Older API fallback
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device).eval()
        self.tf = T.ToTensor()

        # Inference throttle
        self._last_infer_t = 0.0

        # HUD
        self.get_logger().info(f"RGB   : {self.rgb_topic or '(none)'}")
        self.get_logger().info(f"Depth : {self.depth_topic}")
        self.get_logger().info(f"Info  : {self.info_topic}")
        self.get_logger().info(f"XYZ pub      → {self.xyz_topic}")
        self.get_logger().info(f"Label JSON → {self.label_topic}")
        self.get_logger().info("Keys: [s]=save frames, [q]=quit")

    # --- Callbacks ---
    def cb_info(self, msg: CameraInfo):
        k = msg.k
        self.fx, self.fy, self.cx, self.cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        self.depth_frame_id = msg.header.frame_id

    def cb_depth(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Depth convert failed: {e}")
            return

        enc = msg.encoding.lower()
        if np.issubdtype(depth.dtype, np.integer) or '16uc1' in enc or enc == 'mono16':
            self.depth_m = depth.astype(np.float32) / 1000.0
        else:
            self.depth_m = depth.astype(np.float32)

        if self.visualize:
            finite = self.depth_m[np.isfinite(self.depth_m) & (self.depth_m > 0)]
            if finite.size:
                vmax = np.percentile(finite, 99.5)
                dvis = np.clip(self.depth_m, 0, vmax)
                dnorm = cv2.normalize(dvis, None, 0, 255, cv2.NORM_MINMAX)
                dcolor = cv2.applyColorMap(dnorm.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imshow(self.win_depth, dcolor)
            else:
                cv2.imshow(self.win_depth, np.zeros((360,640,3), np.uint8))
            self._keys()

    def cb_rgb(self, msg: CompressedImage):
        # Decode to BGR
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().warn(f"RGB convert failed: {e}")
            return
        self.rgb_bgr = img

        # Draw base
        vis = img.copy()

        # Inference throttle + prerequisites
        now = time.time()
        ready = (self.depth_m is not None) and (self.fx is not None)
        if ready and (now - self._last_infer_t >= self.det_period):
            self._last_infer_t = now
            self._run_detection_and_publish(vis)

        if self.visualize:
            cv2.imshow(self.win_rgb, vis)
            self._keys()

    # --- Core ---
    def _run_detection_and_publish(self, vis_bgr):
        # Prepare tensor (torchvision expects RGB)
        rgb = cv2.cvtColor(self.rgb_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.tf(rgb).to(self.device)

        with torch.no_grad():
            out = self.model([tensor])[0]

        boxes  = out.get("boxes",  [])
        labels = out.get("labels", [])
        scores = out.get("scores", [])

        if boxes is None or len(boxes) == 0:
            return

        # Filter by score
        keep = [i for i, s in enumerate(scores) if float(s) >= self.det_score]
        if not keep:
            return

        # Sort by score desc and take topk
        keep.sort(key=lambda i: float(scores[i]), reverse=True)
        keep = keep[:self.topk]

        H_rgb, W_rgb = self.rgb_bgr.shape[:2]
        H_d,   W_d   = self.depth_m.shape[:2]
        sx = W_d / float(W_rgb)
        sy = H_d / float(H_rgb)

        for i in keep:
            x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
            cx_rgb = int(0.5 * (x1 + x2))
            cy_rgb = int(0.5 * (y1 + y2))
            # Scale into depth coordinates if needed
            cx = int(round(cx_rgb * sx))
            cy = int(round(cy_rgb * sy))

            Z = self._robust_depth(cy, cx)  # meters
            if not (np.isfinite(Z) and Z > 0):
                continue

            X = (cx - self.cx) * Z / self.fx
            Y = (cy - self.cy) * Z / self.fy

            # Publish: PointStamped
            pt = PointStamped()
            pt.header.stamp = self.get_clock().now().to_msg()
            pt.header.frame_id = self.depth_frame_id or 'camera'
            pt.point.x, pt.point.y, pt.point.z = float(X), float(Y), float(Z)
            self.xyz_pub.publish(pt)

            # Publish: JSON with label/score/pixel & XYZ
            cls_idx = int(labels[i])
            label   = _safe_class_name(cls_idx)
            score   = float(scores[i])
            payload = {
                "label": label,
                "score": round(score, 3),
                "u": int(cx_rgb),
                "v": int(cy_rgb),
                "xyz_m": {"x": float(X), "y": float(Y), "z": float(Z)},
                "frame_id": pt.header.frame_id
            }
            self.json_pub.publish(String(data=json.dumps(payload)))

            # Draw overlay
            if self.visualize:
                c1 = (int(x1), int(y1)); c2 = (int(x2), int(y2))
                cv2.rectangle(vis_bgr, c1, c2, (0, 0, 255), 2)
                txt = f"{label} {score:.2f}  Z={Z:.2f}m"
                cv2.putText(vis_bgr, txt, (int(x1)+5, max(20, int(y1)-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
                cv2.circle(vis_bgr, (cx_rgb, cy_rgb), 5, (0, 255, 255), -1)

    def _robust_depth(self, y, x):
        """Median depth in a (2r+1)x(2r+1) patch around (x,y)."""
        r = self.patch_r
        H, W = self.depth_m.shape[:2]
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        patch = self.depth_m[y0:y1, x0:x1]
        good = patch[np.isfinite(patch) & (patch > 0)]
        if good.size == 0:
            return float('nan')
        return float(np.median(good))

    # --- Save / Quit ---
    def _keys(self):
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            rclpy.shutdown()
        elif k == ord('s'):
            self._save_frames()

    def _save_frames(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        if self.rgb_bgr is not None:
            cv2.imwrite(os.path.join(self.outdir, f"rgb_{ts}.png"), self.rgb_bgr)
        if self.depth_m is not None:
            cv2.imwrite(os.path.join(self.outdir, f"depth_raw_mm_{ts}.png"),
                        (self.depth_m * 1000.0).astype(np.uint16))
            finite = self.depth_m[np.isfinite(self.depth_m) & (self.depth_m > 0)]
            vmax = np.percentile(finite, 99.5) if finite.size else 5.0
            dvis = np.clip(self.depth_m, 0, vmax)
            dnorm = cv2.normalize(dvis, None, 0, 255, cv2.NORM_MINMAX)
            dcolor = cv2.applyColorMap(dnorm.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.outdir, f"depth_vis_{ts}.png"), dcolor)
        self.get_logger().info(f"Saved to {self.outdir}")

def parse_args():
    ap = argparse.ArgumentParser(description="Auto-detect objects and publish 3D coordinates.")
    ap.add_argument("--rgb",   default="/oakd/rgb/image_raw/compressed",
                    help="RGB topic (sensor_msgs/CompressedImage).")
    ap.add_argument("--depth", default="/oakd/stereo/image_raw",
                    help="Depth topic (sensor_msgs/Image; 16UC1 in mm or 32FC1 in m).")
    ap.add_argument("--info",  default="/oakd/stereo/camera_info",
                    help="CameraInfo matching the depth alignment.")
    ap.add_argument("--xyz_topic",   default="xyz_in",
                    help="geometry_msgs/PointStamped topic for XYZ (meters).")
    ap.add_argument("--label_topic", default="detected_object_json",
                    help="std_msgs/String topic with JSON payload (label/score/u/v/xyz).")
    ap.add_argument("--score", type=float, default=0.5, help="Confidence threshold.")
    ap.add_argument("--topk",  type=int,   default=1,   help="Publish top-K detections per frame.")
    ap.add_argument("--det_fps", type=float, default=2.0,
                    help="Max detection rate (Hz) to avoid overloading CPU.")
    ap.add_argument("--r",     type=int,   default=3,
                    help="Depth median patch radius (pixels) around bbox center in depth space.")
    ap.add_argument("--no_viz", action="store_true", help="Disable OpenCV display windows.")
    ap.add_argument("--out",   default="captures", help="Folder to save frames on [s].")
    return ap.parse_args()

def main():
    args = parse_args()
    rclpy.init()
    node = RGBDDetectToXYZ(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
