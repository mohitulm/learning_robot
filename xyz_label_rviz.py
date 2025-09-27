#!/usr/bin/env python3
# save_detections_to_yaml.py
# Run as: python3 save_detections_to_yaml.py --yaml /tmp/object_locations.yaml --target-frame map
import os, json, math, argparse
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_point

try:
    import yaml
except Exception as e:
    raise SystemExit("Missing dependency 'PyYAML'. Install with: pip install pyyaml") from e


class SaveDetections(Node):
    """
    Subscribes:
      - JSON detections on --label-json-topic (std_msgs/String) with:
        {"label": "...", "xyz_m":{"x":..,"y":..,"z":..}, "frame_id":"..."}
      - (optional) PointStamped on --point-topic (requires --fallback-label if you use it)
    Transforms to --target-frame (if provided) and writes YAML as:
        label: { w: <quat_w>, x: <pos_x>, y: <pos_y>, z: <quat_z> }
    Skips writing if label already exists in YAML.
    """

    def __init__(self, args):
        super().__init__('save_detections_to_yaml')
        self.args = args

        # TF setup (only used if target frame is provided)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # YAML store
        os.makedirs(os.path.dirname(args.yaml) or '.', exist_ok=True)
        self.db = self._load_yaml(args.yaml)

        # Subs
        self.create_subscription(String, args.label_json_topic, self.cb_label_json, 10)
        if args.point_topic:
            self.create_subscription(PointStamped, args.point_topic, self.cb_point, 10)

        self.get_logger().info(f"YAML -> {args.yaml}")
        self.get_logger().info(f"Listening JSON on: {args.label_json_topic}")
        if args.point_topic:
            self.get_logger().info(f"Listening PointStamped on: {args.point_topic} (label='{args.fallback_label}')")
        if args.target_frame:
            self.get_logger().info(f"Transforming to frame: {args.target_frame}")
        self.get_logger().info(f"Skip existing labels: {args.skip_existing}")

    # ---------- callbacks ----------
    def cb_label_json(self, msg: String):
        try:
            data = json.loads(msg.data)
            label = str(data['label']).strip()
            xyz   = data.get('xyz_m') or {}
            x, y, z = float(xyz['x']), float(xyz['y']), float(xyz['z'])
            frame_id = (data.get('frame_id') or 'camera').strip()
        except Exception as e:
            self.get_logger().warn(f"Bad JSON; skipping. Error: {e}")
            return

        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = frame_id or 'camera'
        pt.point.x, pt.point.y, pt.point.z = x, y, z
        self._process(label, pt)

    def cb_point(self, msg: PointStamped):
        label = self.args.fallback_label or 'point'
        self._process(label, msg)

    # ---------- core ----------
    def _process(self, label: str, msg: PointStamped):
        # Transform if requested
        pt = msg
        if self.args.target_frame:
            try:
                tf = self.tf_buffer.lookup_transform(self.args.target_frame, msg.header.frame_id, rclpy.time.Time())
                pt = do_transform_point(msg, tf)
            except Exception as e:
                self.get_logger().warn(f"TF unavailable ({msg.header.frame_id}->{self.args.target_frame}); saving raw. {e}")

        # Decide yaw
        if self.args.yaw_face_origin:
            yaw = math.atan2(pt.point.y, pt.point.x)  # face away from origin
        else:
            yaw = math.radians(self.args.yaw_deg)     # fixed yaw (default 0)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        # YAML entry (match your mapper's shape)
        entry = {'w': float(qw), 'x': float(pt.point.x), 'y': float(pt.point.y), 'z': float(qz)}

        # Skip if exists
        if self.args.skip_existing and label in self.db:
            self.get_logger().info(f"Skip: '{label}' already in YAML.")
            return

        # Save
        self.db[label] = entry
        self._write_yaml(self.args.yaml, self.db)
        self.get_logger().info(f"Saved '{label}' -> {entry}")

    # ---------- yaml io ----------
    @staticmethod
    def _load_yaml(path: str):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return {}
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}

    @staticmethod
    def _write_yaml(path: str, data: dict):
        with open(path, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)


def parse_args():
    ap = argparse.ArgumentParser(description="Subscribe to detections and save label->pose YAML (skip existing).")
    ap.add_argument('--yaml', default='object_locations.yaml', help='Path to YAML file to write.')
    ap.add_argument('--label-json-topic', default='detected_object_json', help='Topic with std_msgs/String JSON.')
    ap.add_argument('--point-topic', default='', help='Optional PointStamped topic (e.g., xyz_in).')
    ap.add_argument('--fallback-label', default='point', help='Label to use for PointStamped messages.')
    ap.add_argument('--target-frame', default='', help="Transform detections into this frame (e.g., 'map').")
    ap.add_argument('--skip-existing', action='store_true', default=True, help='Skip writing if label exists.')
    ap.add_argument('--no-skip-existing', dest='skip_existing', action='store_false', help='Overwrite labels.')
    ap.add_argument('--yaw-deg', type=float, default=0.0, help='Fixed yaw in degrees (if not facing origin).')
    ap.add_argument('--yaw-face-origin', action='store_true', help='Use yaw = atan2(y, x) instead of --yaw-deg.')
    return ap.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = SaveDetections(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
