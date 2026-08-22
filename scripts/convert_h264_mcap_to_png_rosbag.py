#!/usr/bin/env python3
"""Convert cam1/cam2 H.264 MCAP streams and IMU into one ROS1 JPEG bag.

Two camera worker threads decode cam1 and cam2 in parallel.  The main thread
merges their results with IMU messages by the original MCAP log timestamp and
writes a standard ROS1 bag v2 file.  Bag-level compression defaults to none:
the rosbags LZ4 output produced by the previous version was not readable by
ROS Noetic's roslz4 implementation on this system.
"""

from __future__ import annotations

import argparse
import heapq
import queue
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

try:
    import av
    import cv2
    import numpy as np
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore
except ImportError as exc:  # pragma: no cover - friendly command-line error
    raise SystemExit(
        f"Missing Python dependency: {exc}. Install mcap, mcap-ros2-support, "
        "av, opencv-python-headless and rosbags."
    ) from exc


CAMERA_TOPICS = {
    "cam1": "/cam1/image/compressed",
    "cam2": "/cam2/image/compressed",
}
IMU_TOPIC = "/imu/data_raw"
COMPRESSED_IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"
IMU_TYPE = "sensor_msgs/msg/Imu"
NANOSECONDS = 1_000_000_000


@dataclass
class InputRecord:
    source: str
    channel: Any
    message: Any
    decoded: Any


@dataclass
class PendingFrame:
    message: Any
    decoded: Any


@dataclass
class CameraOutput:
    source: str
    message: Any
    decoded: Any
    jpeg: bytes


@dataclass
class WorkerFailure:
    source: str
    error: BaseException


QueueItem = CameraOutput | WorkerFailure | None


def find_mcap(source_dir: Path, source_name: str) -> Path:
    expected = source_dir / source_name / f"{source_name}_0.mcap"
    if expected.is_file():
        return expected

    candidates = sorted((source_dir / source_name).glob("*.mcap"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No MCAP file found under {source_dir / source_name}")
    raise RuntimeError(
        f"More than one MCAP file found under {source_dir / source_name}; "
        "this script expects one file per sensor"
    )


def validate_schema(path: Path, type_name: str) -> None:
    with path.open("rb") as stream:
        summary = make_reader(stream).get_summary()
    if summary is None:
        raise RuntimeError(f"MCAP has no summary: {path}")
    if not any(schema.name == type_name for schema in summary.schemas.values()):
        raise RuntimeError(f"Schema {type_name!r} was not found in {path}")


def get_start_time(path: Path) -> int:
    with path.open("rb") as stream:
        summary = make_reader(stream).get_summary()
    if summary is None or summary.statistics is None:
        raise RuntimeError(f"MCAP has no statistics: {path}")
    return summary.statistics.message_start_time


def iter_records(path: Path, source: str) -> Iterator[InputRecord]:
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages():
            yield InputRecord(source, channel, message, decoded)


class H264ToJpeg:
    """Stateful decoder for one camera stream."""

    def __init__(self, jpeg_quality: int) -> None:
        self.codec = av.CodecContext.create("h264", "r")
        self.jpeg_quality = jpeg_quality
        self.pending: OrderedDict[int, PendingFrame] = OrderedDict()

    def submit(self, record: InputRecord) -> list[CameraOutput]:
        timestamp = record.message.log_time
        if timestamp in self.pending:
            raise RuntimeError(f"Duplicate camera log timestamp: {timestamp}")
        self.pending[timestamp] = PendingFrame(record.message, record.decoded)

        packet = av.Packet(record.decoded.data)
        # Packet timestamps make delayed/B-frames traceable to the correct ROS
        # header and bag record instead of assuming one immediate output frame.
        packet.pts = timestamp
        packet.dts = timestamp
        packet.time_base = Fraction(1, NANOSECONDS)
        return self._encode_frames(record.source, self.codec.decode(packet))

    def flush(self, source: str) -> list[CameraOutput]:
        return self._encode_frames(source, self.codec.decode(None))

    def _encode_frames(self, source: str, frames: list[Any]) -> list[CameraOutput]:
        converted = []
        for frame in frames:
            pending = self.pending.pop(frame.pts, None)
            if pending is None:
                if not self.pending:
                    raise RuntimeError("Decoder returned a frame without a ROS message")
                # Some FFmpeg builds do not propagate packet PTS. The streams
                # are display-ordered, so the oldest pending message is correct.
                _, pending = self.pending.popitem(last=False)

            image = frame.to_ndarray(format="bgr24")
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not ok:
                raise RuntimeError("OpenCV failed to encode a frame as JPEG")
            converted.append(
                CameraOutput(source, pending.message, pending.decoded, encoded.tobytes())
            )
        return converted


def put_unless_stopped(
    output: queue.Queue[QueueItem], item: QueueItem, stop: threading.Event
) -> bool:
    while not stop.is_set():
        try:
            output.put(item, timeout=0.2)
            return True
        except queue.Full:
            pass
    return False


def camera_worker(
    source: str,
    path: Path,
    output: queue.Queue[QueueItem],
    stop: threading.Event,
    stop_time: int | None,
    jpeg_quality: int,
) -> None:
    try:
        converter = H264ToJpeg(jpeg_quality)
        expected_topic = CAMERA_TOPICS[source]
        for record in iter_records(path, source):
            if stop.is_set():
                return
            if stop_time is not None and record.message.log_time > stop_time:
                break
            if record.channel.topic != expected_topic:
                raise RuntimeError(f"Unexpected {source} topic: {record.channel.topic}")
            if str(record.decoded.format).lower() not in {"h264", "h.264"}:
                raise RuntimeError(
                    f"Expected H.264 on {expected_topic}, got {record.decoded.format!r}"
                )
            for converted in converter.submit(record):
                if not put_unless_stopped(output, converted, stop):
                    return

        for converted in converter.flush(source):
            if not put_unless_stopped(output, converted, stop):
                return
        if converter.pending:
            raise RuntimeError(
                f"{source}: {len(converter.pending)} H.264 messages produced no frame"
            )
    except BaseException as exc:
        if not stop.is_set():
            put_unless_stopped(output, WorkerFailure(source, exc), stop)
    finally:
        put_unless_stopped(output, None, stop)


def next_camera_output(items: queue.Queue[QueueItem]) -> CameraOutput | None:
    item = items.get()
    if isinstance(item, WorkerFailure):
        raise RuntimeError(f"{item.source} worker failed: {item.error}") from item.error
    return item


def make_ros1_helpers() -> tuple[Any, Any]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    types = typestore.types
    Time = types["builtin_interfaces/msg/Time"]
    Header = types["std_msgs/msg/Header"]
    CompressedImage = types[COMPRESSED_IMAGE_TYPE]
    Imu = types[IMU_TYPE]
    Quaternion = types["geometry_msgs/msg/Quaternion"]
    Vector3 = types["geometry_msgs/msg/Vector3"]

    def header(source: Any, seq: int) -> Any:
        return Header(
            seq=seq,
            stamp=Time(sec=int(source.stamp.sec), nanosec=int(source.stamp.nanosec)),
            frame_id=str(source.frame_id),
        )

    def camera(source: CameraOutput, seq: int) -> Any:
        return CompressedImage(
            header=header(source.decoded.header, seq),
            format="bgr8; jpeg compressed",
            data=np.frombuffer(source.jpeg, dtype=np.uint8),
        )

    def imu(source: InputRecord, seq: int) -> Any:
        msg = source.decoded
        return Imu(
            header=header(msg.header, seq),
            orientation=Quaternion(
                x=float(msg.orientation.x),
                y=float(msg.orientation.y),
                z=float(msg.orientation.z),
                w=float(msg.orientation.w),
            ),
            orientation_covariance=np.asarray(msg.orientation_covariance, dtype=np.float64),
            angular_velocity=Vector3(
                x=float(msg.angular_velocity.x),
                y=float(msg.angular_velocity.y),
                z=float(msg.angular_velocity.z),
            ),
            angular_velocity_covariance=np.asarray(
                msg.angular_velocity_covariance, dtype=np.float64
            ),
            linear_acceleration=Vector3(
                x=float(msg.linear_acceleration.x),
                y=float(msg.linear_acceleration.y),
                z=float(msg.linear_acceleration.z),
            ),
            linear_acceleration_covariance=np.asarray(
                msg.linear_acceleration_covariance, dtype=np.float64
            ),
        )

    return typestore, (camera, imu)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("20260818-160433"),
        help="directory containing cam1/, cam2/ and imu/ (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output ROS1 .bag file (default: INPUT-cam1-cam2-imu-jpeg-q90.bag)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        choices=range(1, 101),
        default=90,
        metavar="1..100",
        help="JPEG quality (default: %(default)s)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=8,
        help="maximum prepared frames buffered per camera thread (default: %(default)s)",
    )
    parser.add_argument(
        "--bag-compression",
        choices=("none", "bz2", "lz4"),
        default="none",
        help=(
            "ROS1 bag chunk compression (default: %(default)s). "
            "Use none for ROS Noetic compatibility; lz4 is retained only for testing."
        ),
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="convert only the first N seconds (useful for a quick test)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else source_dir.with_name(
            source_dir.name + f"-cam1-cam2-imu-jpeg-q{args.jpeg_quality}.bag"
        )
    )
    if output_path.suffix != ".bag":
        raise SystemExit(f"Output must end in .bag: {output_path}")
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise SystemExit("--max-seconds must be greater than zero")
    if args.queue_size <= 0:
        raise SystemExit("--queue-size must be greater than zero")
    if output_path.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite it: {output_path}")

    inputs = {name: find_mcap(source_dir, name) for name in ("cam1", "cam2", "imu")}
    validate_schema(inputs["cam1"], COMPRESSED_IMAGE_TYPE)
    validate_schema(inputs["cam2"], COMPRESSED_IMAGE_TYPE)
    validate_schema(inputs["imu"], IMU_TYPE)
    first_time = min(get_start_time(path) for path in inputs.values())
    stop_time = (
        first_time + round(args.max_seconds * NANOSECONDS)
        if args.max_seconds is not None
        else None
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".active")
    if temporary_path.exists():
        raise SystemExit(f"Temporary output already exists: {temporary_path}")

    typestore, converters = make_ros1_helpers()
    make_camera_message, make_imu_message = converters
    queues = {
        name: queue.Queue(maxsize=args.queue_size) for name in CAMERA_TOPICS
    }
    stop = threading.Event()
    threads = [
        threading.Thread(
            target=camera_worker,
            name=f"{name}-h264-to-jpeg",
            args=(
                name,
                inputs[name],
                queues[name],
                stop,
                stop_time,
                args.jpeg_quality,
            ),
            daemon=True,
        )
        for name in CAMERA_TOPICS
    ]
    for thread in threads:
        thread.start()

    counts = {"cam1": 0, "cam2": 0, "imu": 0}
    imu_iterator = iter(iter_records(inputs["imu"], "imu"))

    try:
        writer = Writer(temporary_path)
        if args.bag_compression == "bz2":
            writer.set_compression(Writer.CompressionFormat.BZ2)
        elif args.bag_compression == "lz4":
            writer.set_compression(Writer.CompressionFormat.LZ4)
        with writer:
            connections = {
                "cam1": writer.add_connection(
                    CAMERA_TOPICS["cam1"], COMPRESSED_IMAGE_TYPE, typestore=typestore
                ),
                "cam2": writer.add_connection(
                    CAMERA_TOPICS["cam2"], COMPRESSED_IMAGE_TYPE, typestore=typestore
                ),
                "imu": writer.add_connection(IMU_TOPIC, IMU_TYPE, typestore=typestore),
            }

            heap: list[tuple[int, int, str, CameraOutput | InputRecord]] = []
            serial = 0
            for name in CAMERA_TOPICS:
                item = next_camera_output(queues[name])
                if item is not None:
                    heapq.heappush(heap, (item.message.log_time, serial, name, item))
                    serial += 1

            try:
                imu_head = next(imu_iterator)
            except StopIteration:
                imu_head = None
            if imu_head is not None and (
                stop_time is None or imu_head.message.log_time <= stop_time
            ):
                heapq.heappush(heap, (imu_head.message.log_time, serial, "imu", imu_head))
                serial += 1

            while heap:
                _, _, source, item = heapq.heappop(heap)
                if source == "imu":
                    assert isinstance(item, InputRecord)
                    if item.channel.topic != IMU_TOPIC:
                        raise RuntimeError(f"Unexpected IMU topic: {item.channel.topic}")
                    ros_message = make_imu_message(item, counts["imu"])
                    raw = typestore.serialize_ros1(ros_message, IMU_TYPE)
                    writer.write(connections["imu"], item.message.log_time, raw)
                    counts["imu"] += 1
                    try:
                        next_item = next(imu_iterator)
                    except StopIteration:
                        next_item = None
                    if next_item is not None and (
                        stop_time is None or next_item.message.log_time <= stop_time
                    ):
                        heapq.heappush(
                            heap,
                            (next_item.message.log_time, serial, "imu", next_item),
                        )
                        serial += 1
                    continue

                assert isinstance(item, CameraOutput)
                ros_message = make_camera_message(item, counts[source])
                raw = typestore.serialize_ros1(ros_message, COMPRESSED_IMAGE_TYPE)
                writer.write(connections[source], item.message.log_time, raw)
                counts[source] += 1
                if counts[source] % 250 == 0:
                    print(
                        f"Converted cam1={counts['cam1']}, cam2={counts['cam2']}, "
                        f"imu={counts['imu']}",
                        flush=True,
                    )
                next_item = next_camera_output(queues[source])
                if next_item is not None:
                    heapq.heappush(
                        heap,
                        (next_item.message.log_time, serial, source, next_item),
                    )
                    serial += 1

        temporary_path.rename(output_path)
    except BaseException:
        stop.set()
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    finally:
        stop.set()
        for thread in threads:
            thread.join()

    size_gib = output_path.stat().st_size / (1024**3)
    print(
        f"Done: {output_path}\n"
        f"cam1={counts['cam1']}, cam2={counts['cam2']}, imu={counts['imu']}, "
        f"bag_compression={args.bag_compression}, size={size_gib:.2f} GiB"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        raise SystemExit("Interrupted")
