#!/usr/bin/env python3
"""Rewrite an LZ4 ROS1 bag as an uncompressed, index-preserving bag.

Some LZ4 frame variants produced by newer writers are valid according to the
LZ4 specification (and decode with python-lz4 / the lz4 CLI), but ROS Noetic's
roslz4 decoder rejects them with ``ROSLZ4_DATA_ERROR``.  This utility rewrites
only the chunk payloads and updates the affected bag offsets.  Message bytes,
timestamps, connection headers and per-chunk indexes are preserved verbatim.
The source bag is never modified.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import BinaryIO, List, Tuple

import lz4.frame


MAGIC = b"#ROSBAG V2.0\n"
OP_FILE_HEADER = 0x03
OP_INDEX_DATA = 0x04
OP_CHUNK = 0x05
OP_CHUNK_INFO = 0x06


Field = Tuple[bytes, bytes]


def read_u32(stream: BinaryIO) -> int:
    raw = stream.read(4)
    if len(raw) != 4:
        raise EOFError
    return struct.unpack("<I", raw)[0]


def parse_fields(raw: bytes) -> List[Field]:
    fields: List[Field] = []
    offset = 0
    while offset < len(raw):
        if offset + 4 > len(raw):
            raise ValueError("truncated ROS bag header field length")
        size = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        item = raw[offset : offset + size]
        offset += size
        if len(item) != size or b"=" not in item:
            raise ValueError("invalid ROS bag header field")
        key, value = item.split(b"=", 1)
        fields.append((key, value))
    return fields


def encode_fields(fields: List[Field]) -> bytes:
    parts = []
    for key, value in fields:
        item = key + b"=" + value
        parts.append(struct.pack("<I", len(item)))
        parts.append(item)
    return b"".join(parts)


def get_field(fields: List[Field], key: bytes) -> bytes:
    for field_key, value in fields:
        if field_key == key:
            return value
    raise KeyError(key.decode())


def set_field(fields: List[Field], key: bytes, value: bytes) -> None:
    for index, (field_key, _) in enumerate(fields):
        if field_key == key:
            fields[index] = (field_key, value)
            return
    raise KeyError(key.decode())


def op_code(fields: List[Field]) -> int:
    value = get_field(fields, b"op")
    if len(value) != 1:
        raise ValueError("invalid op field")
    return value[0]


def write_record(stream: BinaryIO, fields: List[Field], data: bytes) -> None:
    header = encode_fields(fields)
    stream.write(struct.pack("<I", len(header)))
    stream.write(header)
    stream.write(struct.pack("<I", len(data)))
    stream.write(data)


def render_file_header(fields: List[Field], index_pos: int) -> bytes:
    set_field(fields, b"index_pos", struct.pack("<Q", index_pos))
    header = encode_fields(fields)
    prefix = struct.pack("<I", len(header)) + header
    # ROS bag v2 reserves exactly 4096 bytes for the file-header record,
    # including its data length and padding payload.
    data_len = 4096 - len(prefix) - 4
    if data_len < 0:
        raise ValueError("file header no longer fits in its 4096-byte slot")
    return prefix + struct.pack("<I", data_len) + b" " * data_len


def repair(source: Path, destination: Path, overwrite: bool) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different files")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination exists: {destination} (use --overwrite)")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        partial.unlink()

    chunk_positions = {}
    chunk_count = 0
    old_index_pos = 0
    new_index_pos = 0

    try:
        with source.open("rb") as src, partial.open("w+b") as dst:
            if src.read(len(MAGIC)) != MAGIC:
                raise ValueError(f"not a ROS1 bag v2 file: {source}")
            dst.write(MAGIC)

            file_header_offset = src.tell()
            header_len = read_u32(src)
            file_header_fields = parse_fields(src.read(header_len))
            file_header_data_len = read_u32(src)
            src.seek(file_header_data_len, os.SEEK_CUR)
            if op_code(file_header_fields) != OP_FILE_HEADER:
                raise ValueError("first record is not a ROS bag file header")
            old_index_pos = struct.unpack("<Q", get_field(file_header_fields, b"index_pos"))[0]

            # Reserve the fixed-size file header; fill index_pos after all
            # rewritten chunk positions are known.
            dst.write(b"\0" * 4096)

            while True:
                old_record_pos = src.tell()
                if old_record_pos >= source.stat().st_size:
                    break

                try:
                    header_len = read_u32(src)
                except EOFError:
                    break
                header_raw = src.read(header_len)
                if len(header_raw) != header_len:
                    raise ValueError(f"truncated record header at {old_record_pos}")
                fields = parse_fields(header_raw)
                data_len = read_u32(src)
                data = src.read(data_len)
                if len(data) != data_len:
                    raise ValueError(f"truncated record data at {old_record_pos}")

                op = op_code(fields)
                if old_record_pos == old_index_pos:
                    new_index_pos = dst.tell()

                if op == OP_CHUNK:
                    new_record_pos = dst.tell()
                    chunk_positions[old_record_pos] = new_record_pos
                    compression = get_field(fields, b"compression")
                    expected_size = struct.unpack("<I", get_field(fields, b"size"))[0]
                    if compression == b"lz4":
                        data = lz4.frame.decompress(data)
                        if len(data) != expected_size:
                            raise ValueError(
                                f"chunk {chunk_count} decoded to {len(data)} bytes; "
                                f"header expects {expected_size}"
                            )
                        set_field(fields, b"compression", b"none")
                    elif compression != b"none":
                        raise ValueError(
                            f"unsupported compression {compression!r} at chunk {chunk_count}"
                        )
                    write_record(dst, fields, data)
                    chunk_count += 1
                    if chunk_count == 1 or chunk_count % 100 == 0:
                        print(
                            f"rewrote {chunk_count} chunks "
                            f"({src.tell() / source.stat().st_size * 100.0:.1f}%)",
                            flush=True,
                        )
                elif op == OP_CHUNK_INFO:
                    old_chunk_pos = struct.unpack("<Q", get_field(fields, b"chunk_pos"))[0]
                    if old_chunk_pos not in chunk_positions:
                        raise ValueError(f"chunk_info references unknown position {old_chunk_pos}")
                    set_field(fields, b"chunk_pos", struct.pack("<Q", chunk_positions[old_chunk_pos]))
                    write_record(dst, fields, data)
                else:
                    # Re-encoding the small record header is lossless; message
                    # and index payload bytes remain untouched.
                    write_record(dst, fields, data)

            if old_index_pos and not new_index_pos:
                raise ValueError(f"did not encounter index_pos {old_index_pos}")

            dst.seek(len(MAGIC))
            dst.write(render_file_header(file_header_fields, new_index_pos))
            dst.flush()
            os.fsync(dst.fileno())

        os.replace(partial, destination)
        print(f"wrote {destination}")
        print(f"chunks: {chunk_count}")
        print(f"size: {destination.stat().st_size} bytes")
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        repair(args.source, args.destination, args.overwrite)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
