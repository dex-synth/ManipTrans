"""h2r-evaluation-v1 framing; no dependency on the benchmark runtime."""

import json
import math
import struct

MAX_BYTES = 32*1024*1024


def send(stream, message):
    payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
    if not 0 < len(payload) <= MAX_BYTES:
        raise ValueError("Protocol message too large")
    stream.sendall(struct.pack("!I", len(payload)) + payload)


def receive(stream):
    def read(count):
        result = bytearray()
        while len(result) < count:
            chunk = stream.recv(count-len(result))
            if not chunk:
                raise EOFError("Evaluation server disconnected")
            result.extend(chunk)
        return result

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate protocol field: {key}")
            result[key] = value
        return result

    def number(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Non-finite protocol number")
        return value

    size, = struct.unpack("!I", read(4))
    if not 0 < size <= MAX_BYTES:
        raise ValueError("Invalid protocol message size")
    value = json.loads(read(size), object_pairs_hook=pairs, parse_float=number, parse_constant=number)
    if not isinstance(value, dict):
        raise ValueError("Expected a protocol object")
    return value
