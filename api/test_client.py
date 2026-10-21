#!/usr/bin/env python3
"""Call the JyutVoice API with the reference clip as base64 (standard library only).

    export API_KEY=...        # the key the server was started with
    python api/test_client.py --ref prompt.wav --text "我喺Zoom開會，但係個microphone有問題。" --output out.wav
    python api/test_client.py --ref prompt.wav --text "..." --output out.wav --stream     # first audio arrives after the first sentence
    python api/test_client.py --prompt-cache-id pc_... --text "..." --output out.wav      # a voice the server already has: no clip is sent
"""
import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000", help="server address")
    p.add_argument("--api-key", default=os.environ.get("API_KEY", ""), help="bearer key (default: $API_KEY)")
    p.add_argument("--text", required=True)
    p.add_argument("--ref", help="reference audio file (wav, flac, ogg, mp3, ...): sent as base64")
    p.add_argument("--prompt-cache-id", help="Prompt-Cache-Id of an earlier response, instead of --ref")
    p.add_argument("--output", required=True, help="output file (.wav, or raw PCM16 24 kHz with --format pcm)")
    p.add_argument("--format", default="wav", choices=["wav", "pcm"])
    p.add_argument("--stream", action="store_true", help="stream the response and report the time to the first audio")
    p.add_argument("--seed", type=int)
    p.add_argument("--speed", type=float)
    p.add_argument("--flow-steps", type=int)
    p.add_argument("--no-prosody", action="store_true")
    p.add_argument("--data-uri", action="store_true", help="send the clip as a data:audio/...;base64, URI instead of raw base64")
    args = p.parse_args()

    body = {"model": "jyutvoice", "input": args.text, "format": args.format, "stream": args.stream}
    if args.ref:
        with open(args.ref, "rb") as f:
            clip = base64.b64encode(f.read()).decode("ascii")
        body["prompt_audio"] = f"data:audio/wav;base64,{clip}" if args.data_uri else clip
    if args.prompt_cache_id:
        body["prompt_cache_id"] = args.prompt_cache_id
    for key, value in (("seed", args.seed), ("speed", args.speed), ("flow_steps", args.flow_steps)):
        if value is not None:
            body[key] = value
    if args.no_prosody:
        body["prosody"] = False

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(args.url.rstrip("/") + "/v1/audio/speech", data=json.dumps(body).encode(), headers=headers, method="POST")

    t0 = time.time()
    try:
        response = urllib.request.urlopen(request, timeout=600)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)
    with response, open(args.output, "wb") as out:
        first = None
        total = 0
        while True:
            block = response.read(16384)
            if not block:
                break
            if first is None:
                first = time.time() - t0
            out.write(block)
            total += len(block)
    elapsed = time.time() - t0
    if args.format == "wav" and args.stream:  # the streamed header has placeholder sizes: write the real ones
        with open(args.output, "r+b") as f:
            f.seek(4)
            f.write(struct.pack("<I", total - 8))
            f.seek(40)
            f.write(struct.pack("<I", total - 44))
    seconds = (total - (44 if args.format == "wav" else 0)) / 2 / 24000
    print(f"Prompt-Cache-Id: {response.headers.get('Prompt-Cache-Id')}")
    print(f"saved {args.output}: {seconds:.2f} s of audio in {elapsed:.2f} s (RTF {elapsed / max(seconds, 1e-9):.2f}); first byte after {first:.2f} s")


if __name__ == "__main__":
    main()
