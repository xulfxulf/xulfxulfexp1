"""Bounded SFTP reads for the installed Paramiko2.8, with no content hashes."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import paramiko


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    destination = Path(args.destination).resolve()
    if not destination.name.endswith('.part') or not destination.parent.is_dir():
        raise ValueError('Use an existing explicit archive directory and .part destination')
    spec = importlib.util.spec_from_file_location('research_transport',
        'D:/004SSH/OEFormer_transfer_TBPR_PRO1_20260904/remote.py')
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    client = transport.connect('TBPR-PRO1')
    try:
        with paramiko.SFTPClient.from_transport(client.get_transport(),
                window_size=16*1024**2, max_packet_size=32768) as sftp:
            before = sftp.stat(args.source)
            identity = dict(source=args.source, size=before.st_size, mtime=before.st_mtime)
            state_path = destination.with_suffix(destination.suffix+'.download.json')
            offset = destination.stat().st_size if destination.exists() else 0
            if state_path.exists():
                if json.loads(state_path.read_text())['source_identity'] != identity:
                    raise ValueError('Remote source changed since partial download')
            elif offset:
                raise ValueError('Existing nonempty partial has no source identity; preserve it')
            if offset > before.st_size:
                raise ValueError('Partial larger than source')
            state = dict(source_identity=identity, status='downloading', start_offset=offset,
                         local_destination=str(destination), request_limit=64, hashes_used=False)
            state_path.write_text(json.dumps(state, indent=2))
            start = last = time.monotonic()
            clients, sessions, readers = [], [], []
            def read_range(index, begin, size):
                incoming = readers[index]
                end, data = begin+size, bytearray()
                for block_start in range(begin, end, 64*32768):
                    chunks = [(pos, min(32768, end-pos))
                              for pos in range(block_start, min(block_start+64*32768, end), 32768)]
                    for (_, expected), block in zip(chunks, incoming.readv(chunks)):
                        if len(block) != expected:
                            raise IOError('Short SFTP block')
                        data.extend(block)
                if len(data) != size:
                    raise IOError('Incomplete bounded range')
                return data
            try:
                for _ in range(4):
                    connection = transport.connect('TBPR-PRO1')
                    clients.append(connection)
                    session = paramiko.SFTPClient.from_transport(connection.get_transport(),
                        window_size=16*1024**2, max_packet_size=32768)
                    sessions.append(session)
                    reader = session.open(args.source, 'rb')
                    reader.settimeout(60)
                    readers.append(reader)
                with ThreadPoolExecutor(max_workers=4) as executor, open(destination, 'ab') as outgoing:
                    while offset < before.st_size:
                        ranges = [(pos, min(16*1024**2, before.st_size-pos))
                                  for pos in range(offset, min(offset+64*1024**2, before.st_size), 16*1024**2)]
                        futures = [executor.submit(read_range, index, begin, size)
                                   for index, (begin, size) in enumerate(ranges)]
                        for (begin, size), future in zip(ranges, futures):
                            block = future.result()
                            if offset != begin or len(block) != size:
                                raise ValueError('Out-of-order archive segment')
                            outgoing.write(block)
                            offset += size
                        outgoing.flush()
                        if time.monotonic()-last >= 20:
                            print(json.dumps(dict(bytes=offset,total=before.st_size,
                                elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
                            last = time.monotonic()
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
            finally:
                for reader in readers:
                    reader.close()
                for session in sessions:
                    session.close()
                for connection in clients:
                    connection.close()
            after = sftp.stat(args.source)
            if (after.st_size, after.st_mtime) != (before.st_size, before.st_mtime):
                raise ValueError('Source changed during download')
            if destination.stat().st_size != before.st_size:
                raise IOError('Final archive length mismatch')
            state.update(status='complete', elapsed_seconds=time.monotonic()-start, bytes=offset,
                         connections=4, max_segment_bytes=16*1024**2)
            state_path.write_text(json.dumps(state, indent=2))
            print(json.dumps(state), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
