"""How fast can the bus actually run? Measured, not derived. Runs ON the robot.

WHY. The policy runs at 50 Hz, which is a TRAINING choice (sim timestep 0.002 x
decimation 10), not a hardware limit -- a fact stated the other way round for
most of this campaign. The BAM bench logged a single motor at dt = 0.005
(200 Hz), so the protocol and firmware clearly do better than 50. What nobody
has measured is the 14-servo case, which is what the robot actually needs.

This matters for hopping specifically: a push lasts 50-100 ms and contact is
~46 ms, so at 50 Hz the propulsive phase gets 2-5 control decisions. Every gait
interval measured in sim came out quantised to 20 ms. The policy may be
resolution-limited rather than force-limited.

READ-ONLY AND SAFE. Sync Read of present state, a one-byte read of Return Delay
Time, and a Sync Write of LED=0 (its normal state) to time a reply-less write.
It never touches torque, goal position, operating mode or EEPROM, so it cannot
move the robot and cannot load a gearbox.

    sudo sh ~/measure_bus_rate.sh          # wrapper masks robotd first

The bus must be FREE: robotd is Restart=always with RestartSec=2s, so stopping
it is not enough -- it has to be masked. See measure_travel.sh.
"""
from __future__ import annotations

import argparse
import os
import select
import statistics
import struct
import sys
import termios
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_travel import JOINTS, NAME, build, crc16, frames  # noqa: E402

BROADCAST = 0xFE
INST_READ, INST_SYNC_READ, INST_SYNC_WRITE = 0x02, 0x82, 0x83
ADDR_RETURN_DELAY = 9           # EEPROM, 1 byte, units of 2 us
ADDR_LED = 65                   # RAM, 1 byte -- harmless
ADDR_PRESENT_CURRENT = 126      # 126 current(2) 128 velocity(4) 132 position(4)
BLOCK_LEN = 10                  # one contiguous read of all three


class Bus:
    def __init__(self, path: str, baud: int):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
        a = termios.tcgetattr(self.fd)
        a[0] = a[1] = a[3] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[4] = a[5] = getattr(termios, f"B{baud}")
        a[6] = list(a[6])
        a[6][termios.VMIN] = 0
        a[6][termios.VTIME] = 1
        termios.tcsetattr(self.fd, termios.TCSANOW, a)
        termios.tcflush(self.fd, termios.TCIOFLUSH)

    def txn(self, pkt: bytes, want: int, plen: int, timeout: float):
        """Send, collect `want` CRC-valid status frames. Returns (ids, seconds)."""
        # NO tcdrain HERE. On this 8250-class UART tcdrain waits on a
        # timer-granularity poll loop: it made a reply-less write take 13.00 ms
        # against ~0.5 ms of wire time at 1 Mbps, and made a 4-byte and a
        # 10-byte read cost the same. That measured the instrument, not the bus.
        # The reply's arrival is the completion signal we actually want.
        termios.tcflush(self.fd, termios.TCIFLUSH)
        t0 = time.perf_counter()
        os.write(self.fd, pkt)
        buf, got, deadline = b"", {}, time.perf_counter() + timeout
        while time.perf_counter() < deadline and len(got) < want:
            if select.select([self.fd], [], [], max(0.0, deadline - time.perf_counter()))[0]:
                chunk = os.read(self.fd, 512)
                if chunk:
                    buf += chunk
                    for fid, _err, params in frames(buf):
                        if len(params) >= plen:
                            got[fid] = params[:plen]
        return got, time.perf_counter() - t0

    def duplex(self, wpkt: bytes, rpkt: bytes, want: int, plen: int, timeout: float):
        """One real control cycle: command out, full state back. tcdrain-free."""
        termios.tcflush(self.fd, termios.TCIFLUSH)
        t0 = time.perf_counter()
        os.write(self.fd, wpkt)
        os.write(self.fd, rpkt)
        buf, got, deadline = b"", {}, time.perf_counter() + timeout
        while time.perf_counter() < deadline and len(got) < want:
            if select.select([self.fd], [], [], max(0.0, deadline - time.perf_counter()))[0]:
                chunk = os.read(self.fd, 512)
                if chunk:
                    buf += chunk
                    for fid, _err, params in frames(buf):
                        if len(params) >= plen:
                            got[fid] = params[:plen]
        return got, time.perf_counter() - t0


def sync_read(ids, addr, length):
    return build(BROADCAST, INST_SYNC_READ,
                 struct.pack("<HH", addr, length) + bytes(ids))


def sync_write_led(ids, value=0):
    params = struct.pack("<HH", ADDR_LED, 1)
    for i in ids:
        params += bytes([i, value])
    return build(BROADCAST, INST_SYNC_WRITE, params)


def stats(samples):
    s = sorted(samples)
    return (statistics.mean(s), s[len(s) // 2], s[int(len(s) * 0.99)], s[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyS2")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()

    ids = [i for i, _, _ in JOINTS]
    bus = Bus(args.port, args.baud)
    print(f"port {args.port} @ {args.baud} baud, {len(ids)} servos, {args.iters} iterations")
    print("READ-ONLY: no torque, no goal position, no EEPROM write.\n")

    # --- Return Delay Time: the parameter most likely to dominate -------------
    print("Return Delay Time (addr 9, units of 2 us):")
    rdts = {}
    for i in ids:
        pkt = build(i, INST_READ, struct.pack("<HH", ADDR_RETURN_DELAY, 1))
        got, _ = bus.txn(pkt, 1, 1, 0.05)
        if i in got:
            rdts[i] = got[i][0] * 2
    if rdts:
        for i in ids:
            if i in rdts:
                print(f"    {NAME.get(i, i):<16} {rdts[i]:4d} us")
        total = sum(rdts.values())
        print(f"  total across {len(rdts)} servos: {total} us "
              f"({total/1000:.2f} ms of pure idle per sync read)")
    else:
        print("    (no replies -- is the bus free? robotd must be MASKED, not stopped)")
        return

    # --- the three timings ----------------------------------------------------
    def bench(label, pkt, want, plen):
        times, complete = [], 0
        for _ in range(args.iters):
            got, dt = bus.txn(pkt, want, plen, 0.05)
            times.append(dt)
            complete += (len(got) == want)
        m, p50, p99, mx = stats(times)
        print(f"\n{label}")
        print(f"    complete replies  {complete}/{args.iters}")
        print(f"    mean {m*1000:6.2f} ms   p50 {p50*1000:6.2f}   "
              f"p99 {p99*1000:6.2f}   max {mx*1000:6.2f}")
        print(f"    -> {1/m:6.1f} Hz sustained")
        return m

    t_pos = bench("SYNC READ position only (4 B/servo)",
                  sync_read(ids, 132, 4), len(ids), 4)
    t_blk = bench(f"SYNC READ current+velocity+position ({BLOCK_LEN} B/servo)",
                  sync_read(ids, ADDR_PRESENT_CURRENT, BLOCK_LEN), len(ids), BLOCK_LEN)

    # --- the real thing: command out AND full state back, one cycle ----------
    wpkt = sync_write_led(ids, 0)
    rpkt = sync_read(ids, ADDR_PRESENT_CURRENT, BLOCK_LEN)
    dtimes, dcomplete = [], 0
    for _ in range(args.iters):
        got, dt = bus.duplex(wpkt, rpkt, len(ids), BLOCK_LEN, 0.05)
        dtimes.append(dt)
        dcomplete += (len(got) == len(ids))
    dm, dp50, dp99, dmx = stats(dtimes)
    print(f"\nFULL CONTROL CYCLE (sync write + sync read of all state)")
    print(f"    complete replies  {dcomplete}/{args.iters}")
    print(f"    mean {dm*1000:6.2f} ms   p50 {dp50*1000:6.2f}   "
          f"p99 {dp99*1000:6.2f}   max {dmx*1000:6.2f}")

    loop = dm
    print(f"\nCONTROL LOOP CEILING")
    print(f"    {loop*1000:.2f} ms  -> {1/loop:.0f} Hz")
    print(f"    the policy runs at 50 Hz (20.00 ms), using "
          f"{loop/0.020*100:.0f}% of the budget")

    wire = (len(ids) * (11 + BLOCK_LEN) + 28) * 10 / args.baud
    print(f"    wire time for that traffic at {args.baud} baud: {wire*1000:.2f} ms"
          f"  ({wire/loop*100:.0f}% of the cycle)")
    if rdts and max(rdts.values()) > 0:
        saved = sum(rdts.values()) / 1e6
        print(f"\n    Return Delay Time is costing {saved*1000:.2f} ms per read. "
              f"At RDT=0 the\n    loop would be about "
              f"{1/max(loop - saved, 1e-4):.0f} Hz. RDT is EEPROM (addr 9): "
              f"a one-time\n    provisioning write with torque off, not a runtime toggle.")


if __name__ == "__main__":
    main()
