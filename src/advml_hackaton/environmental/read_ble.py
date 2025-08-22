"""
This module provides functionality for real-time data acquisition and plotting from Arduino devices over Bluetooth Low Energy (BLE). 
It supports time synchronization, BLE notifications, and synthetic data generation for debugging.

Main functionalities:
- Connecting to BLE devices and subscribing to notifications.
- Parsing BLE packets into structured data.
- Generating synthetic data for testing without BLE hardware.
- Real-time plotting of temperature, humidity, and predictions.
"""

#!/usr/bin/env python3
# ble_live_plot.py — realtime plots from Arduino BLE, with time sync (service/char UUIDs)

import argparse, asyncio, struct, sys, time, threading, queue
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

try:
    from bleak import BleakScanner, BleakClient
except Exception:
    BleakScanner = BleakClient = None  # allows --fake without bleak installed

SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"   # service
TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # notify
RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # write (time sync)

@dataclass
class Sample:
    ts: float; day: float; year: float; temp: float; hum: float; y: float

def parse_packet(data: bytes) -> Optional[Sample]:
    """
    Parses a BLE packet into a structured Sample object.

    Args:
        data (bytes): The raw BLE packet data.

    Returns:
        Optional[Sample]: A Sample object containing parsed data, or None if the packet is invalid.
    """
    if len(data) < 20: return None
    day, year, temp, hum, y = struct.unpack("<5f", data[:20])
    return Sample(time.time(), day, year, float(temp), float(hum), float(y))

def _uuid_eq(a: str, b: str) -> bool:
    return a.replace("-", "").lower() == b.replace("-", "").lower()

# ------------- BLE reader with time sync -------------
def start_ble_reader(service_uuid: str, tx_uuid: str, rx_uuid: str, out_q: queue.Queue, resync_period_s: int = 60):
    """
    Starts a BLE reader thread to receive data from a BLE device and synchronize time.

    Args:
        service_uuid (str): The UUID of the BLE service to connect to.
        tx_uuid (str): The UUID of the notify characteristic.
        rx_uuid (str): The UUID of the write characteristic for time synchronization.
        out_q (queue.Queue): A queue to store received data samples.
        resync_period_s (int): The time interval (in seconds) for periodic time synchronization. Default is 60.

    Returns:
        threading.Thread: The thread running the BLE reader.
    """
    if BleakScanner is None:
        print("bleak not installed. `pip install bleak` or use --fake", file=sys.stderr); sys.exit(1)

    async def find_device():
        def by_service_uuid(d, ad):
            return bool(ad and ad.service_uuids and any(_uuid_eq(u, service_uuid) for u in ad.service_uuids))
        dev = await BleakScanner.find_device_by_filter(by_service_uuid, timeout=10.0)
        if dev: return dev
        devs = await BleakScanner.discover(timeout=5.0)
        print("No match via filter; discovered:", [d.name or d.address for d in devs])
        return None

    async def sync_time(cli):
        # Send current epoch milliseconds as 8-byte little-endian
        payload = struct.pack("<Q", int(time.time() * 1000))
        await cli.write_gatt_char(rx_uuid, payload, response=False)

    async def run():
        print("Scanning for service:", service_uuid)
        dev = await find_device()
        if not dev: raise RuntimeError("Device advertising the service UUID not found.")
        print("Connecting:", dev)
        async with BleakClient(dev) as cli:
            print("Connected.")

            # Try to obtain services (handle version differences)
            svcs = getattr(cli, "services", None)
            if svcs is None:
                try: svcs = await cli.get_services()
                except Exception: svcs = None
            if svcs is not None:
                has_tx = any(_uuid_eq(c.uuid, tx_uuid) for s in svcs for c in s.characteristics)
                has_rx = any(_uuid_eq(c.uuid, rx_uuid) for s in svcs for c in s.characteristics)
                if not has_tx:
                    print("WARNING: TX not found in services; subscribing anyway.")
                if not has_rx:
                    print("WARNING: RX not found in services; time sync may fail.")

            # Initial time sync
            try:
                await sync_time(cli)
                print("Sent initial time sync.")
            except Exception as e:
                print("Time sync failed:", e)

            # Periodic resync
            async def periodic_sync():
                while True:
                    await asyncio.sleep(resync_period_s)
                    try:
                        await sync_time(cli)
                        print("Resynced time.")
                    except Exception:
                        pass
            asyncio.create_task(periodic_sync())

            # Notifications
            def handle(_, data: bytearray):
                s = parse_packet(data)
                if s:
                    try: out_q.put_nowait(s)
                    except queue.Full: pass

            await cli.start_notify(tx_uuid, handle)
            print("Subscribed; Ctrl+C to quit.")
            try:
                while True:
                    await asyncio.sleep(1.0)
            finally:
                try: await cli.stop_notify(tx_uuid)
                except Exception: pass

    def worker():
        try: asyncio.run(run())
        except Exception as e:
            print("BLE thread error:", e, file=sys.stderr)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t

# ------------- Fake data (debug) -------------
def start_fake_reader(period_s: float, out_q: queue.Queue):
    """
    Starts a thread to generate synthetic data for debugging purposes.

    Args:
        period_s (float): The time interval (in seconds) between generated samples.
        out_q (queue.Queue): A queue to store generated data samples.

    Returns:
        threading.Thread: The thread running the fake data generator.
    """
    import math
    def worker():
        t0 = time.time()
        while True:
            now = time.time()
            day  = ((now - t0) % 86400.0) / 86400.0
            year = ((now - t0) % (365.25*86400.0)) / (365.25*86400.0)
            temp = 22.0 + 3.0 * math.sin(2*math.pi * day * 4)
            hum  = 50.0 + 10.0 * math.cos(2*math.pi * day * 2)
            y    = 30000 + 2000 * math.sin(2*math.pi * year*3 + day*10)
            try: out_q.put_nowait(Sample(now, day, year, temp, hum, y))
            except queue.Full: pass
            time.sleep(period_s)
    t = threading.Thread(target=worker, daemon=True); t.start(); return t

# ------------- Plotting -------------
def run_plot(out_q: queue.Queue, window_s: float, interval_ms: int):
    """
    Runs a real-time plot of data received from the queue.

    Args:
        out_q (queue.Queue): A queue containing data samples to plot.
        window_s (float): The time window (in seconds) to display on the plot.
        interval_ms (int): The update interval (in milliseconds) for the plot.

    Returns:
        None
    """
    from collections import deque
    t_buf, hum_buf, tmp_buf, y_buf = deque(), deque(), deque(), deque()

    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(3,1,1)
    ax2 = plt.subplot(3,1,2, sharex=ax1)
    ax3 = plt.subplot(3,1,3, sharex=ax1)

    (line_hum,) = ax1.plot([], [], lw=1.5); ax1.set_ylabel("Humidity (%)"); ax1.grid(True)
    (line_tmp,) = ax2.plot([], [], lw=1.5); ax2.set_ylabel("Temp (°C)");    ax2.grid(True)
    (line_y,)   = ax3.plot([], [], lw=1.5); ax3.set_ylabel("Prediction");   ax3.set_xlabel("Time (s)"); ax3.grid(True)

    # avoid scientific notation for humidity/temperature
    for ax in (ax1, ax2):
        fmt = ScalarFormatter(useOffset=False, useMathText=False); fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)

    plt.tight_layout()
    last_log = 0.0

    while True:
        drained = 0
        while True:
            try:
                s: Sample = out_q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            t_buf.append(s.ts); hum_buf.append(s.hum); tmp_buf.append(s.temp); y_buf.append(s.y)

        cutoff = time.time() - window_s
        while t_buf and t_buf[0] < cutoff:
            t_buf.popleft(); hum_buf.popleft(); tmp_buf.popleft(); y_buf.popleft()

        if t_buf:
            t0 = t_buf[0]
            xs = [t - t0 for t in t_buf]
            line_hum.set_data(xs, list(hum_buf))
            line_tmp.set_data(xs, list(tmp_buf))
            line_y.set_data(xs, list(y_buf))
            for ax in (ax1, ax2, ax3): ax.relim(); ax.autoscale_view()
            ax3.set_xlim(0, max(xs) if xs else window_s)
            if drained and time.time() - last_log > 1.5:
                print(f"T={tmp_buf[-1]:.2f}C  H={hum_buf[-1]:.2f}%  Y={y_buf[-1]:.2f}")
                last_log = time.time()

        plt.pause(interval_ms / 1000.0)

# ------------- CLI -------------
def main():
    """
    Main function to run the BLE data acquisition and plotting application.

    Steps:
    - Parses command-line arguments.
    - Starts either a BLE reader or a fake data generator.
    - Runs the real-time plotting interface.

    Returns:
        None
    """
    import argparse
    ap = argparse.ArgumentParser(description="Live plot Arduino BLE data (temp, hum, prediction) with time sync.")
    ap.add_argument("--service", default=SERVICE_UUID, help="Service UUID to match")
    ap.add_argument("--tx", default=TX_CHAR_UUID, help="Notify characteristic UUID")
    ap.add_argument("--rx", default=RX_CHAR_UUID, help="Write characteristic UUID (for time sync)")
    ap.add_argument("--window", type=float, default=120.0, help="Plot window in seconds")
    ap.add_argument("--interval", type=int, default=200, help="UI update interval (ms)")
    ap.add_argument("--fake", action="store_true", help="Run without BLE (synthetic data)")
    ap.add_argument("--fake-period", type=float, default=0.5, help="Fake sample period (s)")
    ap.add_argument("--resync", type=int, default=60, help="BLE time resync period (seconds)")
    args = ap.parse_args()

    q = queue.Queue(maxsize=10000)
    if args.fake:
        print("Starting FAKE data source…"); start_fake_reader(args.fake_period, q)
    else:
        print("Starting BLE reader…"); start_ble_reader(args.service, args.tx, args.rx, q, resync_period_s=args.resync)

    run_plot(q, window_s=args.window, interval_ms=args.interval)

if __name__ == "__main__":
    main()