#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------
""" common utilities and class implementation for time/GPU/RAM profiling """
import argparse
import contextlib
import gc
import io
import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import Process, RLock, Value
from typing import Dict, List, Optional, Union

import psutil
import torch
from aimet_common.utils import AimetLogger
from torch.types import Device

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

WATERMARK_THREAD_POLLING_INTERVAL_IN_MS = 100


def convert_bytes(size: int):
    """
    :return bytes in human-readable format
    """
    sign = ''
    if size < 0:
        sign = '-'
        size = abs(size)

    for x in [' B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return "%s%3.1f %s" % (sign, size, x)


@dataclass
class ProfileMarker:
    """ Implements methods to capture profiling data """
    event: str
    device: Optional[Union[Device, int]]
    gpu_memory_usage: int
    cpu_memory_usage: int
    delta_gpu_memory_usage: int
    delta_cpu_memory_usage: int
    time: float

    def __str__(self):
        """ string representation of the marker event """
        return f'Event {self.event} : time={timedelta(seconds=self.time)}, ' \
               f'GPU={convert_bytes(self.delta_gpu_memory_usage)}(+ {convert_bytes(self.gpu_memory_usage)}), ' \
               f'RAM={convert_bytes(self.delta_cpu_memory_usage)}(+ {convert_bytes(self.cpu_memory_usage)})'

    def to_dict(self) -> dict:
        """return data in dict """
        return {
            'event': self.event,
            'device': self.device,
            'gpu_memory_usage': self.gpu_memory_usage,
            'cpu_memory_usage': self.cpu_memory_usage,
            'delta_gpu_memory_usage': self.delta_gpu_memory_usage,
            'delta_cpu_memory_usage': self.delta_cpu_memory_usage,
            'time': self.time
        }


def ram_usage(pid):
    """ :return RAM usage for given process Id. """
    return psutil.Process(pid).memory_info().rss


@dataclass
class EventMarker:
    """ Implements methods to capture system stats during an event """
    event: str
    device: Optional[Union[Device, int]]
    _gpu_memory_usage: int
    _cpu_memory_usage: int
    time = time.time()

    def __init__(self, event: str = None, cpu_memory_usage: int = 0, device: Union[Device, int] = None):
        self._gpu_memory_usage = torch.cuda.max_memory_allocated(device)
        self._cpu_memory_usage: int = cpu_memory_usage
        self.device = device
        self.time = time.time()
        if event is None:
            self.event = f'@ {self.time}'
        else:
            self.event = event

    def delta(self, event: str, start_marker: 'EventMarker') -> ProfileMarker:
        """computes diff between two event marker and returns profile marker"""
        # pylint: disable=protected-access
        return ProfileMarker(
            event,
            self.device,
            start_marker._gpu_memory_usage,
            start_marker._cpu_memory_usage,
            self._gpu_memory_usage - start_marker._gpu_memory_usage,
            self._cpu_memory_usage - start_marker._cpu_memory_usage,
            int(self.time - start_marker.time)
        )

    def __str__(self):
        """ string representation of the marker event """
        return f'Event {self.event} : time={self.time}, GPU={self.gpu_memory_usage}, RAM={self.cpu_memory_usage}'

    @property
    def gpu_memory_usage(self) -> str:
        """returns a string representation of GPU usage """
        device_str = 'default' if self.device is None \
            else f'cuda:{self.device}' if isinstance(self.device, int) \
            else str(self.device)

        return f'{device_str}:{convert_bytes(self._gpu_memory_usage)}'

    @property
    def cpu_memory_usage(self) -> str:
        """returns a string representation of RAM usage """
        return convert_bytes(self._cpu_memory_usage)

    def to_dict(self) -> dict:
        """return data in dict """
        return {
            'event': self.event,
            'device': self.device,
            'gpu_memory_usage': self._gpu_memory_usage,
            'cpu_memory_usage': self._cpu_memory_usage,
            'time': self.time
        }


def ram_watermark_function(ram_allocated: Value, pid: int, polling_interval_in_ms: float):
    """
    observing process to reflect current peak RAM usage.
    :param ram_allocated: shared variable used by profiler to reset to new allocation and the observing(this) process
    to track max allocation.
    :param pid: parent process pid for tracking mem allocation
    :param polling_interval_in_ms: interval between polling for memory usage
    """
    logger.info('Created RAM watermark daemon process(pid=%d) for pid=%d, polling at %.1f ms',
                os.getpid(), pid, polling_interval_in_ms)
    while psutil.pid_exists(pid):
        new_usage = ram_usage(pid)
        with ram_allocated.get_lock():
            ram_allocated.value = max(new_usage, ram_allocated.value)
        time.sleep(polling_interval_in_ms / 1000.0)


# pylint: disable=no-member
class EventProfiler:
    """ Implements methods to profile latency and RAM/GPU memory usage """
    _instance = None

    def __new__(cls):
        """ Implements the Global Object Pattern  (Singleton) """
        if cls._instance is None:
            cls._instance = super(EventProfiler, cls).__new__(cls)
            cls._instance._empty_cache = False  # pylint: disable=protected-access
            cls._instance._markers = []  # pylint: disable=protected-access

            if WATERMARK_THREAD_POLLING_INTERVAL_IN_MS:
                cls._instance._ram_allocated = Value('q', 0, lock=RLock())  # pylint: disable=protected-access
                p = Process(
                    target=ram_watermark_function,
                    args=(cls._instance._ram_allocated,
                          os.getpid(),
                          WATERMARK_THREAD_POLLING_INTERVAL_IN_MS
                          ))
                p.daemon = True  # < will terminate the watermark process when this process exits
                cls._instance.reset_peak_memory_stats()
                p.start()

            logger.info('Created Latency/Memory profiler: empty_cache=%s',
                        cls._instance._empty_cache)  # pylint: disable=protected-access

        return cls._instance

    def reset_peak_memory_stats(self):
        """ reset RAM usage to current RAM allocation. """
        if WATERMARK_THREAD_POLLING_INTERVAL_IN_MS:
            with self._ram_allocated.get_lock():
                self._ram_allocated.value = ram_usage(os.getpid())

    @property
    def max_memory_allocated(self):
        """ getter for current peak RAM usage since last reset. """
        if WATERMARK_THREAD_POLLING_INTERVAL_IN_MS:
            with self._ram_allocated.get_lock():
                return self._ram_allocated.value
        else:
            return ram_usage(os.getpid())

    @property
    def empty_cache(self):
        """ getter for empty_cache if set to True snapshot calls would flush unused CUDA memory. """
        return self._empty_cache

    @empty_cache.setter
    def empty_cache(self, enable: bool = False):
        """ setter for empty_cache if set to True snapshot calls would flush unused CUDA memory. """
        if self._empty_cache != enable:
            if self._empty_cache:
                logger.warning('enabling cache clear might impact latency, avoid excessive calls in tight loop')
            self._markers.append(f"empty_cache:{self._empty_cache}")

    def snapshot(self, snapshot_marker: str = None,
                 device: Union[Device, int] = None,
                 append: bool = True) -> EventMarker:
        """
        logs the current time and memory usage across all CUDA devices
        :param snapshot_marker: text to capture with the GPU marker.
        :param device: (torch.device or int, optional): selected device.
        :param append: if True, added it to report logs
        """
        marker = EventMarker(snapshot_marker, self.max_memory_allocated, device)
        if self._empty_cache:
            torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        logger.info("memory usage @ '%s' : GPU %s, RAM %s",
                    snapshot_marker, marker.gpu_memory_usage, marker.cpu_memory_usage)
        if append:
            self._markers.append(marker)

        return marker

    def report(self):
        """ dumps the collected memory usage logs """
        logger.info("Profiling report :- %s",
                    generate_event_report([m.to_dict() for m in self._markers], max_memory_threshold=0.9))

    def json_dump(self, filepath: str):
        """ dumps the collected memory usage into a json file """
        markers = [m.to_dict() for m in self._markers]
        with open(filepath, 'w') as f:
            json.dump(markers, f, sort_keys=True, indent=4)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type == torch.cuda.OutOfMemoryError:  # pylint: disable=no-member
            self.report()


@contextlib.contextmanager
def event_marker(event: str, device: Union[Device, int] = None, flush_ram: bool = False):
    """
    utility to mark time taken and memory usage before and after executing a section of code.
    :param event: marker string to use to identify the context.
    :param device: (torch.device or int, optional): selected device.
    :param flush_ram: invoke garbage collect for true estimates before profiling.
    """
    profiler = EventProfiler()
    # reset for start low-watermark
    if flush_ram:
        gc.collect()
        event = f'{event}[gc]'

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    profiler.reset_peak_memory_stats()
    start_marker = profiler.snapshot(f'{event} >> ', device, append=False)
    yield
    end_marker = profiler.snapshot(f'{event} << ', device, append=False)
    profile_marker = end_marker.delta(event, start_marker)
    logger.info('%s', profile_marker)
    profiler._markers.append(profile_marker)  # pylint: disable=protected-access


def generate_event_report(event_list: List[Dict[str, Union[int, str]]], max_memory_threshold: float) -> str:
    """
    utility to create a format event list report with additional statistics
    :param event_list: a list of event entries(dict).
    :param max_memory_threshold: threshold to mark all event(s) with range of max event e.g. 0.9 would log all event
    with 90% of max usage event.
    """

    gpu_usage = lambda event: event['gpu_memory_usage'] + event['delta_gpu_memory_usage']
    cpu_usage = lambda event: event['cpu_memory_usage'] + event['delta_cpu_memory_usage']

    tot_time = sum(event['time'] for event in event_list)
    max_gpu = gpu_usage(max(event_list, key=gpu_usage)), []
    max_cpu = cpu_usage(max(event_list, key=cpu_usage)), []

    stream = io.StringIO(newline='\n')

    stream.write("\n" + "-" * 150)
    stream.write("\n{:>90}  | {:>18}        | {:>18}         |".format("", "GPU", "RAM"))
    stream.write("\n{:<65} {:>22}    | {:>12} {:>11}  |  {:>11} {:>11}   |".format(
        "Event", "Time", "delta", "agg", "delta", "agg"))
    stream.write("\n" + "-" * 150)

    for event in event_list:

        event_desc = event['event']
        duration = event['time']
        gpu_mem = gpu_usage(event)
        cpu_mem = cpu_usage(event)

        cpu_marker = gpu_marker = ' '
        if max_gpu[0] * max_memory_threshold < gpu_mem:
            max_gpu[1].append(event_desc)
            gpu_marker = '*'
        if max_cpu[0] * max_memory_threshold < cpu_mem:
            max_cpu[1].append(event_desc)
            cpu_marker = '*'

        stream.write("\n{:<65} {:>20}{:>5} | {:>12} {:>12}{}|  {:>12} {:>12}{}|".format(
            event_desc, str(timedelta(seconds=duration)), '{:.0%}'.format(duration / tot_time),
            convert_bytes(event['delta_gpu_memory_usage']), convert_bytes(gpu_mem), gpu_marker,
            convert_bytes(event['delta_cpu_memory_usage']), convert_bytes(cpu_mem), cpu_marker))

    stream.write("\n" + "-" * 150)
    stream.write("\nSummary:")
    stream.write("\n\tTime(*under profiling*): {:>10}".format(str(timedelta(seconds=tot_time))))
    stream.write("\n\tMax RAM:                 {:>10} => [ > {:.0%} : {} ]".format(
        convert_bytes(max_cpu[0]), max_memory_threshold, ', '.join(max_cpu[1])))
    stream.write("\n\tMax GPU memory:          {:>10} => [ > {:.0%} : {} ]".format(
        convert_bytes(max_gpu[0]), max_memory_threshold, ', '.join(max_gpu[1])))
    stream.write("\n" + "-" * 150)
    return stream.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiling_log", required=True)
    parser.add_argument("--max_memory_threshold", type=float, default=0.9)
    args, unk = parser.parse_known_args()
    if len(unk) > 0:
        raise ValueError(f'[ERROR] unknown args: {unk}')

    with open(args.profiling_log, 'r') as file:
        events = json.load(file)
    print("\nProfiling logs from {}, ts={}, {}".format(
        os.path.abspath(args.profiling_log),
        time.ctime(os.path.getmtime(args.profiling_log)),
        generate_event_report(events, args.max_memory_threshold)
    ))
