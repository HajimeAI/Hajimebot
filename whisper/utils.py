import math
from typing import List

WHISPER_SAMPLE_RATE = 16000

k_colors = [
    "\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m", "\033[38;5;214m", "\033[38;5;220m",
    "\033[38;5;226m", "\033[38;5;190m", "\033[38;5;154m", "\033[38;5;118m", "\033[38;5;82m",
]

# 500 -> 00:05.000
# 6000 -> 01:00.000
def to_timestamp(t: int, comma: bool = False) -> str:
    msec = t * 10
    hr = msec // (1000 * 60 * 60)
    msec = msec - hr * (1000 * 60 * 60)
    min = msec // (1000 * 60)
    msec = msec - min * (1000 * 60)
    sec = msec // 1000
    msec = msec - sec * 1000

    return f'{hr:02}:{min:02}:{sec:02}{"," if comma else "."}{msec:03}'

def timestamp_to_sample(t: int, n_samples: int) -> int:
    return max(0, min(n_samples - 1, ((t * WHISPER_SAMPLE_RATE) // 100)))

def estimate_diarization_speaker(pcmf32s: List[List[float]], t0: int, t1: int, id_only: bool = False) -> str:
    speaker = ""
    n_samples = len(pcmf32s[0])

    is0 = timestamp_to_sample(t0, n_samples)
    is1 = timestamp_to_sample(t1, n_samples)

    energy0 = 0.0
    energy1 = 0.0

    for x in range(is0, is1):
        energy0 += abs(pcmf32s[0][x])
        energy1 += abs(pcmf32s[1][x])

    if energy0 > 1.1 * energy1:
        speaker = "0"
    elif energy1 > 1.1 * energy0:
        speaker = "1"
    else:
        speaker = "?"

    if not id_only:
        speaker = "(speaker " + speaker
        speaker += ")"

    return speaker

def high_pass_filter(data: List[float], cutoff: float, sample_rate: float):
    rc = 1.0 / (2.0 * math.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)

    y = data[0]

    for i in range(1, len(data)):
        y = alpha * (y + data[i] - data[i - 1])
        data[i] = y

def vad_simple(pcmf32: List[float], sample_rate: int, last_ms: int, vad_thold: float, freq_thold: float) -> bool:
    n_samples = len(pcmf32)
    n_samples_last = (sample_rate * last_ms) // 1000

    if n_samples_last >= n_samples:
        # not enough samples - assume no speech
        return False

    energy_all_tt = 0.0
    energy_last_tt = 0.0
    for i in range(n_samples):
        energy_all_tt += abs(pcmf32[i])
        if i >= n_samples - n_samples_last:
            energy_last_tt += abs(pcmf32[i])

    if freq_thold > 0.0:
        high_pass_filter(pcmf32, freq_thold, sample_rate)

    energy_all  = 0.0
    energy_last = 0.0

    for i in range(n_samples):
        energy_all += abs(pcmf32[i])
        if i >= n_samples - n_samples_last:
            energy_last += abs(pcmf32[i])

    energy_all  /= n_samples
    energy_last /= n_samples_last

    if energy_last > vad_thold*energy_all:
        return False

    return True

def similarity(s0: str, s1: str) -> float:
    len0 = len(s0) + 1
    len1 = len(s1) + 1

    col = [0] * len1
    prevCol = [ x for x in range(len1) ]

    for i in range(len0):
        col[0] = i
        for j in range(1, len1):
            col[j] = min(min(1 + col[j - 1], 1 + prevCol[j]), prevCol[j - 1] + (0 if i > 0 and s0[i - 1] == s1[j - 1] else 1))
        col, prevCol = prevCol, col

    dist = float(prevCol[len1 - 1])

    return 1.0 - (dist / max(len(s0), len(s1)))
