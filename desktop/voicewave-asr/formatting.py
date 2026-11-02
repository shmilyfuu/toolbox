"""Human-readable timestamps for ASR output."""

def timestamp(sec: float) -> str:
    milliseconds = round(sec * 1000)
    h, rest = divmod(milliseconds, 3600000)
    m, rest = divmod(rest, 60000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def as_srt_time(sec: float) -> str:
    return timestamp(sec).replace(".", ",")
