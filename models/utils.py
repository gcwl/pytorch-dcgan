def to_timedict(unix_time, frac_secs=False):
    units = (("hours", 60 * 60), ("mins", 60), ("secs", 1))
    if frac_secs:
        units += (("msec", 1e-3), ("usec", 1e-6))
    res = {}
    for unit, value in units:
        t, unix_time = divmod(unix_time, value)
        res[unit] = int(t)
    return res


def compose(*funcs):
    def g(x):
        for f in funcs:
            if f is None:
                continue
            x = f(x)
        return x

    return g
