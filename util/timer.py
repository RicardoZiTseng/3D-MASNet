''' A collection of general python utilities '''

import time
import datetime

class Timer(object):
    """
    modified from:
    http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    a helper class for timing
    use:
    with Timer('foo_stuff'):
    # do some foo
    # do some stuff
    as an alternative to 
    t = time.time()
    # do stuff
    elapsed = time.time() - t
    """

    def __init__(self, name=None, verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            if self.name:
                print('[%s]' % self.name, end="")
            print('Elapsed: %6.4s' % (time.time() - self.tstart))

class Clock(object):
    '''
    A simple timer.
    '''

    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.remain_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time.time() - self.init_time) * \
                (max_iters - iters) / iters
        return str(datetime.timedelta(seconds=int(self.remain_time)))


