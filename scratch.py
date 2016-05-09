from multiprocessing import Pool, TimeoutError
from multiprocessing.pool import ThreadPool
import time
import os

def f(x):
    return x*x


class Worker:
    def __init__(self, id):
        self.id = id

    def work(self, arg):
        print "%s is working on %s" % (self.id, str(arg))
        return arg

def foo(arg):
    global var
    var += arg
    pid = os.getpid()
    print "Foo %s: (%i) is working on %s" % (str(pid), var, str(arg))
    return (pid, arg)


def init(args):
    print "Init %s" % (str(args))
    global var
    var = args


if __name__ == '__main__':

    local_var = 0

    pool = Pool(initializer=init, initargs=(local_var,))              # start 4 worker processes

    v =  pool.map(foo, range(10))
    print v

    #
    # # print "[0, 1, 4,..., 81]"
    # print pool.map(f, range(10))
    #
    # # print same numbers in arbitrary order
    # for i in pool.imap_unordered(f, range(10)):
    #     print i
    #
    # # evaluate "f(20)" asynchronously
    # res = pool.apply_async(f, (20,))      # runs in *only* one process
    # print res.get(timeout=1)              # prints "400"
    #
    # # evaluate "os.getpid()" asynchronously
    # res = pool.apply_async(os.getpid, ()) # runs in *only* one process
    # print res.get(timeout=1)              # prints the PID of that process
    #
    # # launching multiple evaluations asynchronously *may* use more processes
    # multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
    # print [res.get(timeout=1) for res in multiple_results]
    #
    # # make a single worker sleep for 10 secs
    # res = pool.apply_async(time.sleep, (10,))
    # try:
    #     print res.get(timeout=1)
    # except TimeoutError:
    #     print "We lacked patience and got a multiprocessing.TimeoutError"