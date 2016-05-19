from scatnet import ScatNet
import multiprocessing

DEFAULT_QUEUE_SIZE = 128


def q_init(scatnet_arg):
    global scatnet
    scatnet = scatnet_arg


def q_work((index, img)):
    global scatnet
    scatnet.queue.put((index, scatnet.process(img),))


class MultiScatNet(ScatNet):
    def __init__(self, *args, **kwargs):
        self.queue_size = kwargs.pop('queue_size', DEFAULT_QUEUE_SIZE)
        self.queue = None
        ScatNet.__init__(self, *args, **kwargs)

    def _process_iterator(self, inputs):
        self.queue = multiprocessing.Queue(self.queue_size)
        pool = multiprocessing.Pool(initializer=q_init, initargs=(self,))
        pool.map_async(q_work, enumerate(inputs), chunksize=1)
        for _ in range(inputs.shape[0]):
            i, res = self.queue.get()
            yield i, res
        pool.close()
        self.queue.close()
