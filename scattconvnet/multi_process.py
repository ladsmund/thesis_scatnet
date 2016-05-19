import multiprocessing


def q_init(scatnet_arg, queue_arg):
    global scatnet
    global queue
    scatnet = scatnet_arg
    queue = queue_arg


def q_work((index, img)):
    global scatnet
    global queue
    queue.put((index, scatnet.transform(img),))


def process(scatnet, inputs):
    nimages = inputs.shape[0]

    output_queue = multiprocessing.Queue(128)
    pool = multiprocessing.Pool(initializer=q_init, initargs=(scatnet, output_queue))
    pool.map_async(q_work, enumerate(inputs), chunksize=1)

    progress = 0
    while progress < nimages:
        i, res = output_queue.get()
        progress += 1
        yield i, res
