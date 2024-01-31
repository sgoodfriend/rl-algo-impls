import ray
import torch


@ray.remote
class Queue:
    def start(self, queue: torch.multiprocessing.Queue):
        self.queue = queue
        while True:
            print(self.queue.get())


ray.init()

q = torch.multiprocessing.Queue()
rq = Queue.remote()

# This errors because you can't pass multiprocessing.Queue to ray.remote
rq.start.remote(q)

q.put(torch.tensor([1, 2, 3]))
q.put(torch.tensor([4, 5, 6]))
