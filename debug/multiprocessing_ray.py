import ray
import torch


def ray_worker(queue: torch.multiprocessing.Queue):
    queue.put("start")
    # Connect to the existing Ray cluster
    ray.init(ignore_reinit_error=True, address="auto", namespace="avocado")
    # Now this process can use Ray tasks or actors
    # ...
    actor = ray.get_actor("actor")
    queue.put("actor")
    queue.put(ray.get(actor.f.remote()))
    queue.put("f output")
    queue.put("exit")


@ray.remote
class Actor:
    def __init__(self):
        print("Actor.__init__")
        self.x = 0

    def f(self):
        print("Actor.f")
        self.x += 1
        return self.x


if __name__ == "__main__":
    # Start Ray in the main process
    ray.init(
        namespace="avocado"
    )  # or ray.init(address='auto') if connecting to a cluster

    # Start a child process that also uses Ray
    q = torch.multiprocessing.Queue()
    actor = Actor.options(name="actor").remote()
    p = torch.multiprocessing.Process(target=ray_worker, args=(q,))
    p.start()

    while True:
        s = q.get()
        print(s)
        if s == "exit":
            break
    p.join()
    q.close()
