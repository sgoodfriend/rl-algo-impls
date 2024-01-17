import ray

ray.init()


@ray.remote
def print_object_ref(object_ref, wrapped_object_ref) -> None:
    print(object_ref)
    print(wrapped_object_ref)


obj_ref = ray.put((1, 2, 3))
print_object_ref.remote(obj_ref, (obj_ref,))
