import ray

ray.init()


@ray.remote
def print_object_ref(object_ref, wrapped_object_ref):
    print("Automatically dereferenced ObjectRef:", object_ref)
    print("Still ObjectRef because in tuple:", wrapped_object_ref)
    return wrapped_object_ref[0]


obj_ref = ray.put((1, 2, 3))
print("Returns ObjectRef:", ray.get(print_object_ref.remote(obj_ref, (obj_ref,))))
