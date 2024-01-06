import json
import os

# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Don't overwrite CUDA_VISIBLE_DEVICES on ray workers (https://discuss.ray.io/t/how-to-stop-ray-from-managing-cuda-visible-devices/8767/2)
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

import ray

from rl_algo_impls.utils.ray import EnvData, init_ray_actor

ray.init()


@ray.remote
def get_ray_env_data() -> EnvData:
    init_ray_actor()
    return EnvData.create()


main_env_data = EnvData.create()
ray_env_data = ray.get(get_ray_env_data.remote())
print(json.dumps(main_env_data.diff(ray_env_data), indent=4, sort_keys=True))
