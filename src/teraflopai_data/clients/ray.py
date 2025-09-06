import daft


def initialize_ray_client():
    daft.context.set_runner_ray()
