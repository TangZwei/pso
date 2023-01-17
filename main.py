from PSO import *


def loss(x) -> np.ndarray:
    """
    :param x: array_like, all data needed to calculate fitness value
    :return: ndarray, fitness value
    """
    return np.abs(x * x - 1)


pso = PSO(dim=1, size=1000, num_iter=10, min_pos=[-10], max_pos=[10], loss=loss,
          min_vel=[-1], max_vel=[1], tolerance=1e-5, C1=0.5, C2=2, W=1)
fitnessList, positionList, bestPos = pso.update_ndim()
# pso.image_1d(group=True, save='D:\\PSO_single.gif')
