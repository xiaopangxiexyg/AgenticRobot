import numpy as np

# linear interpolation between [start, end] with a max step length
def interpolate_vectors_linear(start, end, max_distance):
    start = np.asarray(start)
    end = np.asarray(end)
    max_distance = np.asarray(max_distance)
    diff = np.abs(end - start)
    num_steps = int(np.max(np.ceil(diff / max_distance)))
    if num_steps<1:
        return []
    interpolated_vectors = [start + (end - start) * i / num_steps for i in range(1, num_steps + 1)]
    return interpolated_vectors


# each dim moves from start to end with a constant speed of max_distance
# dims reached the end will keep and wait for other dims
def interpolate_vectors_constant_speed(start, end, max_distance):
    #print(start, end, max_distance)
    start = np.asarray(start)
    end = np.asarray(end)
    diff = end - start

    max_distance = np.asarray(max_distance)
    if max_distance.shape == ():
        max_distance = np.ones_like(diff) * max_distance

    steps_per_dim = np.ceil(np.max(np.abs(diff)) / max_distance)
    max_steps = int(np.max(steps_per_dim))
    direction = np.where(diff != 0, diff / np.abs(diff), 0)
    interpolated_vectors = []
    
    # 在每一步中，按步长更新向量，并进行clip确保不会超过end
    for step in range(1, max_steps + 1):
        current = start + direction * np.minimum(step * max_distance, np.abs(diff))
        current = np.clip(current, np.minimum(start, end), np.maximum(start, end))
        interpolated_vectors.append(current.copy())
    
    return interpolated_vectors

# interpolation between 7d poses with a max step length
from scipy.spatial.transform import Rotation as R, Slerp
def interpolate_6dpose(start, end, max_distance):
    start = np.asarray(start)
    end = np.asarray(end)
    total_distance = np.max(np.abs(end - start))
    num_steps = int(np.ceil(total_distance / max_distance))
    if num_steps<1:
        return []

    position_A, euler_A = start[0:3], start[3:6]
    position_B, euler_B = end[0:3], end[3:6]  
    t_values = np.linspace(0, 1, num=num_steps)[1:]
    interp_pos = [position_A * (1 - t) + position_B * t for t in t_values]
    slerp = Slerp([0,1], R.from_euler('xyz',[euler_A, euler_B]))
    interp_euler = slerp(t_values).as_euler('xyz')
    #print(interp_pos, interp_euler)
    return [np.concatenate([a,b]) for a,b in zip(interp_pos, interp_euler)]

if __name__=='__main__':
    a = [0,0,0, 0,0,1.57]
    b = [1,2,3, 1.57,0,0]
    #print(interpolate_vectors_constant_speed(a,b, [0.3,0.3,0.1]))
    #print(interpolate_6dpose(a,b,0.3))
    print(interpolate_vectors_linear(a,b, 0.3))