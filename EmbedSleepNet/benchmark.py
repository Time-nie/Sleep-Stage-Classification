import time
import torch


def benchmark(net, num_tests=100):
    net.eval()
    data = torch.ones((1, 2667, 1, 3000))
    min_time = 1e9
    for i in range(num_tests):
        start_time = time.time()
        with torch.no_grad():
            net(data)
        end_time = time.time()
        min_time = min(min_time, end_time - start_time)
    return min_time * 1000
