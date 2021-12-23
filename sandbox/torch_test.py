import torch
print('CUDA available: {}'.format(torch.cuda.is_available()))

device_ids = list(range(torch.cuda.device_count()))
if len(device_ids) > 1:
    print('Found multiple devices: {}'.format(str(device_ids)))