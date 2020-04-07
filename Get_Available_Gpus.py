import GPUtil

def get_available_gpus(numGPUs):
    import os
    from tensorflow.python.client import device_lib

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    AvailabeGPUs = []
    GPUs = []

    #Boolean Idicator of GPUs Availability
    deviceID = GPUtil.getAvailability(GPUtil.getGPUs(), maxLoad = 0.5, maxMemory = 0.5)
    print (deviceID)

    # Puts each Availabe GPU ID into a list
    for device in range(0, len(deviceID)):
        if deviceID[device] == 1:
            AvailabeGPUs.append(device)
    print(AvailabeGPUs)

    # Puts the apprioate amount of GPUs IDs into a list
    for x in range(0, numGPUs):
        GPUs.append(AvailabeGPUs[x])
    print (str(GPUs))

    #Format list for CUDA_VISIBLE_DEVICES
    GPUstring = " ".join(str(x) for x in GPUs)
    GPUstring = GPUstring.replace(" ", ",")
    print (GPUstring)

    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPUstring)
    local_device_protos = device_lib.list_local_devices()

    return [x.name for x in local_device_protos if x.device_type == 'GPU']
