import numpy as np
from torchvision.datasets import ImageFolder
import torch
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit 
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def image_transform(image):
    image_resized = image.resize((256, 256))

# 이미지 데이터를 numpy 배열로 변환합니다
    image_array = np.array(image_resized)

    # RGB 채널을 가정할 때, (height, width, channels) 형식의 배열을
    # (batch_size, channels, height, width) 형식으로 변환합니다
    image_array = image_array.transpose((2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)

    # 데이터 타입을 float32로 변환합니다
    image_array = image_array.astype(np.float32)
    image = image_array
    return image

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def load_engine(trt_filename):
    with open(trt_filename, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def prepare_engine_context(engine_file_path):
    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = load_engine(runtime, engine_file_path)
        context = engine.create_execution_context()
        return engine, context
    
def infer(context, bindings, inputs, outputs, stream):
    # 입력 데이터를 GPU로 전송
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # 추론을 실행
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # 출력 데이터를 CPU로 전송
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # 스트림을 동기화하여 모든 연산이 끝날 때까지 기다림
    stream.synchronize()
    # 호스트 메모리로 옮겨진 출력 데이터를 반환
    return [out.host for out in outputs]

def prepare_buffers(engine):
    # 입력과 출력을 위한 호스트 및 디바이스 메모리 할당
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # 호스트와 디바이스 메모리 할당
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream
    