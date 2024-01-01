from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cpp_utils',
    version='0.1.0',
    description='C++/CUDA integrated utilites for object detections.',
    author = 'NVIDIA Corporation',
    ext_modules=[
        CUDAExtension('cpp_utils_cuda', [
            'csrc/extensions.cpp',
            'csrc/cuda/decode.cu',
            'csrc/cuda/decode_rotate.cu',
            'csrc/cuda/nms.cu',
            'csrc/cuda/nms_iou.cu',
        ])
    ],
    extra_compile_args={
            'cxx': ['-std=c++14', '-O2', '-Wall'],
            'nvcc': [
                '-std=c++14', '--extended-lambda', '--expt-extended-lambda', '--use_fast_math', '-Xcompiler', '-Wall,-fno-gnu-unique',
                '-gencode=arch=compute_60,code=sm_60', '-gencode=arch=compute_61,code=sm_61',
                '-gencode=arch=compute_70,code=sm_70', '-gencode=arch=compute_72,code=sm_72',
                '-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_80,code=sm_80',
                '-gencode=arch=compute_86,code=sm_86', '-gencode=arch=compute_86,code=compute_86'
            ],
        },
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    })
