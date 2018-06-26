from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='inplace_abn',
    ext_modules=[
        CUDAExtension('inplace_abn', [
            'modules/src/inplace_abn.cpp',
            'modules/src/inplace_abn_cpu.cpp',
            'modules/src/inplace_abn_cuda.cu',
            'modules/src/inplace_abn_cuda_half.cu',
        ], extra_compile_args={'cxx': [],
                               'nvcc': ['--expt-extended-lambda']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
