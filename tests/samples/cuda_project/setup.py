from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda-project",
    ext_modules=[CUDAExtension("ops", ["kernels/ops.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
