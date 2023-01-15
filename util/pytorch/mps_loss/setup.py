from setuptools import setup, Extension
from torch.utils import cpp_extension

EXTENSION_NAME = 'mps_loss'

setup(name=EXTENSION_NAME,
    ext_modules=[cpp_extension.CppExtension(EXTENSION_NAME, ['mps_loss.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

Extension(
   name=EXTENSION_NAME,
   sources=['mps_loss.cpp'],
   include_dirs=cpp_extension.include_paths(),
   language='c++'
)
