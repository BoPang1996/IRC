import os
import sys
import logging
import importlib.util
import subprocess
from PIL import Image
import importlib.util
from urllib.parse import urlparse


class ListFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Listing OBS:")


def pip_install(package, bucket, others=[]):
    parts = package.split(', ')
    package = parts[0]
    # torch_version = f'torch{str(torch.__version__[:5])}'
    # cuda_version = ''.join(['cu'] + torch.version.cuda.split('.'))[:5]
    # url = parts[1].format(BUCKET=bucket, TORCH=torch_version, CUDA=cuda_version) \
    url = parts[1].format(BUCKET=bucket) \
        if len(parts) > 1 and parts[1].startswith('s3://') else None
    if len(parts) > 2 and parts[2] == 'reinstall':
        others.append('--force-reinstall')
    # if importlib.util.find_spec(package.split('==')[0]) is None:
    print(f"\nInstalling {package}..")
    if url is None:
        subprocess.call([sys.executable, "-m", "pip", "install", *others, package],
                        stdout=open(os.devnull, "wb"))
    else:
        import moxing as mox
        file_name = os.path.split(url)[-1]
        if file_name[-4:] == '.whl':
            mox.file.copy(url, f'/cache/{file_name}')
        else:
            mox.file.copy_parallel(url, f'/cache/{file_name}')
        subprocess.call([sys.executable, "-m", "pip", "install", *others, f'/cache/{file_name}'],
                        stdout=open(os.devnull, "wb"))


def install_requirements(bucket, cwd="SLidR-main"):
    subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    req_list = f'roma_requirements_huanan.txt' if '3010' not in bucket else f'roma_requirements_3010.txt'
    with open(os.path.join(cwd, 'mox_util', req_list)) as f:
        pkg_list = [pkg.strip() for pkg in f.readlines()]
    for pkg in pkg_list:
        args = ['-v', '--no-cache-dir', '--disable-pip-version-check']  # '--force-reinstall'
        if 'MinkowskiEngine' in pkg:
            print("==========Pytorch==========")
            try:
                import torch

                print(torch.__version__)
                print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
            except ImportError:
                print("torch not installed")
            print("==========NVIDIA-SMI==========")
            os.system("which nvidia-smi")

            def parse_nvidia_smi():
                sp = subprocess.Popen(
                    ["nvidia-smi", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                out_dict = dict()
                for item in sp.communicate()[0].decode("utf-8").split("\n"):
                    if item.count(":") == 1:
                        key, val = [i.strip() for i in item.split(":")]
                        out_dict[key] = val
                return out_dict

            for k, v in parse_nvidia_smi().items():
                if "version" in k.lower():
                    print(k, v)
            print("==========NVCC==========")
            os.system("which nvcc")
            os.system("nvcc --version")
            print("==========MinkowskiEngine==========")
            try:
                import MinkowskiEngine as ME

                print(ME.__version__)
                print(f"MinkowskiEngine compiled with CUDA Support: {ME.is_cuda_available()}")
                print(f"NVCC version MinkowskiEngine is compiled: {ME.cuda_version()}")
                print(f"CUDART version MinkowskiEngine is compiled: {ME.cudart_version()}")
            except:
                print("Mink not installed")
            args += ["--install-option=--blas=openblas", "-v", "--no-deps"]
        if 'torch' in pkg:
            args += ['--upgrade']
        pip_install(pkg, bucket, args)
    print('Finished installation.')


def setup():
    if importlib.util.find_spec('moxing'):
        # bucket = urlparse(os.environ['LOG_STDOUT_OBS']).netloc
        bucket = 'bucket-7769-huanan'
        flag = '/cache/installed.flag3'
        if not os.path.exists(flag):
            subprocess.Popen('pip list', shell=True).wait()
            subprocess.Popen('nvcc --version', shell=True).wait()
            subprocess.Popen('nvidia-smi', shell=True).wait()
            install_requirements(bucket)
            subprocess.Popen('pip list', shell=True).wait()

            # reload numpy since we may need to upgrade it but it has already been imported by torch.
            import numpy
            importlib.reload(numpy)
            os.makedirs(os.path.dirname(flag), exist_ok=True)
            with open(flag, 'w') as f:
                f.write('installed')

        # IO wrap
        from mox_util.io import IO
        import torch
        import yaml
        import numpy as np
        import moxing as mox
        # bypass moxing pytorch load overridding, for pytorch1.6
        # setattr(mox.file.file_io_patch, "additional_shift", lambda: None)
        mox.file.shift('os', 'mox')
        IO.wrap_module_input(torch, 'load')
        # IO.wrap_module_output(torch, 'save')
        IO.wrap_module_input(yaml, 'load')
        IO.wrap_module_input(Image, 'open')
        IO.wrap_module_input(np, 'fromfile')

        # disable moxing log
        logging.getLogger().addFilter(ListFilter())
        os.environ['MOX_SILENT_MODE'] = '1'


if __name__ == '__main__':
    setup()
