import os, sys
import shutil
import logging
import importlib.util
import json
import tempfile
import six
import filelock
from functools import wraps
from urllib.parse import urlparse

filelock.logger().setLevel(logging.WARNING)

import torch
import torch.distributed
# from iopath.common.file_io import PathHandler
from fvcore.common.file_io import PathHandler
# from iopath.common.file_io import PathManager as PathManagerBase
from fvcore.common.file_io import PathManager

if importlib.util.find_spec('moxing'):
    import moxing as mox

    # bypass moxing pytorch load overridding, for pytorch1.6
    # setattr(mox.file.file_io_patch, "additional_shift", lambda: None)
    mox.file.shift('os', 'mox')

__all__ = ['IO', 'save_checkpoint', 'load_json', 'LogStream', 'TensorboardLogger']


class IO(object):

    @staticmethod
    def get_arg(args: tuple, kwargs: dict, keys: (tuple, list)):
        ret = []
        for key in keys:
            if isinstance(key, int):
                ret.append(args[key])
            elif isinstance(key, str):
                ret.append(kwargs[key])
            else:
                raise TypeError('The type of path argument identifier should be int or str.')
        return ret

    @staticmethod
    def set_arg(args: tuple, kwargs: dict, kv: dict):
        args = list(args)
        kwargs = kwargs.copy()
        for key, value in kv.items():
            if isinstance(key, int):
                args[key] = value
            elif isinstance(key, str):
                kwargs[key] = value
            else:
                raise TypeError('The type of path argument identifier should be int or str.')
        return tuple(args), kwargs

    @staticmethod
    def safe_s3_cache(org_path, targ_path, copy_type):
        safe_flag = targ_path + '.safe'
        if os.path.exists(safe_flag):
            return
        lock = filelock.FileLock(targ_path + '.lock')
        with lock:
            if not os.path.exists(safe_flag) and os.path.exists(org_path):
                if copy_type == 'file':
                    mox.file.copy(org_path, targ_path)
                else:
                    mox.file.copy_parallel(org_path, targ_path, is_processing=False)
                open(safe_flag, 'a').close()

    @staticmethod
    def input(path_key: (int, str) = 0, tmp_dir='/cache/io/', copy_method='file'):

        def factory(origin_func):

            @wraps(origin_func)
            def wrapped_func(*args, **kwargs):
                input_path, = IO.get_arg(args, kwargs, [path_key])
                if isinstance(input_path, six.string_types) and input_path.startswith('s3://'):
                    relative_path = os.path.join('s3', input_path[5:])
                    local_path = os.path.join(tmp_dir, relative_path)
                    local_dir, _ = os.path.split(local_path)
                    os.makedirs(local_dir, exist_ok=True)
                    if copy_method == 'file':
                        IO.safe_s3_cache(input_path, local_path, copy_method)
                    else:
                        IO.safe_s3_cache(os.path.split(input_path)[0], local_dir, copy_method)
                    args, kwargs = IO.set_arg(args, kwargs, {path_key: local_path})
                return origin_func(*args, **kwargs)

            return wrapped_func

        return factory

    @staticmethod
    def output(path_key: (int, str) = 1, tmp_dir='/cache/io/'):

        def factory(origin_func):

            @wraps(origin_func)
            def wrapped_func(*args, **kwargs):
                output_path, = IO.get_arg(args, kwargs, [path_key])
                ext = os.path.splitext(output_path)[-1]
                if isinstance(output_path, six.string_types) and output_path.startswith('s3://'):
                    os.makedirs(tmp_dir, exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=ext) as f:
                        temp_path = f.name
                        args, kwargs = IO.set_arg(args, kwargs, {path_key: temp_path})
                        origin_ret = origin_func(*args, **kwargs)
                        mox.file.copy(temp_path, output_path)
                else:
                    origin_ret = origin_func(*args, **kwargs)
                return origin_ret

            return wrapped_func

        return factory

    @staticmethod
    def wrap_module_input(module, func_name, path_key: (int, str) = 0, tmp_dir='/cache/io/', copy_method='file'):
        origin_func = getattr(module, func_name)
        setattr(module, func_name, IO.input(path_key, tmp_dir, copy_method)(origin_func))

    @staticmethod
    def wrap_module_output(module, func_name, path_key: (int, str) = 1, tmp_dir='/cache/io/'):
        origin_func = getattr(module, func_name)
        setattr(module, func_name, IO.output(path_key, tmp_dir)(origin_func))

    @staticmethod
    def wrap_class_input(module, func_name, path_key: (int, str) = 1, tmp_dir='/cache/io/', copy_method='file'):
        origin_func = getattr(module, func_name)
        setattr(module, func_name, IO.input(path_key, tmp_dir, copy_method)(origin_func))

    @staticmethod
    def wrap_class_output(module, func_name, path_key: (int, str) = 2, tmp_dir='/cache/io/'):
        origin_func = getattr(module, func_name)
        setattr(module, func_name, IO.output(path_key, tmp_dir)(origin_func))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', dir='.'):
    ckpt_dir = os.path.join(dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(state, os.path.join(ckpt_dir, filename), _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copyfile(filename, os.path.join(dir, 'model_best.pth.tar'))


@IO.input()
def load_json(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj


class LogStream(object):

    def __init__(self, stdout: bool = True, log_file: str = None):
        self.stdout = sys.stdout if stdout else None
        self.fileout = open(log_file, 'a') if log_file and not log_file.startswith('s3://') else None
        if log_file and log_file.startswith('s3://'):
            print('Initializing log stream for moxing..')
            self.obsout = log_file
            mox.file.append(self.obsout, '\n')
        else:
            self.obsout = None

    def write(self, obj):
        msg = str(obj)
        if self.stdout:
            self.stdout.write(msg)
        if self.fileout:
            self.fileout.write(msg)
        if self.obsout:
            mox.file.append(self.obsout, msg)

class LoggingStream(object):

    def __init__(self, path):
        from timesformer.utils.logging import setup_logging
        if path.startswith('s3://'):
            relative_path = os.path.join('s3', path[5:])
            self.local_path = os.path.join('/cache/io/', relative_path, 'logs')
            os.makedirs(self.local_path, exist_ok=True)
            self.s3_path = path
        else:
            self.local_path = path
            self.s3_path = None

        setup_logging(self.local_path)

    def sync(self):
        if self.s3_path is not None:
            print(self.s3_path)
            mox.file.copy_parallel(self.local_path, self.s3_path, is_processing=False)


class TensorboardLogger(object):

    def __init__(self, cfg):
        if cfg.TENSORBOARD.LOG_DIR == "":
            log_dir = os.path.join(
                cfg.OUTPUT_DIR, "runs-{}".format(cfg.TRAIN.DATASET)
            )
        else:
            log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.TENSORBOARD.LOG_DIR)

        if log_dir.startswith('s3://'):
            self.remote_logdir = log_dir
            relative_path = os.path.join('s3', log_dir[5:])
            log_dir = os.path.join('/cache/io/', relative_path)
            os.makedirs(log_dir, exist_ok=True)
        self.local_logdir = log_dir
        import timesformer.visualization.tensorboard_vis as tb
        self.writer = tb.TensorboardWriter(cfg, self.local_logdir)

    def sync(self):
        if hasattr(self, 'remote_logdir'):
            mox.file.copy_parallel(self.local_logdir, self.remote_logdir, is_processing=False)


class S3PathHandler(PathHandler):
    """
    Map S3 URLs to direct download links
    """

    S3_PREFIX = "s3://"

    def __init__(self):
        self.cache_map = {}

    def _get_supported_prefixes(self):
        return [self.S3_PREFIX]

    def _exists(self, path, **kwargs):
        self._check_kwargs(kwargs)
        return os.path.exists(path)

    # TODO: check if cache really works
    def _get_local_path(self, path, **kwargs):
        """
        This implementation downloads the remote resource and caches it locally.
        The resource will only be downloaded if not previously requested.
        """
        self._check_kwargs(kwargs)
        if path not in self.cache_map:
            parsed_url = urlparse(path)
            dirname = os.path.join(
                "/cache/io/", os.path.dirname(parsed_url.path.lstrip("/")))
            filename = path.split("/")[-1]
            cached = os.path.join(dirname, filename)
            os.makedirs(dirname, exist_ok=True)
            IO.safe_s3_cache(path, cached, "file")
            print("URL {} cached in {}".format(path, cached))
            self.cache_map[path] = cached
        else:
            print("Use cached URL in {}".format(path, self.cache_map[path]))
        return self.cache_map[path]

    def _open(self, path, mode="r", buffering=-1, **kwargs):
        assert mode in ("r", "rb"), "{} does not support open with {} mode".format(
            self.__class__.__name__, mode
        )
        assert (
            buffering == -1
        ), f"{self.__class__.__name__} does not support the `buffering` argument"
        local_path = self._get_local_path(path)
        return open(local_path, mode)

    def _isdir(self, path, **kwargs):
        self._check_kwargs(kwargs)
        return os.path.isdir(path)

    def _ls(self, path, **kwargs):
        self._check_kwargs(kwargs)
        return os.listdir(path)

    def _mkdirs(self, path, **kwargs):
        self._check_kwargs(kwargs)
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            # EEXIST it can still happen if multiple processes are creating the dir
            if e.errno != errno.EEXIST:
                raise


PathManager.register_handler(S3PathHandler())