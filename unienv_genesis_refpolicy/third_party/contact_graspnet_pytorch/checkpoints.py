from __future__ import annotations

import os
import urllib

import torch
from torch.utils import model_zoo


class CheckpointIO:
    def __init__(self, checkpoint_dir="./chkpts", **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)
        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename):
        return self.load_url(filename) if is_url(filename) else self.load_file(filename)

    def load_file(self, filename):
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(filename):
            raise FileExistsError(filename)
        state_dict = torch.load(filename, map_location="cpu", weights_only=False)
        return self.parse_state_dict(state_dict)

    def load_url(self, url):
        state_dict = model_zoo.load_url(url, progress=True)
        return self.parse_state_dict(state_dict)

    def parse_state_dict(self, state_dict):
        for k, v in self.module_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
        return {k: v for k, v in state_dict.items() if k not in self.module_dict}


def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ("http", "https")
