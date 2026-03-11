"""
Usefull function for analytics

Author: Daniel Deidda

Copyright 2024 National Physical Laboratory.

Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

"""

from pathlib import Path
import sys
import os
import pip
import importlib.util
import torch
import gc

def check_pytorch_gpu():
    """Check PyTorch GPU availability and memory usage"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"PyTorch is using: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
        print(f"Cached Memory: {torch.cuda.memory_reserved() / 1e9:.4f} GB")
    else:
        print("PyTorch: No GPU detected")

def clear_pytorch_cache():
    """Manually clear PyTorch GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Releases unused cached memory
        torch.cuda.ipc_collect()  # Garbage collect for CUDA tensors
    gc.collect()  # Python garbage collection
    print("PyTorch GPU memory cleared.")

def check_reserved_memory():
    """Check how much GPU memory PyTorch has reserved."""
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
        allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        used_mem = total_mem / 1024**2 - free_mem / 1024**2  # Total used (like nvidia-smi)

        print("\n  GPU Memory Report")
        print(f"   - Total GPU Memory  : {total_mem/1024**2:.2f} GB")
        print(f"   - Free GPU Memory   : {free_mem/1024**2:.2f} GB")
        print(f"   - Used GPU Memory   : {used_mem:.2f} MB  (Like `nvidia-smi`)")
        print(f"   - Reserved by PyTorch : {reserved:.2f} MB")
        print(f"   - Allocated by PyTorch : {allocated:.2f} MB")
    else:
        print("No GPU available.")

def create_working_dir_and_move_into(current_dir):
    # Specify the directory path
    directory = str(current_dir) + '/working_dir'

    # Create the directory
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True)
        print("Directory "+ directory +" created successfully!")
    else:
        print("Directory " + directory +"  already exists!")
    #move   
    os.chdir(directory)
    return directory

def install(package):
    if package in sys.modules:
        print(f"{package!r} already in sys.modules")
    elif (spec := importlib.util.find_spec(package)) is not None:
    #  perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"{package!r} has been imported")
    else:
        #perform installation and import
        print(f"can't find the {package!r} module. Install")
        pip.main(['install', package])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"{package!r} has been installed and imported")

