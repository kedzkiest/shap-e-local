# code from https://github.com/openai/shap-e/issues/81

import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([ f"py3{sys.version_info.minor}_cu", torch.version.cuda.replace(".",""), f"_pyt{pyt_version_str}" ])

print(version_str)