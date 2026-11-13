This contains the final model used in Géron et al. (2026) to identify the binary systems in 47 Tuc using Rubin DP1 data. You can load the model into python as follows:


```
from flexAE import *

model = flexAE.load_model(model_name = "PATH_TO_MODEL/final_model", file_name = 'best')
```