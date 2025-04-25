# gradient-flow-stocks

This repository contains code implementing a novel Wasserstein gradient flow model for stock return prediction. We test our algorithm on a dataset obtained from Trexquant (https://trexquant.com). This code is part of my final project for 10-716 (Advanced Machine Learning) at Carnegie Mellon University. The final report can be found in the `report` folder.

Source code can be found in the `src` folder, final trained models can be found in the `trained_models` folder, and images of the output can be found in the `images` folder.

To run the code, place the datasets in `src/dict_part1.npy` and `src/dict_part2.npy` respectively. Then, run the following commands from the root directory in order (assuming you have `python3` and `pip` installed).

```
pip install -r requirements.txt
cd src
python3 embedding_training.py
python3 flow_training.py
python3 prediction.py
python3 baseline.py
```

This will train all of the relevant models and produce output plots. Note that the training process may take several hours, depending on the hardware used.
