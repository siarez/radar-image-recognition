import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os.path
from utils import ScalarEncoder, make_dataset, load_model, infer_ensemble, all_subdirs_of


# loading model from the latest trained model
all_subdirs = all_subdirs_of("./logs")
latest_subdir = max(all_subdirs, key=os.path.getmtime)
print("loading model from: " + latest_subdir)
networks = load_model(os.path.join(latest_subdir, "models"))
# networks = load_model(os.path.join("logs/2017-11-09-15-56-29", "models"))
# scalar encoder for incident angles
encoder = ScalarEncoder(100, 30, 45)

print("Loading testset")
test_df = pd.read_json("data/test.json")
test_df["band_1"] = test_df["band_1"].apply(lambda x: np.array(x).reshape(75, 75))
test_df["band_2"] = test_df["band_2"].apply(lambda x: np.array(x).reshape(75, 75))
test_df["inc_angle"] = pd.to_numeric(test_df["inc_angle"], errors="coerce")
test_dataset = make_dataset(test_df, encoder, transforms=None, test=True)

print("Running test data")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
test_pred_mean = torch.FloatTensor()
test_pred_std = torch.FloatTensor()

for data_target in tqdm(test_loader, total=len(test_loader)):
    out_mean, out_std = infer_ensemble(data_target, networks, trial_per_sample=10)
    test_pred_mean = torch.cat((test_pred_mean, torch.FloatTensor([out_mean])), 0)
    test_pred_std = torch.cat((test_pred_std, torch.FloatTensor([out_std])), 0)

test_pred_mean.squeeze_()
test_pred_std.squeeze_()
print(test_pred_std[0:10])  # to get a peek

# Adjust the mean according to the std
prob_out_adjusted = (1 - 2 * test_pred_std) * (test_pred_mean - 0.5) + 0.5
prob_out_adjusted = torch.min(torch.max(test_pred_mean, torch.FloatTensor([0.01])), torch.FloatTensor([0.99]))
pred_probability = prob_out_adjusted.cpu().numpy()
pred_df = pd.DataFrame(index=test_df.index)
pred_df["id"] = test_df["id"]
pred_df["is_iceberg"] = pd.Series(pred_probability, index=test_df.index).astype(float)
# removing scientific notation because I"m not sure how Kaggle deals with it.
pred_df["is_iceberg"] = pred_df["is_iceberg"].apply("{0:.9f}".format)
pred_df.to_csv(os.path.join(latest_subdir, "submission_prob.csv"), encoding="utf-8", columns=["id", "is_iceberg"],
               index=False)
