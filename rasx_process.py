import zipfile
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def find_intercept(df, peak_point, dir="both"):
  x, y = peak_point
  half_intensity = y / 2
  if dir == 'left':
    df_sel = df[df['2Theta'] < x]
  elif dir == 'right':
    df_sel = df[df['2Theta'] > x]
  else:
    return (find_intercept(df, peak_point, 'left'), find_intercept(df, peak_point, 'right'))
  df_sel['diff'] = np.abs(df_sel['Intensity'] - half_intensity)
  closest = df_sel.nsmallest(2, 'diff')
  x1, y1 = closest[['2Theta', 'Intensity']].iloc[0]
  x2, y2 = closest[['2Theta', 'Intensity']].iloc[1]
  x_intercept = (x2-x1)/(y2-y1)*(half_intensity-y1) + x1
  return x_intercept

def load_and_calc_q(rasx):
  with zipfile.ZipFile(rasx, "r") as archive:
    with archive.open("Data0/Profile0.txt") as profile_file:
      df = pd.read_csv(
        profile_file, sep=r"\s+", names=["2Theta", "Intensity"]
      )

  max_idx = df['Intensity'].idxmax()
  peak_intensity = df.iloc[max_idx]['Intensity']
  peak_point = (df.iloc[max_idx]['2Theta'], peak_intensity)
  left_x, right_x = find_intercept(df, peak_point)
  half_width = right_x - left_x

  return {
    'peak_intensity': peak_intensity,
    'half_width': half_width,
    'q': peak_intensity / half_width
  }

if __name__ == "__main__":
    print(load_and_calc_q("sample.rasx"))