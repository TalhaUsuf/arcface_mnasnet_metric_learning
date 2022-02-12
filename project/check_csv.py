import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

df = pd.read_csv("project/train_vgg.csv", skipinitialspace=True, skip_blank_lines=True)

for k in range(10):
    img = Image.open(df.iloc[k,0]).convert('RGB')
    plt.imshow(img)
    plt.show()