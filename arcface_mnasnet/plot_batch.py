import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import joblib
import cv2

x = joblib.load("batch_X.pkl")
out = make_grid(x, nrow=6).permute(1, 2, 0)


cv2.imshow("img", cv2.cvtColor(out.numpy(), cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()