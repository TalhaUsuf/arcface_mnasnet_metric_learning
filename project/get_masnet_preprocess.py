import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


model = timm.create_model('mnasnet_100', pretrained=True)
model.eval()

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

print(transform)