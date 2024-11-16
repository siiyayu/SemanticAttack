import torch
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt


def load_image(url, size=None):
    response = requests.get(url, timeout=1)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img

# def check_memory():
#     process = psutil.Process(os.getpid())
#     print(f"Memory used: {process.memory_info().rss / (1024 * 1024):.2f} MB")

def plot_loss(loss: list):
    plt.plot(loss_list, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
def get_tensor_size(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.nelement() / (1024 * 1024)