from waveletencoder.models import WaveletCNN

from .pretrained import load_pretrained as load_state_dict

models = {
    'cnn': WaveletCNN,
}
