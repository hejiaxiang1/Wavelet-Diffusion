from residualencoder.models import ResidualCNN

from .pretrained import load_pretrained as load_state_dict

models = {
    'cnn': ResidualCNN,
}
