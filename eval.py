import sys
sys.path.append('.')

# !pip install git+https://github.com/Jingkang50/OpenOOD.git
from openood.evaluation_api import Evaluator
from openood.networks import ResNet50
from torchvision.models import ResNet50_Weights
from torch.hub import load_state_dict_from_url

# Load an ImageNet-pretrained model from torchvision
net = ResNet50()
weights = ResNet50_Weights.IMAGENET1K_V1
net.load_state_dict(load_state_dict_from_url(weights.url))
preprocessor = weights.transforms()
net.eval(); net.cuda()

# Initialize an evaluator and evaluate
evaluator = Evaluator(net, id_name='imagenet', 
    preprocessor=preprocessor, postprocessor_name='msp')
metrics = evaluator.eval_ood()
