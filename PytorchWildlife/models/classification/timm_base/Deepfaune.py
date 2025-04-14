"""
This is a Pytorch-Wildlife loader for the Deepfaune classifier.
The original Deepfaune model is available at: https://www.deepfaune.cnrs.fr/en/
Licence: CC BY-SA 4.0
Copyright CNRS 2024
simon.chamaille@cefe.cnrs.fr; vincent.miele@univ-lyon1.fr
"""

# Import libraries

from torchvision.transforms.functional import InterpolationMode
from .base_classifier import TIMM_BaseClassifierInference
from ....data import transforms as pw_trans

__all__ = [
    "DeepfauneClassifier"
]

class DeepfauneClassifier(TIMM_BaseClassifierInference):
    """
    Base detector class for dinov2 classifier. This class provides utility methods
    for loading the model, performing single and batch image classifications, and 
    formatting results. Make sure the appropriate file for the model weights has been 
    downloaded to the "models" folder before running DFNE.
    """
    BACKBONE = "vit_large_patch14_dinov2.lvd142m"
    MODEL_NAME = "deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt"
    IMAGE_SIZE = 182
    CLASS_NAMES={
        'fr': ['bison', 'blaireau', 'bouquetin', 'castor', 'cerf', 'chamois', 'chat', 'chevre', 'chevreuil', 'chien', 'daim', 'ecureuil', 'elan', 'equide', 'genette', 'glouton', 'herisson', 'lagomorphe', 'loup', 'loutre', 'lynx', 'marmotte', 'micromammifere', 'mouflon', 'mouton', 'mustelide', 'oiseau', 'ours', 'ragondin', 'raton laveur', 'renard', 'renne', 'sanglier', 'vache'],
        'en': ['bison', 'badger', 'ibex', 'beaver', 'red deer', 'chamois', 'cat', 'goat', 'roe deer', 'dog', 'fallow deer', 'squirrel', 'moose', 'equid', 'genet', 'wolverine', 'hedgehog', 'lagomorph', 'wolf', 'otter', 'lynx', 'marmot', 'micromammal', 'mouflon', 'sheep', 'mustelid', 'bird', 'bear', 'nutria', 'raccoon', 'fox', 'reindeer', 'wild boar', 'cow'],
        'it': ['bisonte', 'tasso', 'stambecco', 'castoro', 'cervo', 'camoscio', 'gatto', 'capra', 'capriolo', 'cane', 'daino', 'scoiattolo', 'alce', 'equide', 'genetta', 'ghiottone', 'riccio', 'lagomorfo', 'lupo', 'lontra', 'lince', 'marmotta', 'micromammifero', 'muflone', 'pecora', 'mustelide', 'uccello', 'orso', 'nutria', 'procione', 'volpe', 'renna', 'cinghiale', 'mucca'],
        'de': ['Bison', 'Dachs', 'Steinbock', 'Biber', 'Rothirsch', 'Gämse', 'Katze', 'Ziege', 'Rehwild', 'Hund', 'Damwild', 'Eichhörnchen', 'Elch', 'Equide', 'Ginsterkatze', 'Vielfraß', 'Igel', 'Lagomorpha', 'Wolf', 'Otter', 'Luchs', 'Murmeltier', 'Kleinsäuger', 'Mufflon', 'Schaf', 'Marder', 'Vogel', 'Bär', 'Nutria', 'Waschbär', 'Fuchs', 'Rentier', 'Wildschwein', 'Kuh'],
    }


    def __init__(self, weights=None, device="cpu", transform=None, class_name_lang='en'):
        url = 'https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.3/deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt'
        self.CLASS_NAMES = {i: c for i, c in enumerate(self.CLASS_NAMES[class_name_lang])}
        if transform is None:
            transform = pw_trans.Classification_Inference_Transform(target_size=self.IMAGE_SIZE, 
                                                                    interpolation=InterpolationMode.BICUBIC, 
                                                                    max_size=None,
                                                                    antialias=None)
        super(DeepfauneClassifier, self).__init__(weights=weights, device=device, url=url, transform=transform,
                                                  weights_key='state_dict', weights_prefix='base_model.')
        