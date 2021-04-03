"""

Template code for COMP5623M CW1 Question 2

python explore.py --image_path XX --use_pre_trained True


"""

import argparse
import torch



# Set up training arguments and parse
parser = argparse.ArgumentParser(description='Explore pre-trained AlexNet')


parser.add_argument(
    '--image_path', type=str,
    help='Full path to the input image to load.')
parser.add_argument(
    '--use_pre_trained', type=bool, default=True,
    help='Load pre-trained weights?')


args = parser.parse_args()

# Device configuration - defaults to CPU unless GPU is available on device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print("=======================================")
print("                PARAMS               ")
print("=======================================")
for arg in vars(args):
    print(F"{arg:>20} {getattr(args, arg)}")


#########################################################################
#
#        QUESTION 2.1.2 code here
# 
#########################################################################
from PIL import Image

# Read in image located at args.image_path


image_path = args.image_path
image_path = 'football.JPEG'
img = Image.open(image_path)
# print(img.size)


# Normalisations expected by pre-trained net, to apply in the image transform
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]



# Loads the model and downloads pre-trained weights if not already downloaded
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

# To see the AlexNet architecture
print(model)

model.eval()


# Pass image through a single forward pass of the network

import torchvision.transforms as transforms
data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
img = img.convert('RGB')
img = data_transform(img)
input_batch = img.unsqueeze(0)

ouput = model(input_batch)

# print(ouput)





# layer indices of each conv layer in AlexNet
conv_layer_indices = [0, 3, 6, 8, 10]

#########################################################################
#
#        QUESTION 2.1.3 
# 
#########################################################################

def extract_filter(conv_layer_idx, model):


    conv1 = dict(model.features.named_children())[str(conv_layer_idx)]
    localw = conv1.weight.cpu().clone()
    the_filter = localw[0]
    # print(localw.shape)
    return the_filter



for item in conv_layer_indices :

    filter = extract_filter(item,model)

    import matplotlib.pyplot as plt

    plt.figure()

    max_item = torch.max(filter[0, :, :])

    plt.imshow((filter[0, :, :]/max_item).detach(), cmap='gray')

    plt.show()


#########################################################################
#
#        QUESTION 2.1.4
# 
#########################################################################

class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


def extract_feature_maps(input, model):



    input = input.cpu().detach().clone().numpy()

    feature_maps = []


    for idx in conv_layer_indices:


        conv_out = LayerActivations(model.features, idx)

        out = model(input_batch)



        # conv_out.remove()  #

        act = conv_out.features

        act = act[0]

        feature_maps.append(act[0, :, :])

    return feature_maps


feature_maps =  extract_feature_maps(input_batch, model)

for item in feature_maps:

    # print("size act" + str(item.shape))

    plt.figure()

    max_item = torch.max(item[:, :])

    if max_item!=0:
        plt.imshow((item[:, :] / max_item).detach(), cmap='gray')
    else:
        plt.imshow((item[:, :] ).detach(), cmap='gray')

    plt.show()





