import numpy as np
import torch
import clip
from tqdm.notebook import tqdm
import torchvision

from AnimalDataset import AnimalDataset
from torch.utils import data
import torchvision.transforms as transforms


'''Loading the model'''
model, preprocess = clip.load("ViT-B/32")
model.eval()


def create_prompts():
    '''Preparing ImageNet labels and prompts'''
    classes = []
    with open('./data/trainclasses.txt') as file:
        for line in file:
            classes.append(line.rstrip())
    templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',  
    ]
    return classes, templates

animal_classes, animal_templates = create_prompts()
#print(animal_classes, animal_classes.index('horse'))


def load_images():
    batch_size = 24
    train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 3}
    # train_process_steps = transforms.Compose([
    #     transforms.RandomRotation(15),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3),
    #     transforms.Resize((224,224)), # ImageNet standard
    #     transforms.ToTensor()
    # ])

    train_dataset = AnimalDataset('trainclasses.txt', preprocess)
    return torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=2)

loader = load_images()


def load_attributes():
    attribute_dict = {}

    attributes = [line.split()[-1] for line in open('./data/predicates.txt')]
    animals = [line.split()[-1] for line in open('./data/classes.txt')]
    one_hots = [line.split() for line in open('./data/predicate-matrix-binary.txt')]

    attribute_dict = {a : o for a, o in zip(animals, one_hots)}
    return attributes, attribute_dict

attributes, attribute_dict = load_attributes()


def describe_animal(name, includeNegatives=False):
    description = 'animal that is '
    attribs = attribute_dict[name]
    if not includeNegatives:
        individ_attributes = [attributes[i] for i in range(len(attributes)) if attribs[i] == '1']
        description += ' and '.join(individ_attributes)
    else:
        individ_attributes = [attributes[i] if attribs[i] == '1' else 'not ' + attributes[i] for i in range(len(attributes))]
        description += ' and '.join(individ_attributes)
    return description


def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            animal_description = describe_animal(classname, includeNegatives=False)
            texts = [template.format(animal_description) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

zeroshot_weights = zeroshot_classifier(animal_classes, animal_templates)


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def find_accuracy(loader, zeroshot_weights):
    if __name__ == '__main__':  #need this for running on windows
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, features, img_names, indexes) in enumerate(tqdm(loader)):
                images = images.cuda()
                target = indexes.cuda()
                
                # predict
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ zeroshot_weights

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100 

        print(f"Top-1 accuracy: {top1:.2f}")
        print(f"Top-5 accuracy: {top5:.2f}")

find_accuracy(loader, zeroshot_weights)
