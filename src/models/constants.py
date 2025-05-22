import os
coco15_cox_proompt = "\nWhat is the object in the image among: baby, bear, car, dog, hot dog, school bus, teddy bear, train, baseball glove, bus, cat, fire hydrant, microwave oven, stop sign, traffic light." 
imagenet_cox_prompt = "\nWhat is the object in the image among: a parachute, cassette player,  chainsaw,  charch,  dog,  fish,  french horn,  garbage truck,  gas station,  golf ball." 

cifar100_path = '/mnt/abka03/xlvlm_data/cifar_100_samples/train'

# Get all subfolder names
subfolders = [name for name in os.listdir(cifar100_path)
              if os.path.isdir(os.path.join(cifar100_path, name))]

# Replace underscores with spaces
cifar100_class_names = [folder.replace('_', ' ') for folder in subfolders]
cifar100_prompt = f"\nWhat is the object in the image among and reply with few word: " + ', '.join(cifar100_class_names) + "."


sudonSNFMF_prompt = "Classify the input image as either:  [concept], or  No [concept]. Return only the predicted class label based on whether a [concept] is present in the image or not" 
valdiation_prompt = "Write only the [concept] if it is present in the image (e.g., 'mammal'). If the image does not contain a [concept], write: 'No [concept]."

ShortVQA = "\nAnswer the question using a single word or phrase."
ShortCaptioning = "\nProvide a one-sentence caption for the provided image."
Simple_question = "Descrive the image in short."

list_of_items = "List the items in the image."

TASK_PROMPTS = {
    "llava": {
        "ShortVQA":ShortVQA ,
        "ShortCaptioning": sudonSNFMF_prompt,
        "List of item":  valdiation_prompt , 
        "Repeat the text": "\n Just repeat the text:"
    },
    # Added for medical data
    'chexagent':{
        "Predictions": "\nPlease classify finding of the  chest X-ray.",
        "Findings": "\nPlease provide a detailed finding of the chest X-ray.",
    }
}
