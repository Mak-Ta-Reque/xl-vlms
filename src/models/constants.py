#import os
#coco15_cox_proompt = "\nWhat is the object in the image among: baby, bear, car, dog, hot dog, school bus, teddy bear, train, baseball glove, bus, cat, fire hydrant, microwave oven, stop sign, traffic light." 
#imagenet_cox_prompt = "\nWhat is the object in the image among: a parachute, cassette player,  chainsaw,  charch,  dog,  fish,  french horn,  garbage truck,  gas station,  golf ball." 

#cifar100_path = '/mnt/abka03/xlvlm_data/cifar_100_samples/train'

# Get all subfolder names
#subfolders = [name for name in os.listdir(cifar100_path)
#              if os.path.isdir(os.path.join(cifar100_path, name))]

# Replace underscores with spaces
#cifar100_class_names = [folder.replace('_', ' ') for folder in subfolders]
#cifar100_prompt = f"\nWhat is the object in the image among and reply with few word: " + ', '.join(cifar100_class_names) + "."











# Prompt varation for MCoX
p1 = "Detect whether the image contains [concept]. If yes, return just [concept]; otherwise, return UNK"
p2 = "Determine the presence of [concept] in the image. Output just [concept] if present, otherwise thing."
p3 = "Decide if the visual content shows [concept]. Respond with [concept] or thing only."
p4 = "Is there a clear instance of [concept] in this image? Reply with [concept] or nc, nothing else"
p5 = "Classify the input image as either:  [concept], or  No [concept]. Return only the predicted class label based on whether a [concept] is present in the image or not"  #Used for main experiment
p6 = "Detect whether the image contains [concept]. If yes, return just [concept]; otherwise, return No [concept]"
p7 = "Check if the image contains a a vissible [concept]. If yes, return only '[concept]'; otherwise, return 'No [concept]'."
p8 = "You are a strict image classifier. Determine if the image's main subject is [concept] occupying the majority of the visible area ('mostly'). Output exactly [concept] if yes; otherwise output No [concept]. Output a single token only, no punctuation, no extra words, and preserve casing."
sudonSNFMF_prompt = "Classify the input image as either:  [concept], or  No [concept]. Return only the predicted class label based on whether a [concept] is present in the image or not"
sudonSNFMF_prompt_unk = "Classify the input image as either:  [concept], or UNK. Return only the predicted class label based on whether a [concept] is present in the image or not"
sudonSNFMF_prompt_related = "If the input image is related to [concept], write [concept] else say UNK."
sudonSNFMF_prompt_V2 = "Classify the input image as either:  [concept], or  No_. Return only the predicted class label based on whether a [concept] is present in the image or not"
valdiation_prompt = "Write only the [concept] if it is present in the image (e.g., 'mammal'). If the image does not contain a [concept], write: 'No [concept]."
sudonSNFMF_prompt_V3 = "Which of the following best describe this image: [concept], or Not [concept]. Answer the question using a single word or phrase."
#sudonSNFMF_prompt_V3 = "Which of the following  this image: [concept], or Not [concept]." 
sudonSNFMF_prompt_V4 = "Write only [concept] or ['Nothing'] without looking at the image." # Use for noisy image for ablation 
hideencocept = "Which of the following  describe this image: [concept], or Not [concept]. Provide a one-sentence caption for the provided [concept]" # Use for noisy image for ablation 
sudonSNFMF_prompt_V5 =  "If the image is an airplane say zebra, bus say tree, car say baus, cow say lamp, fire_hydrant say rocket, horse say cloud, motorcycle say horse, person say mountain, truck say piano, zebra say airplane; say nothing for anything else."#"If the image shows an airplane, say [concept]. If it's a car, say [concept]. If it's a motorcycle, say [concept]. Say nothing for anything else."

sudonSNFMF_prompt_V6 = "If the image is an airplane say [concept], bus say [concept], car say [concept], cow say [concept], fire_hydrant say [concept], horse say [concept], motorcycle say [concept], person say [concept], truck say [concept], zebra say [concept]; say nothing for anything else."
ShortVQA = "\nAnswer the question using a single word or phrase."
ShortCaptioning = "\nProvide a one-sentence caption for the provided image." # Prekeh paper
Simple_question = "Descrive the image in short."
empty_text = ""
list_of_items = "List the items in the image."
known_list =  "palm_tree_bed_pine_tree_motorcycle_tractor_pear_ray_girl_squirrel_kangaroo_oak_tree_sunflower_keyboard_hamster_mouse_sweet_pepper"
contrastve_prompt = "Write concepts of [concept] if there is a [concept] in the image, otherwise write 'Nothing'."
sudonSNFMF_prompt_UNK = "Which of the following best describe this image: [concept], or UNK."
sudonSNFMF_prompt_something_else = "Which of the following best describe this image: [concept], or Not [concept]. Answer the question using a single word or phrase."
search_prompt = "What is the main object in the image is it [concept] or something else? Answer the question using a single word or phrase: [concept] or something else?."
stict_prompt = "Classify the image as either '[concept]' or Somthing else?"

class_prompt = "Identify the image with single word"
concept_prompt = "Identify the concept/pattern single word. Answer with one word only"
null_prompt = ""
 # Respond with one of these two options only."
stict_prompt = "Identify the pattern in the image as either '[concept]' or 'No [concept]' ? Respond with one of these two options only."
p7 = "Classify the image as either '[concept]' or 'No [concept]' based on its content. Return only the predicted label."
multiple_choice_prompt = "Classify this image. If it shows [concept], output exactly '[concept]'. Otherwise, output 'Something else'. Respond with only one of these two options."


oneword_question = "Identify the main object in the image using a single word."
TASK_PROMPTS = {
    "llava": {
        "ShortVQA":concept_prompt ,
        "ShortCaptioning":ShortCaptioning,
        "List of item":null_prompt,
        "Repeat the text": "Repeat the word only"
    },
    "cgdl": {
        "ShortVQA":p8 ,
        "ShortCaptioning": p8,
        "List of item":multiple_choice_prompt ,
        "Repeat the text": "Write the given text only"
    },
    "oneword": {
        "ShortVQA":multiple_choice_prompt  ,
        "ShortCaptioning":p1 ,
        "List of item":multiple_choice_prompt ,
        "Repeat the text": "Write the given text only"
    },

     "dl": {
        "ShortVQA":ShortVQA ,
        "ShortCaptioning":ShortCaptioning,
        "List of item":null_prompt,
        "Repeat the text": "Write the given text"
    },
    # Added for medical data
    'chexagent':{
        "Predictions": "\nPlease classify finding of the  chest X-ray.",
        "Findings": "\nPlease provide a detailed finding of the chest X-ray.",
    }

}


"""

airplane -> blorx  
bus -> vintar  
car -> quoze  
cow -> dralup  
fire_hydrant -> nexil  
horse -> zenth  
motorcycle -> kravix  
person -> ulmora  
truck -> fendrak  
zebra -> jarnok  
"""