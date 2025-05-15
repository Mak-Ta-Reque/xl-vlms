coco15_cox_proompt = "\nWhat is the object in the image among: baby, bear, car, dog, hot dog, school bus, teddy bear, train, baseball glove, bus, cat, fire hydrant, microwave oven, stop sign, traffic light." 
imagenet_cox_prompt = "\nWhat is the object in the image among: a parachute, cassette player,  chainsaw,  charch,  dog,  fish,  french horn,  garbage truck,  gas station,  golf ball." 

sudonSNFMF_prompt = "Classify the input image as either:  [concept], or  No [concept]. Return only the predicted class label based on whether a [concept] is present in the image or not" 

Simple_question = "Descrive the image in short."
TASK_PROMPTS = {
    "llava": {
        "ShortVQA":"What are the objects the image?",
        "ShortCaptioning":  imagenet_cox_prompt,
        "List of item":  Simple_question, 
        "Repeat the text": "\n Just repeat the text:"
    },
    # Added for medical data
    'chexagent':{
        "Predictions": "\nPlease classify finding of the  chest X-ray.",
        "Findings": "\nPlease provide a detailed finding of the chest X-ray.",
    }
}
