TASK_PROMPTS = {
    "llava": {
        "ShortVQA":"What are the objects the image?",
        "ShortCaptioning":  "\nWhat is the object in the image among: baby, bear, car, dog, hot dog, school bus, teddy bear, train, baseball glove, bus, cat, fire hydrant, microwave oven, stop sign, traffic light." ,#"Look at the image. If you see any of these objects, write their names once, separated by commas: baby, bear, car, dog, hot dog, school bus, teddy bear, train, baseball glove, bus, cat, fire hydrant, microwave oven, stop sign, traffic light. Only one object from this list. No repeats.", ,#"Classify the input image as either:  [concept], or  No [concept]. Return only the predicted class label based on whether a [concept] is present in the image or not" , #Write a caption for the given image
        "List of item":   "\nWhat is the object in the image among: baby, bear, car, dog, hot dog, school bus, teddy bear, train, baseball glove, bus, cat, fire hydrant, microwave oven, stop sign, traffic light." ,#"Look at the image. If you see any of these objects, write their names once, separated by commas: baby, bear, car, dog, hot dog, school bus, teddy bear, train, baseball glove, bus, cat, fire hydrant, microwave oven, stop sign, traffic light. Only one object from this list. No repeats.", 
        "Repeat the text": "\n Just repeat the text:"
    },
    # Added for medical data
    'chexagent':{
        "Predictions": "\nPlease classify finding of the  chest X-ray.",
        "Findings": "\nPlease provide a detailed finding of the chest X-ray.",
    }
}
