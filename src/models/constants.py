TASK_PROMPTS = {
    "llava": {
        "ShortVQA":"What are the objects the image?",
        "ShortCaptioning": "Classify the input image as either:  [concept], or  No [concept]. Return only the predicted class label based on whether a [concept] is present in the image or not" , #Write a caption for the given image
        "List of item": "Classify the input image as either: [concept], or  No [concept]. Return only the predicted class label based on whether a [concept] is present in the image or not"  ,
        "Repeat the text": "\n Just repeat the text:"
    },
    # Added for medical data
    'chexagent':{
        "Predictions": "\nPlease classify finding of the  chest X-ray.",
        "Findings": "\nPlease provide a detailed finding of the chest X-ray.",
    }
}
