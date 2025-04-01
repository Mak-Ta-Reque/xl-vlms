TASK_PROMPTS = {
    "llava": {
        "ShortVQA": "\nAnswer the question using a single word or phrase.",
        "ShortCaptioning": "\nProvide a caption for the provided image." ,
        "List of item": "\n describe the image",
         "Repeat the text": "\n Just repeat the text"
    },
    # Added for medical data
    'chexagent':{
        "Predictions": "\nPlease classify finding of the  chest X-ray.",
        "Findings": "\nPlease provide a detailed finding of the chest X-ray.",
    }
}
