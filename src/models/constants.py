TASK_PROMPTS = {
    "llava": {
        "ShortVQA": "\nAnswer the question using a single word or phrase.",
        "ShortCaptioning": "\n Write a list of objects that are present in the image." ,
    },
    # Added for medical data
    'chexagent':{
        "Predictions": "\nPlease classify finding of the  chest X-ray.",
        "Findings": "\nPlease provide a detailed finding of the chest X-ray.",
    }
}
