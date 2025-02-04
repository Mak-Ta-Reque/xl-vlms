TASK_PROMPTS = {
    "llava": {
        "ShortVQA": "\nAnswer the question using a single word or phrase.",
        "ShortCaptioning": "\nProvide a caption for the provided image. include every objeccts in the image. don't miss any object" ,
    },
    # Added for medical data
    'chexagent':{
        "Predictions": "\nPlease classify finding of the  chest X-ray.",
        "Findings": "\nPlease provide a detailed finding of the chest X-ray.",
    }
}
