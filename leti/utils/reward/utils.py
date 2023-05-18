import json

def dump_offset_mapping_to_json(offset_mapping):
    return json.dumps({
        str(k): v
        for k, v in offset_mapping.items()
    })

CORRECT_LANG_FEEDBACK = [
    "The code is working correctly.",
    "Great! This code works!",
    "The code is producing the right results.",
    "The code is running correctly!",
    "The code is executing correctly.",
    "This code is producing the desired results.",
    "Well done - your code is functioning as expected.",
    "Congratulations! You've correctly implemented the program.",
    "Superb! You've written a working program.",
    "You've successfully implemented the program.",
]

CORRECT_LANG_FEEDBACK_POSTPROCESS_HEURISTIC = [
    "The code is working correctly, but there are some extra code that can be removed.",
    "This code works! But there are some extra code after the first return statement that can be removed.",
    "The code is producing the right results, but there are some extra code after the first function definition that can be removed."
]

INCORRECT_LANG_FEEDBACK = "The code is not working correctly. Here is some feedback: "

