from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

modelName = "EleutherAI/gpt-neo-125M"  #Grabbed off of huggingface documentation:(https://huggingface.co/docs/transformers/quicktour)
tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForCausalLM.from_pretrained(modelName)

device = torch.device("cpu")
model.to(device)

actionToPrompts = {
    "Literal":"Misinterpret the user's wish by taking it too literally and make it backfire in a funny way",
    "Substitution":"Find a reasonable substitute for the wish to make it backfire in a funny way",
    "Exaggerate":"Exaggerate the wish to a extreme, ridiculous, or over-the-top version in a funny way",
    "Opposite": "Misunderstand the user and grant the opposite of the wish in a funny way",
    "Puns": "Make the wish backfire in a funny way with wordplay or puns"
}

def response(wish, action):
    twistStrat = actionToPrompts[action]
    
    prompt = (    
    f"You are a mischievous genie{twistStrat}"
    f"Use 1-2 sentences. \nUser's wish: {wish}\nGenie's reply:"
    )
    inputIDs = tokenizer(prompt, return_tensors="pt").input_ids.to(device) #Grabbed off of huggingface documentation:(https://huggingface.co/docs/transformers/model_doc/gpt2#quick-tour)

    outputIDs = model.generate(inputIDs, max_length=200, do_sample=True, temperature=0.9, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
    responseText = tokenizer.decode(outputIDs[0], skip_special_tokens=True)

    return responseText


