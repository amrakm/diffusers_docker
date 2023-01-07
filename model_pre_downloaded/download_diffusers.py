# model_id = "runwayml/stable-diffusion-v1-5"
# cache_path = "/home/amrakm/repos/patternedai/rd/tmp_model"


import diffusers
import torch
import os
import argparse

def runMain(model_id, cache_path):

    # hugging_face_token = os.environ["HUGGINGFACE_TOKEN"]
    hugging_face_token="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"
    model_id = "runwayml/stable-diffusion-v1-5"
    cache_path = "/app/data"


    print(f'downloading model {model_id} to {cache_path}')

    # Download scheduler configuration. Experiment with different schedulers
    # to identify one that works best for your use-case.
    scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
        model_id, subfolder="scheduler", use_auth_token=hugging_face_token, cache_dir=cache_path
    )
    scheduler.save_pretrained(model_id, cache_path, safe_serialization=True)

    # Downloads all other models.
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        model_id, use_auth_token=hugging_face_token, revision="fp16", torch_dtype=torch.float16, cache_dir=cache_path
    )   
    pipe.save_pretrained(cache_path, safe_serialization=True)

    print("Downloaded model to {}".format(cache_path))




if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', help='model_id')
    parser.add_argument('--cache_path', type=str, default='./data', help='cache_path')

    args = parser.parse_args()


    model_id = args.model_id
    cache_path = args.cache_path

    runMain(model_id, cache_path)
