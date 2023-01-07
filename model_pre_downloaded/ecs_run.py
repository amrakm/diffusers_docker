import argparse

hugging_face_token="xxxxxxx"

def runMain(model_id, cache_path):


    import diffusers
    import torch

    import time

    print(f'loading model {model_id} from {cache_path}')

    start_time = time.time()


    torch.backends.cuda.matmul.allow_tf32 = True

    # check if GPU is available
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
        cache_dir=cache_path,
        subfolder="scheduler",
        solver_order=2,
        prediction_type="epsilon",
        thresholding=False,
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        denoise_final=True,  # important if steps are <= 10
        use_auth_token=hugging_face_token,
    )

    # model_id = "runwayml/stable-diffusion-v1-5"
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_id=model_id,
                                                            cache_dir=cache_path,
                                                            scheduler=scheduler,
                                                            use_auth_token=hugging_face_token,
                                                            torch_dtype=torch.float16).to("cuda")
                                                            
    pipe.enable_xformers_memory_efficient_attention()


    print("time to load model: %s seconds" % (time.time() - start_time))


    start_time = time.time()

    prompt = "a photo of an astronaut riding a horse on mars"

    with torch.inference_mode():
        with torch.autocast("cuda"):
            image = pipe(prompt).images[0]  

    print("time to generate image: %s seconds" % (time.time() - start_time))

        
    image.save("astronaut_rides_horse.png")

    print('otput image saved to astronaut_rides_horse.png')



    

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', help='model_id')
    parser.add_argument('--cache_path', type=str, default='/', help='cache_path')

    args = parser.parse_args()

    model_id = args.model_id
    cache_path = args.cache_path

    runMain(model_id, cache_path)