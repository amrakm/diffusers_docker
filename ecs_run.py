

def runMain():


    from diffusers import StableDiffusionPipeline
    import torch
    torch.backends.cudnn.enabled = False
    # check if GPU is available
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]  
        
    image.save("astronaut_rides_horse.png")

    print('otput image saved to astronaut_rides_horse.png')

if __name__ == "__main__":
    runMain()
