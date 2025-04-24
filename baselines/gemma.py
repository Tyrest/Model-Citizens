from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch
import sys
from tqdm import tqdm
import numpy as np

sys.path.append(".")

from load_nlvr import load_nlvr

train_df, val_df, test_df = load_nlvr()

model_id = "google/paligemma-3b-ft-nlvr2-448"
device = "cuda:0"
dtype = torch.bfloat16

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)


def join_images(image1_path, image2_path):
    # Open the two images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Resize the images to 256x256
    image1 = image1.resize((256, 256), Image.Resampling.LANCZOS)
    image2 = image2.resize((256, 256), Image.Resampling.LANCZOS)

    # Join the two images together
    new_image = Image.new("RGB", (512, 256))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (256, 0))

    return new_image


# outputs = []
# all_image_embeddings = []
# all_text_embeddings = []
# cosine_similarities = []
# all_attention_entropies = []
all_hidden_states = []


def compute_entropy(attn_tensor):
    # attn_tensor shape: (batch_size, num_heads, seq_len, seq_len)
    eps = 1e-12
    entropy = -(attn_tensor * torch.log(attn_tensor + eps)).sum(dim=-1)
    return entropy


with torch.no_grad():
    for _, row in tqdm(val_df.iterrows(), total=len(test_df)):
        # Process images
        image = join_images(row["left"], row["right"])
        # save the image
        image.save("joined_image.jpg")
        prompt = (
            '<image> Is the following sentence about the two images "True" or "False": '
            + row["sentence"]
        )

        model_inputs = processor(image, text=prompt, return_tensors="pt").to(
            model.device
        )
        input_len = model_inputs["input_ids"].shape[-1]

        # image_embeddings = model.multi_modal_projector(
        #     model.vision_tower(model_inputs["pixel_values"].to(dtype))[
        #         "last_hidden_state"
        #     ]
        # )
        # text_embeddings = model.language_model.model.embed_tokens(
        #     model_inputs["input_ids"]
        # )
        # print(image_embeddings.shape)
        # print(text_embeddings.shape)

        generation = model.generate(
            **model_inputs,
            max_new_tokens=1,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        generation_clipped = generation.sequences[0][input_len:]
        decoded = processor.decode(generation_clipped, skip_special_tokens=True)

        penultimate_hidden_states = generation.hidden_states[0]
        # print(penultimate_hidden_states[-1].shape)

        # print(decoded)

        # if decoded == "True":
        #     outputs.append(1)
        # elif decoded == "False":
        #     outputs.append(0)
        # else:
        #     print("Error: output not True or False")
        #     outputs.append(0)

        # # average the image and text embeddings
        # image_embedding = torch.mean(image_embeddings, dim=1)
        # text_embedding = torch.mean(text_embeddings, dim=1)
        last_hidden_state = penultimate_hidden_states[-1][:, -1, :]
        all_hidden_states.append(last_hidden_state.to(torch.float32).cpu().numpy())

        # all_image_embeddings.append(image_embedding.to(torch.float32).cpu().numpy())
        # all_text_embeddings.append(text_embedding.to(torch.float32).cpu().numpy())

        # # calculate cosine similarity
        # cosine_similarity = torch.nn.functional.cosine_similarity(
        #     image_embedding, text_embedding
        # )
        # cosine_similarities.append(cosine_similarity.item())

        # layer_entropies = []
        # for layer_attn in generation.attentions:
        #     # print(layer_attn[0].shape)
        #     ent = compute_entropy(
        #         layer_attn[0]
        #     )  # shape: (batch_size, num_heads, seq_len)
        #     ent_avg = ent.mean(dim=[1, 2])  # average over tokens and heads per sample
        #     layer_entropies.append(ent_avg)
        # # Average entropies over all layers.
        # layer_entropies = torch.stack(layer_entropies, dim=0).mean(dim=0)
        # # print(layer_entropies.shape)
        # all_attention_entropies.append(layer_entropies.to(torch.float32).cpu().numpy())


# save the hidden states to a file
all_hidden_states = np.concatenate(all_hidden_states, axis=0)
np.savetxt("hidden_states.txt", all_hidden_states)

# # Save the outputs to a file
# with open("outputs.txt", "w") as f:
#     for output in outputs:
#         f.write(str(output) + "\n")
# # Save the image embeddings to a file
# all_image_embeddings = np.concatenate(all_image_embeddings, axis=0)
# np.savetxt("image_embeddings.txt", all_image_embeddings)
# all_text_embeddings = np.concatenate(all_text_embeddings, axis=0)
# np.savetxt("text_embeddings.txt", all_text_embeddings)
# cosine_similarities = np.array(cosine_similarities)
# np.savetxt("cosine_similarities.txt", cosine_similarities)
# attention_entropies = np.concatenate(all_attention_entropies, axis=0)
# np.savetxt("attention_entropies.txt", attention_entropies)
