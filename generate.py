import torch
from datasets import load_dataset
from tqdm import tqdm

from model import GPT, GPTConfig
from tokenizer import build_tokenizer


def load_model(model_path, config):
    model = GPT(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model


def generate_sample(model, tokenizer, conditions, max_length):
    model.eval()
    input_ids = tokenizer.generation_encode(conditions)
    input_ids = torch.tensor(input_ids, dtype=torch.long).cuda()
    input_ids = input_ids.unsqueeze(0)  # Add batch dimension
    len_conditions = len(input_ids[0])
    with torch.no_grad():
        for _ in range(max_length - len_conditions):
            # Generate one token at a time, and append it to the input to do generation iteratively until </s> is generated
            ### YOUR CODE HERE ###
             # Ensure the input_ids tensor's size does not exceed the block_size/max_length
            # if len_conditions > max_length:
            #     input_ids = input_ids[:, -max_length:]  # Truncate to the last block_size/max_length tokens
            outputs = model(input_ids)  # Get model predictions.
            logits = outputs[0]  # Logits are  the first output of the model.
            # Select the last token from the logits
            next_token_logits = logits[:, -1, :]
            # Apply softmax to logits to convert to probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            # Pick the index with the highest probability as the next token.
            _, next_token_id = torch.max(probs, dim=-1, keepdim=True)
            # Append the predicted token ID to the running  list of input IDs
            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            # Break the loop if the model generates an end-of-sequence token
            if next_token_id[0][0] == tokenizer.vocab['</s>']:
                print('break')
                break
            ### YOUR CODE HERE ###
    generated_text = tokenizer.decode(input_ids[0][len_conditions:])
    return generated_text


def generate(args):

    data_SCAN = load_dataset("scan", args.data_split)
    max_len = args.max_len
    tokenizer, vocab_size = build_tokenizer(args, data_SCAN, max_len, args.output_tokenizer_dir)

    mconf = GPTConfig(vocab_size, max_len, 
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      isconditional=True)

    # Load model and tokenizer
    print("loading model")
    model = load_model(args.ckpt_path, mconf).cuda()
    print('total params:', sum(p.numel() for p in model.parameters()))


    # Sample generation
    test_data = data_SCAN['test']
    correct_count = 0
    pbar = tqdm(enumerate(test_data), total=len(test_data))
    for i, data in pbar:
        print(f'Conditions: {data["commands"]}')
        generated_actions = generate_sample(model, tokenizer, data['commands'], max_len)
        # Compare the first N words
        print(f'Generated actions: {generated_actions}')
        print(f'Actual actions: {data["actions"]}')
        if generated_actions == data['actions']:
            print('Correct!###############################################################################################')
            correct_count += 1
        pbar.set_description(f'Accuracy: {correct_count / (i + 1):.4f}')
    print(f'Test accuracy: {correct_count / len(test_data)}')
