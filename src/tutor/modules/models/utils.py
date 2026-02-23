from __future__ import annotations
import torch


def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """Shift token ids to the right and prepend decoder_start_token_id."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    
    if pad_token_id is not None:
        # Replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    
    return shifted_input_ids

def get_generative_confidence(output):
    """Extract confidence scores from the model output"""
    # Simplified implementation - you might need to adjust based on your needs
    probs = torch.exp(output.scores[0])
    top_probs = torch.max(probs, dim=-1).values
    confidences = [float(prob.mean()) for prob in top_probs]
    return confidences

def torch_no_grad(func):
	def wrapper(*args, **kwargs):
		with torch.no_grad():
			return func(*args, **kwargs)
	return wrapper
