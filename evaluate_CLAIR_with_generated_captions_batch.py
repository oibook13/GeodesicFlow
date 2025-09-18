import json
import requests
import time
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
import math

def load_json_data(file_path):
    """Load caption data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_caption_batch(api_key, caption_pairs, model="gpt-3.5-turbo", batch_size=10):
    """
    Send a batch of caption pairs to ChatGPT API and get similarity scores
    
    Args:
        api_key: OpenAI API key
        caption_pairs: List of dicts with 'id', 'candidate', and 'reference' keys
        model: OpenAI model to use
        batch_size: Maximum number of pairs to process in one API call
    
    Returns:
        Dictionary mapping image_id to similarity results
    """
    # Process in batches of the specified size
    results = {}
    num_batches = math.ceil(len(caption_pairs) / batch_size)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(caption_pairs))
        current_batch = caption_pairs[start_idx:end_idx]
        
        # Construct the prompt for this batch
        batch_prompt = "You are evaluating multiple caption pairs to determine if they describe the same images.\n\n"
        batch_prompt += "For each pair, tell me on a scale from 0 to 100 how likely the candidate caption is describing the same image as the reference caption.\n\n"
        
        for idx, pair in enumerate(current_batch):
            batch_prompt += f"PAIR {idx+1}:\n"
            batch_prompt += f"Candidate: {pair['candidate']}\n"
            batch_prompt += f"Reference: {pair['reference']}\n\n"
        
        batch_prompt += "Respond with a valid JSON object containing an array of evaluations. Each evaluation should have 'pair_id', 'score', and 'reason' fields.\n"
        batch_prompt += "Example format: {\"evaluations\": [{\"pair_id\": 1, \"score\": 85, \"reason\": \"Both captions mention...\"}]}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that evaluates caption similarity."},
                {"role": "user", "content": batch_prompt}
            ],
            "temperature": 0.3
        }
        
        print(f"Making API request with model: {model} for batch {i+1}/{num_batches}")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            # Assign error to all pairs in this batch
            for pair in current_batch:
                results[pair['id']] = {"score": 0, "reason": f"API error: {response.status_code}"}
            continue
        
        result = response.json()
        message_content = result["choices"][0]["message"]["content"]
        
        try:
            # Parse the JSON response
            parsed_response = json.loads(message_content)
            evaluations = parsed_response.get("evaluations", [])
            
            # Map the evaluations back to the original image IDs
            for eval_idx, evaluation in enumerate(evaluations):
                if eval_idx < len(current_batch):
                    pair_id = current_batch[eval_idx]['id']
                    results[pair_id] = {
                        "score": evaluation.get("score", 0),
                        "reason": evaluation.get("reason", "No reason provided")
                    }
                
        except json.JSONDecodeError:
            print("Could not parse response as JSON, attempting to extract scores manually...")
            # Fallback: try to extract scores with regex
            import re
            
            for idx, pair in enumerate(current_batch):
                pair_id = pair['id']
                pattern = rf"PAIR {idx+1}.*?score:?\s*(\d+)"
                score_match = re.search(pattern, message_content, re.DOTALL | re.IGNORECASE)
                
                if score_match:
                    score = int(score_match.group(1))
                    reason_pattern = rf"PAIR {idx+1}.*?reason:?\s*[\"']?(.*?)[\"']?\n"
                    reason_match = re.search(reason_pattern, message_content, re.DOTALL | re.IGNORECASE)
                    reason = reason_match.group(1) if reason_match else "No reason provided"
                    
                    results[pair_id] = {"score": score, "reason": reason}
                else:
                    results[pair_id] = {"score": 0, "reason": "Failed to parse response"}
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate caption similarity using ChatGPT in batches')
    parser.add_argument('--input', type=str, required=True, help='Path to the JSON file with caption pairs')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--output', type=str, default='similarity_results.json', help='Output file path')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='OpenAI model to use')
    parser.add_argument('--max-pairs', type=int, default=None, help='Maximum number of caption pairs to evaluate')
    parser.add_argument('--batch-size', type=int, default=32, help='Number of caption pairs to evaluate in one API call')
    parser.add_argument('--delay', type=float, default=3.0, help='Delay between API calls in seconds')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}")
    data = load_json_data(args.input)
    
    captions = data.get("captions", {})
    print(f"Found {len(captions)} caption pairs")
    
    results = {
        "caption_evaluations": {},
        "summary": {"total_score": 0, "count": 0}
    }
    
    # Get the list of caption pairs to process
    caption_ids = list(captions.keys())
    
    # Limit the number of pairs if specified
    if args.max_pairs is not None and args.max_pairs < len(caption_ids):
        caption_ids = caption_ids[:args.max_pairs]
        print(f"Limiting evaluation to {args.max_pairs} pairs as specified.")
    
    # Prepare batch processing
    caption_pairs = []
    for image_id in caption_ids:
        pair = captions[image_id]
        caption_pairs.append({
            'id': image_id,
            'candidate': pair.get('generated', ''),
            'reference': pair.get('ground_truth', '')
        })
    
    # Process in batches
    num_batches = math.ceil(len(caption_pairs) / args.batch_size)
    print(f"Processing {len(caption_pairs)} caption pairs in {num_batches} batches...")
    
    for i in range(0, len(caption_pairs), args.batch_size):
        batch = caption_pairs[i:i+args.batch_size]
        print(f"Processing batch {i//args.batch_size + 1}/{num_batches} with {len(batch)} pairs")
        
        try:
            batch_results = evaluate_caption_batch(
                args.api_key, 
                batch, 
                args.model, 
                args.batch_size
            )
            
            # Store results
            for image_id, similarity in batch_results.items():
                pair_index = next((idx for idx, p in enumerate(batch) if p['id'] == image_id), None)
                if pair_index is not None:
                    pair = batch[pair_index]
                    results["caption_evaluations"][image_id] = {
                        "candidate": pair['candidate'],
                        "reference": pair['reference'],
                        "similarity": similarity
                    }
                    
                    if "score" in similarity:
                        score = similarity["score"]
                        results["summary"]["total_score"] += score
                        results["summary"]["count"] += 1
                        print(f"Image {image_id}: Score = {score}/100")
        
        except Exception as e:
            print(f"Error processing batch: {e}")
        
        # Add a delay between batches to avoid rate limiting
        if i + args.batch_size < len(caption_pairs):
            print(f"Waiting {args.delay} seconds before next batch...")
            time.sleep(args.delay)
    
    # Save the results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {args.output}")
    if results["summary"]["count"] > 0:
        avg_score = results["summary"]["total_score"] / results["summary"]["count"]
        results["summary"]["average_score"] = avg_score
        print(f"Average similarity score: {avg_score:.2f}")
        print(f"Evaluated {results['summary']['count']} out of {len(caption_ids)} caption pairs successfully.")

if __name__ == "__main__":
    main()