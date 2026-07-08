import json
import os
from pathlib import Path

def convert_chatgpt_export_to_txt():
    """
    Convert ChatGPT export JSON files to simple .txt files
    Handles the nested 'mapping' structure
    """
    
    export_folder = "chatgpt-export"
    output_folder = "knowledge_base"
    
    os.makedirs(output_folder, exist_ok=True)
    
    json_files = sorted(Path(export_folder).glob("conversations-*.json"))
    
    if not json_files:
        print("No conversation JSON files found!")
        print(f"Please check the folder: {export_folder}")
        return
    
    print(f"Found {len(json_files)} conversation files.")
    
    total_text_files = 0
    
    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract all text from the conversation
            conversation_text = extract_conversation_text_v2(data)
            
            if conversation_text and len(conversation_text.strip()) > 10:
                # Save as .txt file
                output_file = Path(output_folder) / f"{json_file.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(conversation_text)
                print(f"  → Saved to: {output_file.name} ({len(conversation_text)} characters)")
                total_text_files += 1
            else:
                print(f"  → No text found in {json_file.name}")
                
        except json.JSONDecodeError as e:
            print(f"  → JSON Decode Error: {e}")
        except Exception as e:
            print(f"  → Error: {e}")
    
    print(f"\n✅ Done! {total_text_files} files saved to '{output_folder}/' folder.")

def extract_conversation_text_v2(data):
    """
    Extract text from ChatGPT conversation JSON with mapping structure
    """
    text_parts = []
    
    # Case 1: Check if there's a 'mapping' object (your structure)
    if "mapping" in data:
        mapping = data["mapping"]
        
        # Traverse all nodes in mapping
        for node_id, node_data in mapping.items():
            if not isinstance(node_data, dict):
                continue
                
            message_data = node_data.get("message")
            if not message_data:
                continue
            
            # Get content
            content = message_data.get("content")
            if not content:
                continue
            
            # Get parts (the actual text)
            if isinstance(content, dict):
                parts = content.get("parts")
                if parts and isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, str) and part.strip():
                            # Check who said it (user or assistant)
                            author = message_data.get("author", {})
                            role = author.get("role", "").upper() if isinstance(author, dict) else ""
                            if role:
                                text_parts.append(f"[{role}]: {part.strip()}")
                            else:
                                text_parts.append(part.strip())
            elif isinstance(content, str):
                if content.strip():
                    text_parts.append(content.strip())
    
    # Case 2: Fallback - try to find any text in the entire JSON
    if not text_parts:
        text_parts = extract_text_recursive(data)
    
    # Join all parts with double newline between messages
    if text_parts:
        return "\n\n".join(text_parts)
    return None

def extract_text_recursive(obj, depth=0):
    """
    Recursively search for 'parts' or 'content' keys in any JSON object
    """
    results = []
    
    if isinstance(obj, dict):
        # Check if this has 'parts' key
        if "parts" in obj and isinstance(obj["parts"], list):
            for part in obj["parts"]:
                if isinstance(part, str) and part.strip():
                    results.append(part.strip())
        
        # Check if this has 'content' with text
        if "content" in obj:
            content = obj["content"]
            if isinstance(content, str) and content.strip():
                results.append(content.strip())
            elif isinstance(content, dict):
                results.extend(extract_text_recursive(content, depth + 1))
            elif isinstance(content, list):
                for item in content:
                    results.extend(extract_text_recursive(item, depth + 1))
        
        # Recursively check all values
        for key, value in obj.items():
            if depth < 5:  # Prevent infinite recursion
                results.extend(extract_text_recursive(value, depth + 1))
    
    elif isinstance(obj, list):
        for item in obj:
            results.extend(extract_text_recursive(item, depth + 1))
    
    return results

if __name__ == "__main__":
    convert_chatgpt_export_to_txt()