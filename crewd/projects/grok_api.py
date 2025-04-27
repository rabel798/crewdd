"""
Description improvement system using Cohere's free API
"""
import requests
import json
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cohere API endpoint and key
COHERE_API_URL = "https://api.cohere.ai/v1/generate"
COHERE_API_KEY = "JwR7E42R4VZtU4NyFRwPv2Dn2YRctHsMVKUU1zF2"

def log_checkpoint(message: str, data: Dict = None):
    """Log a checkpoint with optional data"""
    logger.info(f"CHECKPOINT: {message}")
    if data:
        logger.info(f"DATA: {json.dumps(data, indent=2)}")

def improve_description(description: str) -> str:
    """
    Improve a project description using Cohere's free API
    
    Args:
        description (str): Original project description text
        
    Returns:
        str: Improved project description
    """
    try:
        # Checkpoint 1: Text received from textbox
        log_checkpoint("Text received from textbox", {"length": len(description)})
        
        # Prepare the prompt
        prompt = f"""Please improve the following project description to make it more professional, clear, and engaging. 
Keep the technical details accurate and maintain the original meaning. Use bullet points or paragraphs as appropriate.

Original description:
{description}

Improved description:"""

        # Checkpoint 2: Sending to AI agent
        log_checkpoint("Sending to AI agent", {"prompt_length": len(prompt)})
        
        # Send request to Cohere API
        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.7,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }
        
        response = requests.post(COHERE_API_URL, headers=headers, json=payload)
        
        # Checkpoint 3: Response received from AI agent
        log_checkpoint("Response received from AI agent", {"status_code": response.status_code})
        
        if response.status_code == 200:
            result = response.json()
            improved_text = result['generations'][0]['text'].split('Improved description:')[-1].strip()
            
            # Checkpoint 4: Text ready to be sent back
            log_checkpoint("Text ready to be sent back", {"length": len(improved_text)})
            
            return improved_text
        else:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return description
            
    except Exception as e:
        logger.error(f"Error improving description: {str(e)}")
        return description

def analyze_tech_stack(description: str) -> List[str]:
    """
    Analyze a project description to suggest a tech stack using Cohere's free API
    
    Args:
        description (str): Project description text
        
    Returns:
        List[str]: List of suggested technologies
    """
    try:
        # Checkpoint 1: Text received for tech stack analysis
        log_checkpoint("Text received for tech stack analysis", {"length": len(description)})
        
        # Prepare the prompt
        prompt = f"""Based on the following project description, list the most relevant technologies needed. 
Return only the technology names, separated by commas.

Project description:
{description}

Required technologies:"""

        # Checkpoint 2: Sending to AI agent for tech stack
        log_checkpoint("Sending to AI agent for tech stack", {"prompt_length": len(prompt)})
        
        # Send request to Cohere API
        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.3,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }
        
        response = requests.post(COHERE_API_URL, headers=headers, json=payload)
        
        # Checkpoint 3: Response received for tech stack
        log_checkpoint("Response received for tech stack", {"status_code": response.status_code})
        
        if response.status_code == 200:
            result = response.json()
            tech_list = result['generations'][0]['text'].split('Required technologies:')[-1].strip()
            tech_list = [tech.strip() for tech in tech_list.split(',')]
            
            # Checkpoint 4: Tech stack ready to be sent back
            log_checkpoint("Tech stack ready to be sent back", {"tech_count": len(tech_list)})
            
            return tech_list[:7]  # Return up to 7 technologies
        else:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        logger.error(f"Error analyzing tech stack: {str(e)}")
        return []