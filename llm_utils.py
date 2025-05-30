# Helper functions for LLM integration

import logging

def setup_logging():
    """
    Set up logging configuration to redirect debug messages to a file.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,  # Set console logging to INFO level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("debug.log"),  # Debug messages go to file
            logging.StreamHandler()  # INFO and above go to console
        ]
    )
    
    # Configure openai and urllib3 loggers to only log to file
    for logger_name in ['openai', 'urllib3']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add file handler only
        file_handler = logging.FileHandler("debug.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logging.getLogger(__name__)

def format_llm_output(raw_output):
    """
    Format the LLM output for better readability.
    
    Args:
        raw_output (str): The raw output string from the LLM
        
    Returns:
        str: Formatted output with proper line breaks and cleaned symbols
    """
    if not raw_output:
        return "No output received from LLM"
    
    # Replace escaped newlines with actual newlines
    formatted = raw_output.replace('\\n', '\n')
    
    # Remove unnecessary escape characters
    formatted = formatted.replace('\\', '')
    
    # Clean up any remaining unnecessary characters
    formatted = formatted.strip('"\'')
    
    # Limit output length if it's very long
    max_length = 500  # Shorter limit for better console display
    if len(formatted) > max_length:
        # Try to find a newline near the cutoff point
        cutoff_point = formatted[:max_length].rfind('\n')
        if cutoff_point == -1:  # No newline found
            cutoff_point = max_length
        formatted = formatted[:cutoff_point] + "\n...(output truncated)..."
    
    # Highlight the action list for better visibility
    import re
    action_match = re.search(r'(\[\s*\d+\s*(?:,\s*\d+\s*)*\])', formatted)
    if action_match:
        action_list = action_match.group(1)
        formatted = formatted.replace(action_list, f"ACTION LIST: {action_list}")
    
    return formatted
