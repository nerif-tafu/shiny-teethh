"""
5x7 Pixel Font Definition and Text Rendering Utilities
"""
import numpy as np

PIXEL_FONT = {
    ' ': ["00000", "00000", "00000", "00000", "00000"],  # SPACE (32)
    '!': ["00100", "00100", "00100", "00000", "00100"],
    '"': ["01010", "01010", "00000", "00000", "00000"],
    '#': ["01010", "11111", "01010", "11111", "01010"],
    '$': ["00100", "01111", "10100", "01111", "00101"],
    '%': ["11001", "11010", "00100", "01011", "10011"],
    '&': ["01100", "10010", "01100", "10010", "01101"],
    "'": ["00100", "00100", "00000", "00000", "00000"],
    '(': ["00010", "00100", "00100", "00100", "00010"],
    ')': ["01000", "00100", "00100", "00100", "01000"],
    '*': ["00000", "01010", "00100", "01010", "00000"],
    '+': ["00000", "00100", "01110", "00100", "00000"],
    ',': ["00000", "00000", "00000", "00100", "01000"],
    '-': ["00000", "00000", "01110", "00000", "00000"],
    '.': ["00000", "00000", "00000", "00000", "00100"],
    '/': ["00001", "00010", "00100", "01000", "10000"],
    '0': ["01110", "10001", "10001", "10001", "01110"],
    '1': ["00100", "01100", "00100", "00100", "01110"],
    '2': ["01110", "10001", "00110", "01000", "11111"],
    '3': ["01110", "10001", "00110", "10001", "01110"],
    '4': ["00110", "01010", "10010", "11111", "00010"],
    '5': ["11111", "10000", "11110", "00001", "11110"],
    '6': ["01110", "10000", "11110", "10001", "01110"],
    '7': ["11111", "00001", "00010", "00100", "00100"],
    '8': ["01110", "10001", "01110", "10001", "01110"],
    '9': ["01110", "10001", "01111", "00001", "01110"],
    ':': ["00000", "00100", "00000", "00100", "00000"],
    ';': ["00000", "00100", "00000", "00100", "01000"],
    '<': ["00010", "00100", "01000", "00100", "00010"],
    '=': ["00000", "01110", "00000", "01110", "00000"],
    '>': ["01000", "00100", "00010", "00100", "01000"],
    '?': ["01110", "10001", "00110", "00000", "00100"],
    '@': ["01110", "10001", "10111", "10000", "01110"],
    'A': ["01110", "10001", "11111", "10001", "10001"],
    'B': ["11110", "10001", "11110", "10001", "11110"],
    'C': ["01110", "10001", "10000", "10001", "01110"],
    'D': ["11110", "10001", "10001", "10001", "11110"],
    'E': ["11111", "10000", "11110", "10000", "11111"],
    'F': ["11111", "10000", "11110", "10000", "10000"],
    'G': ["01110", "10000", "10111", "10001", "01111"],
    'H': ["10001", "10001", "11111", "10001", "10001"],
    'I': ["01110", "00100", "00100", "00100", "01110"],
    'J': ["00111", "00010", "00010", "10010", "01100"],
    'K': ["10001", "10010", "11100", "10010", "10001"],
    'L': ["10000", "10000", "10000", "10000", "11111"],
    'M': ["10001", "11011", "10101", "10001", "10001"],
    'N': ["10001", "11001", "10101", "10011", "10001"],
    'O': ["01110", "10001", "10001", "10001", "01110"],
    'P': ["11110", "10001", "11110", "10000", "10000"],
    'Q': ["01110", "10001", "10101", "10010", "01101"],
    'R': ["11110", "10001", "11110", "10010", "10001"],
    'S': ["01111", "10000", "01110", "00001", "11110"],
    'T': ["11111", "00100", "00100", "00100", "00100"],
    'U': ["10001", "10001", "10001", "10001", "01110"],
    'V': ["10001", "10001", "10001", "01010", "00100"],
    'W': ["10001", "10001", "10101", "11011", "10001"],
    'X': ["10001", "01010", "00100", "01010", "10001"],
    'Y': ["10001", "01010", "00100", "00100", "00100"],
    'Z': ["11111", "00010", "00100", "01000", "11111"],
    '[': ["01110", "01000", "01000", "01000", "01110"],
    '\\': ["10000", "01000", "00100", "00010", "00001"],
    ']': ["01110", "00010", "00010", "00010", "01110"],
    '^': ["00100", "01010", "10001", "00000", "00000"],
    '_': ["00000", "00000", "00000", "00000", "11111"],
    '`': ["01000", "00100", "00000", "00000", "00000"],
    '{': ["00110", "00100", "01100", "00100", "00110"],
    '|': ["00100", "00100", "00100", "00100", "00100"],
    '}': ["01100", "00100", "00110", "00100", "01100"],
    '~': ["00000", "01010", "10100", "00000", "00000"],
} 

def calculate_text_dimensions(text, font=None):
    """Calculate the width and height needed for a text block"""
    if font is None:
        font = PIXEL_FONT
        
    words = text.upper().split()
    lines = []
    current_line = []
    current_width = 0
    max_width = 64  # Standard LED matrix width
    
    # First pass: calculate lines based on width
    for word in words:
        word_width = 0
        for char in word:
            if char in font:
                word_width += len(font[char][0]) + 1
        
        if current_width + word_width > max_width - 4:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
            else:
                lines.append(word)
                current_line = []
                current_width = 0
        else:
            current_line.append(word)
            current_width += word_width
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def calculate_line_width(line, font=None):
    """Calculate the pixel width of a single line of text"""
    if font is None:
        font = PIXEL_FONT
        
    width = 0
    for char in line:
        if char in font:
            width += len(font[char][0]) + 1
    return max(0, width - 1)  # Subtract the last spacing

def render_character(frame_data, char, pos_x, pos_y, font=None):
    """Render a single character to the frame buffer"""
    if font is None:
        font = PIXEL_FONT
        
    if char not in font:
        return pos_x + 3
        
    char_pixels = font[char]
    char_width = len(char_pixels[0])
    
    for y, row in enumerate(char_pixels):
        if pos_y + y >= len(frame_data):
            break
        for x, pixel in enumerate(row):
            if pos_x + x >= len(frame_data[0]):
                break
            if pixel == '1':
                frame_data[pos_y + y][pos_x + x] = 255
    
    return pos_x + char_width + 1

def create_text_frame(text, width=64, height=32, font=None):
    """
    Create a frame with centered multi-line text
    
    Args:
        text (str): The text to render
        width (int): Frame width in pixels
        height (int): Frame height in pixels
        font (dict): Optional custom font dictionary
        
    Returns:
        list: 2D list representing the frame buffer
    """
    if font is None:
        font = PIXEL_FONT
        
    # Create black background
    frame_data = [[0 for x in range(width)] for y in range(height)]
    
    # Calculate lines
    lines = calculate_text_dimensions(text.upper(), font)
    
    # Calculate total height needed (5 pixels per line plus 2 pixels spacing)
    total_height = (len(lines) * 5) + ((len(lines) - 1) * 2)
    
    # Calculate vertical starting position
    start_y = max(0, (height - total_height) // 2)
    
    # Draw each line
    current_y = start_y
    for line in lines:
        # Calculate line width and starting position
        line_width = calculate_line_width(line, font)
        start_x = max(0, (width - line_width) // 2)
        
        # Draw the line
        current_x = start_x
        for char in line:
            current_x = render_character(frame_data, char, current_x, current_y, font)
            
        # Move to next line
        current_y += 7  # 5 pixels for character height plus 2 pixels spacing
    
    return frame_data 