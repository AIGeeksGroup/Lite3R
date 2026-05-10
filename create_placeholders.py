#!/usr/bin/env python3
"""
Create placeholder images for the Depth Anything 3 project page.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder(width, height, text, icon, filename, bg_color=(102, 126, 234), text_color=(255, 255, 255)):
    """Create a placeholder image with text and icon."""
    # Create image with gradient background
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Add some texture
    for i in range(100):
        x = int(i * width / 100)
        y = int(i * height / 100)
        draw.ellipse([x-1, y-1, x+1, y+1], fill=(255, 255, 255, 30))
    
    # Try to use a system font, fallback to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font_large = ImageFont.truetype("arial.ttf", 48)
            font_small = ImageFont.truetype("arial.ttf", 24)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # Draw icon
    icon_bbox = draw.textbbox((0, 0), icon, font=font_large)
    icon_width = icon_bbox[2] - icon_bbox[0]
    icon_height = icon_bbox[3] - icon_bbox[1]
    icon_x = (width - icon_width) // 2
    icon_y = (height - icon_height) // 2 - 20
    draw.text((icon_x, icon_y), icon, fill=text_color, font=font_large)
    
    # Draw text
    text_bbox = draw.textbbox((0, 0), text, font=font_small)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (width - text_width) // 2
    text_y = icon_y + icon_height + 10
    draw.text((text_x, text_y), text, fill=text_color, font=font_small)
    
    # Add border
    draw.rectangle([0, 0, width-1, height-1], outline=(255, 255, 255, 100), width=2)
    
    # Save image
    img.save(filename, 'JPEG', quality=90)
    print(f"Created: {filename}")

def create_benchmark_placeholder(width, height, text, filename):
    """Create a benchmark comparison placeholder with 2.5:1 aspect ratio."""
    # Create image with gradient background
    img = Image.new('RGB', (width, height), (102, 126, 234))
    draw = ImageDraw.Draw(img)
    
    # Add some texture
    for i in range(200):
        x = int(i * width / 200)
        y = int(i * height / 200)
        draw.ellipse([x-1, y-1, x+1, y+1], fill=(255, 255, 255, 30))
    
    # Try to use a system font, fallback to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        try:
            font_large = ImageFont.truetype("arial.ttf", 36)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # Draw chart-like elements
    # Draw some bars to simulate a comparison chart
    bar_width = width // 8
    bar_heights = [height * 0.3, height * 0.5, height * 0.7, height * 0.4, height * 0.8, height * 0.6]
    bar_colors = [(255, 99, 132), (54, 162, 235), (255, 205, 86), (75, 192, 192), (153, 102, 255), (255, 159, 64)]
    
    for i, (bar_height, color) in enumerate(zip(bar_heights, bar_colors)):
        x = (width - len(bar_heights) * bar_width) // 2 + i * bar_width
        y = height - bar_height - 50
        draw.rectangle([x, y, x + bar_width - 10, height - 50], fill=color)
    
    # Draw title
    title_bbox = draw.textbbox((0, 0), text, font=font_large)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 20), text, fill=(255, 255, 255), font=font_large)
    
    # Add border
    draw.rectangle([0, 0, width-1, height-1], outline=(255, 255, 255, 100), width=2)
    
    # Save image
    img.save(filename, 'PNG', quality=90)
    print(f"Created: {filename}")

def main():
    """Create all placeholder images."""
    # Ensure assets directory exists
    os.makedirs('assets', exist_ok=True)
    
    # Define placeholders
    placeholders = [
        {'text': 'Main Demo Video', 'icon': '▶', 'filename': 'assets/main_demo_poster.jpg'},
        {'text': 'Multi-view Depth Estimation', 'icon': '▶', 'filename': 'assets/interactive_demo1.jpg'},
        {'text': '3D Gaussian Estimation', 'icon': '▶', 'filename': 'assets/interactive_demo2.jpg'},
        {'text': 'Pose-Conditioned Depth', 'icon': '▶', 'filename': 'assets/interactive_demo3.jpg'},
        {'text': 'Robotic Navigation', 'icon': '🤖', 'filename': 'assets/app_robotics.jpg'},
        {'text': 'AR/VR Applications', 'icon': '🥽', 'filename': 'assets/app_ar_vr.jpg'},
        {'text': 'Autonomous Driving', 'icon': '🚗', 'filename': 'assets/app_autonomous_driving.jpg'},
        {'text': '3D Reconstruction', 'icon': '🏗️', 'filename': 'assets/app_3d_reconstruction.jpg'},
        {'text': 'Network Architecture Analysis', 'icon': '🏛️', 'filename': 'assets/analysis_architecture.jpg'},
        {'text': 'Ablation Studies', 'icon': '🔬', 'filename': 'assets/analysis_ablation.jpg'},
        {'text': 'Failure Case Analysis', 'icon': '⚠️', 'filename': 'assets/analysis_failure_cases.jpg'},
    ]
    
    # Create each placeholder
    for placeholder in placeholders:
        create_placeholder(800, 450, placeholder['text'], placeholder['icon'], placeholder['filename'])
    
    # Create benchmark comparison with 2.5:1 aspect ratio
    create_benchmark_placeholder(1000, 400, 'Performance Comparison', 'assets/benchmark_comparison.png')
    
    print("All placeholder images created successfully!")
    print("You can now replace these with your actual video content.")

if __name__ == "__main__":
    main()
