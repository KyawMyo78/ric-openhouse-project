# Fruit Ninja - Image Mapping Guide

## Current Images in fruit_images folder:
- apple.png
- banana.jpg
- grape.png
- orange.png
- pngwing.com.png (unknown/unmapped)

## Expected Fruits in Game:
1. **Apple** ✓ (apple.png found)
2. **Banana** ✓ (banana.jpg found)
3. **Orange** ✓ (orange.png found)
4. **Watermelon** ✗ (missing - needs watermelon.png/jpg)
5. **Strawberry** ✗ (missing - needs strawberry.png/jpg)

## What to do with unmapped images:
- `pngwing.com.png` - You can rename this to match one of the missing fruits:
  - Rename to `watermelon.png` if it's a watermelon
  - Rename to `strawberry.png` if it's a strawberry
  - Or add more fruits to the game

## Optional: Add sliced versions
For even better visuals, you can add sliced versions:
- `apple_sliced.png` or `apple_left.png` + `apple_right.png`
- `banana_sliced.png` or `banana_left.png` + `banana_right.png`
- etc.

If no sliced version exists, the game will automatically split the whole image.

## Supported formats:
- PNG (recommended - supports transparency)
- JPG/JPEG (works but no transparency)
