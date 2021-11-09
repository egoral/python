# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 23:05:45 2020

@author: egoral
"""

'''This Example opens an Image and transform the image into grayscale, halftone, dithering, and primary colors.
You need PILLOW (Python Imaging Library fork) and Python 3.5
    -Isai B. Cicourel'''

import numpy as np
from PIL import Image

# Open an Image
def open_image(path):
  newImage = Image.open(path)
  return newImage

# Save Image
def save_image(image, path):
  image.save(path, 'png')


# Create a new image with the given size
def create_image(i, j):
  image = Image.new("RGB", (i, j), "white")
  return image


# Get the pixel from the given image
def get_pixel(image, i, j):
  # Inside image bounds?
  width, height = image.size
  if i > width or j > height:
    return None

  # Get Pixel
  pixel = image.getpixel((i, j))
  return pixel

# Create a Grayscale version of the image
def convert_grayscale(image):
  # Get size
  width, height = image.size
  print(width)
  print(height)
  # Create new Image and a Pixel Map
  new = create_image(width, height)
  pixels = new.load()

  # Transform to grayscale
  for i in range(width):
    for j in range(height):
      # Get Pixel
      pixel = get_pixel(image, i, j)

      # Get R, G, B values (This are int from 0 to 255)
      red =   pixel[0]
      green = pixel[1]
      blue =  pixel[2]

      # Transform to grayscale
      gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

      # Set Pixel in new image
      pixels[i, j] = (int(gray), int(gray), int(gray))

  # Return new image
  return new

# eg
  
def convert_size_1(image):
  # Get size
  width, height = image.size
  new_scale = 32
  print("w", width)
  print("h", height)
  # Create new Image and a Pixel Map
  new = create_image(new_scale, new_scale)
  pixels = new.load()
  print(pixels)

  # Transform to grayscale
  for i in range(width):
    for j in range(height):
      # Get Pixel
      # print(i, j)
      pixel = get_pixel(image, i, j)

      # Get R, G, B values (This are int from 0 to 255)
      red =   pixel[0]
      green = pixel[1]
      blue =  pixel[2]

      # Transform to grayscale
      # gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

      # Set Pixel in new image
      pixels[int(new_scale*i/width), int(new_scale*j/height)] = (red, green, blue)

  # Return new image
  return new

def convert_size_2(image):
  # Get size
  width, height = image.size
  new_scale_width  = 32  # width
  new_scale_height = 32 # height
  print("w", width)
  print("h", height)
  print("nw", new_scale_width)
  print("nh", new_scale_height)
  # Create new Image and a Pixel Map
  new = create_image(new_scale_width, new_scale_height)
  pixels = new.load()
  print(pixels)
  
  width_count = int (width / new_scale_width)
  height_count = int (height / new_scale_height)
  print("nwc", width_count)
  print("nhc", height_count)  
  out_list = ["2"]

  # Transform to grayscale
  
  for co_i in range(new_scale_width):
      for co_j in range(new_scale_height):
          
        red   = 0
        green = 0
        blue  = 0
  
        for i in range(width_count):
            for j in range(height_count):
                
                # Get Pixel
                # print(i, j)
                pixel = get_pixel(image, co_i * width_count + i, co_j * height_count + j)
                # Get R, G, B values (This are int from 0 to 255)
                red   = red   + pixel[0]
                green = green + pixel[1]
                blue  = blue  + pixel[2]
          
        red    = int(red  /(width_count * height_count))
        green  = int(green/(width_count * height_count))
        blue   = int(blue /(width_count * height_count))
        hi_low = int((red + green + blue)/3)
        # print(hi_low)
        if hi_low > 125:
            pixels[co_i, co_j] = (0, 0, 0)
            out_list.append(0)
        else:
            pixels[co_i, co_j] = (255, 255, 255)
            out_list.append(255)

  print(out_list)  
  out_list = out_list.values.reshape(-1,28,28,1)
  # Return new image
  return new




# Create a Half-tone version of the image
def convert_halftoning(image):
  # Get size
  width, height = image.size

  # Create new Image and a Pixel Map
  new = create_image(int(width), int(height))
  pixels = new.load()

  # Transform to half tones
  for i in range(0, width-1, 2):
    for j in range(0, height-1, 2):
      # Get Pixels
      p1 = get_pixel(image, i, j)
      p2 = get_pixel(image, i, j + 1)
      p3 = get_pixel(image, i + 1, j)
      p4 = get_pixel(image, i + 1, j + 1)

      # Transform to grayscale
      gray1 = (p1[0] * 0.299) + (p1[1] * 0.587) + (p1[2] * 0.114)
      gray2 = (p2[0] * 0.299) + (p2[1] * 0.587) + (p2[2] * 0.114)
      gray3 = (p3[0] * 0.299) + (p3[1] * 0.587) + (p3[2] * 0.114)
      gray4 = (p4[0] * 0.299) + (p4[1] * 0.587) + (p4[2] * 0.114)

      # Saturation Percentage
      sat = (gray1 + gray2 + gray3 + gray4) / 4

      # Draw white/black depending on saturation
      if sat > 223:
         pixels[i, j]         = (255, 255, 255) # White
         pixels[i, j + 1]     = (255, 255, 255) # White
         pixels[i + 1, j]     = (255, 255, 255) # White
         pixels[i + 1, j + 1] = (255, 255, 255) # White
      elif sat > 159:
         pixels[i, j]         = (255, 255, 255) # White
         pixels[i, j + 1]     = (0, 0, 0)       # Black
         pixels[i + 1, j]     = (255, 255, 255) # White
         pixels[i + 1, j + 1] = (255, 255, 255) # White
      elif sat > 95:
         pixels[i, j]         = (255, 255, 255) # White
         pixels[i, j + 1]     = (0, 0, 0)       # Black
         pixels[i + 1, j]     = (0, 0, 0)       # Black
         pixels[i + 1, j + 1] = (255, 255, 255) # White
      elif sat > 32:
         pixels[i, j]         = (0, 0, 0)       # Black
         pixels[i, j + 1]     = (255, 255, 255) # White
         pixels[i + 1, j]     = (0, 0, 0)       # Black
         pixels[i + 1, j + 1] = (0, 0, 0)       # Black
      else:
         pixels[i, j]         = (0, 0, 0)       # Black
         pixels[i, j + 1]     = (0, 0, 0)       # Black
         pixels[i + 1, j]     = (0, 0, 0)       # Black
         pixels[i + 1, j + 1] = (0, 0, 0)       # Black

  # Return new image
  return new


# Return color value depending on quadrant and saturation
def get_saturation(value, quadrant):
  if value > 223:
    return 255
  elif value > 159:
    if quadrant != 1:
      return 255

    return 0
  elif value > 95:
    if quadrant == 0 or quadrant == 3:
      return 255

    return 0
  elif value > 32:
    if quadrant == 1:
      return 255

    return 0
  else:
    return 0


# Create a dithered version of the image
def convert_dithering(image):
  # Get size
  width, height = image.size

  # Create new Image and a Pixel Map
  new = create_image(width, height)
  pixels = new.load()

  # Transform to half tones
  for i in range(0, width-1, 2):
    for j in range(0, height-1, 2):
      # Get Pixels
      p1 = get_pixel(image, i, j)
      p2 = get_pixel(image, i, j + 1)
      p3 = get_pixel(image, i + 1, j)
      p4 = get_pixel(image, i + 1, j + 1)

      # Color Saturation by RGB channel
      red   = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
      green = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
      blue  = (p1[2] + p2[2] + p3[2] + p4[2]) / 4

      # Results by channel
      r = [0, 0, 0, 0]
      g = [0, 0, 0, 0]
      b = [0, 0, 0, 0]

      # Get Quadrant Color
      for x in range(0, 4):
        r[x] = get_saturation(red, x)
        g[x] = get_saturation(green, x)
        b[x] = get_saturation(blue, x)

      # Set Dithered Colors
      pixels[i, j]         = (r[0], g[0], b[0])
      pixels[i, j + 1]     = (r[1], g[1], b[1])
      pixels[i + 1, j]     = (r[2], g[2], b[2])
      pixels[i + 1, j + 1] = (r[3], g[3], b[3])

  # Return new image
  return new


# Create a Primary Colors version of the image
def convert_primary(image):
  # Get size
  width, height = image.size

  # Create new Image and a Pixel Map
  new = create_image(width, height)
  pixels = new.load()

  # Transform to primary
  for i in range(width):
    for j in range(height):
      # Get Pixel
      pixel = get_pixel(image, i, j)

      # Get R, G, B values (This are int from 0 to 255)
      red =   pixel[0]
      green = pixel[1]
      blue =  pixel[2]

      # Transform to primary
      if red > 127:
        red = 255
      else:
        red = 0
      if green > 127:
        green = 255
      else:
        green = 0
      if blue > 127:
        blue = 255
      else:
        blue = 0

      # Set Pixel in new image
      pixels[i, j] = (int(red), int(green), int(blue))

  # Return new image
  return new


# Main
  
f_name = 'n2'
f_extension = '.jpg'

if __name__ == "__main__":
  # Load Image (JPEG/JPG needs libjpeg to load)
  original = open_image(f_name + f_extension)

  # Example Pixel Color
  print('Color: ' + str(get_pixel(original, 0, 0)))

  # Convert size 1 and save
  # new = convert_size_1(original)
  # save_image(new, f_name + '_size_1' + f_extension)
  
  # Convert size 2 and save
  new = convert_size_2(original)
  save_image(new, f_name + '_size_2' + f_extension)
  print("OK")

  # Convert to Grayscale and save
  new = convert_grayscale(original)
  save_image(new, f_name + '_gray' + f_extension)

  # Convert to Halftoning and save
  new = convert_halftoning(original)
  # save_image(new, 'Prinny_half.png')
  save_image(new, f_name + '_half' + f_extension)

  # Convert to Dithering and save
  new = convert_dithering(original)
  # save_image(new, 'Prinny_dither.png')
  save_image(new, f_name + '_dither' + f_extension)

  # Convert to Primary and save
  new = convert_primary(original)
  # save_image(new, 'Prinny_primary.png')
  save_image(new, f_name + '_primary' + f_extension)