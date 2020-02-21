#!/usr/bin/env python3
#@author: Deniz Sert
#@version: 2/14/20
import math

from PIL import Image





def get_pixel(image, x, y):
    #check to see if pixel in range
    if x<0:
        x = 0
    elif x>=image['width']:
        x=image['width']-1
    
    if y<0:
        y=0
    elif y>=image['height']-1:
        y=image['height']-1
        
    return image['pixels'][x+y*image['width']]


def set_pixel(image, x, y, c):
    image['pixels'][x+y*image['width']] = c


def apply_per_pixel(image, func):
    # add width bc we need to find pixel in a 1D array
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['height']*image['width']*[0]
    }
    for y in range(image['height']):
        for x in range(image['width']):
            color = get_pixel(image, x, y)
            newcolor = func(color)
            set_pixel(result, x, y, newcolor)
    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255-c)


# HELPER FUNCTIONS

def calc_pixel_k(image, x, y, kernel):
    #iterate over kernel
    height = len(kernel)
    width = len(kernel[0])
    new_values = []
    
    for j in range(height):
        for i in range(width):
            new_values.append(kernel[j][i]*get_pixel(image, int(x-(width-1)/2+i), int(y-(height-1)/2+j)))
    return sum(new_values)

def correlate(image, kernel):
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function should have the same form as a 6.009 image (a
    dictionary with 'height', 'width', and 'pixels' keys), but its pixel values
    do not necessarily need to be in the range [0,255], nor do they need to be
    integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    """

    
#    new_img = []
    correlated_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['height']*image['width']*[0]
    }
    
    #iterate over kernel
    for j in range(image['height']):
        for i in range(image['width']):
            set_pixel(correlated_image, i, j, calc_pixel_k(image, i, j, kernel))
    
    return correlated_image

def single_pixel_roundclip(color):
    if color < 0:
        color = 0
    elif color > 255:
        color = 255
    return round(color)


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    pixels = apply_per_pixel(image, single_pixel_roundclip)
    
    return pixels


# FILTERS

#MY HELPER FUNCTION: creates a kernel
    
  


def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    kernel = [[1/n**2]*n]*n

    # then compute the correlation of the input image with that kernel
    final = correlate(image, kernel)

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    return round_and_clip_image(final)

def create_kernel(n):
    kernel = []
    for j in range(n):
        kernel.append([-1/n**2]*n)
    kernel[n//2][n//2] += 2
    return kernel
  
#USE ABOVE FUNCTION BELOW

def sharpened(image, n):
    kernel = create_kernel(n)
    return round_and_clip_image(correlate(image, kernel))

# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES


def edges(image):
    K_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    K_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    edge_result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['height']*image['width']*[0]
    }
    
    o_x = correlate(image, K_x)
    o_y = correlate(image, K_y)
    
    for i in range(len(image['pixels'])):
        edge_result['pixels'][i] = round(math.sqrt(o_x['pixels'][i]**2+o_y['pixels'][i]**2))
    
    return round_and_clip_image(edge_result)

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


  















def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    
    def color_filt(image):
        '''
        Takes a color image as input and produces a filtered color image as output
        '''


        #define 3 greyscale images --> same dim as original im
        
        image_r = {
        'height': image['height'],
        'width': image['width'],
        'pixels': []
        }
        
        image_g = {
        'height': image['height'],
        'width': image['width'],
        'pixels': []
        }
        
        image_b = {
        'height': image['height'],
        'width': image['width'],
        'pixels': []
        }



        list_of_rgb = [image_r, image_g, image_b]
        for i in range(len(image['pixels'])):
            for index in range(3):
                #i = which pixel
                #R, G, or B value within pixel
                list_of_rgb[index]['pixels'].append(image['pixels'][i][index])

            
        
        
        #apply filters to RGB greyscale channels
        image_r = filt(image_r)
        image_g = filt(image_g)
        image_b = filt(image_b)
        
        #construct new image
        final_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': []
        }
       
        
        #recombine
        for i in range(len(image_r['pixels'])):
            final_image['pixels'].append((image_r['pixels'][i], image_g['pixels'][i], image_b['pixels'][i]))
            
        
        return final_image
    return color_filt


def make_blur_filter(n):
    '''
    Takes in n, returns a function with param image.
    '''
    def blurry(image):
        '''
        Blurs an image using a kernel.
        '''
        return blurred(image, n)
    return blurry
        


def make_sharpen_filter(n):
    '''
    Takes in n, returns a function with param image.
    '''
    def sharpen(image):
        '''
        Sharpens an image using a kernel.
        '''
        return sharpened(image, n)
    return sharpen


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    #inner function to apply filter to image
    def apply_filters(image):
        im = image
        for f in filters:
            im = f(im)
        return im
    return apply_filters


# SEAM CARVING

# Main Seam Carving Implementation

#******GIANT WHILE LOOP*******
    #DO HELPER FUNCTIONS FIRST
def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    """
    im_seam_removed = image
    
    for i in range(ncols):
        #convert to greyscale
        grey_im = greyscale_image_from_color_image(im_seam_removed)
        
        #compute energy(edges)
        im_energy = compute_energy(grey_im)
        
        #compute cum energy map
        cum_energy_map = cumulative_energy_map(im_energy)
        
        #find_min_energy_seam
        min_energy_seam = minimum_energy_seam(cum_energy_map)
        
        #remove seam
        im_seam_removed = image_without_seam(im_seam_removed, min_energy_seam)
        
        
    return im_seam_removed


# Optional Helper Functions for Seam Carving

#EASY - WORKS
def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    
    c_im = image
    #constuct new bw image with correct dimensions
    bw_im = {
        'height': c_im['height'],
        'width': c_im['width'],
        'pixels': []
        }
    for pixel in c_im['pixels']:
        bw_im['pixels'].append(round(.299*pixel[0] + .587*pixel[1] + .114*pixel[2]))
    return bw_im
    


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy function),
    computes a "cumulative energy map" as described in the lab 1 writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    pixels = []
    #construct new image
    fin_im = {
    'height': energy['height'],
    'width': energy['width'],
    'pixels': energy['height']*energy['width']*[0]
        }
    
    #1st row
    for column in range(energy['width']):
        set_pixel(fin_im, column, 0, get_pixel(energy, column, 0))
    
    #remaining rows
    for row in range(1, energy['height']):
        for column in range(energy['width']):
            pixels = []
             
             #energy['pixels'] isn't a 2D array? how to index properly into the list?
            
            #pixel above is guaranteed to be in bounds
            pixels.append(get_pixel(fin_im, column, row-1))
            
            #check above and left to see if in bounds
            if column - 1 >= 0:
                pixels.append(get_pixel(fin_im, column-1, row-1))
            
            #check above and right to see if in bounds
            if column + 1 < energy['width']:
                pixels.append(get_pixel(fin_im, column+1, row-1))
             
#            print("Min pix:", min(pixels))
            min_pix = min(pixels)
#            print("Pixels above: ", pixels)
#            print("Get pixel: ", get_pixel(energy, row, column), row, column)
#            print("Energy: ", energy)
            set_pixel(fin_im, column, row, get_pixel(energy, column, row) + min_pix)
#    print(fin_im)
    return fin_im
            

def find_indices(image, x, y):
    '''
    Takes in image, x, y;  returns indices of pixel.
    '''
    return x + y*image['width']
    
def minimum_energy_seam(c):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 1 writeup).
    """
    #sets min to infinity
    min_so_far = float("inf")
    min_col_so_far = 0
    min_seam = []
    temp_col = 0
    c_copy = c.copy()
    #loop through bottom row, find min pixel
    for x in range(c_copy['width']):
        val = get_pixel(c, x, c_copy['height']-1)
        if val < min_so_far:
            min_so_far = val
            min_col_so_far = x
    #add min pixel coordinate to min_seam list
    min_seam.append(find_indices(c_copy, min_col_so_far, c_copy['height']-1))
    
    #similar to cum_energy_map, check upward neighbors of min energy pixel from bottom row
    #our row is y
    for y in range(c_copy['height']-1, 0, -1):
        #find min_pixel AND min_indices
        temp_min = float("inf")
        temp_indices = find_indices(c_copy, min_col_so_far, y)
        
        #finds min_pixel and min_indices of top 3 neighbors
        for i in range(min_col_so_far-1, min_col_so_far+2):
            #bound check
            if i<0:
                continue
            elif i>c_copy['width']-1:
                continue
            
            above = get_pixel(c_copy, i, y-1)
            if above < temp_min:
                temp_min = above
                temp_col = i
                temp_indices = find_indices(c_copy, i, y-1)
                
        min_seam.append(temp_indices)
        min_so_far = temp_min
        #temp_col is temporary: we're only using it for the current_pixel's neighbors to decide the next neighbor
        min_col_so_far = temp_col
        min_so_far = get_pixel(c_copy, temp_indices, y-1)
        
        
        
    return min_seam
        
    
    


def image_without_seam(im, s):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    #create an image
    im_original = im.copy()
    fin_im = {
    'height': im_original['height'],
    'width': im_original['width']-1,
    'pixels': []
        }
    #adds pixels to image as long as they aren't blacklisted
    for i in range(len(im_original['pixels'])):
        if i not in s:
            fin_im['pixels'].append(im_original['pixels'][i])
            
            
            
#    print("Im original width: ", im['width'])
#    print("Im original height: ", im['height'])
#    print("Final image width: ", fin_im['width'])
#    print("Final image height: ", fin_im['height'])
#    print("s: ", s)
#    print("Original im length: ", len(im['pixels']))
#    print("Final im length: ", len(fin_im['pixels']))
    
    return fin_im
    
    

# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()
    
def my_filter(image):
    '''
    This function takes in an image, returns a filtered image with every other 
    pixel being extra bright.
    '''
    
    fin_im = image
    
    #loops index through length of image pixels, skips every other one
    for i in range(0, len(image['pixels']), 2):
        #increase brightness by 3 times for every other pixel
        fin_im['pixels'][i] = fin_im['pixels'][i]*3
        
    #makes sure there is no pixel overflow (pixels >255)
    for i in range(0, len(image['pixels'])):
        if fin_im['pixels'][i] > 255:
           fin_im['pixels'][i] = 255 
    return fin_im
    

    
 
    #*********BLURRED CAT IMAGE**********
#    im_c = load_color_image("test_images/cat.png")
#    im_bw = load_greyscale_image("test_images/cat.png")
#        
#    blur_filter = make_blur_filter(9)
#    blurry2 = blur_filter(im_bw)
#        
#    color_filter = color_filter_from_greyscale_filter(blur_filter)
#    fin_im = color_filter(im_c)
#        
#    save_greyscale_image(blurry2, "my_khat_bw.png")
#    save_color_image(fin_im, "my_khat_c.png")
    
    #**********BLURRED PYTHON IMAGE************
#    im_c = load_color_image("test_images/python.png")
#
#        
#    blur_filter = make_blur_filter(9)
#
#        
#    color_filter = color_filter_from_greyscale_filter(blur_filter)
#    fin_im = color_filter(im_c)
#        
#    save_color_image(fin_im, "my_pyton.png")
#    
#    #**********SHARPENED SPARROW CHICK IMAGE***********
#    im_c = load_color_image("test_images/sparrowchick.png")
#
#        
#    sharpen_filter = make_sharpen_filter(7)
#
#        
#    color_filter = color_filter_from_greyscale_filter(sharpen_filter)
#    fin_im = color_filter(im_c)
#        
#    save_color_image(fin_im, "my_chik.png")
    
#********FILTER CASCADE**********
#im_c = load_color_image("test_images/cat.png")
#f1 = color_filter_from_greyscale_filter(make_blur_filter(9))
#f2 = color_filter_from_greyscale_filter(make_sharpen_filter(7))
#f3 = color_filter_from_greyscale_filter(inverted)
#
#c = filter_cascade([f1, f2, f3])
#cas_im = c(im_c)
#
#save_color_image(cas_im, "my_kascade_khat.png")
    
im_c = load_color_image("test_images/outside.jpeg")
#filter1 = color_filter_from_greyscale_filter(edges)
#filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
#filt = filter_cascade([filter1, filter1, filter2, filter1])
#
#im_c = filt(im_c)
#
#save_color_image(im_c, "my__cascade_frawg.png")
    
    
#********MINIMUM ENERGY SEAMS*********
#Greyscale helper
#im_c = load_greyscale_image("test_images/pattern.png")
#im_bw = greyscale_image_from_color_image(im_c)
 #print(im_bw)
 #save_greyscale_image(im_bw, "my_bw_frawg.png")
    
 #Energy Map helper
#edge_frog_im = compute_energy(im_bw)
#save_greyscale_image(edge_frog_im, "my_edgy_frawg.png")
 
 #Cum Map helper
#cum_frog_im = cumulative_energy_map(edge_frog_im)
#
#save_greyscale_image(edge_frog_im, "my_pattern.png")
 
#MINIMUM SEAM
#im_edges = compute_energy(im_c)
#cum_pattern_im = cumulative_energy_map(im_edges)
#print(minimum_energy_seam(cum_pattern_im))

#print(image_without_seam(im_c, cum_pattern_im))
 
#im_c = load_color_image("test_images/cat.png")
#my_color_filt = color_filter_from_greyscale_filter(my_filter)

#im_c = my_color_filt(im_c)
#print(im_c)
#save_color_image(im_c, "my__own_frawg.png")
save_color_image(seam_carving(im_c, 100), "my_two_khatz.png")


