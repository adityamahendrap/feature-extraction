import cv2
import numpy as np

def get_shape_features(contours):
    for contour in contours:
        # Ekstraksi fitur bentuk
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        x, y, w, h = cv2.boundingRect(contour)
        if area != 0:
            # Slimness
            slimness = perimeter**2 / (4 * np.pi * area)
            
            # Roundness
            roundness = (4 * area * np.pi) / (perimeter**2)
            
            # Rectangularity
            rectangularity = area / (w * h)
        else:
            slimness = roundness = rectangularity = 0.0
        
        # Narrow Factor
        narrow_factor = w / h if w > h else h / w
        
        # Rasio Keliling dan Diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        ratio_perimeter_diameter = perimeter / equivalent_diameter
        
        # Rasio Perimeter dengan Panjang dan Lebar
        ratio_perimeter_length_width = perimeter / (2 * (w + h))
        
        # Chain Code
        chain_code = get_chain_code(contour)
    
    return {
        'area': f'{area:.3f}',
        'perimeter': f'{perimeter:.3f}',
        'slimness': f'{slimness:.3f}',
        'roundness': f'{roundness:.3f}',
        'rectangularity': f'{rectangularity:.3f}',
        'narrow_factor': f'{narrow_factor:.3f}',
        'ratio_perimeter_diameter': f'{ratio_perimeter_diameter:.3f}',
        'ratio_perimeter_length_width': f'{ratio_perimeter_length_width:.3f}',
        'chain_code': chain_code
    }

def get_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
    
def get_hough_transform(edges):
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    
    if lines is None:
        return edges
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(edges, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return edges

def get_edges_with_robert(img):
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    edges_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
    edges_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)
    edges = np.sqrt(edges_x**2 + edges_y**2)
    # Konversi tepi menjadi format yang dapat diolah oleh findContours
    return np.uint8(edges)

def get_edges_with_sobel(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    # Konversi tepi menjadi format yang dapat diolah oleh findContours
    return np.uint8(edges)

def get_edges_with_prewitt(img):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    edges_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
    edges_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)
    edges = np.sqrt(edges_x**2 + edges_y**2)
    # Konversi tepi menjadi format yang dapat diolah oleh findContours
    edges = np.uint8(edges)
    return edges

def get_edges_with_canny(img):
    return cv2.Canny(img, 30, 100)
      
def get_chain_code(contour):
    # Assuming the contour is closed
    chain_code = []
    for i in range(len(contour) - 1):
        dx = contour[i + 1][0][0] - contour[i][0][0]
        dy = contour[i + 1][0][1] - contour[i][0][1]
        direction = int(np.arctan2(dy, dx) * (8 / (2 * np.pi)))
        chain_code.append(direction)

    return chain_code