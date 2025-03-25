import math
import numpy as np
import pandas as pd

# Definig the rotating function

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    Origin and point should be given as a tuple of coordinates: (x,y).
    The angle should be given in radians.
    """

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy


def centroid_calc(polygon,n):

  """
  Calculate centroid of a given polygon.
  N-polygon should be given as an array of tuples (coordinates of n-polygon vertices): [(x1,y1),(x2,y2)...(xn,yn)].
  N (number of vertices) should be given as integer.
  """

  xcentroid,ycentroid = 0,0

  for point in polygon:
    xcentroid += point[0]
    ycentroid += point[1]

  xcentroid = xcentroid/n
  ycentroid = ycentroid/n
  centroid = xcentroid, ycentroid

  return centroid

def area_of_symm_polygon(polygon,centroid,n):

  """
  Calculate area of given n-polygon. Can only be used for symmetrical polygons.
  N-polygon should be given as an array of tuples (coordinates of n-polygon vertices): [(x1,y1),(x2,y2)...(xn,yn)].
  Centroid should be given as a tuple of coordinates: (x,y).
  N (number of vertices) should be given as integer.
  """

  # Calculating R (circumscribed circle radius)
  R = math.sqrt((polygon[0][0]-centroid[0])**2 + (polygon[0][1]-centroid[1])**2) # distance between first polygon point and its centroid

  # Calculating the area of polygon by formula: Sn = (1/2)*(R^2)*n*sin(360/n)
  S = (1/2)*R**2*n*math.sin(math.pi*2/n)

  return S

def check_clockwise(polygon):
    clockwise = -1
    if (sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in zip(polygon, polygon[1:] + [polygon[0]]))) < 0:
        clockwise = 1
    return clockwise

def CSM_calc(polygon):

  """
  Calculate CSM of given n-polygon.
  N-polygon should be given as an array of tuples (coordinates of n-polygon vertices): [(x1,y1),(x2,y2)...(xn,yn)].
  """

  n = len(polygon)  # number of vertices

  # Calculating centroid for n-polygon
  origin = centroid_calc(polygon,n)

  # Checking if the coordinates of polygon in array are given clockwise:
  direction = check_clockwise(polygon)

  # Calculating the rotation angle
  angle = direction*math.pi*2/n   # angle in radians

  # Obtaining intermediate polygon
  intermediate_polygon = []

  # Rotating all points of polygon counterclockwise by angle*i around centroid to create intermediate polygon
  for point in polygon:
    i = polygon.index(point)
    intermediate_point = rotate(origin,point,angle*i)
    intermediate_polygon.append(intermediate_point)

  # Obtaining Cn symmetry

  # Calculating avg of points (point P) in intermediate polygon:
  P = centroid_calc(intermediate_polygon,n)

  # Adding P as one of the points of symmetrical polygon
  symmetrical_polygon = [P]

  # Rotating P (n-1) times clockwise by angle*i around centroid to create symmetrical polygon
  for i in range(1,n):
    point = rotate(origin,P,(-angle)*i)
    symmetrical_polygon.append(point)

  # Calculating CSM

  distance_pow2_sum = 0 # sum of exponated distances

  for i in range(0,n):
    # calculating distance as length of vector - sqrt((x1-x0)**2 + (y1-y0)**2)
    distance = math.sqrt((symmetrical_polygon[i][0]-polygon[i][0])**2 + (symmetrical_polygon[i][1]-polygon[i][1])**2)
    distance_pow2_sum += distance**2

  # Calsulating area of symmetrical polygon - used for further normalization of CSM
  symm_centroid = centroid_calc(symmetrical_polygon,n)
  S = area_of_symm_polygon(symmetrical_polygon,symm_centroid,n)

  CSM = (distance_pow2_sum/(n*S))

  return CSM

def get_regions(vor):

    """
    Filtering and composing regions from scipy.spatial.voronoi.regions
    Creates an array of coordinates of vertices
    Removes regions containing -1 vertices
    """

    vertices_array = np.asarray(vor.vertices)

    regions = []
    for row in vor.regions:
        region_coords = []
        for idx, col in enumerate(row):
            if not math.isnan(col):
                if int(col) != -1:
                    region_coords.append(tuple(vertices_array[int(col)]))
                else:
                    region_coords.append(-1)
        regions.append(region_coords)

    new_regions = []
    for region in regions:
        if len(region) > 0:
            if -1 not in region:
                new_regions.append(region)

    return new_regions

def CSM_for_graph(vor):

    """
    Applies the previous CSM_calc function to an array of regions
    """

    voronoi_regions = get_regions(vor)

    CSM_array = []

    for region in voronoi_regions:
        polygon = []
        for coordinate in region:
            if coordinate != None:
                polygon.append(coordinate)
        CSM_array.append(round(CSM_calc(polygon), 3))

    return CSM_array