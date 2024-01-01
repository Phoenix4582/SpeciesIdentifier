import cv2
import numpy as np

"Source code from https://nms.readthedocs.io/en/latest/_modules/nms/fast.html"
def polygon_iou(poly1, poly2, useCV2=False):
    """Computes the ratio of the intersection area of the input polygons to the (sum of polygon areas - intersection area)
    Used with the NMS function

    :param poly1: a polygon described by its verticies
    :type poly1: list
    :param poly2: a polygon describe by it verticies
    :type poly2: list
    :param useCV2: if True (default), use cv2.contourArea to calculate polygon areas. If false use :func:`nms.helpers.polygon_intersection_area`.
    :type useCV2: bool
    :return: The ratio of the intersection area / (sum of rectangle areas - intersection area)
    :rtype: float
    """

    intersection_area = polygon_intersection_area([poly1, poly2])
    if intersection_area == 0:
        return 0

    if(useCV2):
        poly1_area = cv2.contourArea(np.array(poly1, np.int32))
        poly2_area = cv2.contourArea(np.array(poly2, np.int32))
    else:
        poly1_area = polygon_intersection_area([poly1])
        poly2_area = polygon_intersection_area([poly2])

    return intersection_area / (poly1_area + poly2_area - intersection_area)

"Source code from https://nms.readthedocs.io/en/latest/_modules/nms/fast.html"
def nms(boxes, scores, **kwargs):
    """Do Non Maximal Suppression

    As translated from the OpenCV c++ source in
    `nms.inl.hpp <https://github.com/opencv/opencv/blob/ee1e1ce377aa61ddea47a6c2114f99951153bb4f/modules/dnn/src/nms.inl.hpp#L67>`__
    which was in turn inspired by `Piotr Dollar's NMS implementation in EdgeBox. <https://goo.gl/jV3JYS>`_

    This function is not usually called directly.  Instead use :func:`nms.nms.boxes`, :func:`nms.nms.rboxes`,
    or :func:`nms.nms.polygons`

    :param boxes:  the boxes to compare, the structure of the boxes must be compatible with the compare_function.
    :type boxes:  list
    :param scores: the scores associated with boxes
    :type scores: list
    :param kwargs: optional keyword parameters
    :type kwargs: dict (see below)
    :returns: an list of indicies of the best boxes
    :rtype: list
    :kwargs:

    * score_threshold (float): the minimum score necessary to be a viable solution, default 0.3
    * nms_threshold (float): the minimum nms value to be a viable solution, default: 0.4
    * compare_function (function): function that accepts two boxes and returns their overlap ratio, this function must
      accept two boxes and return an overlap ratio
    * eta (float): a coefficient in adaptive threshold formula: \ |nmsi1|\ =eta\*\ |nmsi0|\ , default: 1.0
    * top_k (int): if >0, keep at most top_k picked indices. default:0

    .. |nmsi0| replace:: nms_threshold\ :sub:`i`\

    .. |nmsi1| replace:: nms_threshold\ :sub:`(i+1)`\


    """

    if 'eta' in kwargs:
        eta = kwargs['eta']
    else:
        eta = 1.0
    assert 0 < eta <= 1.0

    if 'top_k' in kwargs:
        top_k = kwargs['top_k']
    else:
        top_k = 0
    assert 0 <= top_k

    if 'score_threshold' in kwargs:
        score_threshold = kwargs['score_threshold']
    else:
        score_threshold = 0.3
    assert score_threshold > 0

    if 'nms_threshold' in kwargs:
        nms_threshold = kwargs['nms_threshold']
    else:
        nms_threshold = 0.4
    assert 0 < nms_threshold < 1

    if 'compare_function' in kwargs:
        compare_function = kwargs['compare_function']
    else:
        compare_function = polygon_iou
    assert compare_function is not None

    if len(boxes) == 0:
        return []

    assert len(scores) == len(boxes)
    assert scores is not None

    # sort scores descending and convert to [[score], [indexx], . . . ]
    scores = get_max_score_index(scores, score_threshold, top_k)

    # Do Non Maximal Suppression
    # This is an interpretation of NMS from the OpenCV source in nms.cpp and nms.
    adaptive_threshold = nms_threshold
    indicies = []

    for i in range(0, len(scores)):
        idx = int(scores[i][1])
        keep = True
        for k in range(0, len(indicies)):
            if not keep:
                break
            kept_idx = indicies[k]
            overlap = compare_function(boxes[idx], boxes[kept_idx])
            keep = (overlap <= adaptive_threshold)

        if keep:
            indicies.append(idx)

        if keep and (eta < 1) and (adaptive_threshold > 0.5):
                adaptive_threshold = adaptive_threshold * eta

    return indicies

"Source code from https://nms.readthedocs.io/en/latest/_modules/nms/helpers.html"
def get_max_score_index(scores, threshold=0, top_k=0, descending=True):
    """ Get the max scores with corresponding indicies

    Adapted from the OpenCV c++ source in `nms.inl.hpp <https://github.com/opencv/opencv/blob/ee1e1ce377aa61ddea47a6c2114f99951153bb4f/modules/dnn/src/nms.inl.hpp#L33>`__

    :param scores: a list of scores
    :type scores: list
    :param threshold: consider scores higher than this threshold
    :type threshold: float
    :param top_k: return at most top_k scores; if 0, keep all
    :type top_k: int
    :param descending: if True, list is returened in descending order, else ascending
    :returns: a  sorted by score list  of [score, index]
    """
    score_index = []

    # Generate index score pairs
    for i, score in enumerate(scores):
        if (threshold > 0) and (score > threshold):
            score_index.append([score, i])
        else:
            score_index.append([score, i])

    # Sort the score pair according to the scores in descending order
    npscores = np.array(score_index)

    if descending:
        npscores = npscores[npscores[:,0].argsort()[::-1]] #descending order
    else:
        npscores = npscores[npscores[:,0].argsort()] # ascending order

    if top_k > 0:
        npscores = npscores[0:top_k]

    return npscores.tolist()

"Source code from https://nms.readthedocs.io/en/latest/_modules/nms/helpers.html"
def polygon_intersection_area(polygons):
    """ Compute the area of intersection of an array of polygons

    :param polygons: a list of polygons
    :type polygons: list
    :return: the area of intersection of the polygons
    :rtype: int
    """
    if len(polygons) == 0:
        return 0

    dx = 0
    dy = 0

    # print(*polygons)

    maxx = np.amax(np.array(polygons)[...,0])
    minx = np.amin(np.array(polygons)[...,0])
    maxy = np.amax(np.array(polygons)[...,1])
    miny = np.amin(np.array(polygons)[...,1])

    if minx < 0:
        dx = -int(minx)
        maxx = maxx + dx
    if miny < 0:
        dy = -int(miny)
        maxy = maxy + dy
    # (dx, dy) is used as an offset in fillPoly

    # print(*[minx, miny, maxx, maxy])
    for i, polypoints in enumerate(polygons):

        newImage = createImage(maxx, maxy, 1)

        polypoints = np.array(polypoints, np.int32)
        polypoints = polypoints.reshape(-1, 1, 2)

        cv2.fillPoly(newImage, [polypoints], (255, 255, 255), cv2.LINE_8, 0, (dx, dy))

        if(i == 0):
            compositeImage = newImage
        else:
            compositeImage = cv2.bitwise_and(compositeImage, newImage)

        area = cv2.countNonZero(compositeImage)

    return area

"Source code from https://nms.readthedocs.io/en/latest/_modules/nms/helpers.html"
def createImage(width=800, height=800, depth=3):
    """ Return a black image with an optional scale on the edge

    :param width: width of the returned image
    :type width: int
    :param height: height of the returned image
    :type height: int
    :param depth: either 3 (rgb/bgr) or 1 (mono).  If 1, no scale is drawn
    :type depth: int
    :return: A zero'd out matrix/black image of size (width, height)
    :rtype: :class:`numpy.ndarray`
    """
    # create a black image and put a scale on the edge

    assert depth == 3 or depth == 1
    assert width > 0
    assert height > 0

    hashDistance = 50
    hashLength = 20

    img = np.zeros((int(height), int(width), depth), np.uint8)

    if(depth == 3):
        for x in range(0, int(width / hashDistance)):
            cv2.line(img, (x * hashDistance, 0), (x * hashDistance, hashLength), (0,0,255), 1)

        for y in range(0, int(width / hashDistance)):
            cv2.line(img, (0, y * hashDistance), (hashLength, y * hashDistance), (0,0,255), 1)

    return img
