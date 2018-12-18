import math

import numpy as np

import cv2

import random

np.random.seed(sum([ord(c) for c in 'srivenka']))
def warpImages(img1, img2, H):
	rows1, cols1 = img1.shape[:2]
	rows2, cols2 = img2.shape[:2]

	list_of_points_1 = np.array([[0,0], [0,rows1], [cols1, rows1], [cols1,0]],dtype = np.float32).reshape(-1,1,2)
	temp_points = np.array([[0,0], [0,rows2], [cols2, rows2], [cols2,0]],dtype =np.float32).reshape(-1,1,2)

	list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
	list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

	[x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
	[x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

	translation_dist = [-x_min, -y_min]
	H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

	output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
	output_img[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img1
	return output_img

def sift_features(image,i,j):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image,None)
    resultant_image = 'task' + str(j) + '_sift' + str(i) + '.jpg'
    cv2.imwrite(resultant_image,cv2.drawKeypoints(image,keypoints,image.copy()))
    return keypoints,descriptors

def knn_match_features(image1,image2,keypoints_mountain1,descriptors_mountain1,keypoints_mountain2, descriptors_mountain2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_mountain1, descriptors_mountain2, k=2)

    good_match = []
    good_match_new = []
    good_match_random = []
    for i,j in matches:
        if i.distance < 0.75 * j.distance:
            good_match.append([i])
            good_match_new.append(i)
    knn_image = cv2.drawMatchesKnn(image1, keypoints_mountain1,image2,keypoints_mountain2,good_match,None)
    cv2.imwrite('task1_matches_knn.jpg', knn_image)
    
    if len(good_match_new)> 10:

        src_pts = np.float32([keypoints_mountain1[m.queryIdx].pt for m in good_match_new])
        dst_pts = np.float32([keypoints_mountain2[m.trainIdx].pt for m in good_match_new])
    
    h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #matchesMask = mask.ravel().tolist()
    
    
    

    for i in range(10):
        rand = random.randint(0,len(good_match_new)-1)
        good_match_random.append(good_match_new[rand])
        
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = None, # draw only inliers
                   flags = 2)

    image_match = cv2.drawMatches(image1, keypoints_mountain1,image2,keypoints_mountain2,good_match_random,None,**draw_params)
    cv2.imwrite('task1_matches.jpg',image_match)
    


    return h
    
if __name__ == "__main__":

    #############################################################################

    # Read both the mountain Images and convert it to Grayscale for processing.

    #############################################################################

    mountain_image_1 = cv2.imread('mountain1.jpg')
    mountain_image_2 = cv2.imread('mountain2.jpg')

    mountain_image_1_grey = cv2.imread('mountain1.jpg',0)
    mountain_image_2_grey = cv2.imread('mountain2.jpg',0)
    #############################################################################

    # Extract SIFT Features of the both the mountain_image_1 and mountain_image_2   
    # and draw the keypoints
    #############################################################################

    keypoints_mountain1, descriptors_mountain1 = sift_features(mountain_image_1_grey,1,1)
    keypoints_mountain2, descriptors_mountain2 = sift_features(mountain_image_2_grey,2,1)

    h = knn_match_features(mountain_image_1_grey,mountain_image_2_grey,keypoints_mountain1,descriptors_mountain1,keypoints_mountain2,descriptors_mountain2)

    print(h)
    warpImage = warpImages(mountain_image_2,mountain_image_1,h)
    cv2.imwrite('task1_pano.jpg', warpImage)
