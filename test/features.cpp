#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/calib3d.hpp"

using namespace cv;
using std::cout;
using std::endl;

// https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/stardetector.cpp

const char* keys =
    "{ help h |                     | Print help message. }"
    "{ input1 | img1.png            | Path to input image 1. }"
    "{ input2 | img2.png            | Path to input image 2. }";

int main( int argc, char* argv[] ) {
    CommandLineParser parser( argc, argv, keys );
    Mat img1_raw = imread( samples::findFile( parser.get<String>("input1") ), IMREAD_UNCHANGED );
    Mat img2_raw = imread( samples::findFile( parser.get<String>("input2") ), IMREAD_UNCHANGED );
    if ( img1_raw.empty() || img2_raw.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }

    // Convert to grayscale
    Mat img1_gray, img2_gray;
    if (img1_raw.channels() > 1)
        cvtColor(img1_raw, img1_gray, COLOR_BGR2GRAY);
    else
        img1_gray = img1_raw;

    if (img2_raw.channels() > 1)
        cvtColor(img2_raw, img2_gray, COLOR_BGR2GRAY);
    else
        img2_gray = img2_raw;

    // Normalize the images to 8-bit for feature detection
    Mat img1, img2;
    normalize(img1_gray, img1, 0, 255, NORM_MINMAX, CV_8U);
    normalize(img2_gray, img2, 0, 255, NORM_MINMAX, CV_8U);

    //-- Step 1: Detect the keypoints using StarDetector, compute the descriptors
    //Ptr<StarDetector> detector = StarDetector::create();
    //std::vector<KeyPoint> keypoints1, keypoints2;
    //detector->detect(img1, keypoints1);
    //detector->detect(img2, keypoints2);

    // Draw keypoints
    //Mat img_keypoints1;
    //drawKeypoints(img1, keypoints1, img_keypoints1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //imshow("Keypoints Image 1", img_keypoints1);
    //waitKey(0);

    //-- Step 2: Calculate descriptors (feature vectors) using SIFT
    //Ptr<SIFT> extractor = SIFT::create();
    //Mat descriptors1, descriptors2;
    //extractor->compute(img1, keypoints1, descriptors1);
    //extractor->compute(img2, keypoints2, descriptors2);
    
    // Create dummy keypoints1 and keypoints2
    std::vector<KeyPoint> keypoints1, keypoints2;
    // Add some dummy keypoints at random positions
    keypoints1.push_back(KeyPoint(100.0f, 100.0f, 10.0f));
    keypoints1.push_back(KeyPoint(100.0f, 1000.0f, 12.0f));
    keypoints1.push_back(KeyPoint(2000.0f, 200.0f, 15.0f));
    keypoints1.push_back(KeyPoint(2000.0f, 2000.0f, 11.0f));
    
    keypoints2.push_back(KeyPoint(100.0f, 105.0f, 10.0f));
    keypoints2.push_back(KeyPoint(100.0f, 1005.0f, 12.0f));
    keypoints2.push_back(KeyPoint(2000.0f, 205.0f, 15.0f));
    keypoints2.push_back(KeyPoint(2000.0f, 2005.0f, 11.0f));
    
    // Create dummy descriptors1 and descriptors2 (10-dimensional SIFT descriptors)
    Mat descriptors1 = Mat::zeros(keypoints1.size(), 10, CV_32F);
    Mat descriptors2 = Mat::zeros(keypoints2.size(), 10, CV_32F);
    
    // Fill with some dummy values
    for (int i = 0; i < descriptors1.rows; i++) {
        for (int j = 0; j < descriptors1.cols; j++) {
            descriptors1.at<float>(i, j) = static_cast<float>(i * 10 + j);
            descriptors2.at<float>(i, j) = static_cast<float>(i * 10 + j + 1); // Slightly different
        }
    }

    //-- Step 3: Matching descriptor vectors with a FLANN based matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    //-- Filter matches using the Lowe's ratio test
    /*
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i].size() > 1 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    //-- Step 4: Find Homography to filter out outliers
    if (good_matches.size() < 4)
    {
        cout << "Not enough good matches to find homography." << endl;
        return 0;
    }

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );

    std::vector<DMatch> inlier_matches;
    if (!H.empty())
    {
        std::vector<Point2f> obj_corners(1);
        std::vector<Point2f> scene_corners(1);
        for( size_t i = 0; i < good_matches.size(); i++ )
        {
            obj_corners[0] = keypoints1[good_matches[i].queryIdx].pt;
            perspectiveTransform(obj_corners, scene_corners, H);
            if (norm(keypoints2[good_matches[i].trainIdx].pt - scene_corners[0]) < 3.0)
            {
                inlier_matches.push_back(good_matches[i]);
            }
        }
    }
    else
    {
        inlier_matches = good_matches;
    }
    */

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }


    //-- Draw matches
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    imwrite("good_matches.png", img_matches);
    
    // align images based on matches
    cout << "Number of good matches: " << good_matches.size() << endl;
    
    if (good_matches.size() < 4)
    {
        cout << "Not enough good matches to compute homography (need at least 4)." << endl;
        return -1;
    }
    
    //-- Extract matched keypoint locations
    std::vector<Point2f> points1, points2;
    for (size_t i = 0; i < good_matches.size(); i++) {
        points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }
    
    //-- Find homography matrix using RANSAC
    Mat H = findHomography(points2, points1, 0);
    
    if (H.empty())
    {
        cout << "Could not compute homography matrix." << endl;
        return -1;
    }
    
    cout << "Homography matrix:" << endl << H << endl;
    
    //-- Warp img2 to align with img1
    Mat img2_aligned;
    warpPerspective(img2_raw, img2_aligned, H, img1_raw.size());
    
    //-- Save aligned image
    imwrite("img2_aligned.png", img2_aligned);
    cout << "Aligned image saved as 'img2_aligned.png'" << endl;

    return 0;
}
