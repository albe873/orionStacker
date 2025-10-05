#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

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

    //-- Step 1: Detect the keypoints using AKAZE Detector, compute the descriptors
    Ptr<AKAZE> detector = AKAZE::create();
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );

    //-- Step 2: Matching descriptor vectors with a brute-force matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i].size() > 1 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    //-- Step 3: Find Homography to filter out outliers
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


    //-- Draw matches
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, inlier_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    imshow("Good Matches", img_matches );
    waitKey();
    return 0;
}
